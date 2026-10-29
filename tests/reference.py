"""A dense, float64 reference implementation of ATLAS's likelihood.

This module shares **no code** with ATLAS.  It evaluates

.. math::  \\ln p(r) = -\\tfrac12 \\left[ r^T C^{-1} r + \\ln\\det C
                       + N_{\\rm toa}\\ln 2\\pi \\right],
           \\qquad C = N + T \\phi T^{T}

directly, by forming ``C`` and Cholesky-factorising it.  At two pulsars and a
few hundred TOAs that is a sub-millisecond dense solve, which is the point:
it is slow, obvious, and independent, so it can arbitrate whether ATLAS's
Woodbury/Schur algebra is right.

Conventions -- what ATLAS drops
-------------------------------

ATLAS's three likelihoods all omit constants.  They omit the *same* ones,
which is why they are comparable with each other; but they are **not** equal
to a normalised log-density, and the difference is not negligible.

1. ``-(N_toa / 2) * ln(2*pi)``, from the likelihood -- so ATLAS's value is
   *larger* than a normalised one by ``+(N_toa/2) ln 2pi``.  Dropped by all three of
   ``ln_likelihood_curn``, ``lnposterior_reparam`` and
   ``partial_marg_lnposterior``.  The Gaussian prior's own ``(n_a/2)ln 2pi``
   cancels against the coefficient integral / the standardising transform, so
   this TOA term is the only survivor.  At NG15 scale it is ~5.5e5 nats.

2. On the ``linear_timing=True`` path only: ``-0.5 * n_tm * ln(1e40)`` per
   pulsar, where ``n_tm`` counts *real* (unpadded) timing columns.  The flat
   timing prior enters as ``phi^-1 = 1e-40`` in ``Sigma``, but its
   ``-0.5 ln det phi`` contribution is never added -- ``partial_reparm_helper``
   builds ``logdet_phi_non_gwb`` from the IRN modes alone, and
   ``lnposterior_reparam``'s ``logdet_phimat`` likewise covers only the
   IRN/GWB bins.  Since ``1e40`` is an arbitrary stand-in for a flat prior,
   dropping it is defensible -- but it means the ``ltm`` path's log-density is
   offset from a normalised one by a constant nobody has written down.

Both are constant in every sampled parameter, so neither affects a posterior,
an MCMC acceptance ratio, or a Bayes factor between models fitted to the same
data.  They **do** affect any absolute evidence, any comparison across
different ``N_toa`` (including "with and without J1713"), and any comparison
against a code that normalises properly.

:func:`atlas_offset` computes the predicted difference, and
``tests/test_identities.py`` asserts it rather than assuming it.

Padded timing columns
---------------------

``SuperSignal`` pads every pulsar's design matrix to the array-wide maximum
width.  Padded columns are identically zero in ``T``, and are given prior
precision 1 (not ``1e-40``) so HMC sees unit curvature instead of a flat
direction.  They therefore contribute a factor ``N(0, 1)`` per padded column
to the joint density and *nothing* to the likelihood.  The reference models
this by putting ``1.0`` on the diagonal of ``phi`` for those columns, which
reproduces it exactly: a zero column of ``T`` contributes nothing to
``T phi T^T``, so ``C`` is unchanged.

Model layout
------------

The global coefficient vector is ordered pulsar-major::

    [ psr_0 col_0 ... psr_0 col_{ncol-1} | psr_1 col_0 ... ]

and the global design matrix is ``block_diag(T_0, ..., T_{npsr-1})``.  Cross-
pulsar structure lives entirely in ``phi``, in the GWB bins only.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

__all__ = [
    "FYR",
    "powerlaw_psd",
    "hd_orf",
    "fourier_basis",
    "white_nvec",
    "dense_N",
    "build_phi",
    "marginal_logL",
    "conditional_logL",
    "atlas_offset",
    "chromatic_weight",
    "chrom_quad_basis",
    "interp_basis",
    "sw_kernel",
    "woodbury_marginal_logL",
]

# ATLAS uses ``fref = 1 / (365.25 * 86400)`` in ``psd_functions``.
FYR = 1.0 / (365.25 * 24 * 60 * 60)


# --------------------------------------------------------------------------- #
#  Spectral and geometric ingredients, written from the textbook definitions
# --------------------------------------------------------------------------- #

def powerlaw_psd(f, df, log10_A, gamma):
    """Power-law PSD in s^2, matching ``ATLAS.psd_functions.powerlaw``.

    ``P(f) = df * A^2 / (12 pi^2 f^3) * (f / f_yr)^(3 - gamma)``
    """
    f = np.atleast_1d(np.asarray(f, dtype=np.float64))
    return df * 10.0 ** (2 * log10_A) / (12 * np.pi**2 * f**3) * (f / FYR) ** (3 - gamma)


def hd_orf(zeta):
    """Hellings--Downs correlation for pulsar separation ``zeta`` (radians).

    Normalised so that the *off-diagonal* value tends to 0.5 as ``zeta -> 0``
    and the autocorrelation is 1.  Written from
    ``Gamma(zeta) = 1.5 x ln x - 0.25 x + 0.5``, ``x = (1 - cos zeta) / 2``.
    """
    zeta = np.asarray(zeta, dtype=np.float64)
    x = (1.0 - np.cos(zeta)) / 2.0
    out = np.where(
        x > 0.0,
        1.5 * x * np.log(np.where(x > 0.0, x, 1.0)) - 0.25 * x + 0.5,
        1.0,
    )
    return out


def fourier_basis(toas, freqs):
    """``[sin(2 pi f_0 t), cos(2 pi f_0 t), sin(2 pi f_1 t), ...]``.

    Columns are interleaved per frequency, matching
    ``signals_utils.get_fourier_design_matrix``; the ``jnp.repeat(phi, 2)``
    used throughout ATLAS assumes exactly this ordering.
    """
    toas = np.asarray(toas, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    phase = 2 * np.pi * toas[:, None] * freqs[None, :]
    F = np.empty((toas.size, 2 * freqs.size), dtype=np.float64)
    F[:, 0::2] = np.sin(phase)
    F[:, 1::2] = np.cos(phase)
    return F


# --------------------------------------------------------------------------- #
#  The white-noise covariance, formed densely
# --------------------------------------------------------------------------- #

def white_nvec(toaerrs, backend_idx, efac, log10_equad):
    """``nvec = efac^2 (sigma^2 + equad^2)``, per TOA.

    Matches ``SinglePulsarWhiteCov.get_nvec_jvec``, which applies EFAC to the
    EQUAD as well (the "t2equad" convention).
    """
    toaerrs = np.asarray(toaerrs, dtype=np.float64)
    efac = np.asarray(efac, dtype=np.float64)
    eq2 = 10.0 ** (2.0 * np.asarray(log10_equad, dtype=np.float64))
    return efac[backend_idx] ** 2 * (toaerrs**2 + eq2[backend_idx])


def dense_N(nvec, epochs=None, jvec=None):
    """``N = diag(nvec) + sum_e jvec_e u_e u_e^T``, formed densely.

    ``epochs`` is a sequence of index arrays (one per ECORR epoch) and
    ``jvec`` the matching per-epoch variances.  Pass ``epochs=None`` for the
    diagonal case, which is what ``include_ecorr=False`` produces.
    """
    N = np.diag(np.asarray(nvec, dtype=np.float64))
    if epochs is None or jvec is None:
        return N
    for idx, j in zip(epochs, np.atleast_1d(jvec)):
        idx = np.asarray(idx, dtype=int)
        N[np.ix_(idx, idx)] += j
    return N


# --------------------------------------------------------------------------- #
#  The prior covariance over coefficients
# --------------------------------------------------------------------------- #

def build_phi(
    npsr,
    ncol,
    n_tm,
    tm_widths,
    irn_psd,
    gwb_psd=None,
    orf=None,
    dm_psd=None,
    gtm_psd=None,
    tm_prior=1e40,
    pad_prior=1.0,
):
    """Assemble the global ``phi``, pulsar-major, shape ``(npsr*ncol,)^2``.

    Parameters
    ----------
    n_tm : int
        Padded timing-block width (``linear_timing_model_size``); 0 for a
        model string with no ``ltm`` prefix.
    tm_widths : sequence of int
        Per-pulsar *real* timing-column counts; columns beyond these are
        padding and receive ``pad_prior``.
    irn_psd : (nbin_irn, npsr) array
        Per-pulsar IRN PSD at bin resolution.
    gwb_psd : (nbin_gwb,) array or None
        Common PSD at bin resolution.
    orf : (npsr, npsr) array or None
        Overlap reduction function, ``orf[i, i] == 1``.

    dm_psd : (nbin_dm, npsr) array or None
        Per-pulsar DM PSD at bin resolution, occupying its own block after the
        IRN block -- the layout ``";dm"`` produces.
    gtm_psd : (n_gtm_modes, npsr) array or None
        Adaptus prior variance, already at *mode* resolution (a directly
        supplied ``gtm_psd`` is not duplicated across quadratures), in its own
        block after IRN and DM.

    The GWB occupies the *first* ``2 * nbin_gwb`` Fourier columns, nested at
    the head of the IRN block -- the layout ``"unc+cor->unc"`` produces.
    """
    n = npsr * ncol
    phi = np.zeros((n, n), dtype=np.float64)

    # Timing block: 1e-40 precision (i.e. 1e40 variance) on real columns,
    # unit variance on padded ones.
    for p in range(npsr):
        base = p * ncol
        for c in range(n_tm):
            phi[base + c, base + c] = tm_prior if c < tm_widths[p] else pad_prior

    # IRN: block-diagonal, each bin duplicated across its two quadrature modes.
    irn_modes = np.repeat(np.asarray(irn_psd, dtype=np.float64), 2, axis=0)
    n_irn_modes = irn_modes.shape[0]
    for p in range(npsr):
        base = p * ncol + n_tm
        phi[base:base + n_irn_modes, base:base + n_irn_modes] += np.diag(irn_modes[:, p])

    # DM and Adaptus: separate blocks after the IRN block, in that order.
    offset = n_irn_modes
    if dm_psd is not None:
        dm_modes = np.repeat(np.asarray(dm_psd, dtype=np.float64), 2, axis=0)
        for p in range(npsr):
            base = p * ncol + n_tm + offset
            phi[base:base + dm_modes.shape[0], base:base + dm_modes.shape[0]] += \
                np.diag(dm_modes[:, p])
        offset += dm_modes.shape[0]
    if gtm_psd is not None:
        gtm = np.asarray(gtm_psd, dtype=np.float64)
        for p in range(npsr):
            base = p * ncol + n_tm + offset
            phi[base:base + gtm.shape[0], base:base + gtm.shape[0]] += np.diag(gtm[:, p])
        offset += gtm.shape[0]

    # GWB: nested at the head of the IRN block, with ORF cross-terms.
    if gwb_psd is not None:
        gwb_modes = np.repeat(np.asarray(gwb_psd, dtype=np.float64), 2)
        orf = np.eye(npsr) if orf is None else np.asarray(orf, dtype=np.float64)
        for i in range(npsr):
            for j in range(npsr):
                bi = i * ncol + n_tm
                bj = j * ncol + n_tm
                for m, val in enumerate(gwb_modes):
                    phi[bi + m, bj + m] += orf[i, j] * val
    return phi


# --------------------------------------------------------------------------- #
#  The likelihoods
# --------------------------------------------------------------------------- #

def marginal_logL(r, T_blocks, phi, N, include_2pi=True):
    """``ln p(r)`` for ``r ~ N(0, N + T phi T^T)``, formed densely."""
    r = np.asarray(r, dtype=np.float64)
    T = block_diag(*[np.asarray(t, dtype=np.float64) for t in T_blocks])
    C = np.asarray(N, dtype=np.float64) + T @ np.asarray(phi, dtype=np.float64) @ T.T
    cf = cho_factor(C, lower=True)
    quad = r @ cho_solve(cf, r)
    logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
    out = -0.5 * (quad + logdet)
    if include_2pi:
        out -= 0.5 * r.size * np.log(2 * np.pi)
    return float(out)


def conditional_logL(r, T_blocks, a, phi, N, include_2pi=True):
    """``ln p(r | a) + ln p(a)`` -- the un-marginalised joint."""
    r = np.asarray(r, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    T = block_diag(*[np.asarray(t, dtype=np.float64) for t in T_blocks])
    phi = np.asarray(phi, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)

    resid = r - T @ a
    cfN = cho_factor(N, lower=True)
    ll = -0.5 * (resid @ cho_solve(cfN, resid) + 2.0 * np.sum(np.log(np.diag(cfN[0]))))

    cfp = cho_factor(phi, lower=True)
    lp = -0.5 * (a @ cho_solve(cfp, a) + 2.0 * np.sum(np.log(np.diag(cfp[0]))))

    if include_2pi:
        ll -= 0.5 * r.size * np.log(2 * np.pi)
        lp -= 0.5 * a.size * np.log(2 * np.pi)
    return float(ll + lp)


def atlas_offset(ntoa, tm_widths=(), tm_prior=1e40):
    """Predicted ``ATLAS - reference``, i.e. ``atlas == reference + offset``.

    ATLAS omits negative terms, so its log-density is *larger* than a
    normalised one; the offset is positive.

    ``ntoa`` is the total TOA count; ``tm_widths`` the per-pulsar count of
    *real* timing columns (empty when the model string has no ``ltm``).
    """
    off = 0.5 * ntoa * np.log(2 * np.pi)
    off += 0.5 * np.sum(tm_widths) * np.log(tm_prior)
    return float(off)


def precision_logdet(T_blocks, N_blocks, phiinv_diag):
    """``0.5 * ln det(Sigma^-1)`` for the block-diagonal posterior precision.

    ``Sigma^-1_p = T_p^T N_p^-1 T_p + diag(phiinv_diag[p])``, which is what
    ATLAS Cholesky-factorises to build its standardising transform.  The
    Jacobian of that transform is ``lndet_Jac = -0.5 ln det(Sigma^-1)``, so
    this is the term to add back when comparing a reparameterised density
    against :func:`conditional_logL`.
    """
    total = 0.0
    for T, N, pinv in zip(T_blocks, N_blocks, phiinv_diag):
        T = np.asarray(T, dtype=np.float64)
        cfN = cho_factor(np.asarray(N, dtype=np.float64), lower=True)
        S = T.T @ cho_solve(cfN, T) + np.diag(np.asarray(pinv, dtype=np.float64))
        cf = cho_factor(S, lower=True)
        total += np.sum(np.log(np.diag(cf[0])))
    return float(total)


def epochs_from_toas(toas, backend_flags, dt=1.0):
    """Group TOA indices into ECORR epochs.

    An independent reimplementation of the rule ``nMatrix._get_psr_WN_helpers``
    documents: split by backend first, then group TOAs whose sorted adjacent
    separation is below ``dt``, then drop any epoch holding a single TOA (a
    one-TOA epoch makes ECORR exactly degenerate with EQUAD, so ATLAS discards
    it and its ``jvec`` entry).

    Returns ``(epochs, backend_of_epoch)``.
    """
    toas = np.asarray(toas, dtype=np.float64)
    flags = np.asarray(backend_flags)
    epochs, which = [], []
    for b in np.unique(flags):
        idx = np.flatnonzero(flags == b)
        order = idx[np.argsort(toas[idx])]
        breaks = np.flatnonzero(np.diff(toas[order]) >= dt) + 1
        for grp in np.split(order, breaks):
            if grp.size > 1:
                epochs.append(np.sort(grp))
                which.append(b)
    return epochs, np.array(which)


# --------------------------------------------------------------------------- #
#  Chromatic and solar-wind ingredients (NG20 v2.0 vanilla noise model)
# --------------------------------------------------------------------------- #
#
# Written from the definitions in enterprise_extensions
# (``chromatic.chromatic_quad_basis``) and PINT (``noise_model.TimeDomainSWNoise``
# and its kernel functions), not from ATLAS.  The interpolation basis is built
# by hand rather than through scipy, so it is independent of the scipy call
# ATLAS shares with PINT.

def chromatic_weight(freqs, chi, nu_ref=1400.0):
    """``(nu_ref / nu)**chi`` per TOA."""
    return (nu_ref / np.asarray(freqs, dtype=np.float64)) ** float(chi)


def chrom_quad_basis(toas, freqs, chi, timing_basis=None, nu_ref=1400.0):
    """enterprise_extensions ``chromatic_quad_basis``, optionally projected.

    ``q_k = (t - t_mid)**k (nu_ref/nu)**chi``, k = 0, 1, 2, with ``t_mid`` the
    midpoint of the TOA range; then ``q_k <- (I - U U^T) q_k`` when an
    orthonormal ``timing_basis`` U is given; then each column to unit norm.
    """
    t = np.asarray(toas, dtype=np.float64)
    t0 = 0.5 * (t.max() + t.min())
    w = chromatic_weight(freqs, chi, nu_ref)
    q = np.stack([(t - t0) ** k * w for k in range(3)], axis=1)
    if timing_basis is not None:
        U = np.asarray(timing_basis, dtype=np.float64)
        q = q - U @ (U.T @ q)
    return q / np.linalg.norm(q, axis=0)


def interp_basis(toas_s, dt_days=None, nodes_mjd=None):
    """Linear-interpolation ("tent") basis of PINT ``make_interpolation_basis``.

    Nodes: ``nodes_mjd`` if given, else ``arange(t_min, t_max + dt, dt)`` in
    days, exactly as PINT builds them.  Column j is 1 at node j, falls linearly
    to 0 at its neighbours, and is 0 outside the node range.  Empty columns are
    dropped.  Returns ``(L, nodes_s)``.
    """
    t = np.asarray(toas_s, dtype=np.float64)
    if nodes_mjd is None:
        nodes = np.arange(t.min() / 86400, t.max() / 86400 + dt_days, dt_days)
    else:
        nodes = np.sort(np.asarray(nodes_mjd, dtype=np.float64))
    n = nodes * 86400
    L = np.zeros((t.size, n.size))
    for j in range(n.size):
        if j > 0:
            sel = (t >= n[j - 1]) & (t <= n[j])
            L[sel, j] = (t[sel] - n[j - 1]) / (n[j] - n[j - 1])
        if j < n.size - 1:
            sel = (t >= n[j]) & (t < n[j + 1])
            L[sel, j] = (n[j + 1] - t[sel]) / (n[j + 1] - n[j])
    # a TOA exactly on an interior node belongs to that node alone
    keep = L.sum(axis=0) != 0
    return L[:, keep], n[keep]


def sw_kernel(nodes_s, kind, log10_sigma, log10_ell=None, nu=1.5,
              log10_gam_p=None, log10_p=None):
    """PINT ``TimeDomainSWNoise`` node covariance, regulariser included.

    ``ell`` in log10 days, ``p`` in log10 years, ``sigma`` in cm^-3.  All
    non-RIDGE kernels carry PINT's ``(sigma / 50000)**2`` diagonal term.
    """
    x = np.asarray(nodes_s, dtype=np.float64)
    tau = np.abs(x[:, None] - x[None, :])
    s2 = 10.0 ** (2 * log10_sigma)
    if kind == "RIDGE":
        return s2 * np.eye(x.size)
    ell = 10.0 ** log10_ell * 86400
    if kind == "SQEXP":
        k = np.exp(-tau ** 2 / (2 * ell ** 2))
    elif kind == "MATERN":
        r = tau / ell
        if nu == 0.5:
            k = np.exp(-r)
        elif nu == 1.5:
            k = (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
        else:
            k = (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)
    elif kind == "QUASI_PERIODIC":
        p = 10.0 ** log10_p * 365.25 * 86400
        k = np.exp(-tau ** 2 / (2 * ell ** 2)
                   - 10.0 ** log10_gam_p * np.sin(np.pi * tau / p) ** 2)
    else:
        raise ValueError(kind)
    return s2 * k + np.eye(x.size) * s2 / 50000 ** 2


def woodbury_marginal_logL(r, T_blocks, phi, N, include_2pi=True):
    """``ln p(r)`` for ``r ~ N(0, N + T phi T^T)``, through Woodbury.

    Same quantity as :func:`marginal_logL`.  Needed once ``phi`` holds a
    near-flat block (the chromatic quadratic: prior s.d. 1 s on unit-norm
    columns): forming ``C`` densely then costs ~1e-3 nats in float64, while
    ``N^-1``, ``phi^-1`` and ``Sigma^-1 = T^T N^-1 T + phi^-1`` all stay well
    conditioned.
    """
    r = np.asarray(r, dtype=np.float64)
    T = block_diag(*[np.asarray(t, dtype=np.float64) for t in T_blocks])
    cN = cho_factor(np.asarray(N, dtype=np.float64), lower=True)
    NiT, Nir = cho_solve(cN, T), cho_solve(cN, r)
    cp = cho_factor(np.asarray(phi, dtype=np.float64), lower=True)
    S = T.T @ NiT + cho_solve(cp, np.eye(T.shape[1]))
    cS = cho_factor(S, lower=True)
    b = T.T @ Nir
    quad = r @ Nir - b @ cho_solve(cS, b)
    logdet = 2.0 * (np.sum(np.log(np.diag(cN[0]))) + np.sum(np.log(np.diag(cp[0])))
                    + np.sum(np.log(np.diag(cS[0]))))
    out = -0.5 * (quad + logdet)
    if include_2pi:
        out -= 0.5 * r.size * np.log(2 * np.pi)
    return float(out)