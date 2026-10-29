"""Chromatic and solar-wind noise blocks for the NG20 v2.0 "vanilla" model.

Three per-pulsar signals that plug into ``SuperSignal`` as *extra blocks*
(``is_extra_block = True``).  Each owns a separate column block of the
T-matrix and a bin-resolved, pulsar-diagonal phi block, exactly like ``dm``.

``ChromaticRed``  (model-string name ``chrom``)
    Fourier GP scaled by ``(f_ref / nu)**chi`` -- the scattering / non-DM
    chromatic GP.  Any Atlas PSD (power law by default).

``ChromaticQuad`` (name ``cquad``)
    The chromatic quadratic of enterprise_extensions
    ``chromatic.chromatic_quad_basis``: columns ``(t - t_mid)**k (f_ref/nu)**chi``
    for k = 0, 1, 2, each normalised, with an effectively flat prior.  This is
    the marginalised stand-in for CM0/CM1/CM2 (frozen in the par file for
    noise / GWB runs).  Optionally projects the timing model out first, as the
    discovery implementation does.

``TimeDomainSolarWind`` (name ``sw``)
    Port of PINT ``noise_model.TimeDomainSWNoise``: an interpolation basis on
    nodes (uniform ``TDSWDT`` grid, or explicit ``TDSWNODE_`` MJDs -- which is
    how Bayesian-block edges reach the GP), times the solar-wind geometry and
    the nu^-2 dispersion law, with a RIDGE / SQEXP / MATERN / QUASI_PERIODIC
    kernel on the node coefficients (n_E at 1 AU, cm^-3).

The chromatic index ``chi`` shared by ``chrom`` and ``cquad`` is either fixed
per pulsar at construction or sampled in ``model_maker`` with
``varied_chrom_gp_index=True`` (``SuperSignal.update_chrom_gp_basis``).

Conventions
-----------
* Chromatic scaling uses ``data.dm_ref_freq`` (1400 MHz by default) and the
  per-TOA radio frequencies already stored on ``PTA_Data``.
* All blocks have an even column count (padded with zero columns), because
  Atlas phi is bin-resolved: one value per column pair.  A zero column has no
  data constraint and a finite prior, so it changes nothing.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from scipy import interpolate

from ATLAS.signals.factorized.base import Red
from ATLAS.signals.factorized.utils import make_irn_model
from ATLAS.psd_functions import ridge

__all__ = [
    "ChromaticRed", "ChromaticQuad", "TimeDomainSolarWind",
    "DEFAULT_CHROM_INDEX", "DMCONST_PINT",
    "make_interpolation_basis", "ridge_kernel", "square_exponential_kernel",
    "matern_kernel", "periodic_kernel",
    "sw_geometry_from_pint", "sw_geometry_from_enterprise", "align_to_toas",
    "read_tdsw_par",
]

#: Scattering-like default chromatic index for ``chrom`` / ``cquad``.
DEFAULT_CHROM_INDEX = 4.0

#: PINT's dispersion constant, ``pint.DMconst`` [s MHz^2 cm^3 / pc].  The
#: solar-wind basis uses PINT's value so ``TDSWLOGSIG`` transfers unchanged.
DMCONST_PINT = 1.0 / 2.41e-4


def _per_pulsar(value, data, what, default=None):
    """Broadcast a scalar / sequence / {psr_name: value} to a per-pulsar list."""
    names = [str(n) for n in data.psr_names]   # npz loaders give 0-d arrays
    if value is None:
        return [default] * len(names)
    if isinstance(value, dict):
        value = {str(k): v for k, v in value.items()}
        missing = [n for n in names if n not in value and default is None]
        if missing:
            raise ValueError(f"{what}: no entry for pulsars {missing}")
        return [value.get(n, default) for n in names]
    if not isinstance(value, (list, tuple)) and np.ndim(value) == 0:
        return [value] * len(names)
    if len(value) != len(names):
        raise ValueError(f"{what}: got {len(value)} entries for {len(names)} pulsars")
    return list(value)


def _psr_split(arr_cat, data):
    """Split a TOA-concatenated array back into per-pulsar pieces."""
    counts = [len(t) for t in data.toas]
    return np.split(np.asarray(arr_cat), np.cumsum(counts)[:-1])


# ===========================================================================
# 1. Chromatic Fourier GP
# ===========================================================================

class ChromaticRed(Red):
    """Fourier red-noise block scaled by ``(f_ref / nu)**chrom_index``.

    Identical to ``Red`` (same PSD machinery, frequencies, bounds) except the
    basis rows are chromatically scaled per pulsar.

    Parameters
    ----------
    chrom_index : float, sequence or dict, default 4.0
        Reference chromatic index per pulsar.  With ``varied_chrom_gp_index``
        the basis is re-indexed in the model and this value only sets the
        starting point.
    Other parameters are those of ``Red``.
    """
    is_extra_block = True

    def __init__(self, data, psd_function, lower_bound_psd, upper_bound_psd,
                 nfreqs=30, chrom_index=DEFAULT_CHROM_INDEX, name="chrom",
                 **red_kwargs):
        # Needed by get_basis(), which Red.__init__ calls.
        self.chrom_ref_index = jnp.asarray(
            np.asarray(_per_pulsar(chrom_index, data, "chrom_index"), dtype=float))
        self._psr_ratio = [data.dm_ref_freq / np.asarray(p.freqs, dtype=float)
                           for p in data.psrs]
        super().__init__(name=name, data=data, psd_function=psd_function,
                         lower_bound_psd=lower_bound_psd,
                         upper_bound_psd=upper_bound_psd, nfreqs=nfreqs,
                         **red_kwargs)

    def get_basis(self):
        F = super().get_basis()
        idx = np.asarray(self.chrom_ref_index)
        return [Fp * jnp.asarray(r ** c)[:, None]
                for Fp, r, c in zip(F, self._psr_ratio, idx)]

    def rescale(self, block, chrom_index):
        """Re-index a block built at ``chrom_ref_index`` to ``chrom_index``."""
        rel = (chrom_index - self.chrom_ref_index)[self.data.dm_exploder_idxs]
        return block * (self.data.ref_over_radio_freqs ** rel)[:, None]

    def extra_component(self):
        return dict(name=self.name, psd_func=self.psd_function,
                    helper_dictionary=self.psd_reparam_helper,
                    bins=self.nfreqs, freqs=self.freqs)


# ===========================================================================
# 2. Chromatic quadratic (chrom_quad)
# ===========================================================================

class ChromaticQuad:
    """Marginalised chromatic quadratic, coupled to the chromatic GP's index.

    Columns per pulsar, before padding::

        (t - t_mid)**k * (f_ref / nu)**chi,   k = 0, 1, 2

    with ``t_mid`` the midpoint of that pulsar's TOA span, each column
    normalised to unit Euclidean norm -- the same basis as
    ``enterprise_extensions.chromatic.chromatic_quad_basis``.  A fourth zero
    column pads the block to a column pair.

    Parameters
    ----------
    chrom_index : float, sequence or dict, default 4.0
        Reference index; pass the same value as the ``chrom`` block.
    log10_amp : float, default 0.0
        Fixed prior s.d. of the (normalised-column) coefficients, log10 of
        seconds.  enterprise_extensions uses a variance of 1e80; a unit-norm
        column with a 1 s prior s.d. is equally uninformative for timing data
        but keeps the Cholesky factorisations well conditioned.
    project_timing : bool, default True
        Remove each pulsar's timing-model column space from the basis before
        normalising (the discovery option).  Avoids the near-degeneracy with
        offset / DM / DM1 / DM2 when chi is near 0 or 2; columns that project
        to ~0 are zeroed.
    """
    is_extra_block = True

    def __init__(self, data, chrom_index=DEFAULT_CHROM_INDEX, log10_amp=0.0,
                 project_timing=True, name="cquad", rel_tol=1e-8):
        self.name = name
        self.data = data
        self.npsrs = data.npsrs
        self.nmodes, self.nfreqs = 4, 2
        self.freqs = jnp.ones(self.nfreqs)
        self.project_timing = bool(project_timing)
        self.rel_tol = float(rel_tol)
        self.chrom_ref_index = jnp.asarray(
            np.asarray(_per_pulsar(chrom_index, data, "chrom_index"), dtype=float))

        rows = []
        for t in data.toas:
            t = np.asarray(t, dtype=float)
            tc = t - 0.5 * (t.max() + t.min())
            rows.append(np.stack([np.ones_like(tc), tc, tc ** 2,
                                  np.zeros_like(tc)], axis=1))
        self._T_raw = jnp.asarray(np.concatenate(rows))
        self._row_psr = jnp.asarray(data.dm_exploder_idxs)
        self._ratio = jnp.asarray(data.ref_over_radio_freqs)

        if self.project_timing:
            qs = [np.asarray(U, dtype=float) for U in data.Mmat]
            kmax = max(q.shape[1] for q in qs)
            self._Q = jnp.asarray(np.concatenate(
                [np.pad(q, ((0, 0), (0, kmax - q.shape[1]))) for q in qs]))

        self.psd_function, self.psd_reparam_helper = make_irn_model(
            ridge, jnp.array([]), jnp.array([]),
            fixed_psd_params=("log10_amp",), fixed_psd_param_values=(log10_amp,))

    def basis_from_index(self, chrom_index):
        """Full ``[n_toas_total, 4]`` block for a per-pulsar index (jax)."""
        seg = lambda x: jax.ops.segment_sum(x, self._row_psr, num_segments=self.npsrs)
        B = self._T_raw * (self._ratio ** chrom_index[self._row_psr])[:, None]
        norm2_raw = seg(B ** 2)                                           # [npsr, 4]
        if self.project_timing:
            QtB = seg(self._Q[:, :, None] * B[:, None, :])                # [npsr, K, 4]
            B = B - jnp.einsum("nk,nkc->nc", self._Q, QtB[self._row_psr])
        norm2 = seg(B ** 2)
        # Squared norms throughout, and the sqrt only of a guarded positive
        # value: sqrt'(0) is inf, and the padding column is exactly zero, so an
        # unguarded sqrt makes every gradient through the index NaN.
        keep = norm2 > (self.rel_tol ** 2) * norm2_raw
        scale = jnp.where(keep, jax.lax.rsqrt(jnp.where(keep, norm2, 1.0)), 0.0)
        return B * scale[self._row_psr]

    def get_basis(self):
        B = self.basis_from_index(self.chrom_ref_index)
        return [jnp.asarray(b) for b in _psr_split(B, self.data)]

    def rescale(self, block, chrom_index):
        return self.basis_from_index(chrom_index)

    def extra_component(self):
        return dict(name=self.name, psd_func=self.psd_function,
                    helper_dictionary=self.psd_reparam_helper,
                    bins=self.nfreqs, freqs=self.freqs)


# ===========================================================================
# 3. Time-domain solar-wind GP (port of PINT TimeDomainSWNoise)
# ===========================================================================

def make_interpolation_basis(toas, nodes=None, dt=None, kind="linear"):
    """Port of ``pint.models.noise_model.make_interpolation_basis``.

    ``toas`` in seconds; ``nodes`` in MJD; ``dt`` in days.  Returns the
    interpolation matrix with empty columns dropped, and the kept nodes in
    SECONDS (as PINT does).
    """
    toas = np.asarray(toas, dtype=float)
    if nodes is None:
        if dt is None:
            raise ValueError("Must provide either nodes or dt for linear interpolation basis.")
        t_min, t_max = np.min(toas) / 86400, np.max(toas) / 86400
        nodes = np.arange(t_min, t_max + dt, dt)
    nodes = np.asarray(nodes, dtype=float) * 86400
    basis = np.identity(len(nodes))
    interp = interpolate.interp1d(nodes, basis, kind=kind, axis=0,
                                  bounds_error=False, fill_value=0.0,
                                  assume_sorted=True)
    M = interp(toas)
    idx = M.sum(axis=0) != 0
    if not np.any(idx):
        raise RuntimeError("Interpolation basis has no support in the TOA range. Perhaps check units.")
    return M[:, idx], nodes[idx]


# Kernels: ports of the PINT functions (nodes in seconds, ell in log10 days,
# p in log10 years, including PINT's (sigma/50000)^2 diagonal regulariser).

def ridge_kernel(nodes, log10_sigma=-7):
    return np.eye(len(nodes)) * 10 ** (log10_sigma * 2)


def square_exponential_kernel(nodes, log10_sigma=-7, log10_ell=2):
    nodes = np.asarray(nodes, dtype=np.float64)
    r = np.abs(nodes[None, :] - nodes[:, None])
    l = 10 ** log10_ell * 86400
    sigma = 10 ** log10_sigma
    d = np.eye(r.shape[0]) * (sigma / 50000) ** 2
    return sigma ** 2 * np.exp(-(r ** 2) / 2 / l ** 2) + d


def matern_kernel(nodes, log10_sigma=-7, log10_ell=2, nu=1.5):
    if nu not in (0.5, 1.5, 2.5):
        raise ValueError("matern_kernel currently supports nu in {0.5, 1.5, 2.5}.")
    nodes = np.asarray(nodes, dtype=np.float64)
    r = np.abs(nodes[None, :] - nodes[:, None])
    l = 10 ** log10_ell * 86400
    sigma = 10 ** log10_sigma
    rr = r / l
    if nu == 0.5:
        k = np.exp(-rr)
    elif nu == 1.5:
        c = np.sqrt(3.0)
        k = (1.0 + c * rr) * np.exp(-c * rr)
    else:
        c = np.sqrt(5.0)
        k = (1.0 + c * rr + (5.0 / 3.0) * rr ** 2) * np.exp(-c * rr)
    d = np.eye(r.shape[0]) * (sigma / 50000) ** 2
    return sigma ** 2 * k + d


def periodic_kernel(nodes, log10_sigma=-7, log10_ell=2, log10_gam_p=0, log10_p=0):
    nodes = np.asarray(nodes, dtype=np.float64)
    r = np.abs(nodes[None, :] - nodes[:, None])
    sigma = 10 ** log10_sigma
    l = 10 ** log10_ell * 86400
    p = 10 ** log10_p * 365.25 * 86400
    gam_p = 10 ** log10_gam_p
    d = np.eye(r.shape[0]) * (sigma / 50000) ** 2
    return sigma ** 2 * np.exp(-(r ** 2) / 2 / l ** 2 - gam_p * np.sin(np.pi * r / p) ** 2) + d


_KERNELS = {"RIDGE", "SQEXP", "MATERN", "QUASI_PERIODIC"}


class TimeDomainSolarWind:
    """Time-domain solar-wind GP, a port of PINT ``TimeDomainSWNoise``.

    For pulsar p the block is ``U_p * (G_p * DMconst / nu^2)[:, None] @ L_p``:

    * ``U_p``  interpolation basis on the nodes (``TDSWDT`` grid or explicit
      ``TDSWNODE_`` MJDs; Bayesian-block edges enter this way);
    * ``G_p``  solar-wind geometry in pc -- the DM for n_E = 1 cm^-3, from
      PINT's ``solar_wind_geometry`` (any SWM/SWP) or enterprise_extensions;
    * ``L_p``  Cholesky factor of the UNIT-amplitude kernel on the nodes, or
      the identity for RIDGE.

    The coefficients then have prior ``N(0, sigma^2 I)`` with
    ``sigma = 10**log10_amp`` (= PINT ``TDSWLOGSIG``), so the implied node
    covariance is ``sigma^2 K_unit`` -- PINT's kernel, regulariser included.
    The kernel SHAPE parameters (``log10_ell``, ``nu``, ``log10_gam_p``,
    ``log10_p``) are fixed at construction; the amplitude is sampled (or
    fixed with ``fixed_log10_sigma``).

    Parameters
    ----------
    data : PTA_Data
    sw_geometry : list / dict of arrays
        Per-pulsar geometry [pc], aligned with ``data.toas[p]``.  See
        ``sw_geometry_from_pint`` / ``sw_geometry_from_enterprise`` and
        ``align_to_toas``.
    kernel : str or per-pulsar, default "RIDGE"
    dt_days : float or per-pulsar, default 30.0
        Uniform node spacing (PINT ``TDSWDT``); ignored where nodes are given.
    nodes_mjd : array, per-pulsar list, or {psr: array}, optional
        Explicit nodes (PINT ``TDSWNODE_``), e.g. Bayesian-block edges.
    interp_kind : str, default "linear"
    log10_ell, nu, log10_gam_p, log10_p : float or per-pulsar
        Kernel shape (days, -, -, years), as in PINT.
    lower_bound_psd, upper_bound_psd : sequence of one float
        Prior on ``log10_amp`` (log10 of cm^-3).  PINT notes values of order
        unity are physical.
    fixed_log10_sigma : float, optional
        Hold the amplitude fixed (e.g. at the par-file ``TDSWLOGSIG``).
    radio_freqs : per-pulsar list of arrays, optional
        Override ``psr.freqs`` (PINT uses barycentric radio frequencies).
    """
    is_extra_block = True

    def __init__(self, data, sw_geometry, kernel="RIDGE", dt_days=30.0,
                 nodes_mjd=None, interp_kind="linear", log10_ell=None, nu=1.5,
                 log10_gam_p=None, log10_p=None, lower_bound_psd=(-3.0,),
                 upper_bound_psd=(2.0,), fixed_log10_sigma=None,
                 radio_freqs=None, name="sw"):
        self.name = name
        self.data = data
        self.npsrs = data.npsrs

        geoms = _per_pulsar(sw_geometry, data, "sw_geometry")
        kernels = [str(k).upper() for k in _per_pulsar(kernel, data, "kernel")]
        dts = _per_pulsar(dt_days, data, "dt_days")
        if nodes_mjd is not None and not isinstance(nodes_mjd, dict) \
                and np.ndim(np.asarray(nodes_mjd[0], dtype=object)) == 0:
            nodes_mjd = [nodes_mjd] * self.npsrs        # one node array for all
        nodes = _per_pulsar(nodes_mjd, data, "nodes_mjd", default=False)
        ells = _per_pulsar(log10_ell, data, "log10_ell", default=False)
        nus = _per_pulsar(nu, data, "nu")
        gams = _per_pulsar(log10_gam_p, data, "log10_gam_p", default=False)
        ps = _per_pulsar(log10_p, data, "log10_p", default=False)
        freqs = (_per_pulsar(radio_freqs, data, "radio_freqs") if radio_freqs is not None
                 else [np.asarray(p.freqs, dtype=float) for p in data.psrs])

        blocks, self.nodes_s = [], []
        for i, toas in enumerate(data.toas):
            k = kernels[i]
            if k not in _KERNELS:
                raise ValueError(f"kernel must be one of {sorted(_KERNELS)}, got {k!r}")
            t = np.asarray(toas, dtype=float)
            g = np.asarray(geoms[i], dtype=float)
            if g.shape != t.shape:
                raise ValueError(
                    f"{data.psr_names[i]}: sw_geometry has shape {g.shape}, TOAs {t.shape}; "
                    "align it with align_to_toas().")
            nd = nodes[i]
            if nd is not False and nd is not None:
                U, nsec = make_interpolation_basis(t, nodes=np.sort(np.asarray(nd, float)),
                                                   kind=interp_kind)
            else:
                U, nsec = make_interpolation_basis(t, dt=float(dts[i]), kind=interp_kind)
            dt_dm = g * DMCONST_PINT / np.asarray(freqs[i], dtype=float) ** 2
            B = U * dt_dm[:, None]
            if k != "RIDGE":
                if ells[i] is False or ells[i] is None:
                    raise ValueError(f"{data.psr_names[i]}: kernel {k} needs log10_ell")
                if k == "SQEXP":
                    K = square_exponential_kernel(nsec, 0.0, ells[i])
                elif k == "MATERN":
                    K = matern_kernel(nsec, 0.0, ells[i], nus[i])
                else:
                    if gams[i] is False or ps[i] is False:
                        raise ValueError(f"{data.psr_names[i]}: QUASI_PERIODIC needs log10_gam_p and log10_p")
                    K = periodic_kernel(nsec, 0.0, ells[i], gams[i], ps[i])
                B = B @ np.linalg.cholesky(K)
            blocks.append(B)
            self.nodes_s.append(nsec)

        ncol = max(b.shape[1] for b in blocks)
        ncol += ncol % 2
        self.nmodes, self.nfreqs = ncol, ncol // 2
        self.ncols_per_psr = [b.shape[1] for b in blocks]
        self._blocks = [jnp.asarray(np.pad(b, ((0, 0), (0, ncol - b.shape[1]))))
                        for b in blocks]
        self.freqs = jnp.ones(self.nfreqs)

        if fixed_log10_sigma is not None:
            self.psd_function, self.psd_reparam_helper = make_irn_model(
                ridge, jnp.array([]), jnp.array([]),
                fixed_psd_params=("log10_amp",),
                fixed_psd_param_values=(float(fixed_log10_sigma),))
        else:
            self.psd_function, self.psd_reparam_helper = make_irn_model(
                ridge, jnp.asarray(lower_bound_psd, dtype=float),
                jnp.asarray(upper_bound_psd, dtype=float))

    def get_basis(self):
        return list(self._blocks)

    def extra_component(self):
        return dict(name=self.name, psd_func=self.psd_function,
                    helper_dictionary=self.psd_reparam_helper,
                    bins=self.nfreqs, freqs=self.freqs)


def sw_geometry_from_pint(model, toas):
    """Solar-wind geometry [pc] from PINT, exactly as TimeDomainSWNoise uses it.

    Adds an SWM=0 ``SolarWindDispersion`` (NE_SW = 0, so the timing model is
    unchanged) if the model has none.

    Returns
    -------
    geometry_pc : np.ndarray
    """
    import astropy.units as u
    if "SolarWindDispersion" not in model.components:
        from pint.models.timing_model import Component
        sw = Component.component_types["SolarWindDispersion"]()
        model.add_component(sw, validate=False)
        model["NE_SW"].quantity = 0.0
        model["SWM"].value = 0
    geom = model.solar_wind_geometry(toas).to_value(u.pc)
    return np.asarray(geom)


def sw_geometry_from_enterprise(epsr):
    """Solar-wind geometry [pc] for the 1/r^2 model (PINT SWM=0) from an
    ``enterprise`` Pulsar, via enterprise_extensions.  Already in ``epsr.toas``
    order."""
    from enterprise_extensions.chromatic.solar_wind import theta_impact, dm_solar
    theta, r_earth, _, _ = theta_impact(epsr.planetssb, epsr.sunssb, epsr.pos_t)
    return np.asarray(dm_solar(1.0, theta, r_earth))


def align_to_toas(values, src_toas_s, dst_toas_s, tol_s=1.0):
    """Reorder ``values`` (indexed like ``src_toas_s``) to match ``dst_toas_s``.

    Matches TOAs by sorted order and checks that they agree to ``tol_s``.
    """
    src, dst = np.asarray(src_toas_s, float), np.asarray(dst_toas_s, float)
    if src.shape != dst.shape:
        raise ValueError(f"TOA counts differ: {src.size} vs {dst.size}")
    i_src, i_dst = np.argsort(src), np.argsort(dst)
    worst = np.max(np.abs(src[i_src] - dst[i_dst])) if src.size else 0.0
    if worst > tol_s:
        raise ValueError(f"TOAs do not match (worst offset {worst:.3g} s > {tol_s} s)")
    out = np.empty_like(np.asarray(values, dtype=float))
    out[i_dst] = np.asarray(values, dtype=float)[i_src]
    return out


_TDSW_SCALARS = {"TDSWKERNEL": str, "TDSWDT": float, "TDSWLOGSIG": float,
                 "TDSWLOGELL": float, "TDSWNU": float, "TDSWLOGGAMP": float,
                 "TDSWLOGP": float, "TDSWINTERP_KIND": str}


def read_tdsw_par(parfile):
    """Read the ``TDSW*`` settings from a par file into ``TimeDomainSolarWind``
    keyword names.

    Returns
    -------
    dict with keys ``kernel``, ``dt_days``, ``nodes_mjd`` (sorted, or None),
    ``interp_kind``, ``log10_sigma``, ``log10_ell``, ``nu``, ``log10_gam_p``,
    ``log10_p`` -- absent settings are None (PINT defaults apply).
    """
    vals, nodes = {}, []
    with open(parfile) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            key = parts[0].upper()
            if key in _TDSW_SCALARS and len(parts) > 1:
                vals[key] = _TDSW_SCALARS[key](parts[1].replace("D", "E")
                                               if _TDSW_SCALARS[key] is float else parts[1])
            elif re.fullmatch(r"TDSWNODE_\d+", key) and len(parts) > 1:
                nodes.append(float(parts[1].replace("D", "E")))
    if nodes and "TDSWDT" in vals and vals["TDSWDT"] != 30.0:
        warnings.warn("par file sets both TDSWDT and TDSWNODE_; using the nodes.")
    return dict(
        kernel=vals.get("TDSWKERNEL", "RIDGE").upper(),
        dt_days=vals.get("TDSWDT", 30.0),
        nodes_mjd=np.sort(nodes) if nodes else None,
        interp_kind=vals.get("TDSWINTERP_KIND", "LINEAR").lower(),
        log10_sigma=vals.get("TDSWLOGSIG"),
        log10_ell=vals.get("TDSWLOGELL"),
        nu=vals.get("TDSWNU", 1.5),
        log10_gam_p=vals.get("TDSWLOGGAMP"),
        log10_p=vals.get("TDSWLOGP"),
    )
