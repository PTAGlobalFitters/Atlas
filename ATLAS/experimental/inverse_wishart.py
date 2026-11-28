"""
Inverse-Wishart Gibbs sampling on top of ``SuperSignal.partial_marg_lnposterior``.

Combines the multi-pulsar Gibbs scheme of Laal et al. (arXiv:2410.11944) with the
partially-marginalised posterior of ``ATLAS.signals.factorized.base.SuperSignal``.

Model
-----
The GWB covariance is no longer parametric.  For every GWB frequency bin ``k`` it
is a free, dense ``Npulsars x Npulsars`` matrix ``Phi_k`` with the paper's prior
(Eq. 18)::

    p(Phi) = prod_k |Phi_k|^{-Npulsars}      (IW conjugate prior, nu = Npulsars + 1)

truncated so that ``0.5*log10(Phi_k,II)`` stays inside ``[lower, upper]``.

Everything else (timing model, IRN, DM, GTM) is handled exactly as in
``__partial_marg_lnposterior``: those coefficients are marginalised analytically
and their PSD parameters ``theta`` stay parametric.

Sampler (Metropolis-within-Gibbs, two blocks)
---------------------------------------------
State: ``(theta, g, Phi)`` with ``g`` the GWB Fourier coefficients ``[npsr, 2*crn_bins]``.

1. ``(theta, g) | Phi`` -- HMC in the whitened coordinates ``(theta, z)`` with
   ``g = g_hat + L^{-T} z``, using the log-density of ``__partial_marg_lnposterior``
   with the parametric GWB block replaced by the current ``Phi``.
2. ``Phi | g`` -- exact draw, per bin (Eq. 19)::

       Phi_k | g ~ InverseWishart(Npulsars + 1, S_k),  S_k = g_k^c g_k^c^T + g_k^s g_k^s^T

   via the Bartlett decomposition, with rejection on the diagonal bounds.
3. Bridge -- block 2 must hold ``g`` fixed, not ``z``.  Since the map ``z -> g``
   depends on ``Phi``, ``z`` is recomputed as ``z = L_new^T (g - g_hat_new)``
   after every ``Phi`` update.  Skipping this silently changes ``g`` and breaks
   detailed balance.

Note on the basis
-----------------
With ``"unc+cor->unc"`` the GWB coefficients ``g`` and the IRN coefficients share
columns but are separate latent processes, so ``Phi`` is the *GWB-only*
covariance.  A free ``Phi_k,II`` and a parametric IRN at the same frequency are
degenerate (only their sum is constrained by the auto-power).  Either accept that
(the IRN power-law shape regularises it) or start the IRN above the GWB bins.

Usage
-----
::

    rn = ModelBuilder(data=data).make_red_noise("ltm|unc+cor->unc", ...)   # needs 'cor'
    helpers = rn.get_helpers(reff=..., white_noise_params=...)
    gibbs = IWPartialMargGibbs(rn, half_logphi_ii_lower=-9, half_logphi_ii_upper=-2)
    out = gibbs.run(jax.random.PRNGKey(0), helpers, niter=20000,
                    init_red_params=x0, n_warmup=3000, n_leapfrog=16, thin=5,
                    savedir="chains/iw_pm")
    # out['phi'] / renorm_const -> Phi_k in s^2;  out['theta'] -> IRN/DM/GTM params

Author: built on N. Laal's ``inverse_wishart.py`` (04/11/2024) and ATLAS ``SuperSignal``.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsl

from ATLAS.signals import signals_utils as sutils


class IWPartialMargGibbs:
    """
    Parameters
    ----------
    signal : SuperSignal
        Built with a ``'cor'`` block (anything ``partial_marg_lnposterior`` accepts).
    half_logphi_ii_lower, half_logphi_ii_upper : float
        Bounds on ``0.5*log10(Phi_k,II)`` in seconds (shifted internally by the
        model's ``0.5*log10(renorm_const)``).  The improper prior needs them.
    df : int or None
        Inverse-Wishart degrees of freedom of the conditional.  ``None`` gives the
        paper's ``Npulsars + 1``; change it only if you also change the prior.
    max_rejections : int
        Bartlett re-draws allowed per Gibbs step before keeping the old ``Phi``.
    scale_jitter : (float, float)
        Range of the random diagonal perturbation added to the *correlation*
        matrix of ``S_k`` (paper Sec. on stabilisation).  ``S_k`` has rank 2, so
        it is singular without it whenever ``Npulsars > 2``.
    whitening : {'dense', 'diag'}
        Map from the HMC coordinates ``z`` to ``g`` (see ``_whitening``).  ``'diag'``
        reproduces the library exactly; ``'dense'`` (default) mixes far better
        once ``Phi`` is a free, correlated matrix.
    D_params : tuple or None
        Fixed ``(det_params, psr_phases, psr_dists)`` when the model has a ``det``
        block.  Sampling them is not handled here.
    """

    def __init__(self, signal,
                 half_logphi_ii_lower=-9.0,
                 half_logphi_ii_upper=-2.0,
                 df=None,
                 max_rejections=1000,
                 scale_jitter=(1e-8, 1e-5),
                 whitening='dense',
                 D_params=None):
        if signal.signal_comb_idxs.get('cor') is None:
            raise ValueError("The signal model needs a 'cor' (GWB) block.")
        if signal.has_det and D_params is None:
            raise ValueError("Model has a 'det' block: pass fixed D_params.")

        self.sig = signal
        self.model = signal.model
        self.D_params = D_params
        if whitening not in ('dense', 'diag'):
            raise ValueError("whitening must be 'dense' or 'diag'")
        self.whitening = whitening
        self.npsr = int(self.model.Npulsars)
        self.ncrn = int(self.model.crn_bins)
        self.df = self.npsr + 1 if df is None else int(df)
        self.max_rejections = int(max_rejections)
        self.log_jitter = (float(np.log(scale_jitter[0])), float(np.log(scale_jitter[1])))

        off = float(self.model.logrenorm_offset)
        self.lower_auto = half_logphi_ii_lower + off
        self.upper_auto = half_logphi_ii_upper + off

        # theta = the non-GWB PSD parameters (IRN | DM | GTM); the GWB/ORF tail of
        # the flat vector is replaced by a fixed in-bounds placeholder because
        # ``partial_reparm_helper`` always builds the (here unused) parametric GWB.
        self.n_theta = int(self.model.gtm_end_idx)
        lo = jnp.asarray(self.model.lower_prior_lim_all)
        hi = jnp.asarray(self.model.upper_prior_lim_all)
        self.theta_lower, self.theta_upper = lo[:self.n_theta], hi[:self.n_theta]
        self._gwb_placeholder = 0.5 * (lo[self.n_theta:] + hi[self.n_theta:])

        # Bartlett-decomposition index bookkeeping (as in ``wishart_helper``).
        I, J = np.tril_indices(self.npsr, k=-1)
        self.i_cross, self.j_cross = jnp.asarray(I), jnp.asarray(J)
        self.diag_idx = jnp.arange(self.npsr)
        self.bartlett_dfs = jnp.asarray(self.df - np.arange(self.npsr), dtype=float)
        self.eye_batched = jnp.broadcast_to(jnp.eye(self.npsr), (self.ncrn, self.npsr, self.npsr))

        self.param_names = list(self.model.get_param_names())[:self.n_theta]

    # ------------------------------------------------------------------ #
    #  Parameter / Phi helpers                                             #
    # ------------------------------------------------------------------ #
    def _full_params(self, theta):
        return jnp.concatenate([theta, self._gwb_placeholder])

    def lnprior_theta(self, theta):
        inside = jnp.all((theta > self.theta_lower) & (theta < self.theta_upper))
        return jnp.where(inside, 0.0, -jnp.inf)

    def phi_to_phiinv(self, phi):
        """``Phi [ncrn, np, np]`` -> (``Phi^-1`` at mode resolution ``[2*ncrn, np, np]``,
        ``logdet`` summed over both quadratures), the format
        ``partial_reparm_helper`` returns."""
        cp = jsl.cho_factor(phi, lower=True)
        phiinv = jsl.cho_solve(cp, self.eye_batched)
        logdet = 4.0 * jnp.sum(jnp.log(jnp.diagonal(cp[0], axis1=-2, axis2=-1)))
        return jnp.repeat(phiinv, 2, axis=0), logdet

    def parametric_phi(self, red_params):
        """Dense GWB ``Phi [ncrn, np, np]`` of the parametric model at a full
        ``red_params`` vector -- handy as a starting point."""
        phiinv_G, _, _, _ = self.model.partial_reparm_helper(red_params, self.sig._pad_mask)
        return jnp.linalg.inv(phiinv_G[0::2])

    # ------------------------------------------------------------------ #
    #  The partially-marginalised posterior, GWB block injected            #
    # ------------------------------------------------------------------ #
    def _marg_pieces(self, helpers, theta):
        """Everything in ``__partial_marg_lnposterior`` that does not depend on the
        GWB covariance: ``U, V, W_inv_without_prior``, the P-block normalisation
        and ``rNr``.  Same algebra, same order of operations."""
        TNT, TNr, rNr, logdet_N = helpers
        cor_idx, P_idx, det_idx = self.sig.partial_marg_lnposterior_helper
        GG = TNT[:, *sutils.block_slice(cor_idx)]
        PP = TNT[:, *sutils.block_slice(P_idx)]
        GP = TNT[:, *sutils.block_slice(cor_idx, P_idx)]
        Gr = TNr[:, sutils.vec_slice(cor_idx), None]
        Pr = TNr[:, sutils.vec_slice(P_idx), None]

        _, _, phiinv_P, logdet_phi_P = self.model.partial_reparm_helper(
            self._full_params(theta), self.sig._pad_mask)

        diag_P = jnp.arange(PP.shape[-1])
        Sigma_inv_P = PP.at[:, diag_P, diag_P].add(phiinv_P)
        Sigma_inv_P_chol = jsl.cho_factor(Sigma_inv_P, lower=True)
        I_P = jnp.broadcast_to(jnp.identity(Sigma_inv_P.shape[-1]), Sigma_inv_P.shape)
        Sigma_P = jsl.cho_solve(Sigma_inv_P_chol, I_P)
        logdet_Sigma_inv_P = 2 * jnp.sum(jnp.log(jnp.diagonal(Sigma_inv_P_chol[0], axis1=-2, axis2=-1)))

        U = 0.5 * Pr.mT @ Sigma_P @ Pr
        V = -GP @ Sigma_P @ Pr + Gr
        if self.sig.has_det:
            DD = TNT[:, *sutils.block_slice(det_idx)]
            GD = TNT[:, *sutils.block_slice(cor_idx, det_idx)]
            PD = TNT[:, *sutils.block_slice(P_idx, det_idx)]
            Dr = TNr[:, sutils.vec_slice(det_idx), None]
            d = self.sig.det_signal.get_coeffs_func(*self.D_params)[..., None]
            U = U + 0.5 * d.mT @ PD.mT @ Sigma_P @ PD @ d \
                - Pr.mT @ Sigma_P @ PD @ d - 0.5 * d.mT @ DD @ d + d.mT @ Dr
            V = V + GP @ Sigma_P @ PD @ d - GD @ d

        W_np = -GP @ Sigma_P @ GP.mT + GG
        norm_P = -0.5 * (logdet_N + logdet_phi_P + logdet_Sigma_inv_P)
        return U, V, W_np, norm_P, rNr

    def _whitening(self, W_np, V, phiinv_G):
        """Affine map ``z -> g`` (any invertible one gives the same target in
        ``(theta, g)``; it only changes how hard the HMC geometry is).

        ``'diag'``  -- the library's: per-pulsar ``W_np + diag(Phi^-1)``.  With a
                       free, strongly correlated ``Phi`` the cross-pulsar part of
                       ``Phi^-1`` is left out, so ``z`` is badly conditioned.
        ``'dense'`` -- the exact conditional precision of ``g``,
                       ``blockdiag_psr(W_np) + Phi^-1 (x) I_mode``; ``z | theta, Phi``
                       is then exactly N(0, I).  Costs one Cholesky of size
                       ``npsr*2*crn_bins``.

        Returns ``(to_g, to_z, lndet_Jac)``; ``g`` is ``[npsr, nmode, 1]``.
        """
        npsr, nm = W_np.shape[0], W_np.shape[-1]
        if self.whitening == 'diag':
            idx = jnp.arange(nm)
            W = W_np.at[:, idx, idx].add(jnp.diagonal(phiinv_G, axis1=1, axis2=2).T)
            cf = jsl.cho_factor(W, lower=True)
            L, g_hat = jnp.tril(cf[0]), jsl.cho_solve(cf, V)

            def to_g(z):
                return g_hat + jax.lax.linalg.triangular_solve(
                    L, z[..., None], left_side=True, lower=True, transpose_a=True)

            def to_z(g):
                return (L.mT @ (g - g_hat))[..., 0]
        else:
            # flat index = psr * nm + mode
            W = jnp.zeros((npsr, nm, npsr, nm))
            W = W.at[jnp.arange(npsr), :, jnp.arange(npsr), :].set(W_np)
            m = jnp.arange(nm)
            W = W.at[:, m, :, m].add(phiinv_G)
            W = W.reshape(npsr * nm, npsr * nm)
            cf = jsl.cho_factor(W, lower=True)
            L = jnp.tril(cf[0])
            g_hat = jsl.cho_solve(cf, V.reshape(-1))

            def to_g(z):
                g = g_hat + jsl.solve_triangular(L, z.reshape(-1), lower=True, trans='T')
                return g.reshape(npsr, nm, 1)

            def to_z(g):
                return (L.T @ (g.reshape(-1) - g_hat)).reshape(npsr, nm)
        lndet_Jac = -jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)))
        return to_g, to_z, lndet_Jac

    def logdensity(self, theta, z, phiinv_G, logdet_phi_G, helpers):
        """``log p(theta, z | Phi, data)``.  With ``whitening='diag'`` and ``Phi`` the
        parametric GWB block this equals ``partial_marg_lnposterior`` exactly (plus
        the uniform prior on ``theta``).  Returns ``(log_density, g [npsr, 2*ncrn])``."""
        U, V, W_np, norm_P, rNr = self._marg_pieces(helpers, theta)
        to_g, _, lndet_Jac = self._whitening(W_np, V, phiinv_G)
        g = to_g(z)
        ln_like = U + g.mT @ V - 0.5 * g.mT @ W_np @ g
        lnprior_g = -0.5 * (g.transpose(1, 2, 0) @ phiinv_G @ g.transpose(1, 0, 2)).sum()
        lp = (norm_P - 0.5 * logdet_phi_G + lnprior_g + lndet_Jac
              + jnp.sum(ln_like) - 0.5 * rNr + self.lnprior_theta(theta))
        return lp, g[..., 0]

    def g_to_z(self, theta, g, phiinv_G, helpers):
        """Inverse of the reparameterisation: ``z = L^T (g - g_hat)``."""
        _, V, W_np, _, _ = self._marg_pieces(helpers, theta)
        _, to_z, _ = self._whitening(W_np, V, phiinv_G)
        return to_z(g[..., None])

    # ------------------------------------------------------------------ #
    #  Phi | g  (exact Inverse-Wishart, truncated by rejection)            #
    # ------------------------------------------------------------------ #
    def scale_matrix(self, g):
        """``S_k = g_k^c g_k^c^T + g_k^s g_k^s^T`` -> ``[ncrn, np, np]``."""
        gk = g.reshape(self.npsr, self.ncrn, 2)
        return jnp.einsum('ikq,jkq->kij', gk, gk)

    def _stabilised_chol(self, key, S):
        """Cholesky of ``S`` after perturbing the diagonal of its correlation matrix
        (rank-2 ``S`` is otherwise singular)."""
        d = jnp.sqrt(jnp.diagonal(S, axis1=1, axis2=2))
        corr = S / (d[:, :, None] * d[:, None, :])
        eps = jnp.exp(jr.uniform(key, (self.ncrn, self.npsr),
                                 minval=self.log_jitter[0], maxval=self.log_jitter[1]))
        corr = corr.at[:, self.diag_idx, self.diag_idx].add(eps)
        return jnp.linalg.cholesky(corr) * d[:, :, None]

    def _standard_wishart_chol(self, key):
        """Lower-triangular Bartlett factor ``A`` (``A A^T ~ Wishart(df, I)``)."""
        k1, k2 = jr.split(key)
        A = jnp.zeros((self.ncrn, self.npsr, self.npsr))
        A = A.at[:, self.i_cross, self.j_cross].set(
            jr.normal(k1, (self.ncrn, self.i_cross.shape[0])))
        chi2 = 2.0 * jr.gamma(k2, 0.5 * self.bartlett_dfs, (self.ncrn, self.npsr))
        return A.at[:, self.diag_idx, self.diag_idx].set(jnp.sqrt(chi2))

    def sample_phi_given_g(self, key, g):
        """One truncated IW draw for all bins. ``phi = (L^-T A A^T L^-1)^-1`` with
        ``S = L L^T``.  Returns ``(phi, accepted, n_tries)``; on failure ``phi`` is
        NaN-free garbage and the caller must keep the previous value."""
        kj, kloop = jr.split(key)
        sp = self._stabilised_chol(kj, self.scale_matrix(g))
        spT = sp.mT

        def draw(k):
            A = self._standard_wishart_chol(k)
            yT = jsl.solve_triangular(A, spT, lower=True)       # A^-1 L^T
            phi = yT.mT @ yT
            diags = 0.5 * jnp.log10(jnp.diagonal(phi, axis1=1, axis2=2))
            ok = jnp.all((diags > self.lower_auto) & (diags < self.upper_auto))
            return phi, ok & jnp.all(jnp.isfinite(diags))

        def cond(c):
            _, _, ok, n = c
            return (~ok) & (n < self.max_rejections)

        def body(c):
            k, _, _, n = c
            k, sub = jr.split(k)
            phi, ok = draw(sub)
            return k, phi, ok, n + 1

        k0, sub = jr.split(kloop)
        phi0, ok0 = draw(sub)
        _, phi, ok, n = jax.lax.while_loop(cond, body, (k0, phi0, ok0, 1))
        return phi, ok, n

    # ------------------------------------------------------------------ #
    #  HMC for (theta, z) | Phi                                            #
    # ------------------------------------------------------------------ #
    def _split(self, x):
        return x[:self.n_theta], x[self.n_theta:].reshape(self.npsr, 2 * self.ncrn)

    def _logp_flat(self, x, phiinv_G, logdet_phi_G, helpers):
        theta, z = self._split(x)
        lp, _ = self.logdensity(theta, z, phiinv_G, logdet_phi_G, helpers)
        return lp

    def _hmc_step(self, key, x, phiinv_G, logdet_phi_G, helpers, step, inv_mass, n_leap):
        vg = jax.value_and_grad(self._logp_flat)
        kp, ku = jr.split(key)
        p0 = jr.normal(kp, x.shape) / jnp.sqrt(inv_mass)
        lp0, gr0 = vg(x, phiinv_G, logdet_phi_G, helpers)

        def leap(_, c):
            xx, pp, gg, _ = c
            pp = pp + 0.5 * step * gg
            xx = xx + step * inv_mass * pp
            lpn, gg = vg(xx, phiinv_G, logdet_phi_G, helpers)
            pp = pp + 0.5 * step * gg
            return xx, pp, gg, lpn

        x1, p1, _, lp1 = jax.lax.fori_loop(0, n_leap, leap, (x, p0, gr0, lp0))
        dH = (lp1 - 0.5 * jnp.sum(inv_mass * p1 ** 2)) - (lp0 - 0.5 * jnp.sum(inv_mass * p0 ** 2))
        acc = jnp.where(jnp.isfinite(dH), jnp.minimum(1.0, jnp.exp(dH)), 0.0)
        take = jr.uniform(ku) < acc
        return jnp.where(take, x1, x), acc

    # ------------------------------------------------------------------ #
    #  One full Gibbs sweep                                                #
    # ------------------------------------------------------------------ #
    @partial(jax.jit, static_argnums=(0, 7, 8))
    def sweep(self, key, x, phi, helpers, step, inv_mass, n_leap, n_hmc):
        """``n_hmc`` HMC moves on ``(theta, z) | Phi``, then ``Phi | g`` and the
        ``z`` re-map that keeps ``g`` fixed.  Returns the new state plus the mean
        HMC acceptance, the IW success flag and the number of Bartlett tries."""
        phiinv_G, logdet_G = self.phi_to_phiinv(phi)
        keys = jr.split(key, n_hmc + 1)

        def one(c, k):
            xx, _ = c
            xx, acc = self._hmc_step(k, xx, phiinv_G, logdet_G, helpers, step, inv_mass, n_leap)
            return (xx, acc), acc

        (x, _), accs = jax.lax.scan(one, (x, 0.0), keys[:-1])

        theta, z = self._split(x)
        _, g = self.logdensity(theta, z, phiinv_G, logdet_G, helpers)
        phi_try, ok, ntry = self.sample_phi_given_g(keys[-1], g)
        phi_new = jnp.where(ok, phi_try, phi)
        phiinv_new, _ = self.phi_to_phiinv(phi_new)
        z_new = self.g_to_z(theta, g, phiinv_new, helpers)
        x_new = jnp.concatenate([theta, z_new.ravel()])
        return x_new, phi_new, g, jnp.mean(accs), ok, ntry

    # ------------------------------------------------------------------ #
    #  Driver                                                              #
    # ------------------------------------------------------------------ #
    def run(self, key, helpers, niter, init_red_params=None, init_phi=None,
            n_warmup=1000, n_leapfrog=16, n_hmc_per_gibbs=1, thin=1,
            target_accept=0.8, init_step=1e-2, save_g=False, savedir=None,
            phi_format='full', progress_every=500):
        """
        Parameters
        ----------
        helpers : tuple
            ``(TNT, TNr, rNr, logdet_N)`` from ``signal.get_helpers``.
        init_red_params : array, optional
            Full parametric vector (IRN | DM | GTM | GWB | ORF).  Gives ``theta0`` and,
            unless ``init_phi`` is passed, ``Phi0`` from the parametric GWB.
        init_phi : array ``[crn_bins, npsr, npsr]``, optional
        phi_format : {'full', 'tril', 'diag'}
            What to store per saved sample: the whole matrix, its lower triangle
            incl. the diagonal (``np.tril_indices(npsr)`` order) or the diagonal.
        savedir : str, optional
            Chains are written straight to ``.npy`` memmaps there as they run, so a
            long run neither has to fit in RAM nor loses everything if killed.
        n_warmup : int
            Warm-up sweeps: dual-averaging step size plus a diagonal mass matrix
            from doubling windows (as in Stan).  Use a few thousand for real data.
        Returns
        -------
        dict of numpy arrays: ``theta [n, n_theta]``, ``phi [n, crn_bins, npsr, npsr]``
        (in TNT units; divide by ``renorm_const`` for s^2), ``hmc_accept``,
        ``iw_ok``, ``iw_tries`` and optionally ``g``.
        """
        key, k0 = jr.split(key)
        if init_red_params is None:
            init_red_params = self.model.make_initial_guess(k0)
        init_red_params = jnp.asarray(init_red_params)
        theta = init_red_params[:self.n_theta]
        phi = self.parametric_phi(init_red_params) if init_phi is None else jnp.asarray(init_phi)
        x = jnp.concatenate([theta, jnp.zeros(self.npsr * 2 * self.ncrn)])

        inv_mass = jnp.concatenate([((self.theta_upper - self.theta_lower) / 20.0) ** 2,
                                    jnp.ones(self.npsr * 2 * self.ncrn)])
        step = float(init_step)

        # --- warm-up: Stan-style windowed adaptation -----------------------
        # fast window (step size only) | slow windows doubling in length (diagonal
        # mass from the window's variance, regularised) | final fast window.
        def windows(n):
            if n < 20:
                return []
            init, term = max(int(0.15 * n), 1), max(int(0.1 * n), 1)
            ends, w, start, slow_end = [], 25, init, n - term
            while start < slow_end:
                end = start + w
                if end + 2 * w > slow_end:     # absorb remainder into last window
                    end = slow_end
                ends.append((start, end))
                start, w = end, 2 * w
            return ends
        slow = windows(n_warmup)
        slow_end_at = {e: (s0, e) for s0, e in slow}

        def da_reset(step):
            return dict(mu=np.log(10 * step), hbar=0.0, lbar=0.0, t=0)
        da = da_reset(step)
        buf = []
        for i in range(n_warmup):
            key, sk = jr.split(key)
            x, phi, _, acc, _, _ = self.sweep(sk, x, phi, helpers, step, inv_mass,
                                              n_leapfrog, n_hmc_per_gibbs)
            acc = float(acc) if np.isfinite(float(acc)) else 0.0
            da['t'] += 1
            t = da['t']
            da['hbar'] = (1 - 1 / (t + 10.0)) * da['hbar'] + (target_accept - acc) / (t + 10.0)
            logeps = da['mu'] - np.sqrt(t) / 0.05 * da['hbar']
            da['lbar'] = t ** -0.75 * logeps + (1 - t ** -0.75) * da['lbar']
            step = float(np.exp(logeps))
            if any(s0 <= i < e for s0, e in slow):
                buf.append(np.asarray(x))
            if (i + 1) in slow_end_at:
                n = len(buf)
                var = np.var(np.stack(buf), axis=0)
                var = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))
                inv_mass = jnp.asarray(var)
                buf = []
                step = float(np.exp(da['lbar']))
                da = da_reset(step)
        if n_warmup:
            step = float(np.exp(da['lbar'])) if da['t'] else step

        # --- sampling ------------------------------------------------------
        import os
        nsave = niter // thin
        ti, tj = np.tril_indices(self.npsr)
        dg = np.arange(self.npsr)
        if phi_format == 'full':
            phi_shape, pack = (self.ncrn, self.npsr, self.npsr), (lambda p: p)
        elif phi_format == 'tril':
            phi_shape, pack = (self.ncrn, len(ti)), (lambda p: p[:, ti, tj])
        elif phi_format == 'diag':
            phi_shape, pack = (self.ncrn, self.npsr), (lambda p: p[:, dg, dg])
        else:
            raise ValueError("phi_format must be 'full', 'tril' or 'diag'")

        shapes = dict(theta=((nsave, self.n_theta), float), phi=((nsave,) + phi_shape, float),
                      hmc_accept=((niter,), float), iw_ok=((niter,), bool),
                      iw_tries=((niter,), np.int64))
        if save_g:
            shapes['g'] = ((nsave, self.npsr, 2 * self.ncrn), float)
        if savedir is not None:
            os.makedirs(savedir, exist_ok=True)
            out = {k: np.lib.format.open_memmap(os.path.join(savedir, f'{k}.npy'), mode='w+',
                                                dtype=dt, shape=sh)
                   for k, (sh, dt) in shapes.items()}
        else:
            out = {k: np.empty(sh, dt) for k, (sh, dt) in shapes.items()}

        for i in range(niter):
            key, sk = jr.split(key)
            x, phi, g, acc, ok, ntry = self.sweep(sk, x, phi, helpers, step, inv_mass,
                                                  n_leapfrog, n_hmc_per_gibbs)
            out['hmc_accept'][i], out['iw_ok'][i], out['iw_tries'][i] = acc, ok, ntry
            if (i + 1) % thin == 0:
                j = (i + 1) // thin - 1
                out['theta'][j] = np.asarray(x[:self.n_theta])
                out['phi'][j] = pack(np.asarray(phi))
                if save_g:
                    out['g'][j] = np.asarray(g)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"{i + 1}/{niter}  HMC acc={out['hmc_accept'][:i + 1].mean():.2f}  "
                      f"IW failures={int((~out['iw_ok'][:i + 1]).sum())}  step={step:.3g}",
                      flush=True)
                if savedir is not None:
                    for v in out.values():
                        v.flush()

        out['step_size'], out['inv_mass'] = step, np.asarray(inv_mass)
        out['param_names'] = self.param_names
        if savedir is not None:
            for k in shapes:
                out[k].flush()
            np.save(os.path.join(savedir, 'inv_mass.npy'), out['inv_mass'])
            with open(os.path.join(savedir, 'param_names.txt'), 'w') as f:
                f.write('\n'.join(self.param_names))
        return out