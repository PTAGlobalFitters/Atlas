"""The white-noise solver after the speed-ups.

1. ``left^T N^{-1} right`` is now formed by applying N^{-1} (segment-summed
   Sherman-Morrison) to the narrower side and taking one matrix product, the
   timing-marginalised solve goes through the same route, and
   ``get_red_helpers`` takes TNT, TNr and rNr from ONE Gram per pulsar.
2. ``get_static_helpers`` / ``get_red_helpers_from_static``: with T and the
   white noise fixed, each step forms only T^T N^{-1} r and r^T N^{-1} r.

Pinned against a dense float64 reference (``harness.noise_blocks``; for the
marginalised path D^{-1} = N^{-1} - N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1}
and log|D| = log|N| + log|M^T N^{-1} M| + Mprior, the convention ATLAS uses),
for ECORR, marginalised, diagonal and no-epoch pulsars; and
``model_maker`` with a sampled timing model gives the same density and
gradient through the static path as through the full rebuild.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from numpyro.infer.util import log_density

from . import harness as H
from .fixtures.synthetic import make_synth_pta
from ATLAS.data import PTA_Data
from ATLAS.model import model_maker
from ATLAS.nMatrix.base import WhiteCov

TIGHT = 1e-10


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


def setup(kind):
    """(psrs, WhiteCov, wn_vec, include_ecorr)."""
    if kind == "no-epochs":
        psrs = make_synth_pta(2, toas_per_session=1)
        ecorr = False
    else:
        psrs, ecorr = H.load_psrs("synth", 2), True
    data = PTA_Data(psrs, num_gwb_bins=4, num_irn_bins=6, linear_timing=True,
                    marg_timing=(kind == "marg"), diag_white_cov=(kind == "diag"),
                    dm_ref_freq=H.DM_REF_FREQ)
    wn = WhiteCov(data=data, stabilize_TNT=False, include_ecorr=ecorr)
    wn_vec = None if kind == "diag" else H.white_noise_vector(psrs, ecorr)
    return psrs, wn, wn_vec, ecorr


def dense_inv_and_logdet(cov, p, kind, ecorr):
    """Dense C^{-1} and ATLAS's log|C| for one pulsar."""
    if kind == "diag":
        N = np.diag(np.asarray(p.toaerrs, float) ** 2)
    else:
        N = H.noise_blocks(SimpleNamespace(psrs=[p], include_ecorr=ecorr))[0]
    Ni = np.linalg.inv(N)
    logdet = np.linalg.slogdet(N)[1]
    if cov.marg:
        M = np.asarray(cov.Mmat, float)
        K = M.T @ Ni @ M
        Ni = Ni - Ni @ M @ np.linalg.solve(K, M.T @ Ni)
        logdet = logdet + np.linalg.slogdet(K)[1] + float(cov.Mprior)
    return Ni, logdet


def basis(psrs, ncol=9, seed=0):
    rng = np.random.default_rng(seed)
    f = np.arange(1, ncol // 2 + 1) / 3e8
    blocks = [np.hstack([H.ref.fourier_basis(p.toas, f),
                         rng.normal(size=(len(p.toas), ncol - 2 * f.size))]) for p in psrs]
    return jnp.asarray(np.vstack(blocks))


KINDS = ["ecorr", "marg", "diag", "no-epochs"]


# --------------------------------------------------------------------------- #
#  1. Solves and full helpers against the dense reference
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind", KINDS)
def test_get_red_helpers_matches_dense(kind):
    psrs, wn, wn_vec, ecorr = setup(kind)
    T = basis(psrs)
    r = jnp.concat([jnp.asarray(p.residuals) for p in psrs])
    TNT, TNr, rNr, logdet = wn.get_red_helpers(T, r, wn_vec)
    T, r = np.asarray(T), np.asarray(r)
    want_rNr = want_logdet = 0.0
    for i, (cov, s, e) in enumerate(zip(wn.cov_matrices, wn.toa_starts, wn.toa_ends)):
        Ci, ld = dense_inv_and_logdet(cov, psrs[i], kind, ecorr)
        assert rel(TNT[i], T[s:e].T @ Ci @ T[s:e]) < TIGHT
        assert rel(TNr[i], T[s:e].T @ Ci @ r[s:e]) < TIGHT
        want_rNr += r[s:e] @ Ci @ r[s:e]
        want_logdet += ld
    assert rel(rNr, want_rNr) < TIGHT and rel(logdet, want_logdet) < TIGHT
    assert np.array_equal(np.asarray(TNT), np.swapaxes(np.asarray(TNT), 1, 2))  # symmetric


@pytest.mark.parametrize("kind", ["ecorr", "marg"])
@pytest.mark.parametrize("shape", [(3, 11), (11, 3), (5, 5)])
def test_solve_matches_dense_either_side_narrow(kind, shape):
    """The narrower side gets N^{-1}; both orientations give the same matrix."""
    psrs, wn, wn_vec, ecorr = setup(kind)
    cov, p = wn.cov_matrices[0], psrs[0]
    wh = cov.get_nvec_jvec(wn_vec[:cov.n_params])
    rng = np.random.default_rng(1)
    L = rng.normal(size=(len(p.toas), shape[0]))
    R = rng.normal(size=(len(p.toas), shape[1]))
    Ci, ld = dense_inv_and_logdet(cov, p, kind, ecorr)
    got, got_ld = cov.solve_with_logdet(wh, jnp.asarray(L), jnp.asarray(R))
    assert rel(got, L.T @ Ci @ R) < TIGHT and rel(got_ld, ld) < TIGHT
    assert rel(cov.apply_inv(wh, jnp.asarray(R)), Ci @ R) < TIGHT


def test_epoch_index_covers_exactly_the_ecorr_epochs():
    psrs, wn, _, _ = setup("ecorr")
    for cov in wn.cov_matrices:
        idx = np.asarray(cov.toa_epoch)
        for b in range(cov.nepochs):
            members = np.asarray(cov.U_pad[b])[np.asarray(cov.U_mask[b]) > 0]
            assert set(np.flatnonzero(idx == b)) == set(members)
        assert np.all((idx >= 0) & (idx <= cov.nepochs))


# --------------------------------------------------------------------------- #
#  2. Residuals-only helpers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("stabilize", [False, True])
def test_static_helpers_equal_full_recompute(kind, stabilize):
    psrs, wn, wn_vec, _ = setup(kind)
    wn.stabilize = stabilize
    T = basis(psrs)
    static = wn.get_static_helpers(T, wn_vec)
    rng = np.random.default_rng(4)
    for _ in range(3):                                   # several "sampled" residuals
        r = jnp.asarray(rng.normal(scale=1e-6, size=T.shape[0]))
        got = wn.get_red_helpers_from_static(static, r, wn_vec)
        want = wn.get_red_helpers(T, r, wn_vec)
        for g, w, what in zip(got, want, ("TNT", "TNr", "rNr", "logdet")):
            assert rel(g, w) < TIGHT, what


def test_static_helpers_under_jit_and_grad():
    psrs, wn, wn_vec, _ = setup("ecorr")
    T = basis(psrs)
    static = wn.get_static_helpers(T, wn_vec)
    r0 = jnp.concat([jnp.asarray(p.residuals) for p in psrs])
    d = jnp.asarray(np.random.default_rng(2).normal(scale=1e-7, size=r0.size))

    def f(s, fast):
        r = r0 + s * d
        h = (wn.get_red_helpers_from_static(static, r, wn_vec) if fast
             else wn.get_red_helpers(T, r, wn_vec))
        return jnp.sum(h[1] ** 2) + h[2]

    for fast in (True, False):
        assert np.isfinite(float(jax.jit(jax.grad(lambda s: f(s, fast)))(0.3)))
    gf = float(jax.jit(jax.grad(lambda s: f(s, True)))(0.3))
    gs = float(jax.jit(jax.grad(lambda s: f(s, False)))(0.3))
    assert abs(gf - gs) <= 1e-9 * abs(gs)


# --------------------------------------------------------------------------- #
#  3. model_maker with a sampled timing model
# --------------------------------------------------------------------------- #

class FakeTimingModel:
    """Stands in for JUG's MultiPsrTimingModel: residuals linear in z."""

    def __init__(self, raw, k=5, seed=0):
        self.raw = raw
        self.nparams_total = k
        self.A = jnp.asarray(np.random.default_rng(seed).normal(scale=1e-7, size=(raw.size, k)))

    def residuals(self, z):
        return self.raw + self.A @ z


@pytest.mark.parametrize("marg", [False, True])
def test_model_maker_static_path_matches_full(marg):
    m = H.build(**H.CORPUS["ltm"])
    raw = jnp.concat(m.data.raw_residuals)
    tm = FakeTimingModel(raw)
    kw = dict(raw_residuals=raw, super_sig=m.rn, marg_over_non_gwb=marg, vary_white=False,
              tm_model=tm, fixed_white_noise_params=m.wn_vec)
    nz = 2 * m.n_gwb if marg else m.rn.nmodes
    p = dict(red_noise=m.red_params(), timing_lam=jnp.asarray(1.3),
             timing_k=jnp.asarray(np.random.default_rng(1).normal(size=tm.nparams_total)),
             z_a=jnp.asarray(np.random.default_rng(5).normal(size=(m.npsr, nz))))
    static = m.rn.get_static_helpers(white_noise_params=m.wn_vec)
    f_full = lambda q: log_density(model_maker, (), kw, q)[0]
    f_fast = lambda q: log_density(model_maker, (), dict(kw, static_helpers=static), q)[0]
    a, b = float(f_fast(p)), float(f_full(p))
    assert abs(a - b) <= 1e-10 * abs(b)
    ga, gb = jax.grad(f_fast)(p), jax.grad(f_full)(p)
    for k in ("red_noise", "timing_k", "timing_lam"):
        assert np.max(np.abs(np.asarray(ga[k]) - np.asarray(gb[k]))) \
            <= 1e-8 * max(1.0, np.max(np.abs(np.asarray(gb[k]))))


def test_static_helpers_guards():
    m = H.build(**H.CORPUS["ltm"])
    raw = jnp.concat(m.data.raw_residuals)
    static = m.rn.get_static_helpers(white_noise_params=m.wn_vec)
    with pytest.raises(ValueError, match="tm_model"):
        model_maker(raw, m.rn, False, helpers=m.helpers, static_helpers=static)
    with pytest.raises(ValueError, match="fixed basis"):
        model_maker(raw, m.rn, False, vary_white=True, tm_model=FakeTimingModel(raw),
                    wn_lower_bound=m.wn_vec - 1, wn_upper_bound=m.wn_vec + 1,
                    static_helpers=static)
