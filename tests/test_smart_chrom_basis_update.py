"""Partial TNT / TNr updates: only some columns of T changed.

``WhiteCov.get_red_helpers_partial`` recomputes ``C^T N^-1 [T | r]`` for the
changed columns ``C`` and writes it into the previous helpers.  The identity
pinned here is that the result equals a full ``get_red_helpers`` on the new T,
for every white-noise solver path:

* ECORR (Sherman-Morrison epochs), timing columns in T;
* timing model marginalised into N (``_solve_marg``);
* diagonal white noise (``diag_white_cov``).

and that ``model_maker`` with a sampled chromatic index, fixed white noise and
pre-built helpers -- which now takes the partial path -- gives the same
log-density and gradient as the full rebuild.
"""
from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from numpyro.infer.util import log_density

from . import harness as H
from ATLAS.data import PTA_Data
from ATLAS.model import model_maker
from ATLAS.nMatrix.base import WhiteCov

TIGHT = 1e-12


def assert_helpers_equal(a, b, tol=TIGHT):
    for x, y, what in zip(a, b, ("TNT", "TNr", "rNr", "logdet_N")):
        x, y = np.asarray(x), np.asarray(y)
        scale = max(np.max(np.abs(y)), 1e-300)
        assert np.max(np.abs(x - y)) <= tol * scale, what


def perturbed(T, cols, seed=0):
    """T with ``cols`` replaced by new, unrelated columns."""
    T = np.array(T)
    rng = np.random.default_rng(seed)
    T[:, cols] = rng.normal(size=(T.shape[0], len(cols))) * np.std(T[:, cols])
    return jnp.asarray(T)


SOLVERS = {
    "ecorr-ltm":   lambda: H.build(**H.CORPUS["ltm-dm"]),
    "ecorr-margtm": lambda: H.build(**H.CORPUS["curn-margtm"]),
}


@pytest.mark.parametrize("solver", list(SOLVERS))
@pytest.mark.parametrize("cols", [[3], [0, 1, 2], [2, 7, 11], "last5"])
def test_partial_equals_full(solver, cols):
    m = SOLVERS[solver]()
    T0 = m.rn.get_Fmat_concat
    n = T0.shape[1]
    cols = list(range(n - 5, n)) if cols == "last5" else cols
    r = jnp.concat(m.data.raw_residuals)
    N = m.data.Nmat
    base = N.get_red_helpers(T0, r, m.wn_vec)
    T1 = perturbed(T0, cols)
    got = N.get_red_helpers_partial(T1, np.array(cols), base, r, m.wn_vec)
    want = N.get_red_helpers(T1, r, m.wn_vec)
    assert_helpers_equal(got, want)


def test_partial_equals_full_diagonal_white_noise():
    psrs = H.load_psrs("synth", 2)
    data = PTA_Data(psrs, num_gwb_bins=4, num_irn_bins=6, linear_timing=False,
                    marg_timing=False, diag_white_cov=True, dm_ref_freq=H.DM_REF_FREQ)
    N = WhiteCov(data=data, stabilize_TNT=False)
    t = [np.asarray(p.toas) for p in psrs]
    f = np.arange(1, 7) / float(data.pta_tspan)
    T0 = jnp.asarray(np.vstack([H.ref.fourier_basis(x, f) for x in t]))
    r = jnp.concat(data.raw_residuals)
    base = N.get_red_helpers(T0, r, None)
    cols = [4, 5, 9]
    T1 = perturbed(T0, cols, seed=1)
    assert_helpers_equal(N.get_red_helpers_partial(T1, np.array(cols), base, r, None),
                         N.get_red_helpers(T1, r, None))


def test_partial_accepts_slices_and_no_columns():
    m = SOLVERS["ecorr-ltm"]()
    T0, r, N = m.rn.get_Fmat_concat, jnp.concat(m.data.raw_residuals), m.data.Nmat
    base = N.get_red_helpers(T0, r, m.wn_vec)
    T1 = perturbed(T0, list(range(4, 9)))
    assert_helpers_equal(N.get_red_helpers_partial(T1, slice(4, 9), base, r, m.wn_vec),
                         N.get_red_helpers(T1, r, m.wn_vec))
    same = N.get_red_helpers_partial(T1, np.zeros(0, int), base, r, m.wn_vec)
    assert all(a is b for a, b in zip(same, base))


def test_partial_under_jit_and_grad():
    """Columns static, basis traced; the gradient through the partial update
    equals the gradient through the full one."""
    m = SOLVERS["ecorr-ltm"]()
    T0, r, N = m.rn.get_Fmat_concat, jnp.concat(m.data.raw_residuals), m.data.Nmat
    cols = np.array([5, 6, 7])
    base = N.get_red_helpers(T0, r, m.wn_vec)
    direction = perturbed(jnp.zeros_like(T0), cols, seed=3)

    def f(scale, partial):
        T = T0 + scale * direction
        h = (N.get_red_helpers_partial(T, cols, base, r, m.wn_vec) if partial
             else N.get_red_helpers(T, r, m.wn_vec))
        return jnp.sum(h[0] ** 2) + jnp.sum(h[1] ** 2)

    gp = jax.jit(jax.grad(lambda s: f(s, True)))(0.3)
    gf = jax.jit(jax.grad(lambda s: f(s, False)))(0.3)
    assert abs(float(gp) - float(gf)) <= 1e-10 * abs(float(gf))


@pytest.mark.parametrize("cols,message", [([99999], "out of range"), ([2, 2], "repeat")])
def test_partial_rejects_bad_columns(cols, message):
    m = SOLVERS["ecorr-ltm"]()
    T0, r, N = m.rn.get_Fmat_concat, jnp.concat(m.data.raw_residuals), m.data.Nmat
    base = N.get_red_helpers(T0, r, m.wn_vec)
    with pytest.raises(ValueError, match=message):
        N.get_red_helpers_partial(T0, np.array(cols), base, r, m.wn_vec)


def test_partial_refuses_stabilised_helpers():
    psrs = H.load_psrs("synth", 2)
    data = PTA_Data(psrs, num_gwb_bins=4, num_irn_bins=6, linear_timing=False,
                    marg_timing=False, diag_white_cov=False, dm_ref_freq=H.DM_REF_FREQ)
    N = WhiteCov(data=data, stabilize_TNT=True, include_ecorr=True)
    with pytest.raises(NotImplementedError, match="stabilize"):
        N.get_red_helpers_partial(jnp.zeros((sum(len(p.toas) for p in psrs), 4)),
                                  np.array([0]), (None,) * 4, None, None)


# --------------------------------------------------------------------------- #
#  SuperSignal and model_maker
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["curn-all", "ltm-all"])
def test_supersignal_partial_after_index_update(name):
    """Default columns are chrom + cquad, exactly what update_chrom_gp_basis
    rewrites."""
    m = H.build_ng20(**H.NG20_CORPUS[name])
    r = jnp.concat(m.data.raw_residuals)
    T1 = m.rn.update_chrom_gp_basis(jnp.array([2.7, 5.9]))
    got = m.rn.get_helpers_partial(T1, m.helpers, reff=r, white_noise_params=m.wn_vec)
    want = m.rn.get_helpers(red_noise_basis=T1, reff=r, white_noise_params=m.wn_vec)
    assert_helpers_equal(got, want)
    expect = np.concatenate([np.arange(m.rn.signal_comb_idxs[k].start,
                                       m.rn.signal_comb_idxs[k].stop) for k in ("chrom", "cquad")])
    assert np.array_equal(m.rn.chrom_gp_cols, expect)


@pytest.mark.parametrize("marg", [False, True])
def test_model_maker_partial_path_matches_full_rebuild(marg):
    m = H.build_ng20(**H.NG20_CORPUS["ltm-all"])
    nz = 2 * m.n_gwb if marg else m.rn.nmodes
    p = dict(red_noise=m.red_params(m.theta(2)), chrom_gp_index=jnp.array([3.2, 4.9]),
             z_a=jnp.asarray(np.random.default_rng(5).normal(size=(m.npsr, nz))))
    kw = dict(raw_residuals=jnp.concat(m.data.raw_residuals), super_sig=m.rn,
              marg_over_non_gwb=marg, vary_white=False, fixed_white_noise_params=m.wn_vec,
              varied_chrom_gp_index=True)
    f_full = lambda q: log_density(model_maker, (), kw, q)[0]
    f_part = lambda q: log_density(model_maker, (), dict(kw, helpers=m.helpers), q)[0]
    a, b = float(f_part(p)), float(f_full(p))
    assert abs(a - b) <= 1e-10 * abs(b)
    ga, gb = jax.grad(f_part)(p), jax.grad(f_full)(p)
    for k in ("red_noise", "chrom_gp_index"):
        assert np.max(np.abs(np.asarray(ga[k]) - np.asarray(gb[k]))) \
            <= 1e-8 * max(1.0, np.max(np.abs(np.asarray(gb[k]))))
