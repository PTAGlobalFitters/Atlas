"""The chromatic index of a `dm` block reaches the likelihood exactly once.

`SuperSignal` builds the dm block with the DM index 2, and `update_red_basis`
re-indexes it.  Both used to go wrong without raising.  First the scaling was
discarded, so dm was a second achromatic red-noise block.  Then, once kept,
`update_red_basis` compounded with it (index 2 + c) whenever JAX traced it
afresh.  `model_maker`'s keyword call happened to reuse the trace compiled
inside `__init__`, before the scaling was stored, so it was right only by
accident; with pre-built helpers it ignored the index altogether.  Everything
here is checked against an independently built basis.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from numpyro.infer.util import log_density

from . import harness as H
from . import reference as ref
from ATLAS.model import model_maker

TIGHT = 1e-12          # basis columns, elementwise
LOOSE = 1e-9           # log-densities through a Cholesky of a correlated phi


def dm_model():
    return H.build(**H.CORPUS["ltm-dm"])


def reference_dm_block(m, chrom_index):
    """F_dm with each pulsar's rows scaled by ``(f_ref / f_radio)**index``."""
    f_dm = np.arange(1, m.n_dm + 1) / m.data.pta_tspan
    index = np.broadcast_to(np.asarray(chrom_index, dtype=np.float64), (m.npsr,))
    return np.vstack([
        ref.fourier_basis(p.toas, f_dm)
        * ((H.DM_REF_FREQ / np.asarray(p.freqs, dtype=np.float64)) ** c)[:, None]
        for p, c in zip(m.psrs, index)])


def assert_dm_block(m, T, chrom_index):
    """`T`'s dm block has index `chrom_index`; every other column is untouched."""
    T = np.asarray(T)
    F = np.asarray(m.rn.get_Fmat_concat)
    dm = m.rn.chrom_idxs
    want = reference_dm_block(m, chrom_index)
    assert np.max(np.abs(T[:, dm] - want)) < TIGHT * np.max(np.abs(want))
    rest = np.ones(F.shape[1], dtype=bool)
    rest[dm] = False
    assert np.array_equal(T[:, rest], F[:, rest])


def test_dm_basis_carries_the_dm_index():
    m = dm_model()
    assert_dm_block(m, m.rn.get_Fmat_concat, 2.0)


@pytest.mark.parametrize("call", ["default basis", "basis passed", "under jit"])
@pytest.mark.parametrize("chrom_index", [(2.0, 2.0), (4.4, 4.4), (0.0, 3.0)])
def test_update_red_basis_sets_the_index(chrom_index, call):
    """The index is absolute, not added to the DM index the basis carries."""
    m = dm_model()
    c = jnp.asarray(chrom_index)
    if call == "default basis":
        T = m.rn.update_red_basis(c)
    elif call == "basis passed":
        T = m.rn.update_red_basis(c, red_noise_basis=m.rn.get_Fmat_concat)
    else:
        T = jax.jit(lambda c: m.rn.update_red_basis(c))(c)
    assert_dm_block(m, T, chrom_index)


def model_kwargs(m, marg, **kw):
    return dict(raw_residuals=jnp.concat(m.data.raw_residuals), super_sig=m.rn,
                marg_over_non_gwb=marg, vary_white=True,
                wn_lower_bound=m.wn_vec - 1.0, wn_upper_bound=m.wn_vec + 1.0, **kw)


def sampled_vs_reference(marg, pass_basis):
    """Sampling the index at `c` gives the model built on the reference basis
    with index `c`, plus the Uniform(0, 6) prior on the index."""
    m = dm_model()
    c = np.array([1.0, 3.5])
    n_z = 2 * m.rn.model.crn_bins if marg else m.rn.nmodes
    params = dict(white_noise=m.wn_vec, red_noise=m.red_params(),
                  z_a=jnp.asarray(np.random.default_rng(7).normal(size=(m.npsr, n_z))))

    basis = m.rn.get_Fmat_concat if pass_basis else None
    got, _ = log_density(
        model_maker, (),
        model_kwargs(m, marg, varied_chrom_index=True, red_noise_basis=basis),
        dict(params, chromatic_index=jnp.asarray(c)))

    T_ref = np.array(m.rn.get_Fmat_concat)
    T_ref[:, m.rn.chrom_idxs] = reference_dm_block(m, c)
    want, _ = log_density(
        model_maker, (), model_kwargs(m, marg, red_noise_basis=jnp.asarray(T_ref)), params)
    want = float(want) - m.npsr * np.log(6.0)

    assert abs(float(got) - want) < LOOSE * abs(want)


@pytest.mark.parametrize("pass_basis", [False, True])
@pytest.mark.parametrize("marg", [False, True])
def test_sampled_chromatic_index_reaches_the_likelihood(marg, pass_basis):
    sampled_vs_reference(marg, pass_basis)


def test_sampled_chromatic_index_needs_no_cached_trace():
    """The same identity from an empty JAX cache.  This is the case that used
    to compound: nothing may depend on a trace compiled before the dm scaling
    was stored."""
    jax.clear_caches()
    sampled_vs_reference(marg=False, pass_basis=False)


def test_varied_chrom_index_rejects_prebuilt_helpers():
    m = dm_model()
    with pytest.raises(ValueError, match="rebuilt inside the model"):
        model_maker(jnp.concat(m.data.raw_residuals), m.rn, False,
                    helpers=m.helpers, varied_chrom_index=True)


def test_varied_chrom_index_needs_a_dm_block():
    m = H.build(**H.CORPUS["ltm"])
    with pytest.raises(ValueError, match="needs a dm block"):
        model_maker(**model_kwargs(m, False, varied_chrom_index=True))
