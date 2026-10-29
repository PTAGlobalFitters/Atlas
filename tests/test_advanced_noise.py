"""The NG20 v2.0 vanilla noise blocks: chrom, cquad, sw.

Every check is against an independently built reference
(``reference.chromatic_weight`` / ``chrom_quad_basis`` / ``interp_basis`` /
``sw_kernel``, assembled by ``harness.ng20_dense``), never against ATLAS
itself.  What is pinned:

* each new T-matrix block equals its definition column by column
  (enterprise ``chromatic_quad_basis``; PINT ``TimeDomainSWNoise``);
* ``lnposterior_reparam`` with the new blocks is the dense Gaussian marginal,
  up to exactly the constant ATLAS documents dropping (``ref.atlas_offset``);
* ``partial_marg_lnposterior`` agrees with it -- the non-GWB phi block is laid
  out IRN | DM | GTM | extra blocks and has to line up with TNT;
* a sampled chromatic index (``varied_chrom_gp_index``) re-indexes ``chrom``
  and ``cquad`` together, absolutely, without compounding, and its gradient is
  the true one;
* input handling that has already gone wrong once.

Regressions recorded here:

* The chromatic-index gradient was NaN: the zero padding column of ``cquad``
  went through ``sqrt``, whose derivative at 0 is infinite.
* NG15 ``.npz`` loaders give pulsar names as 0-d numpy arrays, which broke
  every per-pulsar ``{name: value}`` argument.
* ``varied_chrom_gp_index`` with fixed white noise used to demand
  ``vary_white=True``; it now rebuilds the helpers from
  ``fixed_white_noise_params``.

Tolerances are measured.  The dense reference goes through Woodbury because a
near-flat ``cquad`` prior makes ``C`` too ill-conditioned to Cholesky-factor
directly to better than ~1e-3 nats.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.linalg import block_diag
from numpyro.infer.util import log_density

from . import harness as H
from . import reference as ref
from ATLAS.model import model_maker
from ATLAS.signals.factorized import chromatic as C

TIGHT = 1e-12          # basis columns, elementwise
LOOSE = 1e-9           # log-densities through Cholesky / Woodbury
GRAD_TOL = 1e-5        # central differences, relative


def build(name, **kw):
    return H.build_ng20(**{**H.NG20_CORPUS[name], **kw})


def dense_N(m):
    return block_diag(*H.noise_blocks(SimpleNamespace(psrs=m.psrs, include_ecorr=True)))


def whitened(m, xs, helpers=None, seed=0):
    """``lnposterior_reparam(z) + |z|^2 / 2`` -- with a zero ORF the transform
    whitens exactly, so this is ln p(r | theta) up to ATLAS's dropped constant."""
    z = np.random.default_rng(seed).normal(size=(m.npsr, m.rn.nmodes))
    lp, _ = m.rn.lnposterior_reparam(m.helpers if helpers is None else helpers,
                                     xs, jnp.asarray(z))
    return float(lp) + 0.5 * np.sum(z ** 2)


def offset(m):
    return ref.atlas_offset(m.residuals.size, [p.Mmat.shape[1] for p in m.psrs]
                            if m.n_tm else ())


def block_cols(m, name):
    sl = m.rn.signal_comb_idxs[name]
    return np.asarray(m.rn.get_Fmat_concat)[:, sl]


def per_psr(m, X):
    counts = [len(p.toas) for p in m.psrs]
    return np.split(np.asarray(X), np.cumsum(counts)[:-1])


# --------------------------------------------------------------------------- #
#  0. The reference ingredients are themselves right
# --------------------------------------------------------------------------- #

def test_reference_interp_basis_is_a_partition_of_unity():
    p = H.load_psrs("synth", 1)[0]
    L, nodes = ref.interp_basis(p.toas, dt_days=30.0)
    assert np.max(np.abs(L.sum(axis=1) - 1.0)) < TIGHT
    assert nodes[0] <= np.min(p.toas) and np.all(np.diff(nodes) > 0)
    # piecewise linear: a linear function of time is reproduced exactly
    assert np.max(np.abs(L @ nodes - np.asarray(p.toas))) < TIGHT * np.max(nodes)


@pytest.mark.parametrize("kind", ["RIDGE", "SQEXP", "MATERN", "QUASI_PERIODIC"])
def test_reference_kernels_are_positive_definite(kind):
    nodes = np.linspace(0, 10, 40) * 3.15e7
    K = ref.sw_kernel(nodes, kind, 0.3, log10_ell=2.0, nu=1.5, log10_gam_p=0.0, log10_p=0.0)
    assert np.allclose(K, K.T) and np.linalg.eigvalsh(K).min() > 0


def test_reference_woodbury_matches_dense():
    """On a well-conditioned problem the two forms of ln p(r) coincide."""
    m = build("curn-chrom")
    T, phi = H.ng20_dense(m, m.theta(1))
    N = dense_N(m)
    a = ref.woodbury_marginal_logL(m.residuals, T, phi, N)
    b = ref.marginal_logL(m.residuals, T, phi, N)
    assert abs(a - b) / abs(b) < LOOSE


def test_ported_sw_functions_match_pint():
    """Only where the installed PINT already ships TimeDomainSWNoise."""
    nm = pytest.importorskip("pint.models.noise_model")
    if not hasattr(nm, "make_interpolation_basis"):
        pytest.skip("this PINT has no TimeDomainSWNoise helpers")
    t = np.sort(np.random.default_rng(0).uniform(53000, 58000, 300)) * 86400
    U_pint, n_pint = nm.make_interpolation_basis(t, dt=30.0)
    U_atl, n_atl = C.make_interpolation_basis(t, dt=30.0)
    assert np.array_equal(n_pint, n_atl) and np.max(np.abs(U_pint - U_atl)) < TIGHT
    assert np.allclose(nm.matern_kernel(n_pint, 0.2, 1.5, 2.5),
                       C.matern_kernel(n_atl, 0.2, 1.5, 2.5), rtol=TIGHT, atol=0)
    assert np.allclose(nm.periodic_kernel(n_pint, 0.2, 1.5, 0.1, 0.0),
                       C.periodic_kernel(n_atl, 0.2, 1.5, 0.1, 0.0), rtol=TIGHT, atol=0)


# --------------------------------------------------------------------------- #
#  1. Each block is its definition
# --------------------------------------------------------------------------- #

def test_chrom_block_is_fourier_times_chromatic_weight():
    m = build("curn-all", chrom_index=4.4)
    f = np.arange(1, m.n_chrom + 1) / float(m.data.pta_tspan)
    want = np.vstack([ref.fourier_basis(p.toas, f)
                      * ref.chromatic_weight(p.freqs, 4.4, H.DM_REF_FREQ)[:, None]
                      for p in m.psrs])
    got = block_cols(m, "chrom")
    assert np.max(np.abs(got - want)) < TIGHT * np.max(np.abs(want))


@pytest.mark.parametrize("project", [True, False])
def test_cquad_block_is_enterprise_chrom_quad(project):
    m = build("curn-all", cquad_project=project)
    for p, B in zip(m.psrs, per_psr(m, block_cols(m, "cquad"))):
        U = H.svd_basis(p.Mmat) if project else None
        want = ref.chrom_quad_basis(p.toas, p.freqs, m.chrom_index, U, H.DM_REF_FREQ)
        assert np.max(np.abs(B[:, :3] - want)) < 1e-10      # through an SVD projection
        assert np.all(B[:, 3] == 0.0)                        # pair padding
        if project:
            assert np.max(np.abs(H.svd_basis(p.Mmat).T @ B)) < 1e-10


@pytest.mark.parametrize("kernel", ["RIDGE", "MATERN"])
def test_sw_block_is_pint_time_domain_basis(kernel):
    """RIDGE: L * G DMconst / nu^2.  MATERN: the same times chol(K_unit) --
    checked through the covariance it implies, ``B B^T == F K F^T``."""
    m = build("curn-all", sw_kernel=kernel)
    for p, B, w in zip(m.psrs, per_psr(m, block_cols(m, "sw")), m.rn.signal_map["sw"].ncols_per_psr):
        L, nodes = ref.interp_basis(p.toas, dt_days=H.NG20_SW_DT_DAYS)
        F = L * (H.sw_geometry_synth(p) * H.NG20_DMCONST / np.asarray(p.freqs) ** 2)[:, None]
        assert w == L.shape[1] and np.all(B[:, w:] == 0.0)
        if kernel == "RIDGE":
            assert np.max(np.abs(B[:, :w] - F)) < TIGHT * np.max(np.abs(F))
        else:
            K = ref.sw_kernel(nodes, "MATERN", 0.0, **{k: v for k, v in H.NG20_MATERN.items()
                                                       if k != "kernel"})
            want = F @ K @ F.T
            assert np.max(np.abs(B @ B.T - want)) < 1e-10 * np.max(np.abs(want))


# --------------------------------------------------------------------------- #
#  2. The likelihoods
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", list(H.NG20_CORPUS))
def test_reparam_density_is_the_dense_marginal(name):
    """``lnposterior_reparam + |z|^2/2 == ln p(r) + atlas_offset`` -- the
    absolute value, not just up to a constant, at several hyperparameters."""
    m = build(name)
    N = dense_N(m)
    for seed in range(3):
        th = m.theta(seed)
        T, phi = H.ng20_dense(m, th)
        got = whitened(m, m.red_params(th))
        want = ref.woodbury_marginal_logL(m.residuals, T, phi, N) + offset(m)
        assert abs(got - want) / abs(want) < LOOSE, (name, seed, got - want)


@pytest.mark.parametrize("name", ["curn-all", "ltm-all", "curn-matern"])
def test_partial_marg_agrees_with_reparam(name):
    """The partial-marginal phi block is IRN | DM | GTM | extras; if it were
    mis-ordered against TNT the two densities would disagree."""
    m = build(name)
    xs = m.red_params(m.theta(3))
    rng = np.random.default_rng(2)
    zp = rng.normal(size=(m.npsr, 2 * m.n_gwb))
    v_p = float(m.rn.partial_marg_lnposterior(m.helpers, xs, jnp.asarray(zp))[0]) \
        + 0.5 * np.sum(zp ** 2)
    v_r = whitened(m, xs)
    assert abs(v_r - v_p) / abs(v_r) < LOOSE


# --------------------------------------------------------------------------- #
#  3. The sampled chromatic index
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("chi", [(4.0, 4.0), (2.6, 4.4), (0.5, 6.0)])
def test_update_sets_the_index_of_chrom_and_cquad_only(chi):
    """Absolute, not relative to the reference index, and nothing else moves."""
    m = build("curn-all")
    T = np.asarray(m.rn.update_chrom_gp_basis(jnp.asarray(chi)))
    F = np.asarray(m.rn.get_Fmat_concat)
    f = np.arange(1, m.n_chrom + 1) / float(m.data.pta_tspan)
    chrom = np.vstack([ref.fourier_basis(p.toas, f)
                       * ref.chromatic_weight(p.freqs, c, H.DM_REF_FREQ)[:, None]
                       for p, c in zip(m.psrs, chi)])
    assert np.max(np.abs(T[:, m.rn.signal_comb_idxs["chrom"]] - chrom)) < TIGHT * np.max(np.abs(chrom))
    for p, c, B in zip(m.psrs, chi, per_psr(m, T[:, m.rn.signal_comb_idxs["cquad"]])):
        want = ref.chrom_quad_basis(p.toas, p.freqs, c, H.svd_basis(p.Mmat), H.DM_REF_FREQ)
        assert np.max(np.abs(B[:, :3] - want)) < 1e-10
    rest = np.ones(F.shape[1], bool)
    for k in ("chrom", "cquad"):
        rest[m.rn.signal_comb_idxs[k]] = False
    assert np.array_equal(T[:, rest], F[:, rest])


def test_update_at_the_reference_index_is_the_identity():
    m = build("curn-all", chrom_index=4.0)
    T = m.rn.update_chrom_gp_basis(jnp.full(m.npsr, 4.0))
    assert np.max(np.abs(np.asarray(T) - np.asarray(m.rn.get_Fmat_concat))) < 1e-10


def test_update_under_jit_does_not_compound():
    """Two traced calls from the as-built basis give the same answer -- the
    dm-block analogue of this once compounded to index 2 + c."""
    m = build("curn-all")
    f = jax.jit(lambda c: m.rn.update_chrom_gp_basis(c))
    a = np.asarray(f(jnp.array([3.0, 5.0])))
    b = np.asarray(f(jnp.array([3.0, 5.0])))
    c = np.asarray(m.rn.update_chrom_gp_basis(jnp.array([3.0, 5.0])))
    assert np.array_equal(a, b) and np.max(np.abs(a - c)) < 1e-12


def kwargs(m, marg, **kw):
    return dict(raw_residuals=jnp.concat(m.data.raw_residuals), super_sig=m.rn,
                marg_over_non_gwb=marg, vary_white=True,
                wn_lower_bound=m.wn_vec - 1.0, wn_upper_bound=m.wn_vec + 1.0, **kw)


def params(m, marg, seed=0):
    nz = 2 * m.n_gwb if marg else m.rn.nmodes
    return dict(white_noise=m.wn_vec, red_noise=m.red_params(m.theta(seed)),
                z_a=jnp.asarray(np.random.default_rng(7).normal(size=(m.npsr, nz))))


@pytest.mark.parametrize("marg", [False, True])
def test_sampled_index_is_the_model_on_the_reference_basis(marg):
    """Sampling chi = c equals the fixed-index model whose chrom and cquad
    columns are built by the reference at c, plus the Uniform(0, 7) log-prior."""
    m = build("ltm-all")
    c = np.array([3.1, 4.7])
    got, _ = log_density(model_maker, (), kwargs(m, marg, varied_chrom_gp_index=True),
                         dict(params(m, marg), chrom_gp_index=jnp.asarray(c)))
    T_ref = np.array(m.rn.get_Fmat_concat)
    f = np.arange(1, m.n_chrom + 1) / float(m.data.pta_tspan)
    T_ref[:, m.rn.signal_comb_idxs["chrom"]] = np.vstack(
        [ref.fourier_basis(p.toas, f) * ref.chromatic_weight(p.freqs, ci, H.DM_REF_FREQ)[:, None]
         for p, ci in zip(m.psrs, c)])
    q = np.zeros_like(T_ref[:, m.rn.signal_comb_idxs["cquad"]])
    for (s, e), p, ci in zip(_row_ranges(m), m.psrs, c):
        q[s:e, :3] = ref.chrom_quad_basis(p.toas, p.freqs, ci, H.svd_basis(p.Mmat), H.DM_REF_FREQ)
    T_ref[:, m.rn.signal_comb_idxs["cquad"]] = q
    want, _ = log_density(model_maker, (),
                          kwargs(m, marg, red_noise_basis=jnp.asarray(T_ref)), params(m, marg))
    want = float(want) - m.npsr * np.log(7.0)
    assert abs(float(got) - want) < LOOSE * abs(want)


def _row_ranges(m):
    edges = np.cumsum([0] + [len(p.toas) for p in m.psrs])
    return list(zip(edges[:-1], edges[1:]))


@pytest.mark.parametrize("marg", [False, True])
def test_index_gradient_is_finite_and_correct(marg):
    """Regression: d/dchi was NaN (sqrt of cquad's zero padding column)."""
    m = build("ltm-all")
    kw = kwargs(m, marg, varied_chrom_gp_index=True)
    p0 = dict(params(m, marg), chrom_gp_index=jnp.array([3.0, 4.5]))
    f = lambda p: log_density(model_maker, (), kw, p)[0]
    g = jax.grad(f)(p0)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(g))
    eps = 1e-5
    for i in range(m.npsr):
        bump = lambda d: dict(p0, chrom_gp_index=p0["chrom_gp_index"].at[i].add(d))
        fd = (float(f(bump(eps))) - float(f(bump(-eps)))) / (2 * eps)
        assert abs(float(g["chrom_gp_index"][i]) - fd) < GRAD_TOL * max(1.0, abs(fd))


def test_sampled_index_with_fixed_white_noise():
    """Helpers rebuilt from ``fixed_white_noise_params``: equal to the
    vary_white path at the same white noise, minus its Uniform(+-1) prior."""
    m = build("curn-all")
    c = jnp.array([3.0, 4.5])
    p = params(m, False)
    fixed = dict(raw_residuals=jnp.concat(m.data.raw_residuals), super_sig=m.rn,
                 marg_over_non_gwb=False, vary_white=False,
                 fixed_white_noise_params=m.wn_vec, varied_chrom_gp_index=True)
    a = log_density(model_maker, (), fixed,
                    dict(red_noise=p["red_noise"], z_a=p["z_a"], chrom_gp_index=c))[0]
    b = log_density(model_maker, (), kwargs(m, False, varied_chrom_gp_index=True),
                    dict(p, chrom_gp_index=c))[0]
    assert abs(float(a) - m.wn_vec.size * np.log(2.0) - float(b)) < LOOSE * abs(float(a))


# --------------------------------------------------------------------------- #
#  4. Layout and guards
# --------------------------------------------------------------------------- #

def test_param_layout():
    """Extras go after everything else (here after the GWB), block by block,
    pulsar-major; cquad has no sampled parameters."""
    m = build("curn-all")
    names = m.rn.model.get_param_names()
    a, b = (str(p.name) for p in m.psrs)
    assert len(names) == m.rn.model.upper_prior_lim_all.shape[0]
    assert names[-8:] == ["gwb_log10_A", "gwb_gamma",
                          f"{a}_chrom_log10_A", f"{a}_chrom_gamma",
                          f"{b}_chrom_log10_A", f"{b}_chrom_gamma",
                          f"{a}_sw_log10_amp", f"{b}_sw_log10_amp"]
    assert not any("cquad" in n for n in names)
    assert m.rn.extra_names == ["chrom", "cquad", "sw"]
    assert m.rn.chrom_gp_names == ["chrom", "cquad"]


def test_models_without_new_blocks_are_unchanged():
    """No extra blocks -> no extra parameters and no extra columns."""
    m = H.build(**H.CORPUS["ltm-dm"])
    assert m.rn.extra_names == [] and m.rn.model.num_extra_params == 0
    assert m.rn.nmodes == m.n_tm + 2 * m.n_irn + 2 * m.n_dm


def test_extra_blocks_must_be_separate():
    with pytest.raises(ValueError, match="separate"):
        H.build_ng20(model_string="unc+cor+chrom->unc;dm,cquad,sw")


def test_partial_marg_rejects_extras_before_dm():
    m = H.build_ng20(model_string="unc+cor->unc;chrom,dm")
    with pytest.raises(ValueError, match="contiguous"):
        m.rn.partial_marg_lnposterior(m.helpers, m.red_params(m.theta()),
                                      jnp.zeros((m.npsr, 2 * m.n_gwb)))


def test_varied_chrom_gp_index_needs_a_chromatic_block():
    m = H.build_ng20(model_string="unc+cor->unc;dm,sw")
    with pytest.raises(ValueError, match="chrom or cquad"):
        model_maker(**kwargs(m, False, varied_chrom_gp_index=True))


def test_varied_chrom_gp_index_needs_a_white_noise_source():
    m = build("curn-all")
    with pytest.raises(ValueError, match="fixed_white_noise_params"):
        model_maker(jnp.concat(m.data.raw_residuals), m.rn, False,
                    helpers=m.helpers, varied_chrom_gp_index=True)


def test_linear_timing_without_ltm_prefix_is_rejected():
    """Regression: linear_timing=True, marg_timing=False and no 'ltm|' gave a
    NaN log-density -- every column offset was shifted by the timing width."""
    with pytest.raises(ValueError, match="ltm"):
        H.build(model_string="unc+cor->unc;dm", linear_timing=True, n_dm=3)


def test_marginalised_linear_timing_needs_no_ltm_prefix():
    """The ChromaticNoise.ipynb setup stays valid: marginalised timing has no
    T-matrix columns, so no prefix is needed and the density is finite."""
    m = H.build(model_string="unc+cor->unc;dm", linear_timing=True,
                marg_timing=True, n_dm=3)
    assert m.rn.linear_timing_model_size == 0
    x0 = m.rn.model.make_initial_guess(jax.random.key(0))
    lp, _ = m.rn.lnposterior_reparam(m.helpers, x0, jnp.zeros((m.npsr, m.rn.nmodes)))
    assert np.isfinite(float(lp))


# --------------------------------------------------------------------------- #
#  5. Inputs
# --------------------------------------------------------------------------- #

def test_per_pulsar_arguments_accept_npz_style_names():
    """Regression: names loaded from .npz are 0-d arrays / np.str_."""
    data = SimpleNamespace(psr_names=[np.array("J0000+0000"), np.array("J0001+0001")])
    got = C._per_pulsar({np.str_("J0001+0001"): 2.0, "J0000+0000": 1.0}, data, "x")
    assert got == [1.0, 2.0]
    assert C._per_pulsar(3.0, data, "x") == [3.0, 3.0]
    assert C._per_pulsar([None, [1, 2]], data, "x") == [None, [1, 2]]
    with pytest.raises(ValueError, match="no entry"):
        C._per_pulsar({"J0000+0000": 1.0}, data, "x")


def test_read_tdsw_par(tmp_path):
    par = tmp_path / "x.par"
    par.write_text("PSR J0000+0000\nF0 100 1\nTDSWKERNEL matern\nTDSWLOGSIG 0.2\n"
                   "TDSWLOGELL 1.5\nTDSWNU 2.5\nTDSWNODE_0002 55100.0\n"
                   "TDSWNODE_0001 55000.0\n")
    d = C.read_tdsw_par(par)
    assert d["kernel"] == "MATERN" and d["nu"] == 2.5
    assert d["log10_sigma"] == 0.2 and d["log10_ell"] == 1.5
    assert list(d["nodes_mjd"]) == [55000.0, 55100.0]
    empty = tmp_path / "y.par"
    empty.write_text("PSR J0000+0000\n")
    assert C.read_tdsw_par(empty) == dict(
        kernel="RIDGE", dt_days=30.0, nodes_mjd=None, interp_kind="linear",
        log10_sigma=None, log10_ell=None, nu=1.5, log10_gam_p=None, log10_p=None)


def test_explicit_nodes_reach_the_basis():
    """TDSWNODE_ (Bayesian-block edges) replace the TDSWDT grid."""
    p = H.load_psrs("synth", 1)[0]
    t = np.asarray(p.toas)
    nodes = np.unique(np.concatenate([np.linspace(t.min(), t.max(), 9) / 86400,
                                      [t.min() / 86400 + 100.0]]))
    U, n = C.make_interpolation_basis(t, nodes=nodes)
    L, n_ref = ref.interp_basis(t, nodes_mjd=nodes)
    assert np.array_equal(n, n_ref) and np.max(np.abs(U - L)) < TIGHT


def test_align_to_toas():
    src, dst = np.array([3.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])
    assert list(C.align_to_toas([30, 10, 20], src, dst)) == [10, 20, 30]
    with pytest.raises(ValueError, match="do not match"):
        C.align_to_toas([1, 2, 3], src, dst + 10.0)