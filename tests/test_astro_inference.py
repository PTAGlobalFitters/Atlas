"""Hierarchical astro inference (astro.HDAstroRedModel / AstroInferenceModel) against Pandora.

Atlas builds TNT/TNr and phi itself; the comparisons feed the same dense
TNT/TNr into Pandora's ``models.UniformPrior`` (``utils.hd_spectrum``) and
``LikelihoodCalculator.AstroInferenceModel``, so they test the phi / Sigma
algebra, the flow coupling, the prior and the jump proposals.

The likelihood agrees to ~1e-14 rather than bitwise: Atlas's intrinsic
power law multiplies by ``df = diff(f)`` where Pandora divides by ``Tspan``.
The common-process PSD, the prior decisions and the flow-draw proposal are
compared exactly.

Needs torch, zuko and an installed Pandora (atlas-env-2); the chain-level
comparison is marked ``slow``.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zuko")
LC = pytest.importorskip("pandora.LikelihoodCalculator")
from pandora import models, nf_dist, utils as putils  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from ATLAS.data import PTA_Data  # noqa: E402
from ATLAS.experimental import astro  # noqa: E402
from ATLAS.experimental import zuko_flows as zf  # noqa: E402
from tests.harness import fixture_available, load_psrs, white_noise_vector  # noqa: E402

pytestmark = pytest.mark.astro

N_BINS = 5
N_ASTRO = 2
PANDORA_DATA = os.path.join(os.path.dirname(os.path.dirname(nf_dist.__file__)), "data")


def _red(fixture, npsr=5):
    psrs = load_psrs(fixture, npsr)
    ecorr = fixture == "synth"      # the MDC fixtures have one TOA per epoch
    data = PTA_Data(psrs, num_gwb_bins=N_BINS, num_irn_bins=N_BINS, linear_timing=False,
                    marg_timing=True)
    red = astro.HDAstroRedModel.from_pta_data(data, white_noise_vector(psrs, ecorr),
                                              include_ecorr=ecorr)
    return red, np.array([p.pos for p in psrs])


@pytest.fixture(scope="module", params=["synth", "mdc1_5"])
def red_and_pos(request):
    if not fixture_available(request.param):
        pytest.skip(f"fixture {request.param} not available")
    return _red(request.param)


def _flow(red, seed=0):
    """A random-weight 'rho|theta' flow at the red model's frequencies."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    flow = zf.make_zuko_flow(N_BINS, N_ASTRO, hidden_features=[16, 16])
    mapping = dict(B=5, mean=np.concatenate([[0.0, 1.0], np.full(N_BINS, -8.0)]),
                   half_range=np.concatenate([[1.0, 2.0], rng.uniform(2, 3, N_BINS)]))
    return zf.ZukoAstroFlow(flow, "rho|theta", mapping, N_ASTRO,
                            metadata=dict(f_conv=list(red.f_common), tspan=red.tspan,
                                          param_names=["a", "b"]))


def _pandora(red, pos, flow=None, n_astro=N_ASTRO, fixed=None, stabilize=False, TNT=None,
             TNr=None):
    T = red.tspan
    f = np.arange(1, N_BINS + 1) / T
    psd, orf, helper = putils.hd_spectrum(renorm_const=1, crn_bins=N_BINS,
                                          lower_halflog10_rho=-12, upper_halflog10_rho=-4)
    run = models.UniformPrior(psd, orf, N_BINS, N_BINS, f, f, 1 / T, pos, T, len(pos), helper,
                              renorm_const=1)
    if flow is None:
        class _NF:
            nf_type = "rho|theta"
        nf = _NF()
    else:
        nf = nf_dist.NFastroinference(flow.flow, "rho|theta", flow.mapping["mean"],
                                      flow.mapping["half_range"], flow.B,
                                      np.arange(n_astro, n_astro + N_BINS), np.arange(n_astro),
                                      "cpu")
    n_var = n_astro - (len(fixed) if fixed else 0)
    kw = {}
    if fixed:
        kw = dict(astro_param_fixed_values=np.array(list(fixed.values()), dtype=float),
                  astro_param_fixed_indices=np.array(list(fixed.keys())))
    return LC.AstroInferenceModel(
        nf, n_astro, -np.ones(n_var), np.ones(n_var), lambda x: 0.0, run, None,
        TNr=jnp.asarray(red.TNr if TNr is None else TNr),
        TNT=jnp.asarray(red.TNT if TNT is None else TNT),
        matrix_stabilization=stabilize, **kw)


# ---- red-noise likelihood --------------------------------------------------------------

def test_order_helpers_roundtrip():
    x = np.arange(10.0)
    y = astro.pandora_to_atlas_order(x, 3)
    assert list(y[:6]) == [1, 0, 3, 2, 5, 4] and list(y[6:]) == [6, 7, 8, 9]
    assert np.array_equal(astro.atlas_to_pandora_order(y, 3), x)


def test_red_likelihood_matches_pandora(red_and_pos):
    red, pos = red_and_pos
    pm = _pandora(red, pos)
    assert red.param_names[0].endswith("irn_log10_A") and red.gwb_slice == slice(10, 15)
    assert np.array_equal(astro.atlas_to_pandora_order(red.lower, len(pos)),
                          np.asarray(pm.run_type_object.lower_prior_lim_all))
    rng = np.random.default_rng(0)
    for _ in range(200):
        xa = rng.uniform(red.lower, red.upper)
        la, psd_a = red.lnlikelihood(xa)
        lp, psd_p = pm.get_lnliklihood_non_astro(jnp.asarray(astro.atlas_to_pandora_order(xa, len(pos))))
        assert np.array_equal(np.asarray(psd_a), np.asarray(psd_p))
        assert np.isclose(float(la), float(lp), rtol=1e-12, atol=0)


def test_phi_inverse_matches_pandora(red_and_pos):
    red, pos = red_and_pos
    run = _pandora(red, pos).run_type_object
    xa = np.random.default_rng(1).uniform(red.lower, red.upper)
    phi_a = red.model.get_phi_mat(jnp.asarray(xa))
    phi_p = run.get_phi_mat(jnp.asarray(astro.atlas_to_pandora_order(xa, len(pos))))
    assert np.allclose(phi_a, phi_p, rtol=1e-13, atol=0)
    inv_a, ld_a = red.model.get_phi_mat_inv(phi_a)
    inv_p, ld_p = run.get_phi_mat_inv(phi_p)
    assert np.allclose(inv_a, inv_p, rtol=1e-12, atol=0)
    assert np.isclose(ld_a, ld_p, rtol=1e-13, atol=0)


def test_include_constant(red_and_pos):
    red, _ = red_and_pos
    x = red.lower / 2 + red.upper / 2
    a, _ = red.lnlikelihood(x)
    b, _ = red.lnlikelihood(x, include_constant=True)
    assert np.isclose(float(a - b), 0.5 * (red.rNr + red.logdet_N))


@pytest.mark.parametrize("stabilize", [None, 1e-6])
def test_from_matrices_on_pandora_15yr(stabilize):
    path = os.path.join(PANDORA_DATA, "15yr_TNT_and_TNr_5Int_5Common.npz")
    if not os.path.exists(path):
        pytest.skip("Pandora's 15-year TNT/TNr not available")
    with np.load(path) as d:
        TNT, TNr = d["TNT"], d["TNr"]
    pos = np.load(os.path.join(PANDORA_DATA, "15yr_pulsar_positions.npy"))
    T = float(np.load(os.path.join(PANDORA_DATA, "15yr_Tspan.npy")))
    red = astro.HDAstroRedModel.from_matrices(TNT, TNr, pos, T, N_BINS,
                                              stabilize_delta=stabilize)
    pm = _pandora(red, pos, TNT=TNT, TNr=TNr, stabilize=bool(stabilize))
    rng = np.random.default_rng(2)
    for _ in range(20):
        xa = rng.uniform(red.lower, red.upper)
        la, _ = red.lnlikelihood(xa)
        lp, _ = pm.get_lnliklihood_non_astro(jnp.asarray(astro.atlas_to_pandora_order(xa, len(pos))))
        assert np.isclose(float(la), float(lp), rtol=1e-11, atol=0)


# ---- astro model ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixed", [None, {1: 0.7}])
def test_astro_likelihood_matches_pandora(red_and_pos, fixed):
    red, pos = red_and_pos
    flow = _flow(red)
    ours = astro.AstroInferenceModel(red, flow, -np.ones(N_ASTRO), np.ones(N_ASTRO),
                                     fixed_astro=fixed)
    pm = _pandora(red, pos, flow, fixed=fixed)
    rng = np.random.default_rng(3)
    for _ in range(50):
        xa = rng.uniform(ours.lower, ours.upper)
        xp = astro.atlas_to_pandora_order(xa, len(pos))
        assert np.isclose(ours.lnlikelihood(xa) - flow.log_jacobian(), pm.get_lnliklihood(xp),
                          rtol=1e-12, atol=1e-9)
        assert np.isfinite(ours.lnprior(xa)) == np.isfinite(pm.get_lnprior(xp))
    assert ours.lnprior(ours.upper + 1) == -np.inf
    assert np.isclose(ours.lnprior(xa), -np.sum(np.log(ours.upper - ours.lower)))


def test_gwb_flow_draw_matches_pandora(red_and_pos):
    red, pos = red_and_pos
    flow = _flow(red)
    ours = astro.AstroInferenceModel(red, flow, -np.ones(N_ASTRO), np.ones(N_ASTRO))
    pm = _pandora(red, pos, flow)
    x = ours.make_initial_guess(seed=5)
    q, lqxy = ours._gwb_flow_draw(x, torch_seed=7)
    torch.manual_seed(7)
    qp, lqxy_p = pm.draw_from_gwb_prior(x.copy(), 0, 1)
    assert np.array_equal(q, qp)
    assert np.isclose(lqxy, float(np.squeeze(lqxy_p)), rtol=1e-12, atol=1e-12)
    assert np.array_equal(q[:red.gwb_slice.start], x[:red.gwb_slice.start])
    assert np.array_equal(q[red.gwb_slice.stop:], x[red.gwb_slice.stop:])


def test_gwb_flow_draw_uses_full_theta_when_fixed(red_and_pos):
    red, _ = red_and_pos
    flow = _flow(red)
    ours = astro.AstroInferenceModel(red, flow, -np.ones(1), np.ones(1), fixed_astro={0: 0.3})
    x = ours.make_initial_guess(seed=6)
    q, _ = ours._gwb_flow_draw(x, torch_seed=8)
    expected = flow.sample(1, np.array([0.3, x[-1]]), seed=8)[0]
    assert np.array_equal(q[red.gwb_slice], expected)


def test_prior_proposals_touch_one_parameter_in_their_block(red_and_pos):
    red, _ = red_and_pos
    ours = astro.AstroInferenceModel(red, _flow(red), -np.ones(N_ASTRO), np.ones(N_ASTRO), seed=1)
    x = ours.make_initial_guess()
    blocks = {ours.draw_from_prior: (0, len(x)),
              ours.draw_from_red_prior: (0, red.n_irn_params),
              ours.draw_from_nonIR_prior: (red.n_irn_params, len(x)),
              ours.draw_from_astro_prior: (ours.n_red, len(x))}
    for draw, (lo, hi) in blocks.items():
        for _ in range(50):
            q, lqxy = draw(x, 0, 1)
            changed = np.flatnonzero(q != x)
            assert lqxy == 0.0 and len(changed) == 1 and lo <= changed[0] < hi
            assert ours.lower[changed[0]] <= q[changed[0]] <= ours.upper[changed[0]]


def test_initial_guess_seed_zero_is_honoured(red_and_pos):
    red, _ = red_and_pos
    ours = astro.AstroInferenceModel(red, _flow(red), -np.ones(N_ASTRO), np.ones(N_ASTRO))
    assert np.array_equal(ours.make_initial_guess(seed=0), ours.make_initial_guess(seed=0))


def test_flow_frequency_mismatch_raises(red_and_pos):
    red, _ = red_and_pos
    flow = _flow(red)
    flow.metadata["f_conv"] = list(red.f_common * 1.01)
    with pytest.raises(ValueError, match="Tspan"):
        astro.AstroInferenceModel(red, flow, -np.ones(N_ASTRO), np.ones(N_ASTRO))
    torch.manual_seed(0)
    short = zf.ZukoAstroFlow(zf.make_zuko_flow(3, N_ASTRO, hidden_features=[8]), "rho|theta",
                             dict(B=5, mean=np.zeros(5), half_range=np.ones(5)), N_ASTRO)
    with pytest.raises(ValueError, match="frequency bins"):
        astro.AstroInferenceModel(red, short, -np.ones(N_ASTRO), np.ones(N_ASTRO))


def test_flow_draw_only_for_free_spectrum():
    from ATLAS.psd_functions import powerlaw

    psrs = load_psrs("synth", 3)
    data = PTA_Data(psrs, num_gwb_bins=N_BINS, num_irn_bins=N_BINS, linear_timing=False,
                    marg_timing=True)
    red = astro.HDAstroRedModel.from_pta_data(
        data, white_noise_vector(psrs, True), include_ecorr=True, gwb_psd_function=powerlaw,
        gwb_bounds=(np.array([-18.0, 0.0]), np.array([-11.0, 7.0])))
    assert not red.gwb_is_free_spectrum and red.gwb_slice == slice(6, 8)


# ---- sampling ------------------------------------------------------------------------------

def _chain(path):
    return np.loadtxt(os.path.join(path, "chain_1.txt"))


def test_seeded_chains_are_reproducible(tmp_path):
    red, _ = _red("synth", 3)
    ours = astro.AstroInferenceModel(red, _flow(red), -np.ones(N_ASTRO), np.ones(N_ASTRO))
    kw = dict(seed=11, resume=False, isave=100, thin=1)
    ours.sample(600, str(tmp_path / "a"), **kw)
    ours.sample(600, str(tmp_path / "b"), **kw)
    a, b = _chain(tmp_path / "a"), _chain(tmp_path / "b")
    assert a.shape[0] >= 500 and np.array_equal(a, b)
    with open(tmp_path / "a" / "pars.txt") as fh:
        assert fh.read().split() == ours.param_names


def _batch_se(x, n_batches=20):
    batches = np.array_split(x, n_batches)
    return np.std([b.mean() for b in batches], ddof=1) / np.sqrt(n_batches)


@pytest.mark.slow
def test_posterior_agrees_with_pandora(tmp_path):
    """Both samplers on the same 3-pulsar posterior; marginals agree within MC error."""
    red, pos = _red("synth", 3)
    flow = _flow(red, seed=3)
    ours = astro.AstroInferenceModel(red, flow, -np.ones(N_ASTRO), np.ones(N_ASTRO))
    pm = _pandora(red, pos, flow)
    niter = 40_000
    x0 = ours.make_initial_guess(seed=1)
    ours.sample(niter, str(tmp_path / "atlas"), x0=x0, seed=2, resume=False, thin=10)
    pm.sample(astro.atlas_to_pandora_order(x0, len(pos)), niter, str(tmp_path / "pandora"),
              resume=False)
    ndim = len(x0)
    a = _chain(tmp_path / "atlas")[:, :ndim]
    p = astro.pandora_to_atlas_order(_chain(tmp_path / "pandora")[:, :ndim], len(pos))
    a, p = a[len(a) // 4:], p[len(p) // 4:]
    for j in list(range(red.gwb_slice.start, ndim)):
        se = np.hypot(_batch_se(a[:, j]), _batch_se(p[:, j]))
        assert abs(a[:, j].mean() - p[:, j].mean()) < 4 * se + 1e-3, red.param_names[j] if j < ours.n_red else j
        assert 0.75 < a[:, j].std() / p[:, j].std() < 1.33
