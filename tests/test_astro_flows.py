"""Flow training and evaluation (zuko_flows.py, astro.train_astro_flow) against Pandora.

The reference is Pandora's own ``pandora.nf_dist`` (NFMaker, DataSplitter,
ValidationHell, NFastroinference) plus the notebook training loop copied into
tests/pandora_reference/notebook_flow.py. Training is compared bit for bit:
same ``torch.manual_seed`` for the initial weights, same ``random`` seed for
the batch indices, CPU, float64.

Needs torch, zuko and an installed Pandora (atlas-env-2); skipped otherwise.
Importing ``pandora.nf_dist`` sets torch's default dtype to float64 for the
whole session, as it does in Pandora.
"""
import os
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zuko")
nf_dist = pytest.importorskip("pandora.nf_dist")

from ATLAS.experimental import astro  # noqa: E402
from ATLAS.experimental import flow_utils as fu  # noqa: E402
from ATLAS.experimental import zuko_flows as zf  # noqa: E402
from tests.pandora_reference import notebook_flow  # noqa: E402

pytestmark = pytest.mark.astro

CPU = torch.device("cpu")
HIDDEN = [16, 16]
N_DRAWS, N_REAL, N_F, N_P = 20, 30, 3, 2


@pytest.fixture(scope="module")
def data():
    """Normalized-looking samples: rho (draws, real, f) and theta (draws, p)."""
    rng = np.random.default_rng(0)
    theta = np.clip(rng.normal(size=(N_DRAWS, N_P)) * 2, -5, 5)
    rho = np.clip(theta[:, None, :1] + rng.normal(size=(N_DRAWS, N_REAL, N_F)), -5, 5)
    return rho, theta


@pytest.fixture(scope="module")
def physical_set():
    rng = np.random.default_rng(1)
    params = rng.normal(size=(N_DRAWS, N_P)) + np.array([3.0, -2.5])
    rho = -8 + 0.3 * rng.normal(size=(N_DRAWS, N_REAL, N_F)) + 0.1 * params[:, None, :1]
    return astro.TrainingSet(params=params, log10_rho=rho, param_names=["a", "b"])


def _state_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


# ---- splitting and batches ---------------------------------------------------------

def test_split_matches_datasplitter(physical_set):
    train, val = physical_set.split(4, 6, seeds=(3, 7))
    p_train, p_val = nf_dist.DataSplitter.splitter_mesh(physical_set.log10_rho,
                                                         np.array([4, 6]), (3, 7))
    q_train, q_val = nf_dist.DataSplitter.splitter(physical_set.params, 4, 3)
    assert np.array_equal(train.log10_rho, p_train) and np.array_equal(val.log10_rho, p_val)
    assert np.array_equal(train.params, q_train) and np.array_equal(val.params, q_val)
    assert set(train.draw_indices).isdisjoint(val.draw_indices)


def _nfmaker(inputs, context, tmp_path, **kw):
    os.makedirs(tmp_path, exist_ok=True)     # NFMaker does not create it
    return nf_dist.NFMaker(inputs, context, str(tmp_path), 5, device=CPU,
                           hidden_dims=HIDDEN, **kw)


@pytest.mark.parametrize("swap", [False, True], ids=["rho|theta", "theta|rho"])
@pytest.mark.parametrize("mode,rep_in,rep_ctx", [("diagonal", False, False),
                                                 ("mesh", True, True)])
def test_batches_match_nfmaker(data, tmp_path, swap, mode, rep_in, rep_ctx):
    rho, theta = data
    inputs, context = (theta, rho) if swap else (rho, theta)
    m = _nfmaker(inputs, context, tmp_path)
    random.seed(5)
    expected = [m.sample_from_dist(4, m.chain_input, m.chain_context, rep_in, rep_ctx, mode)
                for _ in range(10)]
    sampler = fu.BatchSampler(m.chain_input, m.chain_context, 4, mode=mode, repeat_input=rep_in,
                              repeat_context=rep_ctx, rng=random.Random(5))
    for x_ref, c_ref in expected:
        x, c = sampler()
        assert torch.equal(x, x_ref) and torch.equal(c, c_ref)


def test_flat_batches_match_notebook(data):
    rho, theta = data
    flat_rho = torch.tensor(rho.reshape(-1, N_F))
    flat_theta = torch.tensor(np.repeat(theta[:, None], N_REAL, axis=1).reshape(-1, N_P))
    random.seed(9)
    rows = [random.sample(range(N_DRAWS * N_REAL), k=16) for _ in range(10)]
    sampler = fu.BatchSampler(torch.tensor(rho), torch.tensor(theta), 16, mode="flat",
                              rng=random.Random(9))
    for r in rows:
        x, c = sampler()
        assert torch.equal(x, flat_rho[r]) and torch.equal(c, flat_theta[r])


# ---- training ---------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["diagonal", "mesh"])
def test_training_matches_nfmaker(data, tmp_path, mode):
    rho, theta = data
    torch.manual_seed(1)
    m = _nfmaker(rho, theta, tmp_path / "pandora")
    random.seed(2)
    m.train(steps=60, batch_size=4, save_freq=20, repeat_input=False, repeat_context=True,
            do_validation=False, validation_input=None, validation_context=None, mode=mode,
            learning_rate=1e-3, progress_bar=False)

    torch.manual_seed(1)
    t = zf.FlowTrainer(rho, theta, str(tmp_path / "atlas"), 5, device=CPU, hidden_dims=HIDDEN)
    hist = t.train(60, 4, 20, mode=mode, repeat_context=True, learning_rate=1e-3, seed=2,
                   progress_bar=False)

    assert _state_equal(t.flow, m.flow)
    assert [s for s, _ in hist["checkpoints"]] == [20, 40, 59]
    assert sorted(os.listdir(tmp_path / "pandora")) == [f"flow_{s}steps.pkl" for s in (20, 40, 59)]
    assert hist["loss"].shape == (60,)


def test_training_matches_notebook_loop(data, tmp_path):
    rho, theta = data
    torch.manual_seed(3)
    random.seed(4)
    ref = notebook_flow.train_notebook_flow(rho, theta, steps=60, batch_size=16, lr=1e-3,
                                            hidden=HIDDEN)
    torch.manual_seed(3)
    t = zf.FlowTrainer(rho, theta, str(tmp_path), 5, device=CPU, hidden_dims=HIDDEN)
    t.train(60, 16, 1000, mode="flat", learning_rate=1e-3, seed=4, progress_bar=False)
    assert _state_equal(t.flow, ref)


# ---- Hellinger validation ----------------------------------------------------------

def test_hellinger_matches_validationhell(tmp_path):
    import jax.numpy as jnp

    rng = np.random.default_rng(2)
    val = rng.normal(size=(6, 400))
    lower, upper = val.min(axis=-1), val.max(axis=-1)
    vh = nf_dist.ValidationHell(str(tmp_path / "Hell.pdf"), "cpu")
    ref_hist = np.asarray(vh.batched_histogram(jnp.asarray(val), jnp.asarray(lower),
                                               jnp.asarray(upper), bins=15))
    hist = fu.batched_histogram(val, lower, upper, 15)
    assert np.array_equal(hist, ref_hist)

    # a sequence of flow-sample sets that drifts towards the validation set
    gens = [rng.normal(loc=mu, size=(6, 2000)) for mu in (0.8, 0.4, 0.2, 0.19, 0.18, 0.18)]
    vh.plot_hell_dist = lambda: None
    monitor = fu.HellingerMonitor()
    for gen in gens:
        vh.sample_from_nf = lambda *a, gen=gen, **k: jnp.asarray(gen)
        ref_dec = vh.make_decision(jnp.asarray(ref_hist), None, 2000, 1, jnp.asarray(lower),
                                   jnp.asarray(upper), None, progress_bar=False)
        hell = fu.hellinger(hist, fu.batched_histogram(gen, lower, upper, 15))
        assert np.allclose(hell, vh.hell[-1], rtol=1e-12, atol=0)
        assert monitor.update(hell) == ref_dec
    assert monitor.ll == vh.ll and monitor.ul == vh.ul


# ---- evaluation wrapper -------------------------------------------------------------

def _mapping(rng, p=N_P, f=N_F):
    return dict(B=5, mean=rng.normal(size=p + f), half_range=rng.uniform(0.5, 2, size=p + f))


@pytest.mark.parametrize("nf_type", ["rho|theta", "theta|rho"])
def test_wrapper_matches_nfastroinference(nf_type):
    rng = np.random.default_rng(3)
    mapping = _mapping(rng)
    torch.manual_seed(0)
    flow = zf.make_zuko_flow(N_F, N_P, hidden_features=HIDDEN) if nf_type == "rho|theta" \
        else zf.make_zuko_flow(N_P, N_F, hidden_features=HIDDEN)
    ref = nf_dist.NFastroinference(flow, nf_type, mapping["mean"], mapping["half_range"], 5,
                                   np.arange(N_P, N_P + N_F), np.arange(N_P), "cpu")
    ours = zf.ZukoAstroFlow(flow, nf_type, mapping, N_P)
    rho = mapping["mean"][N_P:] + rng.normal(size=(7, N_F)) * 0.5
    theta = mapping["mean"][:N_P] + rng.normal(size=(7, N_P)) * 0.5

    lp = ours.log_prob(rho, theta, jacobian=False)
    assert np.array_equal(lp, ref.log_prob(rho, theta))
    assert np.allclose(ours.log_prob(rho, theta) - lp, ours.log_jacobian(), rtol=0, atol=1e-12)
    half = mapping["half_range"][N_P:] if nf_type == "rho|theta" else mapping["half_range"][:N_P]
    assert np.isclose(ours.log_jacobian(), np.sum(np.log(5 / half)))

    ctx = theta[0] if nf_type == "rho|theta" else rho[0]
    torch.manual_seed(11)
    ref_s = ref.sample((50,), ctx)
    assert np.array_equal(ours.sample(50, ctx, seed=11), ref_s)


def test_unconditional_log_prob():
    """Pandora's ``nf_type='rho'`` log_prob calls a method zuko flows lack."""
    rng = np.random.default_rng(4)
    mapping = _mapping(rng)
    torch.manual_seed(0)
    flow = zf.make_zuko_flow(N_F, 0, hidden_features=HIDDEN)
    rho = mapping["mean"][N_P:] + rng.normal(size=(5, N_F))
    ref = nf_dist.NFastroinference(flow, "rho", mapping["mean"], mapping["half_range"], 5,
                                   np.arange(N_P, N_P + N_F), np.arange(N_P), "cpu")
    with pytest.raises(AttributeError):
        ref.log_prob(rho, None)
    lp = zf.ZukoAstroFlow(flow, "rho", mapping, N_P).log_prob(rho, jacobian=False)
    scaled = torch.tensor(5 * (rho - mapping["mean"][N_P:]) / mapping["half_range"][N_P:])
    assert np.array_equal(lp, flow().log_prob(scaled).detach().numpy())


@pytest.mark.parametrize("payload", ["nfmaker", "notebook"])
def test_from_pandora_pickles(tmp_path, payload):
    rng = np.random.default_rng(5)
    mapping = _mapping(rng)
    torch.manual_seed(0)
    flow = zf.make_zuko_flow(N_F, N_P, hidden_features=[12, 10, 8], transforms=2, bins=6)
    path = str(tmp_path / "flow.pkl")
    torch.save([flow, 5] if payload == "nfmaker" else [flow], path)
    np.savez_compressed(str(tmp_path / "x_mapping_data.npy"), **mapping)
    loaded = zf.ZukoAstroFlow.from_pandora(path, str(tmp_path / "x_mapping_data.npy.npz"))
    assert loaded.n_params == N_P
    rho = mapping["mean"][N_P:] + rng.normal(size=(5, N_F))
    theta = mapping["mean"][:N_P] + rng.normal(size=(5, N_P))
    direct = zf.ZukoAstroFlow(flow, "rho|theta", mapping, N_P)
    assert np.array_equal(loaded.log_prob(rho, theta), direct.log_prob(rho, theta))

    loaded.save(str(tmp_path / "saved"))
    again = zf.ZukoAstroFlow.load(str(tmp_path / "saved"))
    assert np.array_equal(again.log_prob(rho, theta), direct.log_prob(rho, theta))
    if payload == "nfmaker":
        torch.save([flow, 4], path)
        with pytest.raises(ValueError, match="B=4"):
            zf.ZukoAstroFlow.from_pandora(path, str(tmp_path / "x_mapping_data.npy.npz"))


# ---- bug fixes ----------------------------------------------------------------------

def _fail_at(trainer, calls):
    real, count = trainer._log_prob, [0]

    def flaky(x, c):
        count[0] += 1
        if count[0] in calls:
            raise AssertionError("spline")
        return real(x, c)

    trainer._log_prob = flaky


def test_spline_failure_reloads_and_keeps_training(data, tmp_path):
    rho, theta = data
    torch.manual_seed(0)
    t = zf.FlowTrainer(rho, theta, str(tmp_path), 5, device=CPU, hidden_dims=HIDDEN)
    _fail_at(t, {16})                      # step 15; checkpoint at step 10
    hist = t.train(40, 4, 10, learning_rate=1e-3, seed=0, progress_bar=False)
    ckpt10 = torch.load(str(tmp_path / "flow_10steps.pt"))
    # the reloaded flow kept training: its weights moved past the checkpoint
    assert any(not torch.equal(v, ckpt10[k]) for k, v in t.flow.state_dict().items())
    assert len(hist["loss"]) == 39


def test_spline_failure_before_first_checkpoint(data, tmp_path):
    rho, theta = data
    t = zf.FlowTrainer(rho, theta, str(tmp_path), 5, device=CPU, hidden_dims=HIDDEN)
    _fail_at(t, {3})
    with pytest.raises(RuntimeError, match="before the first checkpoint"):
        t.train(40, 4, 10, seed=0, progress_bar=False)


# ---- end to end ------------------------------------------------------------------------

def _end_to_end(tmp_path, device, nf_type="rho|theta"):
    pytest.importorskip("holodeck")
    import logging
    import holodeck
    from holodeck.constants import YR

    holodeck.log.setLevel(logging.WARNING)
    ps = astro.pandora_phenom_param_space(nsamples=8, sam_shape=10, seed=0)
    ts = astro.generate_training_set(ps, str(tmp_path / "set"), 20 * YR, n_real=60, seed=0)
    out = str(tmp_path / "flow")
    flow, hist = astro.train_astro_flow(
        ts, out, nf_type=nf_type, val=(2, 10) if nf_type != "theta|rho" else None,
        steps=300, batch_size=32, save_freq=100, mode="flat", device=device,
        hidden_dims=HIDDEN, train_kwargs=dict(learning_rate=3e-3, seed=0, val_ndraws=2000,
                                              progress_bar=False, plot=False))
    assert hist["loss"][-50:].mean() < hist["loss"][:50].mean()
    reloaded = zf.ZukoAstroFlow.load(out, device=device)
    rho, theta = ts.log10_rho[:, 0], ts.params
    assert np.array_equal(reloaded.log_prob(rho, theta), flow.log_prob(rho, theta))
    assert reloaded.metadata["param_names"] == list(astro.PANDORA_PHENOM_PARAM_NAMES)
    ckpt = zf.ZukoAstroFlow.load(out, device=device, state="flow_299steps.pt")
    assert np.array_equal(ckpt.log_prob(rho, theta), flow.log_prob(rho, theta))
    if nf_type != "theta|rho":
        assert len(hist["decisions"]) == 3
    return flow


@pytest.mark.parametrize("nf_type", zf.NF_TYPES)
def test_train_astro_flow_end_to_end(tmp_path, nf_type):
    _end_to_end(tmp_path, "cpu", nf_type)


@pytest.mark.gpu
def test_train_astro_flow_on_gpu(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    _end_to_end(tmp_path, "cuda")
