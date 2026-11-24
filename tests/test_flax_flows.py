"""Differentiable flax flows (flax_flows.py).

The first part needs only JAX (it also runs in atlas-env-1): JAX
transformations, the numpyro wrapper, native training and persistence of
both architectures. The second part compares ``ZukoNSF`` with zuko's NSF and
Pandora (atlas-env-2): the spline, the masks, converted flows, and training
from identical initial weights and batches.
"""
import copy
import os

import numpy as np
import pytest

pytest.importorskip("flax")
pytest.importorskip("distrax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

from ATLAS.experimental import astro  # noqa: E402
from ATLAS.experimental import flax_flows as ff  # noqa: E402

pytestmark = pytest.mark.astro

F, P = 3, 2


def _mapping(seed=0, p=P, f=F):
    rng = np.random.default_rng(seed)
    return dict(B=5, mean=np.concatenate([rng.normal(size=p), np.full(f, -8.0)]),
                half_range=rng.uniform(0.5, 2, size=p + f))


def _flow(kind="zuko_nsf", nf_type="rho|theta", seed=0, trained_like=True):
    feats, ctx = {"rho|theta": (F, P), "theta|rho": (P, F), "rho": (F, 0)}[nf_type]
    arch = (dict(kind="zuko_nsf", features=feats, context=ctx, bins=6, hidden_features=[16, 16],
                 transforms=3) if kind == "zuko_nsf" else
            dict(kind="coupling_rqs", features=feats, context=ctx, num_layers=3,
                 hidden_sizes=[16, 16], num_bins=6))
    model = ff.make_flax_flow(arch, seed=seed)
    if kind == "coupling_rqs" and trained_like:
        # move the zero-initialized output layers off the identity
        state = nnx.state(model, nnx.Param)
        leaves, tdef = jax.tree.flatten(state)
        keys = jax.random.split(jax.random.key(seed), len(leaves))
        nnx.update(model, jax.tree.unflatten(tdef, [l + 0.05 * jax.random.normal(k, l.shape)
                                                   for l, k in zip(leaves, keys)]))
    return ff.FlaxAstroFlow(model, nf_type, _mapping(seed), P,
                            metadata=dict(param_names=["a", "b"]))


def _points(flow, n=6, seed=1):
    rng = np.random.default_rng(seed)
    rho = np.asarray(flow.mean_gwb) + rng.normal(size=(n, F)) * 0.4
    theta = np.asarray(flow.mean_ast) + rng.normal(size=(n, P)) * 0.4
    return jnp.asarray(rho), jnp.asarray(theta)


# ---- JAX properties ------------------------------------------------------------------------

def test_untrained_coupling_flow_is_its_base():
    flow = _flow("coupling_rqs", trained_like=False)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(5, F)))
    base = -0.5 * jnp.sum(x**2, -1) - 0.5 * F * jnp.log(2 * jnp.pi)
    assert np.allclose(flow.model.log_prob(x, jnp.zeros((5, P))), base, rtol=1e-12, atol=1e-12)


def _to_latent(model, x, c):
    if isinstance(model, ff.ZukoNSF):
        return model.forward_and_log_det(x, c)[0]
    return model._bijector(jnp.broadcast_to(c, x.shape[:-1] + c.shape[-1:])).inverse(x)


@pytest.mark.parametrize("kind", ["zuko_nsf", "coupling_rqs"])
def test_jit_vmap_grad(kind):
    flow = _flow(kind)
    rho, theta = _points(flow)
    lp = flow.log_prob(rho, theta)
    assert lp.shape == (6,)
    assert np.allclose(jax.jit(flow.log_prob)(rho, theta), lp, rtol=1e-12)
    assert np.allclose(jax.vmap(flow.log_prob)(rho, theta), lp, rtol=1e-12)
    # gradients in theta and rho against central differences
    f = lambda r, t: flow.log_prob(r, t)  # noqa: E731
    g_t = jax.grad(f, argnums=1)(rho[0], theta[0])
    g_r = jax.grad(f, argnums=0)(rho[0], theta[0])
    h = 1e-6
    for i in range(P):
        e = jnp.zeros(P).at[i].set(h)
        fd = (f(rho[0], theta[0] + e) - f(rho[0], theta[0] - e)) / (2 * h)
        assert np.isclose(g_t[i], fd, rtol=1e-5, atol=1e-7)
    for i in range(F):
        e = jnp.zeros(F).at[i].set(h)
        fd = (f(rho[0] + e, theta[0]) - f(rho[0] - e, theta[0])) / (2 * h)
        assert np.isclose(g_r[i], fd, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("kind", ["zuko_nsf", "coupling_rqs"])
def test_sample_is_differentiable_and_consistent(kind):
    flow = _flow(kind)
    _, theta = _points(flow)
    key = jax.random.key(3)
    g = jax.grad(lambda t: flow.sample(key, (4,), t).sum())(theta[0])
    assert np.all(np.isfinite(g)) and np.any(g != 0)
    # sampling inverts the density's transform: the draws map back to the base normals
    x = flow.sample(key, (5,), theta[0])
    z = _to_latent(flow.model, flow.transform_rho_to_scaled_interval(x),
                   flow.transform_params_to_scaled_interval(theta[0]))
    assert np.allclose(z, jax.random.normal(key, (5, F)), rtol=1e-9, atol=1e-9)


def test_apply_and_log_prob_fn():
    flow = _flow()
    rho, theta = _points(flow)
    graphdef, params, rest = flow.split()
    assert np.allclose(flow.apply(params, rho, theta), flow.log_prob(rho, theta), rtol=1e-14)
    assert np.allclose(flow.log_prob_fn()(rho, theta), flow.log_prob(rho, theta), rtol=1e-14)
    grads = jax.grad(lambda p: flow.apply(p, rho, theta).sum())(params)
    assert all(np.all(np.isfinite(g)) for g in jax.tree.leaves(grads))
    assert np.isclose(float(flow.log_prob(rho, theta)[0] - flow.log_prob(rho, theta, jacobian=False)[0]),
                      float(flow.log_jacobian()))


@pytest.mark.parametrize("nf_type", ff.NF_TYPES)
def test_nf_types(nf_type):
    flow = _flow(nf_type=nf_type)
    rho, theta = _points(flow)
    lp = flow.log_prob(rho, None if nf_type == "rho" else theta)
    assert lp.shape == (6,) and np.all(np.isfinite(lp))
    ctx = None if nf_type == "rho" else (theta[0] if nf_type == "rho|theta" else rho[0])
    s = flow.sample(jax.random.key(0), (7,), ctx)
    assert s.shape == (7, P if nf_type == "theta|rho" else F)


# ---- numpyro ---------------------------------------------------------------------------------

def test_flow_distribution():
    numpyro = pytest.importorskip("numpyro")
    from numpyro.infer import MCMC, NUTS

    flow = _flow()
    _, theta = _points(flow)
    d = ff.FlowDistribution(flow, theta[0])
    assert d.event_shape == (F,) and d.batch_shape == ()
    key = jax.random.key(4)
    x = d.sample(key, (5,))
    assert np.allclose(x, flow.sample(key, (5,), theta[0]))
    assert np.allclose(d.log_prob(x), flow.log_prob(x, theta[0]))
    assert ff.FlowDistribution(flow, theta).batch_shape == (6,)

    def model():
        numpyro.sample("rho", ff.FlowDistribution(flow, theta[0]))

    mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=3000, progress_bar=False)
    mcmc.run(jax.random.key(5))
    nuts = np.asarray(mcmc.get_samples()["rho"])
    direct = np.asarray(flow.sample(jax.random.key(6), (20000,), theta[0]))
    se = np.sqrt(nuts.var(0) / (len(nuts) / 4) + direct.var(0) / len(direct))
    assert np.all(np.abs(nuts.mean(0) - direct.mean(0)) < 4 * se)


# ---- training ---------------------------------------------------------------------------------

def _gaussian_data(n, seed):
    """x | c ~ N([c, -c/2], diag(0.6^2, 0.4^2)), c ~ U(-2, 2): known density."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(-2, 2, size=(n, 1))
    sd = np.array([0.6, 0.4])
    x = np.concatenate([c, -c / 2], axis=1) + rng.normal(size=(n, 2)) * sd
    true_lp = (-0.5 * (((x - np.concatenate([c, -c / 2], 1)) / sd) ** 2).sum(1)
               - np.log(sd).sum() - np.log(2 * np.pi))
    return x, c, true_lp


@pytest.mark.parametrize("arch", [("coupling_rqs", dict(num_layers=4, hidden_sizes=[32, 32], num_bins=8)),
                                  ("zuko_nsf", dict(bins=8, hidden_features=[32, 32], transforms=3))])
def test_learns_conditional_gaussian(tmp_path, arch):
    x, c, _ = _gaussian_data(20000, 0)
    xt, ct, true_lp = _gaussian_data(4000, 1)
    t = ff.FlaxFlowTrainer(x, c, str(tmp_path), 5, architecture=arch[0], arch_kwargs=arch[1])
    t.train(3000, 256, 1000, mode="diagonal", learning_rate=3e-3, seed=0, progress_bar=False)
    nll = -float(jnp.mean(t.model.log_prob(jnp.asarray(xt), jnp.asarray(ct))))
    assert nll - (-true_lp.mean()) < 0.05


def test_nonfinite_loss_recovery(tmp_path, monkeypatch):
    x, c, _ = _gaussian_data(2000, 2)
    arch = dict(num_layers=2, hidden_sizes=[8], num_bins=4)
    t = ff.FlaxFlowTrainer(x, c, str(tmp_path), 5, architecture="coupling_rqs", arch_kwargs=arch)
    real, count = ff._train_chunk, [0]

    def flaky(*args):
        count[0] += 1
        model, opt, losses = real(*args)
        return (model, opt, losses.at[3].set(jnp.nan)) if count[0] == 2 else (model, opt, losses)

    monkeypatch.setattr(ff, "_train_chunk", flaky)
    # chunks end at the checkpoints 10, 20, 30, 39; the second chunk (steps 11-20) fails
    hist = t.train(40, 32, 10, seed=0, progress_bar=False, learning_rate=1e-3)
    assert len(hist["loss"]) == 11 + 10 + 9
    assert [s for s, _ in hist["checkpoints"]] == [10, 30, 39]

    monkeypatch.setattr(ff, "_train_chunk",
                        lambda *a: (lambda m, o, l: (m, o, l * jnp.nan))(*real(*a)))
    t2 = ff.FlaxFlowTrainer(x, c, str(tmp_path / "b"), 5, architecture="coupling_rqs",
                            arch_kwargs=arch)
    with pytest.raises(RuntimeError, match="before the first checkpoint"):
        t2.train(40, 32, 10, seed=0, progress_bar=False)


@pytest.mark.parametrize("mode", ["diagonal", "mesh", "flat"])
@pytest.mark.parametrize("xp", ["numpy", "jax"])
def test_indices_then_gather_equals_call(mode, xp):
    import random

    from ATLAS.experimental.flow_utils import BatchSampler, gather

    rng = np.random.default_rng(0)
    arr = (lambda a: a) if xp == "numpy" else jnp.asarray
    for inputs, context in [(rng.normal(size=(12, 9, F)), rng.normal(size=(12, P))),
                            (rng.normal(size=(12, P)), rng.normal(size=(12, 9, F))),
                            (rng.normal(size=(12, 9, F)), None)]:
        inputs, context = arr(inputs), None if context is None else arr(context)
        kw = dict(mode=mode, repeat_input=True, repeat_context=True)
        a = BatchSampler(inputs, context, 4, rng=random.Random(3), **kw)
        b = BatchSampler(inputs, context, 4, rng=random.Random(3), **kw)
        for _ in range(5):
            xa, ca = a()
            xi, ci = b.indices()
            assert np.array_equal(xa, gather(inputs, xi))
            assert (ca is None and ci is None) or np.array_equal(ca, gather(context, ci))


def test_scanned_chunks_match_single_steps(tmp_path):
    x, c, _ = _gaussian_data(3000, 3)
    arch = dict(bins=6, hidden_features=[16, 16], transforms=2)
    runs = {}
    for name, spc in [("scan", None), ("steps", 1), ("chunks7", 7)]:
        t = ff.FlaxFlowTrainer(x, c, str(tmp_path / name), 5, arch_kwargs=arch, seed=4)
        h = t.train(60, 32, 25, mode="diagonal", learning_rate=2e-3, seed=5, progress_bar=False,
                    steps_per_call=spc)
        runs[name] = (h, ff._flat_params(t.model))
        assert [s for s, _ in h["checkpoints"]] == [25, 50, 59]
    for name in ("steps", "chunks7"):
        assert np.allclose(runs[name][0]["loss"], runs["scan"][0]["loss"], rtol=1e-10, atol=1e-12)
        for k, v in runs["scan"][1].items():
            assert np.allclose(runs[name][1][k], v, rtol=1e-10, atol=1e-12), (name, k)


def _small_set(n_draws=10, n_real=60, seed=0):
    rng = np.random.default_rng(seed)
    params = rng.normal(size=(n_draws, P)) + np.array([3.0, -2.5])
    rho = (-8 + 0.3 * rng.normal(size=(n_draws, n_real, F))
           + 0.2 * params[:, None, :1] - 0.1 * np.arange(F))
    return astro.TrainingSet(params=params, log10_rho=rho, param_names=["a", "b"],
                             f_conv=np.arange(1, F + 1) / 1e9, tspan=1e9)


@pytest.mark.parametrize("arch", ["zuko_nsf", "coupling_rqs"])
def test_train_astro_flow_flax_end_to_end(tmp_path, arch):
    ts = _small_set()
    out = str(tmp_path / "flow")
    kwargs = (dict(hidden_dims=[16, 16]) if arch == "zuko_nsf" else
              dict(arch_kwargs=dict(num_layers=3, hidden_sizes=[16, 16], num_bins=6)))
    flow, hist = astro.train_astro_flow(
        ts, out, val=(2, 10), steps=300, batch_size=64, save_freq=100, mode="flat",
        backend="flax", architecture=arch,
        train_kwargs=dict(learning_rate=3e-3, seed=0, val_ndraws=2000, progress_bar=False,
                          plot=False), **kwargs)
    assert isinstance(flow, ff.FlaxAstroFlow)
    assert hist["loss"][-50:].mean() < hist["loss"][:50].mean()
    assert len(hist["decisions"]) == 3
    rho, theta = ts.log10_rho[:, 0], ts.params
    reloaded = ff.FlaxAstroFlow.load(out)
    assert np.array_equal(reloaded.log_prob(rho, theta), flow.log_prob(rho, theta))
    assert reloaded.metadata["param_names"] == ["a", "b"]
    ckpt = ff.FlaxAstroFlow.load(out, params="flow_299steps.npz")
    assert np.array_equal(ckpt.log_prob(rho, theta), flow.log_prob(rho, theta))


def test_architectures_reach_similar_fits(tmp_path):
    """Both architectures, trained alike on a small holodeck set, fit held-out realizations.

    Scored on realizations of the training draws that were held out: with only
    a couple dozen draws, held-out *draws* measure overfitting noise rather
    than fit quality. On this set the coupling flow fits ~0.6 nats per sample
    worse than the NSF at the same size and step count, hence the 1-nat margin.
    """
    pytest.importorskip("holodeck")
    import logging
    import holodeck
    from holodeck.constants import YR

    holodeck.log.setLevel(logging.WARNING)
    ps = astro.pandora_phenom_param_space(nsamples=24, sam_shape=10, seed=0)
    ts = astro.generate_training_set(ps, str(tmp_path / "set"), 20 * YR, n_real=200, seed=0)

    def subset(sl):
        return astro.TrainingSet(params=ts.params, log10_rho=np.asarray(ts.log10_rho)[:, sl],
                                 param_names=ts.param_names, f_conv=ts.f_conv, tspan=ts.tspan)

    train, held = subset(slice(0, 150)), subset(slice(150, 200))
    nll = {}
    for arch, kw in [("zuko_nsf", dict(bins=8, hidden_features=[64, 64], transforms=3)),
                     ("coupling_rqs", dict(num_layers=4, hidden_sizes=[64, 64], num_bins=8))]:
        flow, _ = astro.train_astro_flow(train, str(tmp_path / arch), steps=3000, batch_size=256,
                                         save_freq=1000, mode="flat", backend="flax",
                                         architecture=arch, arch_kwargs=kw, seed=1,
                                         train_kwargs=dict(learning_rate=2e-3, seed=0,
                                                           progress_bar=False))
        rho = held.log10_rho.reshape(-1, held.n_freqs)
        theta = np.repeat(held.params, held.n_real, axis=0)
        nll[arch] = -float(jnp.mean(flow.log_prob(rho, theta)))
    assert abs(nll["zuko_nsf"] - nll["coupling_rqs"]) < 1.0, nll


# ---- against zuko and Pandora (atlas-env-2) -------------------------------------------------------

def _zuko():
    torch = pytest.importorskip("torch")
    pytest.importorskip("zuko")
    from ATLAS.experimental import zuko_flows as zf
    return torch, zf


def test_rqs_matches_zuko():
    torch, _ = _zuko()
    from zuko.transforms import MonotonicRQSTransform

    rng = np.random.default_rng(0)
    K, n = 6, 400
    w, h, d = (rng.normal(size=(n, K)) * 2, rng.normal(size=(n, K)) * 2,
               rng.normal(size=(n, K - 1)) * 2)
    x = rng.uniform(-7, 7, size=n)          # includes points outside the bound
    t = MonotonicRQSTransform(*(torch.tensor(a) for a in (w, h, d)))
    b = ff.ZukoRQS(*(jnp.asarray(a) for a in (w, h, d)))
    y_t, l_t = (v.numpy() for v in t.call_and_ladj(torch.tensor(x)))
    y_j, l_j = b.forward_and_log_det(jnp.asarray(x))
    assert np.allclose(y_j, y_t, rtol=1e-13, atol=1e-13)
    assert np.allclose(l_j, l_t, rtol=1e-12, atol=1e-12)
    assert np.allclose(b.inverse(jnp.asarray(y_t)), t.inv(torch.tensor(y_t)).numpy(),
                       rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("F_,C_,hid,T", [(5, 6, [16, 16], 3), (3, 0, [8], 2),
                                          (2, 4, [12, 10, 8], 4), (6, 2, [32], 5)])
def test_masks_and_conversion_match_zuko(F_, C_, hid, T):
    torch, zf = _zuko()
    torch.manual_seed(0)
    nsf = zf.make_zuko_flow(F_, C_, hidden_features=hid, transforms=T, bins=6)
    native = ff.ZukoNSF(F_, C_, bins=6, hidden_features=hid, transforms=T, rngs=nnx.Rngs(0))
    state = nsf.state_dict()
    for t in range(T):
        for j, layer in enumerate(native.layers[t]):
            assert np.array_equal(state[f"transform.transforms.{t}.hyper.{2 * j}.mask"].numpy().astype(bool),
                                  np.asarray(layer.mask[...]).astype(bool))
    m = ff.ZukoNSF.from_zuko(nsf)
    x = torch.randn(9, F_, dtype=torch.float64) * 3
    c = torch.randn(9, C_, dtype=torch.float64) if C_ else None
    dist = nsf(c) if C_ else nsf()
    cj = None if c is None else jnp.asarray(c.numpy())
    assert np.allclose(m.log_prob(jnp.asarray(x.numpy()), cj), dist.log_prob(x).detach().numpy(),
                       rtol=1e-12, atol=1e-12)
    z = torch.randn(9, F_, dtype=torch.float64)
    assert np.allclose(m.inverse(jnp.asarray(z.numpy()), cj), dist.transform.inv(z).detach().numpy(),
                       rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("nf_type", ff.NF_TYPES)
def test_from_zuko_astro_flow(nf_type):
    torch, zf = _zuko()
    feats, ctx = {"rho|theta": (F, P), "theta|rho": (P, F), "rho": (F, 0)}[nf_type]
    torch.manual_seed(1)
    zflow = zf.ZukoAstroFlow(zf.make_zuko_flow(feats, ctx, hidden_features=[16, 16]), nf_type,
                             _mapping(1), P, metadata=dict(param_names=["a", "b"], tspan=1.0))
    flow = ff.FlaxAstroFlow.from_zuko(zflow)
    rho, theta = _points(flow)
    th = None if nf_type == "rho" else np.asarray(theta)
    assert np.allclose(flow.log_prob(rho, th), zflow.log_prob(np.asarray(rho), th),
                       rtol=1e-12, atol=1e-11)
    assert flow.metadata["param_names"] == ["a", "b"]


def test_pandora_pickle_through_conversion(tmp_path):
    torch, zf = _zuko()
    nf_dist = pytest.importorskip("pandora.nf_dist")
    torch.manual_seed(2)
    nsf = zf.make_zuko_flow(F, P, hidden_features=[16, 16])
    mapping = _mapping(2)
    torch.save([nsf, 5], str(tmp_path / "flow.pkl"))
    np.savez_compressed(str(tmp_path / "m_mapping_data.npy"), **mapping)
    flow = ff.FlaxAstroFlow.from_zuko(
        zf.ZukoAstroFlow.from_pandora(str(tmp_path / "flow.pkl"), str(tmp_path / "m_mapping_data.npy.npz")))
    ref = nf_dist.NFastroinference(nsf, "rho|theta", mapping["mean"], mapping["half_range"], 5,
                                   np.arange(P, P + F), np.arange(P), "cpu")
    rho, theta = _points(flow)
    assert np.allclose(flow.log_prob(rho, theta, jacobian=False),
                       ref.log_prob(np.asarray(rho), np.asarray(theta)), rtol=1e-12, atol=1e-11)


@pytest.mark.parametrize("mode", ["diagonal", "mesh", "flat"])
def test_training_matches_zuko_trainer(tmp_path, mode):
    torch, zf = _zuko()
    rng = np.random.default_rng(0)
    theta = np.clip(rng.normal(size=(20, P)) * 2, -5, 5)
    rho = np.clip(theta[:, None, :1] + rng.normal(size=(20, 30, F)), -5, 5)
    torch.manual_seed(1)
    nsf = zf.make_zuko_flow(F, P, hidden_features=[16, 16])
    model = ff.ZukoNSF.from_zuko(copy.deepcopy(nsf))
    zt = zf.FlowTrainer(rho, theta, str(tmp_path / "z"), 5, device="cpu", flow=nsf)
    hz = zt.train(50, 4, 25, mode=mode, repeat_context=True, learning_rate=1e-3, seed=3,
                  progress_bar=False)
    ft = ff.FlaxFlowTrainer(rho, theta, str(tmp_path / "f"), 5, model=model)
    hf = ft.train(50, 4, 25, mode=mode, repeat_context=True, learning_rate=1e-3, seed=3,
                  progress_bar=False)
    assert np.allclose(hf["loss"], hz["loss"], rtol=1e-8, atol=1e-10)
    final = ff.ZukoNSF.from_zuko(zt.flow)
    for (k, a), (_, b) in zip(sorted(ff._flat_params(final).items()),
                              sorted(ff._flat_params(ft.model).items())):
        assert np.allclose(a, b, rtol=1e-7, atol=1e-10), k


# ---- GPU ------------------------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("arch", [("zuko_nsf", dict(bins=8, hidden_features=[64, 64], transforms=3)),
                                  ("coupling_rqs", dict(num_layers=4, hidden_sizes=[64, 64], num_bins=8))])
def test_training_on_gpu_matches_cpu(tmp_path, arch):
    """Run with JAX_PLATFORMS=cuda (conftest only sets a CPU default)."""
    if jax.default_backend() != "gpu":
        pytest.skip("needs JAX on a GPU")
    x, c, _ = _gaussian_data(20000, 0)

    def run(device, name):
        with jax.default_device(device):
            t = ff.FlaxFlowTrainer(x, c, str(tmp_path / name), 5, architecture=arch[0],
                                   arch_kwargs=arch[1], seed=0)
            h = t.train(200, 256, 100, learning_rate=2e-3, seed=0, progress_bar=False)
            assert t.inputs.devices() == {device}
            return h["loss"]

    gpu = run(jax.devices("gpu")[0], "gpu")
    cpu = run(jax.devices("cpu")[0], "cpu")
    assert np.allclose(gpu, cpu, rtol=1e-6, atol=1e-8)


@pytest.mark.gpu
def test_train_astro_flow_flax_on_gpu(tmp_path):
    if jax.default_backend() != "gpu":
        pytest.skip("needs JAX on a GPU")
    ts = _small_set()
    flow, hist = astro.train_astro_flow(
        ts, str(tmp_path / "flow"), val=(2, 10), steps=300, batch_size=64, save_freq=100,
        mode="flat", backend="flax", hidden_dims=[32, 32],
        train_kwargs=dict(learning_rate=3e-3, seed=0, val_ndraws=2000, progress_bar=False,
                          plot=False))
    assert hist["loss"][-50:].mean() < hist["loss"][:50].mean()
    assert len(hist["decisions"]) == 3
    leaf = jax.tree.leaves(nnx.state(flow.model, nnx.Param))[0]
    assert leaf.devices().pop().platform == "gpu"
