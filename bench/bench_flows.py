#!/usr/bin/env python
"""Haiku (experimental/flows.py) vs Flax NNX (experimental/flax_flows.py).

    python bench/bench_flows.py                   # astro.py-sized CW flow
    python bench/bench_flows.py --preset toy      # D=4, C=2: per-call overhead
    python bench/bench_flows.py --D 134 --target 5

Both implementations are built with the same configuration, data and seed and
timed on the default backend. Every row reports Haiku ms, NNX ms and the
ratio NNX / Haiku (below 1 means NNX is faster). Timings go through
bench/harness.timeit, so the fast side is repeated rather than the slow side
measured once.

Sections:
  first call      compile + run of a fresh instance, then of a second
                  instance with the same architecture
  training        one optimiser step, one epoch of fit('pre-loaded'), and
                  (ConditionalFlow) one live_fit call
  evaluation      log_prob / forward_pass at batch 1 and 1024, sample(1024),
                  forward_pass inside an outer jax.jit with the flow static
                  (how astro.py calls it)
  memory          compiled generated-code / argument / temp bytes
  quality         held-out NLL after the same training, over several seeds
  parity          max |Δ log_prob| after copying Haiku weights into NNX

Rows marked "nnx.jit design" time the first NNX design (nnx.jit, module
mutated in place) on the same operations, to show why flax_flows.py uses a
plain jax.jit over the module as a pytree instead.

Caveats the numbers need:
  - Haiku's public jitted methods close over self.params, so its compiled
    log_prob / forward_pass carry the weights as constants (and go stale
    after training). That shows up in the memory section, and it is why the
    quality section evaluates Haiku through log_prob_fn.apply instead.
  - Haiku's sample() is not jitted (op-by-op dispatch); NNX's is.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

import distrax
import flax
import haiku
import optax
from flax import nnx

from ATLAS.experimental import flows as hkf
from ATLAS.experimental import flax_flows as ff
from bench.harness import (Timing, compiled_memory, fmt_bytes, machine,
                           timeit, write_results)

PRESETS = {
    # astro.py make_flow / make_conditional_flow defaults; D = 2 x 67 pulsars
    # x 2 CW bins, a few astrophysical context parameters.
    "astro": dict(D=268, C=6, N=16384, layers=4, hidden=128, mlp_layers=2,
                  bins=8, batch=256, p=0.1, B=4.0),
    "toy": dict(D=4, C=2, N=16384, layers=4, hidden=128, mlp_layers=2,
                bins=8, batch=256, p=0.1, B=4.0),
}
COEFF_MAX = 1e-4          # astro.py's coeff_max_absolute_value


# ─────────────────────────────────────────────────────────────────────────────
# Data and construction
# ─────────────────────────────────────────────────────────────────────────────

def make_data(cfg, n, key):
    """Heavy-tailed CW-like coefficients whose scale depends on the context."""
    kc, kt = jr.split(key)
    ctx = np.asarray(0.1 + jr.uniform(kc, (n, cfg["C"])))
    scale = 1e-5 * (0.5 + ctx[:, :1])
    x = np.clip(scale * np.asarray(jr.t(kt, 3.0, (n, cfg["D"]))), -COEFF_MAX, COEFF_MAX)
    return x, ctx


def flow_kwargs(cfg, seed):
    D = cfg["D"]
    return dict(data_min=np.full(D, -COEFF_MAX), data_max=np.full(D, COEFF_MAX),
                flow_num_layers=cfg["layers"], hidden_size=cfg["hidden"],
                mlp_num_layers=cfg["mlp_layers"], num_bins=cfg["bins"],
                learning_rate=1e-4, B=cfg["B"], p=cfg["p"], seed=seed)


def cond_kwargs(cfg, seed):
    kw = flow_kwargs(cfg, seed)
    kw.update(context_min=np.full(cfg["C"], 0.05), context_max=np.full(cfg["C"], 1.15))
    return kw


def build(impl, kind, cfg, x, c, seed=0):
    mod = hkf if impl == "haiku" else ff
    if kind == "Flow":
        return mod.Flow(x, **flow_kwargs(cfg, seed))
    return mod.ConditionalFlow(x, c, **cond_kwargs(cfg, seed))


# ─────────────────────────────────────────────────────────────────────────────
# Uniform call adapters (so every row times the same operation on both sides)
# ─────────────────────────────────────────────────────────────────────────────

class Stepper:
    """One optimiser step with state threaded through (Haiku's ConditionalFlow
    update donates its input buffers, so they cannot be reused)."""

    def __init__(self, impl, kind, flow):
        self.impl, self.kind, self.flow = impl, kind, flow

    def __call__(self, u, c):
        f = self.flow
        if self.impl == "nnx":
            f.model, f.optimizer, loss = ff._train_step(
                f.model, f.optimizer, u, None if self.kind == "Flow" else c)
            return loss
        if self.impl == "nnx.jit":
            return _nnx_jit_step(f.model, f.optimizer, u, None if self.kind == "Flow" else c)
        if self.kind == "Flow":
            f.params, f.opt_state = f.update(f.params, f.opt_state, u)
        else:
            f.params, f.opt_state = f._update(f.params, f.opt_state, u, c)
        return f.params


# The first NNX design, kept for comparison: nnx.jit mutates the module in
# place, but walks the module graph in Python on every call, and cannot be
# called under an outer jax transform (TraceContextError).
@nnx.jit
def _nnx_jit_step(model, optimizer, u, c):
    loss, grads = nnx.value_and_grad(lambda m: -jnp.mean(m.log_prob(u, c)))(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def _nnx_jit_log_prob(model, x, c, xn, cn):
    u = ff._from_coefficients_jnp(x, *xn)
    return (model.log_prob(u, ff._maybe_normalize(c, cn))
            + ff._logdet_from_coefficients_jnp(x, *xn))


def nnx_jit_log_prob(kind, flow):
    cn = None if kind == "Flow" else flow._cn
    return lambda x, c: _nnx_jit_log_prob(flow.model, x, None if kind == "Flow" else c, flow._xn, cn)


def nnx_jit_epoch(kind, flow, bs):
    """fit('pre-loaded')'s loop with the nnx.jit step."""
    def run():
        flow.key, sub = jr.split(flow.key)
        if kind == "Flow":
            for batch in flow.get_batches(flow.x_norm, bs, sub):
                _nnx_jit_step(flow.model, flow.optimizer, batch, None)
        else:
            for xb, cb in ff._iter_batches(flow._data, flow._context, flow.x_norm, flow.c_norm, bs, sub):
                _nnx_jit_step(flow.model, flow.optimizer, xb, cb)
        return flow.params
    return run


def normalized_batch(kind, flow, x, c):
    if kind == "Flow":
        return flow.to_unit_interval(x), None
    return flow.normalize_data(x), flow.normalize_context(c)


def call_log_prob(kind, flow):
    return (lambda x, c: flow.log_prob(x)) if kind == "Flow" else (lambda x, c: flow.log_prob(x, c))


def call_forward(kind, flow):
    return (lambda z, c: flow.forward_pass(z)) if kind == "Flow" else (lambda z, c: flow.forward_pass(z, c))


def call_sample(kind, flow):
    return (lambda n, c: flow.sample(n)) if kind == "Flow" else (lambda n, c: flow.sample(c, n))


@partial(jax.jit, static_argnums=(1,))
def _nested_flow(z, flow):
    return flow.forward_pass(z=z).sum(axis=0)


@partial(jax.jit, static_argnums=(1,))
def _nested_cond(z, flow, c):
    return flow.forward_pass(c=c, z=z).sum(axis=0)


def call_nested(kind, flow):
    """forward_pass inside an outer jit with the flow as a static argument, as
    in astro.py's get_gwb_coeff_clt / _nonclt."""
    return (lambda z, c: _nested_flow(z, flow)) if kind == "Flow" else (lambda z, c: _nested_cond(z, flow, c))


def epoch_fn(impl, kind, flow, batch):
    def run():
        flow.fit(num_epochs=1, batch_size=batch)
        return flow.params
    return run


def heldout_nll(impl, kind, flow, x, c):
    """Mean NLL per dimension, always from the current weights (Haiku's public
    log_prob would reuse the weights from its first call)."""
    if impl == "nnx":
        lp = flow.log_prob(x) if kind == "Flow" else flow.log_prob(x, c)
    elif kind == "Flow":
        lp = (flow.log_prob_fn.apply(flow.params, flow.to_unit_interval(x))
              + flow.logdet_to_unit_interval(x))
    else:
        lp = (flow.log_prob_fn.apply(flow.params, flow.normalize_data(x), flow.normalize_context(c))
              + flow._logdet_normalize_data(x))
    return float(-jnp.mean(lp)) / x.shape[1]


def haiku_to_nnx(hk_params, model, n_layers, n_hidden):
    """Copy Haiku conditioner weights into the NNX model (benchmark-only)."""
    state = nnx.state(model, nnx.Param)
    pure = nnx.to_pure_dict(state)
    for i, ck in enumerate(sorted(pure["conditioners"])):
        mlp = "mlp" if i == 0 else f"mlp_{i}"
        lin = "linear" if i == 0 else f"linear_{i}"
        cond = pure["conditioners"][ck]
        for j, hkey in enumerate(sorted(cond["hidden"])):
            cond["hidden"][hkey]["kernel"] = hk_params[f"{mlp}/~/linear_{j}"]["w"]
            cond["hidden"][hkey]["bias"] = hk_params[f"{mlp}/~/linear_{j}"]["b"]
        cond["out"]["kernel"] = hk_params[lin]["w"]
        cond["out"]["bias"] = hk_params[lin]["b"]
    nnx.replace_by_pure_dict(state, pure)
    nnx.update(model, state)


def first_call_ms(fn, *args):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) * 1e3


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_kind(kind, cfg, args, x, c, x_te, c_te, rows, extra):
    bs = cfg["batch"]
    xb, cb = jnp.asarray(x[:bs]), jnp.asarray(c[:bs])
    x1, c1 = jnp.asarray(x_te[:1]), jnp.asarray(c_te[:1])
    xk, ck = jnp.asarray(x_te[:1024]), jnp.asarray(c_te[:1024])
    zk = jr.normal(jr.PRNGKey(7), (1024, cfg["D"]))
    tkw = dict(target_s=args.target, rounds=args.rounds)

    def row(section, label, h, n, note=""):
        r = dict(kind=kind, section=section, label=label, haiku=asdict(h), nnx=asdict(n),
                 ratio=n.median_ms / h.median_ms, note=note)
        rows.append(r)
        print(f"  {kind:15s} {label:34s} {h.median_ms:10.3f} {n.median_ms:10.3f}"
              f" {r['ratio']:7.2f}x   (load {n.loadavg:.2f}){'  ' + note if note else ''}")

    # ── first call: must run before anything else compiles this architecture
    compile_rows = {}
    for impl in ("haiku", "nnx"):
        out = {}
        for inst in ("first instance", "second instance"):
            f = build(impl, kind, cfg, x, c, seed=0 if inst == "first instance" else 1)
            u, cn = normalized_batch(kind, f, xb, cb)
            out[inst] = dict(
                log_prob=first_call_ms(call_log_prob(kind, f), xk, ck),
                forward_pass=first_call_ms(call_forward(kind, f), zk, ck),
                train_step=first_call_ms(Stepper(impl, kind, f), u, cn),
            )
        compile_rows[impl] = out
    extra.setdefault("first_call_ms", {})[kind] = compile_rows
    for inst in ("first instance", "second instance"):
        for op in ("log_prob", "forward_pass", "train_step"):
            h, n = compile_rows["haiku"][inst][op], compile_rows["nnx"][inst][op]
            print(f"  {kind:15s} {'first call, ' + inst + ': ' + op:34s} {h:10.1f} {n:10.1f} {n / h:7.2f}x")

    flows = {impl: build(impl, kind, cfg, x, c) for impl in ("haiku", "nnx")}

    # ── training
    t = {}
    for impl, f in flows.items():
        u, cn = normalized_batch(kind, f, xb, cb)
        t[impl] = timeit(Stepper(impl, kind, f), u, cn, label="train step", **tkw)
    row("training", f"train step (batch {bs})", t["haiku"], t["nnx"])

    t = {impl: timeit(epoch_fn(impl, kind, f, bs), label="epoch", **tkw)
         for impl, f in flows.items()}
    row("training", f"epoch, fit('pre-loaded') N={cfg['N']}", t["haiku"], t["nnx"],
        note=f"{cfg['N'] // bs} steps")

    t_hk_epoch = t["haiku"]

    if kind == "ConditionalFlow":
        t = {}
        for impl, f in flows.items():
            u, cn = normalized_batch(kind, f, xb, cb)
            t[impl] = timeit(f.live_fit, u, cn, label="live_fit", **tkw)
        row("training", "live_fit (one step)", t["haiku"], t["nnx"],
            note="NNX includes writing the result back into self.model")

    # First NNX design (nnx.jit), same operations, for comparison.
    fj = build("nnx", kind, cfg, x, c)
    u, cn = normalized_batch(kind, fj, xb, cb)
    t_j = timeit(Stepper("nnx.jit", kind, fj), u, cn, label="train step nnx.jit", **tkw)
    rows_hk_step = [r for r in rows if r["kind"] == kind and r["label"].startswith("train step")][0]
    row("training (nnx.jit design)", f"train step (batch {bs}), nnx.jit",
        Timing(**rows_hk_step["haiku"]), t_j)
    t_j = timeit(nnx_jit_epoch(kind, fj, bs), label="epoch nnx.jit", **tkw)
    row("training (nnx.jit design)", "epoch, nnx.jit loop", t_hk_epoch, t_j)

    # ── evaluation
    for label, maker, args_ in (
        ("log_prob, batch 1", call_log_prob, (x1, c1)),
        ("log_prob, batch 1024", call_log_prob, (xk, ck)),
        ("forward_pass, batch 1", call_forward, (zk[:1], c1)),
        ("forward_pass, batch 1024", call_forward, (zk, ck)),
        ("forward_pass in outer jit, 1024", call_nested, (zk, ck)),
    ):
        t = {impl: timeit(maker(kind, f), *args_, label=label, **tkw) for impl, f in flows.items()}
        row("evaluation", label, t["haiku"], t["nnx"],
            note="flow is a static arg of an outer jax.jit (astro.py)" if "outer" in label else "")
    t = {impl: timeit(call_sample(kind, f), 1024, c1, label="sample", **tkw) for impl, f in flows.items()}
    row("evaluation", "sample(1024)", t["haiku"], t["nnx"], note="Haiku sample() is not jitted")
    hk_eval = {r["label"]: r for r in rows if r["kind"] == kind and r["section"] == "evaluation"}
    for label, args_ in (("log_prob, batch 1", (x1, c1)), ("log_prob, batch 1024", (xk, ck))):
        t_j = timeit(nnx_jit_log_prob(kind, flows["nnx"]), *args_, label=label, **tkw)
        row("evaluation (nnx.jit design)", label + ", nnx.jit",
            Timing(**hk_eval[label]["haiku"]), t_j)

    # ── compiled memory, as each implementation actually compiles its calls
    hf, nf = flows["haiku"], flows["nnx"]
    cn_ = None if kind == "Flow" else nf._cn
    mem = {
        "log_prob": {
            "haiku": compiled_memory(lambda x_, c_: call_log_prob(kind, hf)(x_, c_), xk, ck),
            "nnx": compiled_memory(lambda m, x_, c_: ff._log_prob_phys(
                m, x_, None if kind == "Flow" else c_, nf._xn, cn_), nf.model, xk, ck),
        },
        "forward_pass": {
            "haiku": compiled_memory(lambda z_, c_: call_forward(kind, hf)(z_, c_), zk, ck),
            "nnx": compiled_memory(lambda m, z_, c_: ff._forward_phys(
                m, z_, None if kind == "Flow" else c_, nf._xn, cn_), nf.model, zk, ck),
        },
    }
    extra.setdefault("memory", {})[kind] = mem
    for op, m in mem.items():
        for field in ("generated_code", "argument", "temp"):
            print(f"  {kind:15s} {'memory ' + op + ': ' + field:34s} "
                  f"{fmt_bytes(m['haiku'].get(field)):>10s} {fmt_bytes(m['nnx'].get(field)):>10s}")

    # ── parity: copy mildly perturbed Haiku weights into NNX
    hp = build("haiku", kind, cfg, x, c)
    leaves, tdef = jax.tree_util.tree_flatten(hp.params)
    ks = jr.split(jr.PRNGKey(99), len(leaves))
    hp.params = jax.tree_util.tree_unflatten(
        tdef, [l + 0.05 * jr.normal(k, l.shape) for l, k in zip(leaves, ks)])
    np_ = build("nnx", kind, cfg, x, c)
    haiku_to_nnx(hp.params, np_.model, cfg["layers"], cfg["mlp_layers"])
    lh, ln = call_log_prob(kind, hp)(xk, ck), call_log_prob(kind, np_)(xk, ck)
    par = dict(max_abs=float(jnp.max(jnp.abs(lh - ln))),
               max_rel=float(jnp.max(jnp.abs(lh - ln) / jnp.abs(lh))))
    extra.setdefault("parity", {})[kind] = par
    print(f"  {kind:15s} parity: max|Δ log_prob| = {par['max_abs']:.2e} (rel {par['max_rel']:.1e})")

    # ── quality: held-out NLL after identical training, several seeds
    if args.quality_epochs > 0:
        marks = sorted({1, max(1, args.quality_epochs // 2), args.quality_epochs})
        qual = {"epochs": marks, "haiku": [], "nnx": [], "train_s": {"haiku": [], "nnx": []}}
        for seed in range(args.seeds):
            for impl in ("haiku", "nnx"):
                f = build(impl, kind, cfg, x, c, seed=seed)
                curve, done, t0 = [], 0, time.perf_counter()
                for m in marks:
                    f.fit(num_epochs=m - done, batch_size=bs)
                    done = m
                    curve.append(heldout_nll(impl, kind, f, xk, ck))
                qual["train_s"][impl].append(time.perf_counter() - t0)
                qual[impl].append(curve)
        extra.setdefault("quality", {})[kind] = qual
        for impl in ("haiku", "nnx"):
            finals = [cv[-1] for cv in qual[impl]]
            print(f"  {kind:15s} held-out NLL/dim after {marks[-1]} epochs, {impl:5s}: "
                  f"{statistics.mean(finals):.4f} ± {statistics.pstdev(finals):.4f}  "
                  f"(curve {np.round(np.mean(qual[impl], axis=0), 4).tolist()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="astro", choices=list(PRESETS))
    for k in ("D", "C", "N", "layers", "hidden", "mlp_layers", "bins", "batch"):
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=int, default=None)
    ap.add_argument("--kinds", default="Flow,ConditionalFlow")
    ap.add_argument("--target", type=float, default=2.0,
                    help="seconds per timing round (see bench/harness.timeit)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--quality-epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    cfg = dict(PRESETS[args.preset])
    for k in cfg:
        if getattr(args, k, None) is not None:
            cfg[k] = getattr(args, k)

    x, c = make_data(cfg, cfg["N"], jr.PRNGKey(0))
    x_te, c_te = make_data(cfg, 4096, jr.PRNGKey(1))

    info = machine()
    info.update(preset=args.preset, **cfg, flax=flax.__version__, haiku=haiku.__version__,
                distrax=distrax.__version__, optax=optax.__version__)
    print("  ".join(f"{k}={v}" for k, v in info.items() if k != "recorded"))
    print(f"\n  {'':15s} {'':34s} {'haiku ms':>10s} {'nnx ms':>10s} {'nnx/hk':>8s}")

    rows, extra = [], {}
    for kind in args.kinds.split(","):
        bench_kind(kind, cfg, args, x, c, x_te, c_te, rows, extra)
        print()

    path = write_results(f"flows-haiku-vs-flax-{args.preset}",
                         dict(info=info, rows=rows, **extra))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
