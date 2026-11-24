# bench/

Benchmarks with a protocol, so that two numbers taken on different days are
comparable.

```bash
python bench/bench_gradient.py                                  # 2 synthetic pulsars
python bench/bench_gradient.py --fixture ng15_3 --npsr 3
python bench/bench_gradient.py --case ltm-gtm --target 5
```

Results land in `bench/results/<name>-<host>-<date>.json` with the machine,
backend, JAX version, loadavg and model configuration recorded alongside every
number.

## The protocol

**Repeat the fast side, not the slow one.** `harness.timeit` takes a *target
duration* and derives the repeat count, rather than taking a repeat count and
letting you accidentally measure the slow variant once. This is not pedantry:
contention distorts a 2.5 s measurement roughly six-fold and a 90 s one by under
1%, so a before/after where the "after" is fast and measured once is worthless.

**Record compiled memory, not just wall clock.** `memory_analysis()` reports
generated-code, argument and temp bytes. Those are load-independent and so
quotable without a caveat, unlike wall clock, and they are what actually decides
whether a configuration will load on a given card. A large *generated-code*
figure means constants have been baked into the executable — which is exactly
what a closed-over array on `self` does under `jit_method`, and how the NG15
helper build once reached 9.5 GB of generated code.

**Record loadavg with every timing.** Included in every row and in the JSON.

**Never quote a compile-time host-RAM high-water mark as a sampling footprint.**
Every host-RAM figure in the older profiling notes is a high-water mark during
XLA compilation. After the helper build, 30 consecutive gradient calls add 0 MB.
The two scale differently and conflating them overstates by roughly 2x.

## What the gradient benchmark measures

| row | meaning |
|---|---|
| `helper build (forward)` | one `Tᵀ N⁻¹ T` / `Tᵀ N⁻¹ r` assembly |
| `grad, helpers FROZEN` | a leapfrog step with `vary_white=False` |
| `grad, helpers REBUILT` | a leapfrog step with `vary_white=True` |
| `grad, partial_marg` | the same, through the marginalised-P-block likelihood |

The headline is the **ratio** of the two gradient rows: the price of letting
white noise vary, which is the dominant cost of a genuine global fit.

Backend caveat: the CPU backend attributes closed-over constants differently
from the GPU backend, so `generated_code` can read 0 B on CPU for a computation
that shows megabytes on GPU. Compare memory figures only within one backend —
which is why the backend is recorded in every result file.

## Flow benchmark: Haiku vs Flax NNX

```bash
python bench/bench_flows.py                  # astro.py-sized CW flow (D=268, C=6)
python bench/bench_flows.py --preset toy     # D=4, C=2: fixed per-call overhead
```

Times `experimental/flows.py` (Haiku) against `experimental/flax_flows.py`
(Flax NNX) with identical configuration, data and seed. Every row is Haiku ms,
NNX ms and the ratio NNX / Haiku. A parity check copies Haiku weights into the
NNX model and compares `log_prob` (~1e-12), so any timing difference comes from
the framework and not from the model.

RTX 4070 Laptop, jax 0.11.1, flax 0.12.10, 2026-09-30
(`results/flows-haiku-vs-flax-{astro,toy}-gpu-20260930.json`):

| row | astro, Flow | astro, CondFlow | toy, Flow | toy, CondFlow |
|---|---|---|---|---|
| train step (batch 256) | 1.01x | 1.02x | 1.02x | 1.02x |
| epoch, `fit('pre-loaded')` | 1.01x | 1.01x | 1.11x | 1.05x |
| `log_prob`, batch 1 | 0.95x | 1.00x | 1.18x | 1.29x |
| `log_prob`, batch 1024 | 1.00x | 1.01x | 1.00x | 0.97x |
| `forward_pass` in an outer jit (astro.py) | 1.00x | 1.01x | 1.00x | 1.00x |
| `sample(1024)` | 0.33x | 0.32x | 0.01x | 0.01x |
| second instance, first `log_prob` | 0.06x | 0.06x | 0.01x | 0.01x |

How to read it:

- **At the size astro.py uses, the two are equally fast.** At toy size NNX pays
  a fixed ~15-50 us per call to flatten the module pytree. That is visible only
  when the computation itself takes ~0.1 ms.
- `sample` is faster only because Haiku's `sample()` is not jitted.
- NNX compiles once per architecture. Haiku compiles once per *instance*,
  because its jitted methods close over `self.params`. For the same reason
  Haiku's compiled `forward_pass` holds 28 MB of weights as generated code at
  astro size (NNX: 2 kB), and those weights go stale after training.
- The first `log_prob` compile of `ConditionalFlow` is 1.7x slower under NNX
  (about +0.8 s, once).
- `live_fit` costs 2x at toy size, because each call writes the result back
  into `self.model`. At astro size it is 1.01x.
- Held-out NLL after 20 epochs, 3 seeds: at toy size the two agree within
  seed scatter. At astro size NNX finishes 0.003 nats/dim worse (0.03%, about
  3 standard deviations); the curves agree at epoch 10.
- The rows labelled "nnx.jit design" time the first NNX design (`nnx.jit`,
  module mutated in place). It costs 4.5-10x on batch-1 calls and ~2x per toy
  epoch, because it walks the module graph in Python on every call. It also
  raises `TraceContextError` when called under an outer `jax.jit`, as astro.py
  does. That is why `flax_flows.py` passes the module to a plain `jax.jit`.
