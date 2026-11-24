#!/usr/bin/env python
"""Flax flow training throughput: one jitted call per step vs scanned chunks.

    JAX_PLATFORMS=cpu  python bench/bench_flow_training.py
    JAX_PLATFORMS=cuda python bench/bench_flow_training.py
    python bench/bench_flow_training.py --steps 500 --batch 512 --target 5

``FlaxFlowTrainer`` keeps the training set on the device, draws batch
indices on the host (the same stream as the zuko trainer) and runs the steps
between checkpoints as one ``lax.scan`` -- Pelargir's training loop.
``steps_per_call=1`` dispatches and syncs every step instead, which is what
the trainer did before; the ratio is the cost of per-step dispatch.

The problem is Pandora-sized: 5 frequency bins, 6 astrophysical parameters,
512-wide hidden layers, batches of 512 in the notebook's 'flat' mode.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import jax

from bench.harness import machine, timeit, write_results
from ATLAS.experimental import flax_flows as ff

ARCHITECTURES = {
    "zuko_nsf": dict(bins=8, hidden_features=[512, 512], transforms=3),
    "coupling_rqs": dict(num_layers=4, hidden_sizes=[512, 512], num_bins=8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200, help="training steps per timed call")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--real", type=int, default=200)
    ap.add_argument("--target", type=float, default=3.0,
                    help="seconds per timing round (see bench/harness.timeit)")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    theta = rng.uniform(-5, 5, size=(args.draws, 6))
    rho = np.clip(theta[:, None, :5] * 0.5 + rng.normal(size=(args.draws, args.real, 5)), -5, 5)

    info = machine()
    info.update(steps=args.steps, batch=args.batch, draws=args.draws, real=args.real,
                features=5, context=6, mode="flat", device=str(jax.devices()[0]))
    print("  ".join(f"{k}={v}" for k, v in info.items() if k not in ("recorded",)))
    print()

    rows = {}
    for arch, kw in ARCHITECTURES.items():
        for label, spc in [("per-step dispatch", 1), ("scanned", None)]:
            with tempfile.TemporaryDirectory() as tmp:
                t = ff.FlaxFlowTrainer(rho, theta, tmp, 5, architecture=arch, arch_kwargs=kw)

                def run():
                    return t.train(args.steps, args.batch, args.steps, mode="flat",
                                   learning_rate=1e-4, seed=0, progress_bar=False,
                                   steps_per_call=spc)["loss"]

                timing = timeit(run, label=f"{arch} {label}", target_s=args.target,
                                rounds=args.rounds)
            rate = args.steps / (timing.median_ms / 1e3)
            print(f"{timing}   {rate:9.1f} steps/s")
            rows[f"{arch}/{label}"] = dict(timing.__dict__, steps_per_s=rate)
        speedup = (rows[f"{arch}/per-step dispatch"]["median_ms"]
                   / rows[f"{arch}/scanned"]["median_ms"])
        rows[f"{arch}/speedup"] = speedup
        print(f"-> {arch}: scanned chunks are {speedup:.2f}x faster\n")

    path = write_results("flow-training", dict(machine=info, results=rows))
    print(f"written: {path}")


if __name__ == "__main__":
    main()
