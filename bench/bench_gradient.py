#!/usr/bin/env python
"""What a leapfrog step costs, with and without rebuilding the helpers.

    python bench/bench_gradient.py                        # 2 synthetic pulsars
    python bench/bench_gradient.py --fixture ng15_3 --npsr 3
    python bench/bench_gradient.py --case ltm-gtm --target 5

The headline number is the ratio. With ``vary_white=False`` the helpers are
built once outside the sampler and a leapfrog step is just the phi assembly plus
a batched Cholesky. With ``vary_white=True`` they are rebuilt every step, over
TOA-sized tensors, and that rebuild is the dominant cost of a global fit.

Replaces the ad-hoc bench.py that used to sit in the repo root with hardcoded
paths to one MDC1 pickle.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import jax
import jax.numpy as jnp

from bench.harness import (compiled_memory, fmt_bytes, machine, timeit,
                           write_results)
from tests import harness as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="ltm", choices=list(H.CORPUS))
    ap.add_argument("--fixture", default="synth")
    ap.add_argument("--npsr", type=int, default=2)
    ap.add_argument("--target", type=float, default=2.0,
                    help="seconds per timing round (see bench/harness.timeit)")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    kw = dict(H.CORPUS[args.case])
    ec = kw.get("include_ecorr", True) and args.fixture != "mdc1_5"
    m = H.build(npsr=args.npsr, orf_name="hd", fixture=args.fixture,
                include_ecorr=ec, **kw)
    red = m.red_params()
    D = m.det_params()
    z = jnp.zeros((m.npsr, m.ncol))
    raw = jnp.concat(m.data.raw_residuals)

    info = machine()
    info.update(case=args.case, model_string=kw["model_string"],
                fixture=args.fixture, npsr=m.npsr,
                ntoa=int(sum(p.ntoa for p in m.psrs)),
                nmodes=m.rn.nmodes, include_ecorr=ec,
                latent_dims=int(m.npsr * m.ncol),
                n_red_params=len(m.rn.model.get_param_names()),
                n_wn_params=int(np.asarray(m.wn_vec).size))
    print("  ".join(f"{k}={v}" for k, v in info.items() if k not in ("recorded",)))
    print()

    build = jax.jit(lambda ww: m.rn.get_helpers(reff=raw, white_noise_params=ww))
    frozen = jax.jit(jax.grad(
        lambda q, zz: m.rn.lnposterior_reparam(m.helpers, q, zz, D_params=D)[0],
        argnums=(0, 1)))

    def joint(q, zz, ww):
        hh = m.rn.get_helpers(reff=raw, white_noise_params=ww)
        return m.rn.lnposterior_reparam(hh, q, zz, D_params=D)[0]
    rebuilt = jax.jit(jax.grad(joint, argnums=(0, 1, 2)))

    pmarg = jax.jit(jax.grad(
        lambda q, zz: m.rn.partial_marg_lnposterior(m.helpers, q, zz, D_params=D)[0],
        argnums=(0, 1)))
    zg = jnp.zeros((m.npsr, 2 * m.n_gwb))

    timings = [
        timeit(build, m.wn_vec, label="helper build (forward)",
               target_s=args.target, rounds=args.rounds),
        timeit(frozen, red, z, label="grad, helpers FROZEN",
               target_s=args.target, rounds=args.rounds,
               note="vary_white=False leapfrog step"),
        timeit(rebuilt, red, z, m.wn_vec, label="grad, helpers REBUILT",
               target_s=args.target, rounds=args.rounds,
               note="vary_white=True leapfrog step"),
        timeit(pmarg, red, zg, label="grad, partial_marg (frozen helpers)",
               target_s=args.target, rounds=args.rounds),
    ]
    for t in timings:
        print(t)

    ratio = timings[2].median_ms / timings[1].median_ms
    print(f"\n-> rebuilding the helpers costs {ratio:.2f}x per leapfrog step")

    mem = {
        "helper_build": compiled_memory(
            lambda ww: m.rn.get_helpers(reff=raw, white_noise_params=ww), m.wn_vec),
        "grad_frozen": compiled_memory(
            jax.grad(lambda q, zz: m.rn.lnposterior_reparam(m.helpers, q, zz)[0],
                     argnums=(0, 1)), red, z),
        "grad_rebuilt": compiled_memory(
            jax.grad(joint, argnums=(0, 1, 2)), red, z, m.wn_vec),
    }
    print("\ncompiled memory (load-independent, and what decides whether it loads)")
    print(f"  {'':16s} {'code':>12s} {'arguments':>12s} {'temp':>12s}")
    for k, v in mem.items():
        if "error" in v:
            print(f"  {k:16s} {v['error']}")
            continue
        print(f"  {k:16s} {fmt_bytes(v.get('generated_code')):>12s} "
              f"{fmt_bytes(v.get('argument')):>12s} {fmt_bytes(v.get('temp')):>12s}")

    path = write_results(
        f"gradient-{args.case}-{args.fixture}",
        dict(info=info, ratio=ratio,
             timings=[t.__dict__ for t in timings], memory=mem))
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
