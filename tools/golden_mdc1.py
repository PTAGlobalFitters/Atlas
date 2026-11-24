#!/usr/bin/env python
"""The MDC1 golden run: an end-to-end fit checked against a recorded posterior.

    python tools/golden_mdc1.py                    # run and record
    python tools/golden_mdc1.py --check            # run and compare, don't rewrite

Writes ``tests/golden/mdc1_36psr.json``, which ``tests/test_golden_mdc1.py``
asserts against. Needs a GPU to be quick (~4 min on a 4090); runs on CPU, slowly.

Why this dataset. IPTA MDC1 Open-1 is 36 pulsars over 4.94 yr with a known
injected background, so the answer is checkable rather than merely plausible.
It reads from ``tests/fixtures/data/mdc1_36.npz`` -- pre-loaded through tempo2,
because the par files carry ``EPHVER 5`` and no ``UNITS`` line, so PINT reads
TCB as TDB, mis-scales F0 by L_B, and the residuals come out as a uniform hash
over one pulse period. That failure looks exactly like a broken sampler.

Why the health diagnostics are part of the gate. When this dataset *was* being
mis-read, the chain reported **0 divergences and acceptance probability 0.76** --
both healthy. What actually caught it: the ``z_a`` posterior standard deviation
(~1e-12 against the ~1 the transform guarantees), the fraction of iterations
pinned at the tree-depth cap (100% against ~0), and the final step size (1e-13
against ~1e-2). So those three are asserted, and divergences and acceptance
probability are recorded but explicitly not relied on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# NUTS on this problem grows a BFC pool that leaves CUDA no room to *load* the
# compiled kernel; a smaller pool is the fix, not a bigger one.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpyro
from numpyro.infer import MCMC, NUTS

from ATLAS.data import PTA_Data
from ATLAS.model import model_maker
from ATLAS.model_builder import ModelBuilder
from ATLAS.nMatrix.base import WhiteCov
from ATLAS.psd_functions import hd_orf, powerlaw
from tests.fixtures.pulsar import load_fixture

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "tests" / "golden" / "mdc1_36psr.json"
FIXTURE = ROOT / "tests" / "fixtures" / "data" / "mdc1_36.npz"

# The configuration this result is defined at. Changing any of it invalidates
# the recorded posterior.
CONFIG = dict(
    model_string="ltm|unc+cor->unc",
    n_gwb=8, n_irn=15,
    linear_timing=True, marg_timing=False,
    diag_white_cov=False, include_ecorr=False,
    orf="hd_orf",
    irn_prior=[-18.0, 0.0, -11.0, 7.0],
    gwb_prior=[-18.0, 0.0, -11.0, 7.0],
    warmup=700, samples=700, max_tree_depth=8, seed=170817,
    vary_white=True,
)
INJECTED = dict(gwb_log10_A=-13.301, gwb_gamma=13 / 3)


def _ess(chain):
    """Effective sample size from the initial-positive-sequence autocorrelation."""
    x = np.asarray(chain, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    if not np.any(x):
        return 0.0
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conjugate(f))[:n].real
    ac /= ac[0]
    total, k = 0.0, 1
    while k < n and ac[k] > 0.05:
        total += ac[k]
        k += 1
    return float(n / (1.0 + 2.0 * total))


def run():
    psrs, prov = load_fixture(FIXTURE)
    data = PTA_Data(
        psrs, num_gwb_bins=CONFIG["n_gwb"], num_irn_bins=CONFIG["n_irn"],
        num_dm_bins=None, adaptus_basis=None, adaptus_size=None,
        fixed_white_noise_params=None,
        linear_timing=CONFIG["linear_timing"], marg_timing=CONFIG["marg_timing"],
        diag_white_cov=CONFIG["diag_white_cov"], fixed_res=False,
        timfiles=None, parfiles=None, noise_dict=None, dm_ref_freq=1400,
    )
    m = ModelBuilder(data=data)
    # wn = WhiteCov(data=data, stabilize_TNT=True,
    #               include_ecorr=CONFIG["include_ecorr"])
    ## should run cleanly now
    wn = m.make_white_noise(stabilize_TNT=True)
    ilo, ilo_g, ihi, ihi_g = CONFIG["irn_prior"]
    glo, glo_g, ghi, ghi_g = CONFIG["gwb_prior"]
    rn = m.make_red_noise(
        CONFIG["model_string"], use_pulsar_tspan=False,
        irn_psd_function=powerlaw, gwb_psd_function=powerlaw,
        orf_function=hd_orf, dm_psd_function=None,
        irn_lower_bound_psd=jnp.array([ilo, ilo_g]),
        irn_upper_bound_psd=jnp.array([ihi, ihi_g]),
        gwb_lower_bound_psd=jnp.array([glo, glo_g]),
        gwb_upper_bound_psd=jnp.array([ghi, ghi_g]),
        dm_lower_bound_psd=None, dm_upper_bound_psd=None,
        upper_bound_orf=None, lower_bound_orf=None,
    )
    raw = jnp.concat(data.raw_residuals)
    wn_lo, wn_hi = wn.get_prior_bounds()

    print(f"{data.npsrs} pulsars, {int(sum(p.ntoa for p in psrs))} TOAs, "
          f"Tspan {data.pta_tspan / 31557600:.2f} yr, "
          f"{rn.nmodes} T columns/pulsar, "
          f"{data.npsrs * rn.nmodes} latent + "
          f"{len(rn.model.get_param_names())} red + {np.asarray(wn_lo).size} white")

    kernel = NUTS(model=model_maker, target_accept_prob=0.8,
                  max_tree_depth=CONFIG["max_tree_depth"])
    mcmc = MCMC(kernel, num_warmup=CONFIG["warmup"], num_samples=CONFIG["samples"],
                num_chains=1, progress_bar=True)
    t0 = time.time()
    mcmc.run(jrandom.key(CONFIG["seed"]),
             extra_fields=("diverging", "num_steps", "adapt_state.step_size"),
             raw_residuals=raw, super_sig=rn, vary_white=CONFIG["vary_white"],
             wn_lower_bound=wn_lo, wn_upper_bound=wn_hi, tm_model=None,
             helpers=None, save_red_coeff=False, marg_over_non_gwb=False)
    wall_min = (time.time() - t0) / 60.0

    s = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    red = np.asarray(s["red_noise"])
    names = list(rn.model.get_param_names())
    z_a = np.asarray(s["z_a"])
    nsteps = np.asarray(extra["num_steps"])
    cap = 2 ** CONFIG["max_tree_depth"] - 1

    ia, ig = names.index("gwb_log10_A"), names.index("gwb_gamma")
    summary = dict(
        recorded=time.strftime("%Y-%m-%dT%H:%M:%S"),
        backend=jax.default_backend(),
        jax=jax.__version__, numpyro=numpyro.__version__,
        config=CONFIG, injected=INJECTED, provenance=prov,
        npsr=int(data.npsrs), ntoa=int(sum(p.ntoa for p in psrs)),
        nmodes=int(rn.nmodes), wall_minutes=round(wall_min, 2),
        posterior=dict(
            gwb_log10_A=dict(mean=float(red[:, ia].mean()), std=float(red[:, ia].std()),
                             ess=_ess(red[:, ia])),
            gwb_gamma=dict(mean=float(red[:, ig].mean()), std=float(red[:, ig].std()),
                           ess=_ess(red[:, ig])),
        ),
        health=dict(
            z_a_std=float(z_a.std()),
            frac_at_tree_depth_cap=float(np.mean(nsteps >= cap)),
            mean_num_steps=float(nsteps.mean()),
            final_step_size=float(np.asarray(extra["adapt_state.step_size"])[-1]),
            # Recorded, but NOT part of the gate: both looked healthy while the
            # data was being mis-read.
            n_divergent=int(np.sum(np.asarray(extra["diverging"]))),
        ),
    )
    return summary


def warn_if_no_gpu():
    """Fall back to CPU, loudly. The fit still runs, but takes far longer
    (over 30 min on a many-core CPU against ~4 min on a 4090)."""
    backend = jax.default_backend()
    if backend == "gpu":
        return
    requested = os.environ.get("JAX_PLATFORMS")
    why = (f"JAX_PLATFORMS={requested!r} is set" if requested
           else "no GPU is available to JAX")
    print(f"golden_mdc1: WARNING: {why}; running the golden fit on {backend}, "
          f"which is much slower than on a GPU.", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare against the record instead of rewriting it")
    args = ap.parse_args()

    warn_if_no_gpu()
    summary = run()
    p = summary["posterior"]
    h = summary["health"]
    print("\n" + "=" * 74)
    print(f"gwb_log10_A  {p['gwb_log10_A']['mean']:+.4f} +- {p['gwb_log10_A']['std']:.4f}"
          f"   injected {INJECTED['gwb_log10_A']:+.4f}"
          f"   ({(p['gwb_log10_A']['mean'] - INJECTED['gwb_log10_A']) / p['gwb_log10_A']['std']:+.2f} sigma)")
    print(f"gwb_gamma    {p['gwb_gamma']['mean']:+.4f} +- {p['gwb_gamma']['std']:.4f}"
          f"   injected {INJECTED['gwb_gamma']:+.4f}"
          f"   ({(p['gwb_gamma']['mean'] - INJECTED['gwb_gamma']) / p['gwb_gamma']['std']:+.2f} sigma)")
    print(f"\nhealth   z_a std {h['z_a_std']:.4f} (want ~1)"
          f" | at tree cap {h['frac_at_tree_depth_cap'] * 100:.1f}% (want ~0)"
          f" | step size {h['final_step_size']:.2e} (want ~1e-2)")
    print(f"         mean leapfrog steps {h['mean_num_steps']:.0f}"
          f" | divergences {h['n_divergent']} (recorded, not gated)")
    print(f"\n{summary['wall_minutes']:.2f} min on {summary['backend']}")

    if args.check:
        if not RECORD.exists():
            raise SystemExit(f"no record at {RECORD}; run without --check first")
        old = json.loads(RECORD.read_text())
        for key in ("gwb_log10_A", "gwb_gamma"):
            a, b = old["posterior"][key], summary["posterior"][key]
            mc = np.hypot(a["std"], b["std"]) / np.sqrt(max(a["ess"], 1.0))
            print(f"  {key}: recorded {a['mean']:+.4f}, now {b['mean']:+.4f}, "
                  f"shift {abs(a['mean'] - b['mean']) / max(mc, 1e-12):.2f} MC sigma")
    else:
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps(summary, indent=2))
        print(f"\nwritten: {RECORD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
