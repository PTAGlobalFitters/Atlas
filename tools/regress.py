#!/usr/bin/env python
"""Compare two git revisions' log-densities and gradients, elementwise.

    python tools/regress.py --ref-a HEAD~1 --ref-b HEAD
    python tools/regress.py --null                 # HEAD vs HEAD, must be 0.0
    python tools/regress.py --ref-a v0 --ref-b HEAD --case ltm-gtm --markdown

This is the gate for Stage 1: a registry refactor that changes a number has a
bug. It formalises what the perf branch did by hand.

How it works, and why
---------------------
Each revision is checked out into a throwaway git worktree (sparse, ATLAS/ only
-- a full checkout would pull ~810 MB of committed datasets), and
``tools/_probe.py`` is run against it with that worktree first on ``sys.path``.

Two rules make the comparison trustworthy:

  * the *probe* is invariant. It is always this repo's copy, never the
    worktree's, so both revisions are exercised by identical driver code. An
    in-package harness would compare two revisions using two different
    implementations of the comparison.
  * the *inputs* are invariant. Every array the model is built from is baked
    into one ``.npz`` here, in the driver, so both revisions are evaluated at a
    bit-identical point. Nothing is regenerated inside the worktree.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROBE = Path(__file__).resolve().parent / "_probe.py"

# Arrays whose disagreement is reported but does not fail the gate: these
# describe the model rather than evaluate it.
DESCRIPTIVE = {"param_names", "nmodes", "curn_error"}


def sh(*cmd, cwd=None):
    return subprocess.run([str(c) for c in cmd], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout


def resolve(ref):
    return sh("git", "rev-parse", ref, cwd=ROOT).strip()


def make_inputs(case, path, npsr, fixture):
    """Bake every input array for one corpus case into a single .npz."""
    from tests import harness as H
    from tests.fixtures.synthetic import make_adaptus_basis

    kw = dict(H.CORPUS[case])
    psrs = H.load_psrs(fixture, npsr)
    n_gtm = int(kw.get("n_gtm", 0))
    n_dm = int(kw.get("n_dm", 0))
    n_det = int(kw.get("n_det", 0))
    n_gwb, n_irn = 4, 6

    payload = {"names": np.array([p.name for p in psrs], dtype="<U64")}
    for i, p in enumerate(psrs):
        payload[f"psr{i}_toas"] = np.asarray(p.toas, dtype=np.float64)
        payload[f"psr{i}_residuals"] = np.asarray(p.residuals, dtype=np.float64)
        payload[f"psr{i}_toaerrs"] = np.asarray(p.toaerrs, dtype=np.float64)
        payload[f"psr{i}_freqs"] = np.asarray(p.freqs, dtype=np.float64)
        payload[f"psr{i}_pos"] = np.asarray(p.pos, dtype=np.float64)
        payload[f"psr{i}_Mmat"] = np.asarray(p.Mmat, dtype=np.float64)
        payload[f"psr{i}_backend_flags"] = np.asarray(p.backend_flags, dtype="<U32")
        payload[f"psr{i}_pdist"] = np.asarray(p.pdist, dtype=np.float64)

    if n_gtm:
        basis, gtm_psd = make_adaptus_basis(psrs, n_gtm)
        for i, b in enumerate(basis):
            payload[f"adaptus{i}"] = np.asarray(b, dtype=np.float64)
        payload["gtm_psd"] = np.asarray(gtm_psd, dtype=np.float64)

    if n_det:
        payload["det_bounds"] = np.asarray(H.CW_BOUNDS, dtype=np.float64)
        for key, arr in zip(("det_params", "psr_phases", "psr_dists"),
                            H.cw_point(len(psrs))):
            payload[key] = np.asarray(arr, dtype=np.float64)

    payload["wn_vec"] = np.asarray(
        H.white_noise_vector(psrs, kw.get("include_ecorr", True)), dtype=np.float64)

    vals = []
    for _ in psrs:
        vals += [-15.0, 3.5]
    if n_dm:
        for _ in psrs:
            vals += [-15.5, 2.5]
    vals += [-14.8, 13 / 3]
    payload["red_params"] = np.array(vals, dtype=np.float64)

    # One generous z block; the probe slices what each likelihood needs.
    ncol_max = 64
    payload["z_unit"] = np.random.default_rng(20260831).normal(
        size=(len(psrs), ncol_max))

    np.savez(path, **payload)
    spec = dict(
        model_string=kw["model_string"],
        linear_timing=bool(kw.get("linear_timing", False)),
        marg_timing=bool(kw.get("marg_timing", False)),
        include_ecorr=bool(kw.get("include_ecorr", True)),
        stabilize_TNT=False,
        orf="zero",
        n_gwb=n_gwb, n_irn=n_irn, n_dm=n_dm, n_gtm=n_gtm, n_det=n_det,
    )
    return spec


def add_worktree(sha, dest):
    """Sparse worktree carrying only ATLAS/ -- a full checkout is ~810 MB."""
    sh("git", "worktree", "add", "--no-checkout", "--detach", dest, sha, cwd=ROOT)
    sh("git", "-C", dest, "sparse-checkout", "init", "--cone")
    sh("git", "-C", dest, "sparse-checkout", "set", "ATLAS")
    sh("git", "-C", dest, "checkout")


def run_probe(worktree, inputs, spec_path, out, env_extra=None):
    import os
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env["PYTHONPATH"] = str(worktree)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(
        [sys.executable, str(PROBE), "--worktree", str(worktree),
         "--inputs", str(inputs), "--spec", str(spec_path), "--out", str(out)],
        cwd=str(worktree), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"probe failed in {worktree}:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


def compare(a_path, b_path):
    a = np.load(a_path, allow_pickle=False)
    b = np.load(b_path, allow_pickle=False)
    rows = []
    for key in sorted(set(a.files) | set(b.files)):
        if key not in a.files or key not in b.files:
            rows.append((key, None, None, "present in only one revision"))
            continue
        if key in DESCRIPTIVE:
            same = np.array_equal(a[key], b[key])
            rows.append((key, 0.0 if same else np.nan, 0.0 if same else np.nan,
                         "identical" if same else "CHANGED"))
            continue
        x, y = np.asarray(a[key], dtype=np.float64), np.asarray(b[key], dtype=np.float64)
        if x.shape != y.shape:
            rows.append((key, None, None, f"shape {x.shape} vs {y.shape}"))
            continue
        # A NaN or Inf anywhere is a failure on its own terms, and must be
        # caught here: `d.max()` and `rel.max()` would both come back non-finite,
        # and a non-finite maximum compares False against any tolerance -- so a
        # broken likelihood or gradient would slip through the gate silently.
        n_a, n_b = int((~np.isfinite(x)).sum()), int((~np.isfinite(y)).sum())
        if n_a or n_b:
            rows.append((key, None, None,
                         f"NON-FINITE: {n_a} in A, {n_b} in B, of {x.size}"))
            continue
        d = np.abs(x - y)
        scale = np.maximum(np.abs(x), np.abs(y))
        rel = np.divide(d, scale, out=np.zeros_like(d), where=scale > 0)
        rows.append((key, float(d.max()), float(rel.max()), ""))
    return rows


def _row_ok(row, tol):
    """Whether one compared array passes the gate. Fails closed."""
    key, _, mrel, note = row
    if key in DESCRIPTIVE:
        return note == "identical"
    if mrel is None or not np.isfinite(mrel):
        return False
    return bool(note == "" and mrel <= tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-a", default="HEAD")
    ap.add_argument("--ref-b", default="HEAD")
    ap.add_argument("--null", action="store_true",
                    help="HEAD vs HEAD; every difference must be exactly 0.0")
    ap.add_argument("--case", action="append", default=None,
                    help="corpus case (repeatable); default all")
    ap.add_argument("--npsr", type=int, default=2)
    ap.add_argument("--fixture", default="synth")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="max relative difference treated as a pass")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--keep", action="store_true", help="leave the worktrees in place")
    args = ap.parse_args()

    from tests import harness as H
    cases = args.case or list(H.CORPUS)

    sha_a, sha_b = resolve(args.ref_a), resolve(args.ref_b)
    if args.null:
        sha_a = sha_b = resolve("HEAD")
    print(f"A = {args.ref_a} @ {sha_a[:9]}")
    print(f"B = {args.ref_b} @ {sha_b[:9]}")
    if sha_a == sha_b:
        print("same revision: this is the harness's null test, expect exactly 0.0")

    tmp = Path(tempfile.mkdtemp(prefix="atlas-regress-"))
    worktrees = []
    failures = 0
    try:
        for tag, sha in (("a", sha_a), ("b", sha_b)):
            wt = tmp / f"wt_{tag}_{sha[:9]}"
            add_worktree(sha, wt)
            worktrees.append(wt)

        for case in cases:
            inputs = tmp / f"inputs_{case}.npz"
            spec = make_inputs(case, inputs, args.npsr, args.fixture)
            spec_path = tmp / f"spec_{case}.json"
            spec_path.write_text(json.dumps(spec, indent=2))

            outs = []
            for tag, wt in zip(("a", "b"), worktrees):
                out = tmp / f"out_{case}_{tag}.npz"
                run_probe(wt, inputs, spec_path, out)
                outs.append(out)

            rows = compare(*outs)
            worst = max((r[2] for r in rows if r[2] is not None and np.isfinite(r[2])),
                        default=0.0)
            # Fail closed: a row passes only if it is descriptive-and-identical,
            # or carries a finite relative difference within tolerance. Anything
            # else -- a missing array, a shape change, a non-finite value, an
            # unparseable comparison -- is a failure. The previous form
            # enumerated failure notes, so a note it did not know about read as
            # a pass.
            bad = [r for r in rows if not _row_ok(r, args.tol)]
            status = "PASS" if not bad else "FAIL"
            if bad:
                failures += 1
            print(f"\n{case:14s} [{spec['model_string']}]  {status}"
                  f"   worst rel = {worst:.3e}")
            if args.markdown:
                print("\n| array | max abs | max rel | note |")
                print("|---|---|---|---|")
            for row in rows:
                key, mabs, mrel, note = row
                flag = "" if _row_ok(row, args.tol) else "  <-- FAIL"
                if args.markdown:
                    fa = "-" if mabs is None else f"{mabs:.3e}"
                    fr = "-" if mrel is None else f"{mrel:.3e}"
                    print(f"| `{key}` | {fa} | {fr} | {note} |")
                elif note or flag or mrel is None or mrel > 0:
                    fa = "-" if mabs is None else f"{mabs:.3e}"
                    fr = "-" if mrel is None else f"{mrel:.3e}"
                    print(f"    {key:20s} abs {fa:>11s}  rel {fr:>11s}  {note}{flag}")

        print(f"\n{len(cases) - failures}/{len(cases)} cases within {args.tol:.0e}")
        if sha_a == sha_b and failures == 0:
            print("null test passed")
        return 1 if failures else 0
    finally:
        if not args.keep:
            for wt in worktrees:
                subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                               cwd=ROOT, capture_output=True)
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"\nworktrees kept under {tmp}")


if __name__ == "__main__":
    sys.exit(main())
