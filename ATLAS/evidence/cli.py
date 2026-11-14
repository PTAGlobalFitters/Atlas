"""``atlas-evidence``: thermodynamic integration over an atlas-run beta sweep.

    atlas-evidence estimate run_config.yaml            ln Z per model (group)
    atlas-evidence propose  run_config.yaml --n 32     append the next adaptive pass
    atlas-evidence bf       run_config.yaml --a CURN --b HD

A *group* is one model: all runs whose overrides are identical apart from
``sampler.beta``.  Its label is the run name with the ``__beta=...`` part
removed (and a trailing ``_p<k>`` pass suffix).
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from ATLAS.evidence.ti import point_from_samples, ti_estimate, propose_betas, TIPoint



def _label(name):
    lab = re.sub(r"__beta=[0-9eE.+-]+", "", name)
    return re.sub(r"_p\d+(?=__|$)", "", lab)


def _key(run):
    ov = {k: v for k, v in (run["run"]["overrides"] or {}).items() if k != "sampler.beta"}
    return json.dumps(ov, sort_keys=True, default=str)


def _load_point(run):
    out = Path(run["run"]["output_dir"])
    beta = float(run["sampler"]["beta"])
    summ = out / "summary.json"
    if summ.exists():
        ti = json.loads(summ.read_text()).get("ti")
        if ti:
            return TIPoint(beta, ti["mean"], ti["std"], ti["n_eff"], ti["n"], run["run"]["name"])
    npz = out / "samples.npz"
    if npz.exists():
        z = np.load(npz)
        for site in ("delta_switch", "lnL_partial"):
            if site in z.files:
                return point_from_samples(beta, z[site], run["run"]["name"])
    nc = out / "posterior.nc"
    if nc.exists():
        import xarray as xr
        ds = xr.open_dataset(nc, group="posterior")
        for site in ("delta_switch", "lnL_partial"):
            if site in ds:
                return point_from_samples(beta, ds[site].values, run["run"]["name"])
    return None


def collect(config):
    """{label: dict(key, overrides, points, missing, names)} for the beta runs."""
    from ATLAS.runner.config import load_config, expand_runs
    runs = expand_runs(load_config(config))
    groups = {}
    for r in runs:
        if r["sampler"]["beta"] is None:
            continue
        k = _key(r)
        g = groups.setdefault(k, dict(label=_label(r["run"]["name"]), points=[], missing=[],
                                      switch=bool(r["sampler"]["switch_overrides"]),
                                      overrides={kk: vv for kk, vv in r["run"]["overrides"].items()
                                                 if kk != "sampler.beta"}))
        p = _load_point(r)
        (g["points"].append(p) if p is not None else g["missing"].append(r["run"]["name"]))
    by_label = {}
    for g in groups.values():
        lab = g["label"]
        while lab in by_label:
            lab += "+"
        by_label[lab] = g
    return by_label


def _estimate(groups, require_endpoints=True):
    res = {}
    for lab, g in groups.items():
        if len(g["points"]) < 2:
            res[lab] = None
            continue
        try:
            res[lab] = ti_estimate(g["points"], require_endpoints=require_endpoints)
        except ValueError as exc:
            print(f"{lab}: {exc}", file=sys.stderr)
            res[lab] = None
    return res


def cmd_estimate(a):
    groups = collect(a.config)
    if not groups:
        sys.exit("no runs with sampler.beta set in this config")
    res = _estimate(groups, not a.allow_open_ends)
    out = {}
    for lab, g in groups.items():
        r = res[lab]
        miss = f"  ({len(g['missing'])} beta runs not finished)" if g["missing"] else ""
        txt = str(r) if r is not None else "not enough finished beta runs"
        if r is not None and g.get("switch"):
            txt = (txt.replace("ln Z =", "ln BF[beta=1 model / beta=0 model] =", 1)
                   + f"   log10 BF = {r.ln_z / np.log(10):.3f}")
        print(f"{lab}: {txt}{miss}")
        if r is not None:
            out[lab] = dict(r.as_dict(), missing=g["missing"])
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"-> {a.json}")


def cmd_bf(a):
    groups = collect(a.config)
    res = _estimate(groups, not a.allow_open_ends)
    for x in (a.a, a.b):
        if res.get(x) is None:
            sys.exit(f"no ln Z for group {x!r}; groups: {sorted(groups)}")
    from ATLAS.evidence.ti import bayes_factor
    ln_bf, err = bayes_factor(res[a.b], res[a.a])
    print(f"ln BF[{a.b} / {a.a}] = {ln_bf:.4f} +/- {err:.4f}   (log10 BF = {ln_bf / np.log(10):.3f})")


def cmd_propose(a):
    """Append one sweep per group with the next adaptive betas; write --out."""
    groups = collect(a.config)
    raw = yaml.safe_load(Path(a.config).read_text()) or {}
    sweeps = list(raw.get("sweeps") or [])
    existing = {s.get("name") for s in sweeps}
    added = 0
    for lab, g in groups.items():
        if a.group and lab not in a.group:
            continue
        if len(g["points"]) < 2:
            print(f"{lab}: needs at least two finished beta runs first", file=sys.stderr)
            continue
        new = propose_betas(g["points"], a.n, floor=a.floor)
        if len(new) == 0:
            print(f"{lab}: no new betas proposed", file=sys.stderr)
            continue
        k = 2
        while f"{lab}_p{k}" in existing:
            k += 1
        name = f"{lab}_p{k}"
        existing.add(name)
        sweeps.append(dict(name=name, base_overrides=copy.deepcopy(g["overrides"]),
                           grid={"sampler.beta": [float(round(b, 10)) for b in new]}))
        added += 1
        print(f"{lab}: +{len(new)} betas as sweep {name!r} "
              f"(range {new.min():.3g} .. {new.max():.3g})")
    if not added:
        sys.exit("nothing proposed")
    raw["sweeps"] = sweeps
    out = Path(a.out or a.config)
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    print(f"-> {out}   (atlas-run run {out} runs only the new betas when "
          "output.skip_completed is true)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="atlas-evidence", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("estimate"); s.add_argument("config")
    s.add_argument("--json", help="write the results here")
    s.add_argument("--allow-open-ends", action="store_true",
                   help="integrate even if beta=0 or 1 is missing (biased)")
    s.set_defaults(fn=cmd_estimate)
    s = sub.add_parser("bf"); s.add_argument("config")
    s.add_argument("--a", required=True, help="denominator group")
    s.add_argument("--b", required=True, help="numerator group")
    s.add_argument("--allow-open-ends", action="store_true")
    s.set_defaults(fn=cmd_bf)
    s = sub.add_parser("propose"); s.add_argument("config")
    s.add_argument("--n", type=int, default=32, help="new betas per group")
    s.add_argument("--group", nargs="+", help="only these groups")
    s.add_argument("--floor", type=float, default=0.02)
    s.add_argument("--out", help="write the extended config here (default: overwrite)")
    s.set_defaults(fn=cmd_propose)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
