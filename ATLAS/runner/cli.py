"""``atlas-run``: run every analysis in a config file.

    atlas-run init  DIR                     copy template config + priors into DIR
    atlas-run check CONFIG                  validate every run; print the plan
    atlas-run run   CONFIG [--local] [--only NAME ...]
    atlas-run status CONFIG                 state of every run (from status.json)
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

__all__ = ["main"]

TEMPLATES = Path(__file__).resolve().parent / "templates"


def _cmd_init(a):
    dest = Path(a.dir)
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("run_config.yaml", "priors.yaml"):
        target = dest / f
        if target.exists() and not a.force:
            print(f"{target} exists (use --force to overwrite)")
            continue
        shutil.copy(TEMPLATES / f, target)
        print(f"wrote {target}")
    return 0


def _cmd_check(a):
    from ATLAS.runner.config import ConfigError
    from ATLAS.runner.driver import plan_runs
    try:
        plan = plan_runs(a.config, only=a.only)
    except ConfigError as exc:
        print(f"CONFIG ERROR\n{exc}")
        return 2
    w = max(len(r["run"]["name"]) for r, _ in plan)
    for r, action in plan:
        print(f"{r['run']['name']:<{w}}  {action:<17}  {r['model']['basis_string']!r:44}  "
              f"-> {r['run']['output_dir']}")
    print(f"{sum(a == 'run' for _, a in plan)} to run, "
          f"{sum(a != 'run' for _, a in plan)} skipped")
    return 0


def _cmd_run(a):
    from ATLAS.runner.config import ConfigError
    from ATLAS.runner.driver import run
    try:
        results = run(a.config, executor="local" if a.local else "ray", only=a.only)
    except ConfigError as exc:
        print(f"CONFIG ERROR\n{exc}")
        return 2
    failed = [r for r in results if r.get("state") == "failed"]
    for r in results:
        line = f"{r.get('run_name')}: {r.get('state')}"
        if r.get("state") == "failed":
            line += f"  -- {r.get('error')}"
        elif r.get("seconds_total") is not None:
            line += f"  ({r['seconds_total']} s, {r.get('num_divergent', 0)} divergent)"
        print(line)
    return 1 if failed else 0


def _cmd_status(a):
    from ATLAS.runner.config import expand_runs, load_config
    from ATLAS.runner.execute import read_status
    for r in expand_runs(load_config(a.config)):
        st = read_status(r) or {"state": "not started"}
        extra = st.get("error") or (f"{st.get('seconds_total')} s" if "seconds_total" in st else "")
        print(f"{r['run']['name']:<40} {st['state']:<12} {extra}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="atlas-run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init"); s.add_argument("dir"); s.add_argument("--force", action="store_true")
    s.set_defaults(fn=_cmd_init)
    for name, fn in (("check", _cmd_check), ("run", _cmd_run)):
        s = sub.add_parser(name); s.add_argument("config")
        s.add_argument("--only", nargs="+", help="run only these run names")
        if name == "run":
            s.add_argument("--local", action="store_true",
                           help="run sequentially in this process instead of Ray")
        s.set_defaults(fn=fn)
    s = sub.add_parser("status"); s.add_argument("config"); s.set_defaults(fn=_cmd_status)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("ATLAS.runner").setLevel(logging.INFO)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
