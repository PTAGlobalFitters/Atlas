"""Expand a config into runs and execute them with Ray (or in-process).

    plan = plan_runs("run_config.yaml")      # validate everything, nothing runs
    results = run("run_config.yaml")          # prepare caches, then Ray tasks

Config errors in ANY run stop the whole submission before anything starts.
A run that fails while sampling only fails itself; the rest carry on.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from ATLAS.runner import data_io
from ATLAS.runner.config import ConfigError, expand_runs, load_config, validate_run
from ATLAS.runner.execute import is_completed, run_one

__all__ = ["plan_runs", "prepare", "run"]

log = logging.getLogger("ATLAS.runner")
_ATLAS_ROOT = str(Path(__file__).resolve().parents[2])


def plan_runs(config_path, only=None, check_files=True):
    """Load, expand and validate.  Returns ``[(run, action)]`` with action
    ``run`` | ``skip (completed)``; raises ``ConfigError`` listing every
    problem in every run."""
    cfg = load_config(config_path)
    runs = expand_runs(cfg)
    if only:
        unknown = sorted(set(only) - {r["run"]["name"] for r in runs})
        if unknown:
            raise ConfigError(f"--only names not in the config: {unknown}")
        runs = [r for r in runs if r["run"]["name"] in set(only)]
    errors = []
    for r in runs:
        try:
            validate_run(r, check_files=check_files)
        except ConfigError as exc:
            errors.append(str(exc))
    if errors:
        raise ConfigError("\n".join(errors))
    plan = []
    for r in runs:
        out = Path(r["run"]["output_dir"])
        if r["output"]["skip_completed"] and is_completed(r):
            plan.append((r, "skip (completed)"))
        elif out.exists() and any(out.iterdir()) and not r["output"]["overwrite"] \
                and not r["output"]["skip_completed"]:
            raise ConfigError(f"{out} is not empty; set output.overwrite or output.skip_completed")
        else:
            plan.append((r, "run"))
    return plan


def prepare(runs):
    """Driver-side, once per dataset: load and cache pulsars, and the
    solar-wind geometry for runs that use ``sw`` (PINT is slow and must not
    run once per Ray task)."""
    from ATLAS.signals.signals_utils import parse_basis_string
    done = set()
    for r in runs:
        key = data_io.data_key(r)
        if key not in done:
            data_io.prepare_data(r)
            done.add(key)
        parsed = parse_basis_string(r["model"]["basis_string"])
        if "sw" in parsed["separate_names"]:
            gkey = (key, r["model"]["signals"]["sw"]["geometry"])
            if gkey not in done:
                data_io.sw_geometry(r, data_io.load_pulsars(
                    dict(r, data=dict(r["data"], pulsars="all", exclude=[]))))
                done.add(gkey)


def _ray_runtime_env(run0):
    env = {k: str(v) for k, v in (run0["ray"]["env_vars"] or {}).items()}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (_ATLAS_ROOT, os.environ.get("PYTHONPATH", "")) if p)
    return {"env_vars": env}


def _ray_entry(run):
    """Executed in the Ray worker.  JAX is imported here, after Ray has set
    the environment (CUDA_VISIBLE_DEVICES, XLA flags)."""
    return run_one(run)


def run(config_path, executor="ray", only=None, dry_run=False):
    """Validate, prepare and execute every run.  Returns a list of status dicts.

    ``executor="local"`` runs sequentially in this process (debugging, tests).
    """
    plan = plan_runs(config_path, only=only)
    for r, action in plan:
        log.info("%-40s %s -> %s", r["run"]["name"], action, r["run"]["output_dir"])
    todo = [r for r, a in plan if a == "run"]
    skipped = [dict(run_name=r["run"]["name"], state="skipped") for r, a in plan if a != "run"]
    if dry_run or not todo:
        return skipped + [dict(run_name=r["run"]["name"], state="planned") for r in todo]

    prepare(todo)

    if executor == "local":
        return skipped + [run_one(r) for r in todo]
    if executor != "ray":
        raise ValueError(f"executor must be 'ray' or 'local', got {executor!r}")

    import ray
    r0 = todo[0]
    if not ray.is_initialized():
        ray.init(address=r0["ray"]["address"], runtime_env=_ray_runtime_env(r0),
                 include_dashboard=False, ignore_reinit_error=True,
                 log_to_driver=True)
    remote = ray.remote(_ray_entry)
    pending, results, name_of = [], [], {}
    limit = r0["ray"]["max_concurrent_runs"] or len(todo)
    queue = list(todo)
    while queue or pending:
        while queue and len(pending) < limit:
            r = queue.pop(0)
            opts = dict(num_cpus=r["ray"]["num_cpus_per_run"],
                        num_gpus=r["ray"]["num_gpus_per_run"],
                        max_retries=int(r["ray"]["max_retries"]),
                        retry_exceptions=False, name=r["run"]["name"])
            ref = remote.options(**opts).remote(r)
            name_of[ref] = r["run"]["name"]
            pending.append(ref)
        ready, pending = ray.wait(pending, num_returns=1)
        for ref in ready:
            try:
                res = ray.get(ref)
            except Exception as exc:          # worker died after its retries
                res = dict(state="failed", error=f"{type(exc).__name__}: {exc}",
                           run_name=name_of[ref])
            log.info("%-40s %s", res.get("run_name"), res.get("state"))
            results.append(res)
    return skipped + results
