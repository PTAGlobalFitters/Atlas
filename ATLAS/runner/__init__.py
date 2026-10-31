"""Config-driven, Ray-parallel ATLAS runs.

    from ATLAS.runner import run, plan_runs
    results = run("run_config.yaml")            # every run, one Ray task each
    results = run("run_config.yaml", executor="local")

or from the shell: ``atlas-run check|run|status run_config.yaml``.

Ray is only imported when ``executor="ray"``; the rest needs PyYAML.
"""
from ATLAS.runner.config import ConfigError, expand_runs, load_config, validate_run
from ATLAS.runner.driver import plan_runs, run

__all__ = ["run", "plan_runs", "load_config", "expand_runs", "validate_run", "ConfigError"]
