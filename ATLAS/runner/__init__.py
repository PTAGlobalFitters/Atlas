"""Config-driven, Ray-parallel ATLAS runs.

    from ATLAS.runner import run, plan_runs
    results = run("run_config.yaml")            # every run, one Ray task each
    results = run("run_config.yaml", executor="local")

or from the shell: ``atlas-run check|run|status run_config.yaml``.

Ray is only imported when ``executor="ray"``; the rest needs PyYAML.
"""
__all__ = ["run", "plan_runs", "load_config", "expand_runs", "validate_run", "ConfigError"]

# Imported lazily: the submodules pull in ATLAS modules that create JAX arrays at
# import time, which initialises JAX (and, on a GPU machine, reserves most of each
# GPU's memory).  ``atlas-run`` first pins itself to the CPU, then imports them.
_LAZY = {"ConfigError": "config", "expand_runs": "config", "load_config": "config",
         "validate_run": "config", "plan_runs": "driver", "run": "driver"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f"ATLAS.runner.{_LAZY[name]}"), name)
    raise AttributeError(f"module 'ATLAS.runner' has no attribute {name!r}")
