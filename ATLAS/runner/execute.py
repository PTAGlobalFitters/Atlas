"""Run one resolved config: build, sample, save.

Everything here runs inside a Ray worker (or in-process with the local
executor).  A run's directory holds::

    status.json          state (running | done | failed), timings, error
    run.log              this run's log
    config.yaml          the fully resolved config (if output.save.resolved_config)
    environment.json     ATLAS git commit, package versions (if output.save.environment)
    samples.npz | posterior.nc
    summary.json         mean / std / n_eff / r_hat per named parameter, divergences
"""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
import traceback
from functools import partial
from pathlib import Path

import numpy as np

from ATLAS.runner.config import dump_run, validate_run

__all__ = ["run_one", "is_completed", "read_status"]

log = logging.getLogger("ATLAS.runner")


def _status_path(run):
    return Path(run["run"]["output_dir"]) / "status.json"


def read_status(run):
    p = _status_path(run)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _output_file(run):
    return Path(run["run"]["output_dir"]) / (
        "posterior.nc" if run["output"]["format"] == "netcdf" else "samples.npz")


def is_completed(run):
    st = read_status(run)
    return bool(st and st.get("state") == "done" and _output_file(run).exists())


def _write_status(run, **fields):
    p = _status_path(run)
    st = read_status(run) or {}
    st.update(fields, run_name=run["run"]["name"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, default=str))
    tmp.replace(p)


def _environment():
    import jax
    import numpyro
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                    capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        commit, dirty = None, None
    return dict(atlas_commit=commit, atlas_dirty=dirty, python=platform.python_version(),
                jax=jax.__version__, numpyro=numpyro.__version__, numpy=np.__version__,
                jax_devices=[str(dev) for dev in jax.devices()])


def _model_and_args(run, b):
    """Bind the non-array settings into the model; pass only arrays to
    ``mcmc.run`` so ``jit_model_args=True`` traces them (the T-matrix in
    particular, which would otherwise be folded into the executable)."""
    from ATLAS.model import model_maker

    sig, smp = run["model"]["signals"], run["sampler"]
    vary_white = bool(run["white_noise"]["vary"])
    chrom_idx = bool(sig["chrom"]["varied_index"])
    dm_idx = bool(sig["dm"]["varied_index"])
    bound = dict(super_sig=b.rn,
                 marg_over_non_gwb=bool(smp["marg_over_non_gwb"]),
                 vary_white=vary_white,
                 tm_model=b.tm_model,
                 save_red_coeff=bool(run["output"]["save"]["red_coefficients"]),
                 varied_chrom_index=dm_idx,
                 varied_chrom_gp_index=chrom_idx,
                 chrom_gp_index_bounds=tuple(float(x) for x in sig["chrom"]["index_bounds"]),
                 tm_direct_sampling_type=run["timing"]["nonlinear"]["parametrization"])
    args = dict(raw_residuals=b.residuals)
    basis_moves = chrom_idx or dm_idx or vary_white or b.tm_model is not None
    if vary_white:
        args.update(wn_lower_bound=b.wn_lower, wn_upper_bound=b.wn_upper)
    elif basis_moves:
        args["fixed_white_noise_params"] = b.wn_vec
        if b.tm_model is not None and not (chrom_idx or dm_idx):
            # sampled timing, fixed basis + white noise: only the residual
            # terms of the helpers change each step
            args["static_helpers"] = b.rn.get_static_helpers(white_noise_params=b.wn_vec)
        if chrom_idx and not dm_idx and b.tm_model is None:
            # fixed white noise + residuals: each step updates only the chrom /
            # cquad rows and columns of these helpers (model_maker, partial path)
            args["helpers"] = b.rn.get_helpers(reff=args["raw_residuals"],
                                               white_noise_params=b.wn_vec)
    else:
        args["helpers"] = b.rn.get_helpers(reff=args["raw_residuals"],
                                           white_noise_params=b.wn_vec)
    if basis_moves:
        args["red_noise_basis"] = b.rn.get_Fmat_concat
    return partial(model_maker, **bound), args


def _custom_model(run, b):
    """``sampler.model: path/to/file.py:function`` -> ``function(run, built)``,
    which returns ``(model, kwargs)`` for ``MCMC.run``.  The run's variables
    are in ``run['vars']``."""
    import importlib.util
    from ATLAS.runner.config import resolve_path
    f, fn = run["sampler"]["model"].rsplit(":", 1)
    path = resolve_path(run, f)
    spec = importlib.util.spec_from_file_location(f"atlas_custom_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, fn):
        raise AttributeError(f"{path} has no function {fn!r}")
    model, args = getattr(mod, fn)(run, b)
    return model, dict(args or {})


def _named(samples, b):
    """Site -> list of column names, for the sites that have them."""
    names = {"red_noise": b.rn.model.get_param_names()}
    names["white_noise"] = [n for cov in b.wn.cov_matrices for n in cov.get_param_names()]
    for site in ("chrom_gp_index", "chromatic_index"):
        if site in samples:
            names[site] = [f"{n}_{site}" for n in b.psr_names]
    return names


def _save(run, samples, names, diverging):
    out = _output_file(run)
    flat = {k: np.asarray(v) for k, v in samples.items()}
    if run["output"]["format"] == "npz":
        extra = {f"{k}__names": np.asarray(v) for k, v in names.items()
                 if k in flat and flat[k].shape[-1] == len(v)}
        np.savez(out, diverging=np.asarray(diverging), **flat, **extra)
        return out
    # InferenceData layout written with xarray directly: groups "posterior"
    # and "sample_stats", dims (chain, draw, ...), named parameter coords.
    # Readable by arviz.from_netcdf in ArviZ 0.x and 1.x alike.
    import xarray as xr
    post = {}
    for site, arr in flat.items():
        nm = names.get(site)
        dims = ["chain", "draw"] + [f"{site}_dim_{i}" for i in range(arr.ndim - 2)]
        coords = {}
        if nm is not None and arr.ndim == 3 and arr.shape[-1] == len(nm):
            dims[-1] = f"{site}_param"
            coords[dims[-1]] = list(nm)
        post[site] = xr.DataArray(arr, dims=dims, coords=coords)
    engine = _netcdf_engine()
    xr.Dataset(post).to_netcdf(out, group="posterior", engine=engine, mode="w")
    xr.Dataset({"diverging": (("chain", "draw"), np.asarray(diverging))}).to_netcdf(
        out, group="sample_stats", engine=engine, mode="a")
    return out


def _netcdf_engine():
    from ATLAS.runner.config import netcdf_engine
    eng = netcdf_engine()
    if eng is None:
        raise RuntimeError("output.format netcdf needs netCDF4, or h5netcdf with h5py")
    return eng


def _summary(samples, names, diverging):
    from numpyro.diagnostics import summary
    out = {"num_divergent": int(np.sum(diverging)), "parameters": {}}
    for site, nm in names.items():
        if site not in samples:
            continue
        arr = np.asarray(samples[site])          # (chains, draws, n)
        if arr.ndim != 3 or arr.shape[-1] != len(nm):
            continue
        s = summary(arr, group_by_chain=True)["Param:0"]
        for i, n in enumerate(nm):
            out["parameters"][n] = {k: float(np.asarray(s[k])[i])
                                    for k in ("mean", "std", "median", "n_eff", "r_hat")}
    return out


def run_one(run, prebuilt=None):
    """Build, sample and save one run.  Returns a status dict; never raises
    for a failure inside the run (the error is recorded in status.json)."""
    out_dir = Path(run["run"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out_dir / "run.log", mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger("ATLAS.runner")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    t0 = time.time()
    _write_status(run, state="running", started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                  host_pid=os.getpid())
    try:
        if run["output"]["save"]["resolved_config"]:
            dump_run(run, out_dir / "config.yaml")
        if run["output"]["save"]["environment"]:
            (out_dir / "environment.json").write_text(json.dumps(_environment(), indent=2))

        parsed, resolved = validate_run(run, check_files=False)
        from ATLAS.runner.build import build_model
        b = prebuilt or build_model(run, parsed, resolved)
        t_build = time.time()

        import jax.random as jr
        from numpyro.infer import MCMC, NUTS
        smp = run["sampler"]
        custom = smp["model"] != "atlas"
        model, args = _custom_model(run, b) if custom else _model_and_args(run, b)
        mcmc = MCMC(NUTS(model, **smp["nuts"]),
                    num_warmup=int(smp["num_warmup"]), num_samples=int(smp["num_samples"]),
                    num_chains=int(smp["num_chains"]), thinning=int(smp["thinning"]),
                    chain_method=smp["chain_method"], progress_bar=bool(smp["progress_bar"]),
                    jit_model_args=not custom)
        keep_za = bool(run["output"]["save"]["z_a"])
        mcmc.run(jr.key(int(smp["seed"])),
                 extra_fields=(("diverging",) if (keep_za or custom)
                               else ("diverging", "~z.z_a")),
                 **args)
        t_sample = time.time()

        samples = mcmc.get_samples(group_by_chain=True)
        if not keep_za:
            samples = {k: v for k, v in samples.items() if k != "z_a"}
        diverging = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["diverging"])
        names = _named(samples, b)
        path = _save(run, samples, names, diverging) if run["output"]["save"]["samples"] else None
        summ = _summary(samples, names, diverging)
        (out_dir / "summary.json").write_text(json.dumps(summ, indent=2))
        st = dict(state="done", output=str(path) if path else None,
                  num_divergent=summ["num_divergent"],
                  seconds_build=round(t_build - t0, 1),
                  seconds_sample=round(t_sample - t_build, 1),
                  seconds_total=round(time.time() - t0, 1))
        _write_status(run, **st)
        log.info("%s done in %.0f s (%d divergent)", run["run"]["name"],
                 time.time() - t0, summ["num_divergent"])
        return dict(st, run_name=run["run"]["name"])
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("%s failed:\n%s", run["run"]["name"], tb)
        st = dict(state="failed", error=f"{type(exc).__name__}: {exc}", traceback=tb,
                  seconds_total=round(time.time() - t0, 1))
        _write_status(run, **st)
        return dict(st, run_name=run["run"]["name"])
    finally:
        root.removeHandler(handler)
        handler.close()