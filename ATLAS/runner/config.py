"""Run configuration: defaults, dotted-path overrides, sweeps and validation.

A config file holds one base setup and a list of runs.  Each run is the base
with a few values replaced through dotted paths::

    runs:
      - name: hd_chi_sampled
        overrides:
          model.signals.chrom.varied_index: true

``expand_runs`` turns the file into one fully resolved dict per run, and
``validate_run`` checks a resolved run against ATLAS before anything is
launched -- signal names, PSD/ORF names, prior coverage, bin counts, and the
setting combinations ATLAS cannot run.
"""
from __future__ import annotations

import copy
import glob
import importlib.util
import itertools
import json
from pathlib import Path

import yaml

from ATLAS.runner import registry

__all__ = ["DEFAULTS", "SIGNALS", "load_config", "expand_runs", "validate_run", "substitute_vars",
           "ConfigError", "set_dotted", "get_dotted", "load_priors", "prior_bounds"]

SIGNALS = ("unc", "cor", "dm", "gtm", "chrom", "cquad", "sw", "det")
# relative order the separate blocks must keep (partial marginalisation lays
# the non-GWB phi out IRN | DM | GTM | extras, and det must come last)
_BLOCK_ORDER = ("dm", "gtm", "chrom", "cquad", "sw", "det")

DEFAULTS = {
    "config_version": 1,
    # free-form values substituted into any string as ${name} (after overrides)
    "vars": {},
    "ray": {
        "address": None,
        "num_cpus_per_run": 1,
        "num_gpus_per_run": 0,
        "max_concurrent_runs": None,
        "max_retries": 1,
        "env_vars": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_ENABLE_X64": "true"},
    },
    "output": {
        "root": None,
        "run_dir": "{root}/{run_name}",
        "format": "npz",
        "save": {"samples": True, "red_coefficients": False, "z_a": False,
                 "resolved_config": True, "environment": True},
        "skip_completed": True,
        "overwrite": False,
    },
    "data": {
        "source": "par_tim",
        "par_files": None,
        "tim_files": None,
        "pickle": None,
        "npz_dir": None,
        "npz_files": None,
        "pulsars": "all",
        "exclude": [],
        "timing_package": "pint",
        "ephem": "DE440",
        "noise_dict": None,
        "pulsar_distances": None,
        "dm_ref_freq": 1400.0,
        "cache_dir": None,
        "load_jobs": 4,
        "file_exclude": [],
    },
    "timing": {
        "linear_timing": True,
        "marg_timing": True,
        "nonlinear": {"enabled": False, "params_to_sample": None,
                      "parametrization": "klam"},
    },
    "white_noise": {"vary": False, "include_ecorr": "auto", "stabilize_TNT": False},
    "frequency_bins": {
        "num_irn_bins": 30, "num_gwb_bins": 14, "num_dm_bins": 30,
        "num_chrom_bins": 30, "num_det_bins": None, "use_pulsar_tspan": False,
        "irn_user_freqs": None, "gwb_user_freqs": None,
        "dm_user_freqs": None, "chrom_user_freqs": None,
    },
    "model": {
        "basis_string": "unc+cor->unc ; dm",
        "priors_file": None,
        "signals": {
            "unc": {"psd": "powerlaw", "fixed": {}, "priors": "irn_powerlaw"},
            "cor": {"psd": "powerlaw", "fixed": {}, "orf": "hd_orf", "priors": "gwb_powerlaw_free_gamma"},
            "dm": {"psd": "powerlaw", "fixed": {}, "priors": "dm_powerlaw", "varied_index": False},
            "chrom": {"psd": "powerlaw", "fixed": {}, "priors": "chrom_powerlaw", "chrom_index": 4.0,
                      "varied_index": False, "index_bounds": [0.0, 7.0]},
            "cquad": {"log10_amp": 0.0, "project_timing": True},
            "sw": {"geometry": "pint", "settings_from_par": True,
                   "defaults": {"kernel": "RIDGE", "dt_days": 30.0, "interp_kind": "linear"},
                   "fixed_log10_sigma": None, "priors": "sw_ridge"},
            "gtm": {"basis_file": None, "basis_pattern": None, "mean_pattern": None,
                    "from_pulsar": False, "index": None, "size": None,
                    "psd_file": None, "psd_value": None},
            "det": {"waveform": "cw_delay_evolve_float64", "priors": "cw"},
        },
    },
    "sampler": {
        "kind": "nuts",
        "model": "atlas",
        "seed_offset_var": None,
        "marg_over_non_gwb": False,
        "nuts": {"target_accept_prob": 0.8, "max_tree_depth": 10, "dense_mass": False},
        "num_warmup": 500,
        "num_samples": 1000,
        "num_chains": 1,
        "chain_method": "sequential",
        "thinning": 1,
        "seed": 0,
        "progress_bar": False,
    },
}

# dicts whose keys are free-form (not checked against DEFAULTS)
_FREE_FORM = ("vars", "ray.env_vars", "model.signals.*.fixed", "model.signals.sw.defaults",
              "timing.nonlinear.params_to_sample")


class ConfigError(ValueError):
    """A config that cannot run.  Raised before anything is launched."""


# --------------------------------------------------------------------------- #
#  dict helpers
# --------------------------------------------------------------------------- #

def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and v:
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _is_free_form(path):
    parts = path.split(".")
    for pat in _FREE_FORM:
        pp = pat.split(".")
        if len(parts) >= len(pp) and all(a == b or a == "*" for a, b in zip(pp, parts)):
            return True
    return False


def _check_keys(d, ref, prefix=""):
    """Unknown keys are almost always typos (``num_irn_bin``) -- reject them."""
    for k, v in d.items():
        path = f"{prefix}{k}"
        if _is_free_form(path):
            continue
        if k not in ref:
            raise ConfigError(f"unknown config key {path!r}; expected one of {sorted(ref)}")
        if isinstance(v, dict) and isinstance(ref[k], dict) and ref[k]:
            _check_keys(v, ref[k], path + ".")


def get_dotted(d, path):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def set_dotted(d, path, value):
    """Replace ``d[a][b][c]`` for ``path = "a.b.c"``.  The whole value is
    replaced (a dict is not merged), and the path must already exist."""
    parts = path.split(".")
    cur = d
    for i, part in enumerate(parts[:-1]):
        if not isinstance(cur, dict) or part not in cur:
            raise ConfigError(f"override {path!r}: {'.'.join(parts[:i + 1])!r} does not exist")
        cur = cur[part]
    if not isinstance(cur, dict):
        raise ConfigError(f"override {path!r}: parent is not a section")
    if parts[-1] not in cur and not _is_free_form(".".join(parts[:-1]) + ".x"):
        raise ConfigError(f"override {path!r}: unknown key {parts[-1]!r}; "
                          f"expected one of {sorted(cur)}")
    cur[parts[-1]] = copy.deepcopy(value)


# --------------------------------------------------------------------------- #
#  loading and expansion
# --------------------------------------------------------------------------- #

def load_config(path):
    """Read a run-config YAML file and merge it over ``DEFAULTS``."""
    path = Path(path).resolve()
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    runs = raw.pop("runs", None)
    sweeps = raw.pop("sweeps", None)
    _check_keys(raw, DEFAULTS)
    cfg = _deep_merge(DEFAULTS, raw)
    cfg["runs"] = runs if runs is not None else [{"name": "run"}]
    cfg["sweeps"] = sweeps or []
    cfg["_config_path"] = str(path)
    cfg["_config_dir"] = str(path.parent)
    return cfg


_VAR = __import__("re").compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def substitute_vars(obj, variables, _what="config"):
    """Replace ``${name}`` in every string of ``obj`` by ``variables[name]``.

    A string that is exactly ``${name}`` takes the value itself (so numbers and
    lists stay numbers and lists); otherwise the value is formatted into the
    string.  Values may refer to other variables.
    """
    def sub(v):
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        if isinstance(v, list):
            return [sub(x) for x in v]
        if not isinstance(v, str) or "${" not in v:
            return v
        for _ in range(10):
            m = _VAR.fullmatch(v)
            if m:
                if m.group(1) not in variables:
                    raise ConfigError(f"{_what}: unknown variable ${{{m.group(1)}}}")
                v = copy.deepcopy(variables[m.group(1)])
                if not (isinstance(v, str) and "${" in v):
                    return sub(v) if isinstance(v, (dict, list)) else v
                continue
            missing = [n for n in _VAR.findall(v) if n not in variables]
            if missing:
                raise ConfigError(f"{_what}: unknown variable(s) {missing} in {v!r}; "
                                  f"defined: {sorted(variables)}")
            v = _VAR.sub(lambda mm: str(variables[mm.group(1)]), v)
            if "${" not in v:
                return v
        raise ConfigError(f"{_what}: variables refer to each other in a loop: {v!r}")
    return sub(obj)


def _grid_values(key, vals):
    """A grid entry: a list, or ``{range: [start, stop(, step)]}``."""
    if isinstance(vals, dict):
        if set(vals) != {"range"}:
            raise ConfigError(f"sweep grid {key!r}: a dict entry must be {{range: [start, stop]}}")
        return list(range(*[int(x) for x in vals["range"]]))
    if not isinstance(vals, list) or not vals:
        raise ConfigError(f"sweep grid {key!r}: give a non-empty list or {{range: [start, stop]}}")
    return vals


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "-".join(_fmt(x) for x in v)
    return str(v).replace("/", "_").replace(" ", "")


def expand_runs(cfg):
    """One fully resolved dict per run: base + overrides (+ sweep grids).

    Each resolved run carries ``run.name``, ``run.index``, ``run.output_dir``
    and a seed (``sampler.seed + index`` unless the run sets its own).
    """
    base = {k: v for k, v in cfg.items() if k not in ("runs", "sweeps")}
    specs = []
    for r in cfg["runs"]:
        if "name" not in r:
            raise ConfigError(f"every run needs a name: {r}")
        if "sweep" in r:
            raise ConfigError(f"run {r['name']!r}: put sweeps under the top-level 'sweeps:'")
        extra = set(r) - {"name", "overrides"}
        if extra:
            raise ConfigError(f"run {r['name']!r}: unknown keys {sorted(extra)} "
                              "(use 'overrides' with dotted paths)")
        specs.append((r["name"], dict(r.get("overrides") or {})))
    for sw in cfg["sweeps"]:
        name, grid = sw.get("name"), sw.get("grid") or {}
        if not name or not grid:
            raise ConfigError(f"a sweep needs 'name' and 'grid': {sw}")
        keys = list(grid)
        for combo in itertools.product(*[_grid_values(k, grid[k]) for k in keys]):
            ov = dict(sw.get("base_overrides") or {})
            ov.update(zip(keys, combo))
            tag = "__".join(f"{k.split('.')[-1]}={_fmt(v)}" for k, v in zip(keys, combo))
            specs.append((f"{name}__{tag}", ov))

    names = [n for n, _ in specs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ConfigError(f"duplicate run names: {dup}")

    runs = []
    for i, (name, overrides) in enumerate(specs):
        run = copy.deepcopy(base)
        for path, value in overrides.items():
            set_dotted(run, path, value)
        variables = dict(run["vars"] or {}, run_name=name, run_index=i)
        run["vars"] = substitute_vars(run["vars"] or {}, variables, f"run {name!r} vars")
        variables.update(run["vars"])
        run = {k: (v if k in ("vars", "_config_path", "_config_dir")
                   else substitute_vars(v, variables, f"run {name!r}"))
               for k, v in run.items()}
        if "sampler.seed" not in overrides:
            off_var = run["sampler"]["seed_offset_var"]
            if off_var is not None:
                if off_var not in variables:
                    raise ConfigError(f"run {name!r}: sampler.seed_offset_var {off_var!r} "
                                      "is not a variable")
                run["sampler"]["seed"] = int(run["sampler"]["seed"]) + int(variables[off_var])
            else:
                run["sampler"]["seed"] = int(run["sampler"]["seed"]) + i
        root = run["output"]["root"]
        if root is None:
            raise ConfigError("output.root is required")
        root = str(Path(cfg["_config_dir"], root)) if not Path(root).is_absolute() else root
        run["run"] = {"name": name, "index": i, "overrides": overrides,
                      "output_dir": run["output"]["run_dir"].format(root=root, run_name=name)}
        runs.append(run)

    dirs = [r["run"]["output_dir"] for r in runs]
    if len(set(dirs)) != len(dirs):
        raise ConfigError("two runs write to the same output directory; "
                          "check output.run_dir uses {run_name}")
    return runs


# --------------------------------------------------------------------------- #
#  priors
# --------------------------------------------------------------------------- #

def _resolve(cfg, p):
    if p is None:
        return None
    p = Path(p)
    return str(p if p.is_absolute() else Path(cfg["_config_dir"], p))


def load_priors(run):
    path = _resolve(run, run["model"]["priors_file"])
    if path is None:
        raise ConfigError("model.priors_file is required")
    if not Path(path).exists():
        raise ConfigError(f"priors file not found: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def prior_bounds(entry, names, what):
    """``{param: [lo, hi]}`` -> (lower, upper) lists in ``names`` order.

    A free spectrum may give one pair under ``halflog10_rho`` (broadcast to
    every bin) or a list with one pair per bin.
    """
    entry = {k: v for k, v in (entry or {}).items() if k != "orf"}
    fs = registry.FREE_SPECTRUM_SENTINEL
    if names and all(n.startswith(fs + "_") for n in names) and fs in entry:
        val = entry.pop(fs)
        pairs = [val] * len(names) if _is_pair(val) else list(val)
        if len(pairs) != len(names):
            raise ConfigError(f"{what}: {len(pairs)} bounds for {len(names)} free-spectrum bins")
        entry.update(zip(names, pairs))
    missing = [n for n in names if n not in entry]
    extra = [k for k in entry if k not in names]
    if missing or extra:
        raise ConfigError(f"{what}: priors must cover exactly the sampled parameters "
                          f"{names}; missing {missing}, unexpected {extra}")
    lo, hi = [], []
    for n in names:
        pair = entry[n]
        if not _is_pair(pair) or not float(pair[0]) < float(pair[1]):
            raise ConfigError(f"{what}: bound for {n!r} must be [low, high] with low < high, got {pair}")
        lo.append(float(pair[0])); hi.append(float(pair[1]))
    return lo, hi


def _is_pair(v):
    return isinstance(v, (list, tuple)) and len(v) == 2 and \
        all(isinstance(x, (int, float)) for x in v)


# --------------------------------------------------------------------------- #
#  validation
# --------------------------------------------------------------------------- #

def _bins_key(sig):
    return {"unc": "num_irn_bins", "cor": "num_gwb_bins", "dm": "num_dm_bins",
            "chrom": "num_chrom_bins", "det": "num_det_bins"}.get(sig)


def _files(spec, cfg):
    if spec is None:
        return []
    items = [spec] if isinstance(spec, str) else list(spec)
    out = []
    for it in items:
        it = _resolve(cfg, it)
        hits = sorted(glob.glob(it))
        out += hits if hits else [it]
    return out


def validate_run(run, check_files=True):
    """Raise ``ConfigError`` (listing every problem) if ``run`` cannot run.

    Returns the parsed basis string and the per-signal priors resolved to
    bound arrays, which the builder reuses.
    """
    from ATLAS.signals.signals_utils import parse_basis_string

    errs = []
    name = run["run"]["name"]
    m, fb, tm, smp = run["model"], run["frequency_bins"], run["timing"], run["sampler"]

    try:
        parsed = parse_basis_string(m["basis_string"])
    except Exception as exc:
        raise ConfigError(f"[{name}] bad basis_string {m['basis_string']!r}: {exc}")
    present = parsed["shared_names"] + parsed["separate_names"]
    unknown = [s for s in present if s not in SIGNALS]
    if unknown:
        errs.append(f"unknown signals {unknown} in basis_string; known: {list(SIGNALS)}")

    # -- layout ---------------------------------------------------------------
    for s in ("chrom", "cquad", "sw", "det", "gtm", "dm"):
        if s in parsed["shared_names"] and s != "dm":
            errs.append(f"{s!r} must be a separate block (after ';'), not shared")
    order = [s for s in parsed["separate_names"] if s in _BLOCK_ORDER]
    if order != sorted(order, key=_BLOCK_ORDER.index):
        errs.append(f"separate blocks must keep the order {list(_BLOCK_ORDER)}; got {order}")
    if tm["linear_timing"] and not tm["marg_timing"] and not parsed["include_timing"]:
        errs.append("linear_timing=true with marg_timing=false needs 'ltm|' at the "
                    "start of basis_string (or set marg_timing: true)")
    if parsed["include_timing"] and (tm["marg_timing"] or not tm["linear_timing"]):
        errs.append("'ltm|' needs timing.linear_timing: true and timing.marg_timing: false")

    # -- bins -------------------------------------------------------------------
    for s in present:
        key = _bins_key(s)
        if key and not (isinstance(fb.get(key), int) and fb[key] > 0):
            errs.append(f"signal {s!r} needs frequency_bins.{key} > 0 (got {fb.get(key)})")
    if parsed["representative"] == "unc" and "cor" in parsed["shared_names"] \
            and isinstance(fb["num_gwb_bins"], int) and fb["num_gwb_bins"] > fb["num_irn_bins"]:
        errs.append("with 'unc+cor->unc' the GWB bins sit inside the IRN basis: "
                    "num_gwb_bins must be <= num_irn_bins")

    # -- sampler and flags ----------------------------------------------------------
    if smp["kind"] != "nuts":
        errs.append(f"sampler.kind {smp['kind']!r} is not supported yet (only 'nuts')")
    if smp["chain_method"] not in ("sequential", "vectorized", "parallel"):
        errs.append(f"sampler.chain_method must be sequential|vectorized|parallel")
    if smp["marg_over_non_gwb"] and "cor" not in present:
        errs.append("sampler.marg_over_non_gwb needs a 'cor' signal")
    sig = m["signals"]
    if sig["dm"]["varied_index"]:
        if "dm" not in present:
            errs.append("model.signals.dm.varied_index needs 'dm' in basis_string")
        if not (run["white_noise"]["vary"] or tm["nonlinear"]["enabled"]):
            errs.append("model.signals.dm.varied_index needs white_noise.vary or "
                        "timing.nonlinear.enabled (the helpers must be rebuilt per step)")
    custom = smp["model"] != "atlas"
    if custom and (not isinstance(smp["model"], str) or ":" not in smp["model"]):
        errs.append("sampler.model must be 'atlas' or 'path/to/file.py:function'")
    if sig["chrom"]["varied_index"] and not ({"chrom", "cquad"} & set(present)):
        errs.append("model.signals.chrom.varied_index needs 'chrom' or 'cquad' in basis_string")
    if tm["nonlinear"]["enabled"] and importlib.util.find_spec("jug") is None:
        errs.append("timing.nonlinear.enabled needs JUG installed")
    if tm["nonlinear"]["enabled"] and not run["data"]["par_files"]:
        errs.append("timing.nonlinear.enabled needs data.par_files / data.tim_files "
                    "(JUG reads them, whatever data.source is)")
    pts = tm["nonlinear"]["params_to_sample"]
    if not (pts is None or pts == "from_pulsar" or isinstance(pts, (list, dict))):
        errs.append("timing.nonlinear.params_to_sample must be null, 'from_pulsar', a list, "
                    "{psr: [...]}, or {file: ..., key: ..., index: ...}")
    for k in ("irn_user_freqs", "gwb_user_freqs", "dm_user_freqs", "chrom_user_freqs"):
        v = fb[k]
        if not (v is None or isinstance(v, (list, str, dict))):
            errs.append(f"frequency_bins.{k} must be null, a list, 'file.npz:key' or "
                        "{file: ..., key: ...}")
    if run["output"]["format"] not in ("npz", "netcdf"):
        errs.append("output.format must be npz or netcdf")
    if run["output"]["format"] == "netcdf" and netcdf_engine() is None:
        errs.append("output.format netcdf needs xarray plus netCDF4, or h5netcdf with "
                    "h5py; or use npz")

    # -- PSDs, ORFs, priors -----------------------------------------------------------
    resolved = {}
    try:
        priors = load_priors(run)
    except ConfigError as exc:
        errs.append(str(exc)); priors = {}
    for s in ("unc", "cor", "dm", "chrom"):
        if s not in present:
            continue
        spec = sig[s]
        try:
            fn = registry.psd(spec["psd"], signal=s)
            names = registry.psd_param_names(fn, fixed=tuple(spec["fixed"] or {}),
                                             n_bins=fb[_bins_key(s)])
            if spec["priors"] not in priors:
                raise ConfigError(f"model.signals.{s}.priors = {spec['priors']!r} "
                                  "is not a key of the priors file")
            lo, hi = prior_bounds(priors[spec["priors"]], names, f"priors[{spec['priors']}]")
            resolved[s] = dict(psd=fn, names=names, lower=lo, upper=hi)
            if s == "cor":
                ofn = registry.orf(spec["orf"])
                onames = registry.orf_param_names(ofn)
                olo = ohi = None
                if onames:
                    olo, ohi = prior_bounds(priors[spec["priors"]].get("orf"), onames,
                                            f"priors[{spec['priors']}].orf")
                resolved[s].update(orf=ofn, orf_names=onames, orf_lower=olo, orf_upper=ohi)
        except (KeyError, ConfigError, ValueError) as exc:
            errs.append(f"signal {s!r}: {exc}")
    if "sw" in present and sig["sw"]["fixed_log10_sigma"] is None:
        try:
            if sig["sw"]["priors"] not in priors:
                raise ConfigError(f"model.signals.sw.priors = {sig['sw']['priors']!r} "
                                  "is not a key of the priors file")
            lo, hi = prior_bounds(priors[sig["sw"]["priors"]], ["log10_amp"],
                                  f"priors[{sig['sw']['priors']}]")
            resolved["sw"] = dict(lower=lo, upper=hi)
        except ConfigError as exc:
            errs.append(f"signal 'sw': {exc}")
    if "sw" in present and sig["sw"]["geometry"] not in ("pint", "approx", "stored"):
        errs.append("model.signals.sw.geometry must be pint, approx or stored")
    if "det" in present:
        try:
            registry.waveform(sig["det"]["waveform"])
            entry = priors.get(sig["det"]["priors"]) or {}
            if not entry:
                raise ConfigError(f"priors[{sig['det']['priors']!r}] is empty or missing")
            lo, hi = prior_bounds(entry, list(entry), f"priors[{sig['det']['priors']}]")
            resolved["det"] = dict(names=list(entry), lower=lo, upper=hi)
        except (KeyError, ConfigError) as exc:
            errs.append(f"signal 'det': {exc}")
        if run["data"]["pulsar_distances"] is None:
            errs.append("a 'det' signal needs data.pulsar_distances")
    g = sig["gtm"]
    n_src = sum(bool(x) for x in (g["basis_file"], g["basis_pattern"], g["from_pulsar"]))
    if "gtm" in present and n_src != 1:
        errs.append("a 'gtm' signal needs exactly one of model.signals.gtm.basis_file, "
                    "basis_pattern or from_pulsar")
    if g["psd_file"] and g["psd_value"] is not None:
        errs.append("model.signals.gtm: give psd_file or psd_value, not both")

    # -- files --------------------------------------------------------------------
    if check_files:
        d = run["data"]
        src = d["source"]
        if src == "par_tim":
            pars, tims = _files(d["par_files"], run), _files(d["tim_files"], run)
            if not pars or not all(Path(p).exists() for p in pars):
                errs.append(f"data.par_files matched nothing: {d['par_files']}")
            if not tims or not all(Path(t).exists() for t in tims):
                errs.append(f"data.tim_files matched nothing: {d['tim_files']}")
        elif src == "enterprise_pickle":
            if not d["pickle"] or not Path(_resolve(run, d["pickle"])).exists():
                errs.append(f"data.pickle not found: {d['pickle']}")
        elif src == "atlas_npz":
            if d["npz_files"]:
                pat = _resolve(run, d["npz_files"])
                pulsars = d["pulsars"] if d["pulsars"] != "all" else None
                if "{psr}" in pat and pulsars:
                    miss = [pat.replace("{psr}", n) for n in pulsars
                            if not Path(pat.replace("{psr}", n)).exists()]
                    if miss:
                        errs.append(f"data.npz_files: {len(miss)} file(s) missing, e.g. {miss[:3]}")
                elif not glob.glob(pat.replace("{psr}", "*")):
                    errs.append(f"data.npz_files matched nothing: {d['npz_files']}")
            elif not d["npz_dir"] or not Path(_resolve(run, d["npz_dir"])).is_dir():
                errs.append(f"data.npz_dir not found: {d['npz_dir']} (or set data.npz_files)")
            if tm["nonlinear"]["enabled"]:
                pars, tims = _files(d["par_files"], run), _files(d["tim_files"], run)
                if not pars or not all(Path(p).exists() for p in pars) or \
                        not tims or not all(Path(t).exists() for t in tims):
                    errs.append("nonlinear timing: data.par_files / data.tim_files matched nothing")
        else:
            errs.append("data.source must be par_tim | enterprise_pickle | atlas_npz")
        for key in ("noise_dict", "pulsar_distances"):
            if d[key] is not None and not Path(_resolve(run, d[key])).exists():
                errs.append(f"data.{key} not found: {d[key]}")
        if d["noise_dict"] is None and not run["white_noise"]["vary"]:
            errs.append("fixed white noise (white_noise.vary: false) needs data.noise_dict")
        if d["cache_dir"] is None:
            errs.append("data.cache_dir is required")
        ci = sig["chrom"]["chrom_index"]
        if isinstance(ci, str) and not Path(_resolve(run, ci)).exists():
            errs.append(f"model.signals.chrom.chrom_index file not found: {ci}")
        for key in ("basis_file", "psd_file"):
            f = sig["gtm"][key]
            if "gtm" in present and f and not Path(_resolve(run, f)).exists():
                errs.append(f"model.signals.gtm.{key} not found: {f}")
        if "gtm" in present and g["basis_pattern"] and d["pulsars"] != "all":
            for key in ("basis_pattern", "mean_pattern"):
                pat = g[key]
                if pat:
                    miss = [n for n in d["pulsars"]
                            if not Path(_resolve(run, pat.replace("{psr}", n))).exists()]
                    if miss:
                        errs.append(f"model.signals.gtm.{key}: no file for {miss[:5]}")
        if custom:
            f = smp["model"].rsplit(":", 1)[0]
            if not Path(_resolve(run, f)).exists():
                errs.append(f"sampler.model file not found: {f}")
        for k in ("irn_user_freqs", "gwb_user_freqs", "dm_user_freqs", "chrom_user_freqs"):
            v = fb[k]
            f = v.get("file") if isinstance(v, dict) else (v.rsplit(":", 1)[0] if isinstance(v, str) else None)
            if f and not Path(_resolve(run, f)).exists():
                errs.append(f"frequency_bins.{k}: file not found: {f}")

    if errs:
        raise ConfigError(f"run {name!r} cannot run:\n" + "\n".join(f"  - {e}" for e in errs))
    return parsed, resolved


def netcdf_engine():
    """An xarray engine that can write NetCDF groups, or None."""
    has = lambda m: importlib.util.find_spec(m) is not None
    if not has("xarray"):
        return None
    if has("netCDF4"):
        return "netcdf4"
    if has("h5netcdf") and has("h5py"):
        return "h5netcdf"
    return None


def resolve_path(run, p):
    """Public form of the path resolution used throughout (config-relative)."""
    return _resolve(run, p)


def dump_run(run, path):
    """Write a resolved run as YAML (functions and private keys dropped)."""
    clean = {k: v for k, v in run.items() if not k.startswith("_")}
    clean["run"] = dict(clean["run"])
    with open(path, "w") as fh:
        yaml.safe_dump(json.loads(json.dumps(clean, default=str)), fh, sort_keys=False)
