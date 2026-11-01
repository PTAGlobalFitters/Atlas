"""Loading and caching everything a run reads from disk.

Expensive steps (PINT loading of par/tim files, PINT solar-wind geometry) run
once, in the driver, and are cached under ``data.cache_dir``; Ray workers only
read the caches.  Cache entries are keyed by a hash of the inputs that
determine them, so two configs pointing at the same files share a cache.
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import pickle
import re
from pathlib import Path

import numpy as np

from ATLAS.runner.config import ConfigError, resolve_path

log = logging.getLogger("ATLAS.runner")

__all__ = ["data_key", "prepare_data", "load_pulsars", "pulsar_files",
           "sw_geometry", "sw_kwargs_for", "chrom_index_for", "load_noise_dict",
           "load_distances", "load_gtm", "sw_geometry_approx", "match_toas",
           "user_freqs", "timing_params_for"]


# --------------------------------------------------------------------------- #
#  files and keys
# --------------------------------------------------------------------------- #

def _expand(run, spec):
    if spec is None:
        return []
    items = [spec] if isinstance(spec, str) else list(spec)
    out = []
    for it in items:
        it = resolve_path(run, it)
        out += sorted(glob.glob(it)) or [it]
    return out


def _par_name(par):
    with open(par) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) > 1 and parts[0].upper() in ("PSRJ", "PSR", "PSRB"):
                return parts[1]
    return Path(par).stem.split("_")[0]


def _keep(path, exclude):
    base = Path(path).name
    return not any(x in base for x in exclude)


def pulsar_files(run, names=None):
    """``{pulsar_name: (par, tim)}`` from data.par_files / data.tim_files.

    Pairs are matched by name (from the par file's PSR/PSRJ line, found in the
    tim file's name), not by sort order.  Files whose name contains any of
    ``data.file_exclude`` (e.g. ``[ao, gbt]``) are ignored.  With ``names``,
    only those pulsars are paired, and every one of them must pair; problems
    are reported together.
    """
    d = run["data"]
    excl = [str(x) for x in (d.get("file_exclude") or [])]
    pars = [p for p in _expand(run, d["par_files"]) if _keep(p, excl)]
    tims = [t for t in _expand(run, d["tim_files"]) if _keep(t, excl)]
    by_name = {}
    for par in pars:
        by_name.setdefault(_par_name(par), []).append(par)
    want = list(by_name) if names is None else [str(n) for n in names]
    out, problems = {}, []
    for name in want:
        cand = by_name.get(name, [])
        if len(cand) != 1:
            problems.append(f"{name}: expected 1 par file, found {len(cand)} {cand}")
            continue
        hits = [t for t in tims if Path(t).name.startswith(name)] or \
               [t for t in tims if name in Path(t).name]
        if len(hits) != 1:
            problems.append(f"{name}: expected 1 tim file, found {len(hits)} {hits}")
            continue
        out[name] = (cand[0], hits[0])
    if problems:
        raise ConfigError(f"{len(problems)} pulsar(s) failed par/tim pairing "
                          "(see data.file_exclude):\n  " + "\n  ".join(problems))
    return out


def data_key(run):
    """Hash of everything that determines the loaded pulsars."""
    d = run["data"]
    spec = dict(source=d["source"], ephem=d["ephem"], timing_package=d["timing_package"],
                pars=_expand(run, d["par_files"]), tims=_expand(run, d["tim_files"]),
                pickle=resolve_path(run, d["pickle"]), npz=resolve_path(run, d["npz_dir"]),
                npz_files=resolve_path(run, d.get("npz_files")),
                exclude=list(d.get("file_exclude") or []))
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


def _cache_dir(run):
    p = Path(resolve_path(run, run["data"]["cache_dir"]))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _pulsar_cache(run):
    return _cache_dir(run) / f"pulsars_{data_key(run)}.pkl"


# --------------------------------------------------------------------------- #
#  pulsars
# --------------------------------------------------------------------------- #

def _load_enterprise(par, tim, ephem, timing_package):
    from enterprise.pulsar import Pulsar
    return Pulsar(par, tim, sort=False, drop_pintpsr=False, ephem=ephem,
                  timing_package=timing_package.upper() if timing_package == "pint"
                  else timing_package)


def _load_all(run):
    d = run["data"]
    src = d["source"]
    if src == "par_tim":
        files = pulsar_files(run)
        from joblib import Parallel, delayed
        log.info("loading %d pulsars from par/tim with %s (%d jobs)",
                 len(files), d["timing_package"], d["load_jobs"])
        return Parallel(n_jobs=int(d["load_jobs"]))(
            delayed(_load_enterprise)(par, tim, d["ephem"], d["timing_package"])
            for par, tim in files.values())
    if src == "enterprise_pickle":
        with open(resolve_path(run, d["pickle"]), "rb") as fh:
            return list(pickle.load(fh))
    if src == "atlas_npz":
        from ATLAS.pulsar import PulsarDataLoader
        return [PulsarDataLoader(**np.load(p, allow_pickle=True)) for p in _npz_paths(run)]
    raise ConfigError(f"unknown data.source {src!r}")


def _npz_paths(run):
    """The .npz files of an atlas_npz run, in data.pulsars order when the
    pattern names them with ``{psr}`` (only those files are read)."""
    d = run["data"]
    if d.get("npz_files"):
        pat = resolve_path(run, d["npz_files"])
        want = d["pulsars"]
        if "{psr}" in pat and want != "all":
            excl = set(d["exclude"] or [])
            return [pat.replace("{psr}", str(n)) for n in want if str(n) not in excl]
        return sorted(glob.glob(pat.replace("{psr}", "*")))
    return sorted(glob.glob(str(Path(resolve_path(run, d["npz_dir"])) / "*.npz")))


def prepare_data(run):
    """Load every pulsar of this run's dataset once and cache it (driver side).

    ATLAS .npz files are already fast to read, so they are not cached (a
    realization sweep would otherwise copy the whole dataset once per run).
    """
    if run["data"]["source"] == "atlas_npz":
        return None
    path = _pulsar_cache(run)
    if not path.exists():
        psrs = _load_all(run)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(psrs, fh)
        tmp.replace(path)
        log.info("cached %d pulsars -> %s", len(psrs), path)
    return path


def load_pulsars(run):
    """The pulsars of one run: cached dataset, filtered by pulsars / exclude.

    With an explicit ``data.pulsars`` list the pulsars come back in that order.
    """
    if run["data"]["source"] == "atlas_npz":
        psrs = _load_all(run)
    else:
        path = _pulsar_cache(run)
        if not path.exists():
            prepare_data(run)
        with open(path, "rb") as fh:
            psrs = pickle.load(fh)
    want, excl = run["data"]["pulsars"], set(run["data"]["exclude"] or [])
    names = [str(p.name) for p in psrs]
    if want != "all":
        want = [str(w) for w in want]
        missing = sorted(set(want) - set(names))
        if missing:
            raise ConfigError(f"data.pulsars not in the dataset: {missing}")
        by = {str(p.name): p for p in psrs}
        psrs = [by[w] for w in want]
    psrs = [p for p in psrs if str(p.name) not in excl]
    if not psrs:
        raise ConfigError("no pulsars left after data.pulsars / data.exclude")
    return psrs


# --------------------------------------------------------------------------- #
#  side inputs
# --------------------------------------------------------------------------- #

def _read_mapping(path):
    path = str(path)
    if path.endswith((".yaml", ".yml")):
        import yaml
        with open(path) as fh:
            return yaml.safe_load(fh)
    if path.endswith((".pkl", ".pickle")):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    with open(path) as fh:
        return json.load(fh)


def load_noise_dict(run):
    p = run["data"]["noise_dict"]
    return None if p is None else _read_mapping(resolve_path(run, p))


def load_distances(run):
    p = run["data"]["pulsar_distances"]
    if p is None:
        return {}
    raw = _read_mapping(resolve_path(run, p))
    return {str(k): tuple(float(x) for x in v) for k, v in raw.items()}


def chrom_index_for(run, psrs):
    """``{psr: chi}`` from a number or a yaml/json/pickle mapping (``default``
    key allowed for pulsars not listed)."""
    ci = run["model"]["signals"]["chrom"]["chrom_index"]
    names = [str(p.name) for p in psrs]
    if isinstance(ci, (int, float)):
        return {n: float(ci) for n in names}
    table = {str(k): v for k, v in _read_mapping(resolve_path(run, ci)).items()}
    out = {}
    for n in names:
        if n in table:
            out[n] = float(table[n])
        elif "default" in table:
            out[n] = float(table["default"])
        else:
            raise ConfigError(f"chrom_index file has no entry (and no 'default') for {n}")
    return out


def _pick(arr, index):
    arr = np.asarray(arr)
    return arr if index is None else arr[index]


def load_gtm(run, psrs):
    """(basis list, gtm_psd or None, mean list or None) for the ``gtm`` block.

    The basis comes from one of
      basis_file     npz keyed by pulsar name, one [ntoa, k] array each
      basis_pattern  one .npy per pulsar, path with ``{psr}`` (e.g. Adaptus' U.npy)
      from_pulsar    PulsarDataLoader.adaptus_basis / adaptus_mean
    ``index`` picks ``arr[index]`` first (U.npy / mean.npy stored as [1, ...]
    -> 0), ``size`` keeps the first ``size`` columns.  The mean (mean_pattern
    or from_pulsar) is subtracted from the residuals by the builder.
    """
    g = run["model"]["signals"]["gtm"]
    names = [str(p.name) for p in psrs]
    idx, size = g["index"], g["size"]
    means = None
    if g["basis_file"]:
        z = np.load(resolve_path(run, g["basis_file"]))
        missing = [n for n in names if n not in z.files]
        if missing:
            raise ConfigError(f"gtm basis_file has no basis for {missing}")
        basis = [_pick(z[n], idx) for n in names]
    elif g["basis_pattern"]:
        basis = [_pick(np.load(resolve_path(run, g["basis_pattern"].replace("{psr}", n))), idx)
                 for n in names]
    elif g["from_pulsar"]:
        miss = [n for n, p in zip(names, psrs) if getattr(p, "adaptus_basis", None) is None]
        if miss:
            raise ConfigError(f"gtm.from_pulsar: no adaptus_basis stored for {miss}")
        basis = [np.asarray(p.adaptus_basis) for p in psrs]
        if all(getattr(p, "adaptus_mean", None) is not None for p in psrs):
            means = [np.asarray(p.adaptus_mean, float).reshape(-1) for p in psrs]
    else:
        raise ConfigError("gtm needs basis_file, basis_pattern or from_pulsar")
    if g["mean_pattern"]:
        means = [_pick(np.load(resolve_path(run, g["mean_pattern"].replace("{psr}", n))), idx)
                 .astype(float).reshape(-1) for n in names]

    basis = [np.asarray(b, float) for b in basis]
    for n, b, p in zip(names, basis, psrs):
        if b.ndim != 2 or b.shape[0] != len(p.toas):
            raise ConfigError(f"gtm basis for {n} has shape {b.shape}; expected "
                              f"({len(p.toas)}, k) -- set model.signals.gtm.index?")
    if size is not None:
        short = [n for n, b in zip(names, basis) if b.shape[1] < int(size)]
        if short:
            raise ConfigError(f"gtm.size={size} but the basis of {short} has fewer columns")
        basis = [b[:, :int(size)] for b in basis]
    widths = {b.shape[1] for b in basis}
    if len(widths) != 1:
        raise ConfigError(f"gtm bases must share one width; got {sorted(widths)}")
    if means is not None:
        bad = [n for n, m, p in zip(names, means, psrs) if m.size != len(p.toas)]
        if bad:
            raise ConfigError(f"gtm mean has the wrong length for {bad}")

    k = widths.pop()
    psd = None
    if g["psd_file"]:
        zp = np.load(resolve_path(run, g["psd_file"]))
        psd = (np.asarray(zp["gtm_psd"], float) if "gtm_psd" in zp.files
               else np.stack([np.asarray(zp[n], float) for n in names], axis=1))
    elif g["psd_value"] is not None:
        psd = np.full((k, len(psrs)), float(g["psd_value"]))
    return basis, psd, means


def _array_from_spec(run, spec, what):
    """A list, ``"file.npz:key"`` / ``"file.npy"``, or ``{file:, key:, index:}``."""
    if spec is None:
        return None
    if isinstance(spec, list):
        return np.asarray(spec, float)
    if isinstance(spec, str):
        f, key = (spec.rsplit(":", 1) + [None])[:2] if ":" in spec else (spec, None)
        spec = {"file": f, "key": key}
    f = resolve_path(run, spec["file"])
    z = np.load(f, allow_pickle=True)
    if hasattr(z, "files"):
        if not spec.get("key"):
            raise ConfigError(f"{what}: {f} is an npz; name the array with 'file.npz:key'")
        z = z[spec["key"]]
    return _pick(z, spec.get("index"))


def user_freqs(run, which):
    """frequency_bins.<which>_user_freqs as an array [Hz], or None."""
    v = _array_from_spec(run, run["frequency_bins"][f"{which}_user_freqs"],
                         f"frequency_bins.{which}_user_freqs")
    return None if v is None else np.asarray(v, float).reshape(-1)


def timing_params_for(run, psrs):
    """Per-pulsar list of timing parameters to sample nonlinearly, or None.

    ``params_to_sample`` may be null (every fitted parameter), a list (the
    same for every pulsar), ``{psr: [...]}``, ``from_pulsar``
    (PulsarDataLoader.sample_params), or ``{file: x.npz, key: K, index: i}``
    whose object is ``{psr: [...]}`` (e.g. info.npz SAMPLE_LIST[0]).
    """
    spec = run["timing"]["nonlinear"]["params_to_sample"]
    names = [str(p.name) for p in psrs]
    if spec is None:
        return None
    if spec == "from_pulsar":
        miss = [n for n, p in zip(names, psrs) if getattr(p, "sample_params", None) is None]
        if miss:
            raise ConfigError(f"params_to_sample: from_pulsar, but no sample_params for {miss}")
        return [[str(x) for x in p.sample_params] for p in psrs]
    if isinstance(spec, list):
        return [[str(x) for x in spec] for _ in names]
    if isinstance(spec, dict) and "file" in spec:
        obj = _array_from_spec(run, spec, "timing.nonlinear.params_to_sample")
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
            obj = obj.item()
        table = {str(k): v for k, v in dict(obj).items()}
    else:
        table = {str(k): v for k, v in spec.items()}
    miss = [n for n in names if n not in table]
    if miss:
        raise ConfigError(f"params_to_sample has no entry for {miss}")
    return [[str(x) for x in np.asarray(table[n]).reshape(-1)] for n in names]


# --------------------------------------------------------------------------- #
#  solar wind
# --------------------------------------------------------------------------- #

_AU_PC = 4.84813681109536e-06


def sw_geometry_approx(toas_s, pos):
    """1/r^2 (PINT SWM 0) geometry [pc] from a low-precision Sun position
    (Astronomical Almanac, ~0.01 deg).  ~1% of PINT; for tests and smoke runs."""
    mjd = np.asarray(toas_s, float) / 86400.0
    n = mjd + 2400000.5 - 2451545.0
    g = np.deg2rad(357.528 + 0.9856003 * n)
    lam = np.deg2rad(280.460 + 0.9856474 * n + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.deg2rad(23.439 - 4e-7 * n)
    r_au = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)
    sun = np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=1)
    elong = np.arccos(np.clip(sun @ np.asarray(pos, float), -1.0, 1.0))
    rho = np.pi - elong
    return _AU_PC * rho / (r_au * np.sin(rho))


def match_toas(values, src_toas_s, dst_toas_s, tol_s=600.0):
    """Reorder ``values`` (indexed like ``src``) onto ``dst``.

    The same TOAs may carry different time references (topocentric TDB vs
    barycentric arrival differ by up to ~500 s), so equal-size sets are paired
    by time rank; otherwise each destination TOA takes its nearest source TOA.
    """
    src, dst = np.asarray(src_toas_s, float), np.asarray(dst_toas_s, float)
    vals = np.asarray(values, float)
    if np.array_equal(src, dst):
        return vals
    if src.size == dst.size:
        i_s, i_d = np.argsort(src, kind="stable"), np.argsort(dst, kind="stable")
        worst = np.max(np.abs(src[i_s] - dst[i_d]))
        if worst < tol_s:
            out = np.empty_like(vals)
            out[i_d] = vals[i_s]
            return out
    order = np.argsort(src)
    s = src[order]
    j = np.clip(np.searchsorted(s, dst), 1, len(s) - 1)
    j -= (dst - s[j - 1]) < (s[j] - dst)
    worst = np.max(np.abs(s[j] - dst))
    if worst >= tol_s:
        raise ConfigError(f"solar-wind geometry: TOAs do not match (worst {worst:.3g} s)")
    return vals[order][j]


def _pint_geometry(psr, files, ephem):
    from ATLAS.signals.factorized.chromatic import sw_geometry_from_pint
    if getattr(psr, "model", None) is not None and getattr(psr, "pint_toas", None) is not None:
        _, geom = sw_geometry_from_pint(psr.model, psr.pint_toas)
        return np.asarray(psr.toas, float), geom          # enterprise sort=False order
    name = str(psr.name)
    if name not in files:
        raise ConfigError(f"PINT solar-wind geometry for {name} needs its par/tim files "
                          "(data.par_files / data.tim_files) or an enterprise pulsar "
                          "loaded with drop_pintpsr=False; or use geometry: approx")
    import pint.logging
    from pint.models import get_model_and_toas
    pint.logging.setup(level="WARNING")
    model, toas = get_model_and_toas(files[name][0], files[name][1], ephem=ephem)
    return sw_geometry_from_pint(model, toas)


def _sw_cache(run, method):
    p = _cache_dir(run) / f"sw_geometry_{data_key(run)}_{method}"
    p.mkdir(exist_ok=True)
    return p


def sw_geometry(run, psrs, compute=True):
    """Per-pulsar solar-wind geometry [pc] in each pulsar's TOA order (cached)."""
    method = run["model"]["signals"]["sw"]["geometry"]
    cache = _sw_cache(run, method)
    files = {}
    if run["data"]["par_files"] and method != "approx":
        files = pulsar_files(run, [str(p.name) for p in psrs])
    out = []
    for p in psrs:
        f = cache / f"{str(p.name)}.npz"
        if f.exists():
            c = np.load(f)
            src_t, geom = c["toas"], c["geom"]
        elif not compute:
            raise ConfigError(f"no cached solar-wind geometry for {p.name}; run prepare first")
        else:
            if method == "approx":
                src_t, geom = np.asarray(p.toas, float), sw_geometry_approx(p.toas, p.pos)
            else:
                src_t, geom = _pint_geometry(p, files, run["data"]["ephem"])
            np.savez(f, toas=src_t, geom=geom)
        out.append(match_toas(geom, src_t, p.toas))
    return out


_TDSW_KW = {"TDSWKERNEL": "kernel", "TDSWDT": "dt_days", "TDSWINTERP_KIND": "interp_kind",
            "TDSWLOGELL": "log10_ell", "TDSWNU": "nu", "TDSWLOGGAMP": "log10_gam_p",
            "TDSWLOGP": "log10_p"}


def _tdsw_from_par(par):
    """Only the TDSW* settings actually present in ``par`` (kwarg names)."""
    from ATLAS.signals.factorized.chromatic import read_tdsw_par
    present = set()
    with open(par) as fh:
        for line in fh:
            parts = line.split()
            if parts:
                key = parts[0].upper()
                if key in _TDSW_KW:
                    present.add(_TDSW_KW[key])
                elif re.fullmatch(r"TDSWNODE_\d+", key):
                    present.add("nodes_mjd")
    full = read_tdsw_par(par)
    return {k: full[k] for k in present}


def sw_kwargs_for(run, psrs):
    """``sw_kwargs`` for TimeDomainSolarWind: config defaults, overridden per
    pulsar by the TDSW* lines of its par file (when settings_from_par)."""
    s = run["model"]["signals"]["sw"]
    base = dict(s["defaults"] or {})
    files = {}
    if s["settings_from_par"] and run["data"]["par_files"]:
        try:
            files = pulsar_files(run, [str(p.name) for p in psrs])
        except ConfigError as exc:
            log.warning("TDSW* settings: %s", exc)
    per = []
    for p in psrs:
        kw = dict(base)
        kw.setdefault("nu", 1.5)
        if s["settings_from_par"]:
            if str(p.name) in files:
                kw.update(_tdsw_from_par(files[str(p.name)][0]))
            else:
                log.warning("%s: no par file to read TDSW* from; using model.signals.sw.defaults",
                            p.name)
        per.append(kw)
    kinds = {str(kw.get("interp_kind", "linear")).lower() for kw in per}
    if len(kinds) > 1:
        raise ConfigError(f"TDSWINTERP_KIND differs between pulsars ({sorted(kinds)}); "
                          "ATLAS uses one interpolation kind for all")
    names = [str(p.name) for p in psrs]
    out = dict(interp_kind=kinds.pop())
    for key in ("kernel", "dt_days", "log10_ell", "nu", "log10_gam_p", "log10_p"):
        vals = [kw.get(key) for kw in per]
        if any(v is not None for v in vals):
            out[key] = vals
    nodes = {n: kw.get("nodes_mjd") for n, kw in zip(names, per)}
    if any(v is not None for v in nodes.values()):
        out["nodes_mjd"] = nodes
    if s["fixed_log10_sigma"] is not None:
        out["fixed_log10_sigma"] = float(s["fixed_log10_sigma"])
    return out
