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
           "load_distances", "load_gtm", "sw_geometry_approx", "match_toas"]


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


def pulsar_files(run):
    """``{pulsar_name: (par, tim)}`` from data.par_files / data.tim_files.

    Pairs are matched by name (from the par file's PSR/PSRJ line, found in the
    tim file's name), not by sort order.
    """
    d = run["data"]
    pars, tims = _expand(run, d["par_files"]), _expand(run, d["tim_files"])
    out = {}
    for par in pars:
        name = _par_name(par)
        hits = [t for t in tims if Path(t).name.startswith(name)] or \
               [t for t in tims if name in Path(t).name]
        if len(hits) != 1:
            raise ConfigError(f"cannot pair {par} ({name}) with exactly one tim file; "
                              f"candidates: {hits}")
        out[name] = (par, hits[0])
    return out


def data_key(run):
    """Hash of everything that determines the loaded pulsars."""
    d = run["data"]
    spec = dict(source=d["source"], ephem=d["ephem"], timing_package=d["timing_package"],
                pars=_expand(run, d["par_files"]), tims=_expand(run, d["tim_files"]),
                pickle=resolve_path(run, d["pickle"]), npz=resolve_path(run, d["npz_dir"]))
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
        paths = sorted(glob.glob(str(Path(resolve_path(run, d["npz_dir"])) / "*.npz")))
        return [PulsarDataLoader(**np.load(p, allow_pickle=True)) for p in paths]
    raise ConfigError(f"unknown data.source {src!r}")


def prepare_data(run):
    """Load every pulsar of this run's dataset once and cache it (driver side)."""
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
    """The pulsars of one run: cached dataset, filtered by pulsars / exclude."""
    path = _pulsar_cache(run)
    if not path.exists():
        prepare_data(run)
    with open(path, "rb") as fh:
        psrs = pickle.load(fh)
    want, excl = run["data"]["pulsars"], set(run["data"]["exclude"] or [])
    names = [str(p.name) for p in psrs]
    if want != "all":
        missing = sorted(set(want) - set(names))
        if missing:
            raise ConfigError(f"data.pulsars not in the dataset: {missing}")
        psrs = [p for p in psrs if str(p.name) in set(want)]
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


def load_gtm(run, psrs):
    """(basis list, gtm_psd or None) from npz files keyed by pulsar name."""
    g = run["model"]["signals"]["gtm"]
    names = [str(p.name) for p in psrs]
    z = np.load(resolve_path(run, g["basis_file"]))
    missing = [n for n in names if n not in z.files]
    if missing:
        raise ConfigError(f"gtm basis_file has no basis for {missing}")
    basis = [np.asarray(z[n], float) for n in names]
    widths = {b.shape[1] for b in basis}
    if len(widths) != 1:
        raise ConfigError(f"gtm bases must share one width; got {sorted(widths)}")
    psd = None
    if g["psd_file"]:
        zp = np.load(resolve_path(run, g["psd_file"]))
        psd = (np.asarray(zp["gtm_psd"], float) if "gtm_psd" in zp.files
               else np.stack([np.asarray(zp[n], float) for n in names], axis=1))
    return basis, psd


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
    files = pulsar_files(run) if run["data"]["par_files"] else {}
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
    files = pulsar_files(run) if (s["settings_from_par"] and run["data"]["par_files"]) else {}
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
