"""Name -> function lookup for config files.

A config file names PSDs, ORFs and deterministic waveforms as strings; this
module resolves them against ATLAS and reports the parameters each one
samples, so prior bounds can be checked before anything runs.
"""
from __future__ import annotations

import inspect
from functools import partial

from ATLAS import psd_functions

__all__ = ["psd", "orf", "waveform", "psd_param_names", "orf_param_names",
           "available_psds", "available_orfs"]

_NOT_PSDS = {"clip_psd", "make_free_spectrum", "interp1d", "partial", "lru_cache"}
FREE_SPECTRUM_SENTINEL = "halflog10_rho"


def _public_functions(module, suffix=None, exclude=()):
    out = {}
    for name, obj in vars(module).items():
        if name.startswith("_") or name in exclude or not callable(obj):
            continue
        if inspect.isclass(obj) or inspect.ismodule(obj):
            continue
        if getattr(obj, "__module__", module.__name__) not in (module.__name__, None) \
                and not hasattr(obj, "__wrapped__"):
            continue
        if suffix is None or name.endswith(suffix):
            out[name] = obj
    return out


def available_psds():
    fns = _public_functions(psd_functions, exclude=_NOT_PSDS)
    return {k: v for k, v in fns.items() if not k.endswith("_orf")}


def available_orfs():
    return _public_functions(psd_functions, suffix="_orf")


def psd(name, signal=None):
    """The PSD function called ``name``.

    For the common process ``cor``, ``free_spectrum`` resolves to
    ``gwb_free_spectrum`` (same parameters, column-shaped output, which the
    correlated phi assembly needs).
    """
    table = available_psds()
    if signal == "cor" and name == "free_spectrum":
        name = "gwb_free_spectrum"
    if name not in table:
        raise KeyError(f"unknown PSD {name!r}; available: {sorted(table)}")
    return table[name]


def orf(name):
    table = available_orfs()
    if name not in table:
        raise KeyError(f"unknown ORF {name!r}; available: {sorted(table)}")
    return table[name]


def waveform(name):
    from ATLAS.signals.deterministic import det_signals
    fn = getattr(det_signals, name, None)
    if fn is None or not callable(fn):
        raise KeyError(f"unknown deterministic waveform {name!r} in "
                       "ATLAS.signals.deterministic.det_signals")
    return fn


def _signature(func):
    while isinstance(func, partial):
        func = func.func
    return [p for p in inspect.signature(func).parameters.values()
            if p.kind is not inspect.Parameter.VAR_KEYWORD]


def psd_param_names(func, fixed=(), n_bins=None):
    """Sampled parameter names of a PSD ``f(f, df, *params)``, in order.

    A free spectrum (``*halflog10_rho``) has one parameter per bin, named
    ``halflog10_rho_<i>``; ``n_bins`` is then required.
    """
    names = [p.name for p in _signature(func)[2:]]
    if FREE_SPECTRUM_SENTINEL in names:
        if n_bins is None:
            raise ValueError("a free spectrum needs the number of bins")
        return [f"{FREE_SPECTRUM_SENTINEL}_{i}" for i in range(int(n_bins))]
    unknown = set(fixed) - set(names)
    if unknown:
        raise KeyError(f"fixed parameters {sorted(unknown)} are not parameters of "
                       f"this PSD; it takes {names}")
    return [n for n in names if n not in set(fixed)]


def orf_param_names(func):
    return [p.name for p in _signature(func)[1:]]


def is_free_spectrum(func):
    return FREE_SPECTRUM_SENTINEL in [p.name for p in _signature(func)[2:]]
