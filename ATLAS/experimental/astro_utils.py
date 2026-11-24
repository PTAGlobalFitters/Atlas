"""
Helpers for :mod:`ATLAS.experimental.astro`: training-set file layout,
frequency conventions, the per-draw simulation worker, and GWB prior bounds.
"""

import glob
import json
import os
import re

import numpy as np


def conversion_freqs(freqs, tspan, n_freqs):
    """Frequencies for the ``hc -> log10_rho`` conversion (see ``astro.gwb_frequencies``)."""
    if freqs is None:
        return np.arange(1 / tspan, (n_freqs + .001) / tspan, 1 / tspan)
    if isinstance(freqs, str):
        raise ValueError(f"freqs must be None or an array, got {freqs!r}")
    f = np.asarray(freqs, dtype=np.float64)
    if f.shape != (n_freqs,):
        raise ValueError(f"freqs array has shape {f.shape}, expected ({n_freqs},)")
    return f


def default_tag(tspan):
    """Pandora's file suffix, e.g. ``'20yrs'``."""
    from holodeck.constants import YR

    return f"{round(tspan / YR, 6):g}yrs"


def metadata_path(save_dir, tag):
    return os.path.join(save_dir, f"trainset_{tag}_metadata.json")


def draw_path(save_dir, idx, tag):
    return os.path.join(save_dir, f"{idx}_{tag}.npy")


def read_metadata(save_dir, tag):
    """The training set's metadata JSON, or ``{}`` if there is none."""
    path = metadata_path(save_dir, tag)
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def draw_indices_on_disk(save_dir, tag):
    """Sorted draw indices of the ``{i}_{tag}.npy`` files in ``save_dir``."""
    pattern = re.compile(rf"^(\d+)_{re.escape(tag)}\.npy$")
    found = (pattern.match(os.path.basename(p))
             for p in glob.glob(os.path.join(save_dir, f"*_{tag}.npy")))
    return np.array(sorted(int(m.group(1)) for m in found if m), dtype=int)


def simulate_and_save(pspace, idx, params, path, tspan, n_freqs, n_real, freqs,
                      seed, sam_shape):
    """joblib worker: simulate one draw and save it unless it is non-finite."""
    # looked up at call time so tests can patch astro.simulate_log10_rho
    from ATLAS.experimental import astro

    rho = astro.simulate_log10_rho(pspace, params, tspan, n_freqs, n_real, freqs=freqs,
                                   seed=seed, sam_shape=sam_shape)
    if not np.isfinite(rho).all():
        return idx, False
    np.save(path, rho)
    return idx, True


def gwb_bounds(gwb_psd_function, bounds, rho_bounds, crn_bins):
    """GWB PSD prior bounds: given, or ``rho_bounds`` per bin for a free spectrum."""
    import jax.numpy as jnp
    from ATLAS.psd_functions import gwb_free_spectrum

    if bounds is not None:
        return jnp.asarray(bounds[0]), jnp.asarray(bounds[1])
    if gwb_psd_function is not gwb_free_spectrum:
        raise ValueError("gwb_bounds is required unless the GWB PSD is gwb_free_spectrum")
    return jnp.full(crn_bins, float(rho_bounds[0])), jnp.full(crn_bins, float(rho_bounds[1]))
