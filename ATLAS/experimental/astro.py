"""
Hierarchical astrophysical inference of the GWB (ported from Pandora).

This module will hold everything needed to go from an astrophysical population
model to a GWB likelihood term:

1. Training-set generation: holodeck SAM draws converted to free-spectrum
   amplitudes ``log10_rho``.
2. Conditional normalizing-flow training on those sets
   (:func:`train_astro_flow`; the flows live in ``zuko_flows.py``).
3. The hierarchical PTA posterior and its PTMCMC sampling
   (:class:`HDAstroRedModel`, :class:`AstroInferenceModel`).

Training sets
-------------
Pandora generates training sets in ``examples/AstroInferenceUpdated.ipynb``.
The functions here follow that notebook closely enough to reproduce its output
bit for bit, given the same parameter draws and the same RNG seed:

* Latin-hypercube draws of the astrophysical parameters come from a holodeck
  ``_Param_Space``. Any holodeck parameter space works, and
  :func:`pandora_phenom_param_space` builds the 6-parameter "phenom" space
  that Pandora uses.
* Each draw runs ``sam.gwb_new`` and converts the characteristic strain to
  ``log10_rho = 0.5 * log10(hc^2 / (12 pi^2 f^3 T))``, which is the
  ``halflog10_rho`` / enterprise ``log10_rho`` convention.
* Files on disk use Pandora's names and layouts, so Pandora training sets and
  Atlas training sets can be used interchangeably:

  ========================================================  ===================
  ``test_astro_params.npy``                                  (n_draws, n_pars)
  ``{i}_{tag}.npy``                                          (n_f, n_real)
  ``gwb_spectrum_samples_{tag}.npy``                         (n_kept, n_real, n_f)
  ``gwb_spectrum_samples_{tag}_normalized.npy``              (n_kept, n_real, n_f)
  ``ast_spectrum_samples_{tag}_normalized.npy``              (n_kept, n_pars)
  ``gwb_spectrum_samples_{tag}_mapping_data.npy.npz``        B, mean, half_range
  ========================================================  ===================

  Atlas also writes ``trainset_{tag}_metadata.json``, which records how the
  set was made.

Differences from Pandora
------------------------
* Draws can be seeded. holodeck builds an unseeded ``PCG64()`` for every
  ``gwb_new`` call, so Pandora runs cannot be reproduced. Here draw ``i`` is
  seeded with ``seed + i`` through :func:`seeded_holodeck_rng`.
* Parameters stay aligned with spectra when draws are dropped. Pandora drops a
  draw whose spectrum is not finite, and then pairs spectra with
  ``theta[:n_files]``, which shifts every later draw onto the wrong
  parameters. Atlas tracks the draw index of each spectrum instead.
* Normalization works column by column instead of building the full
  ``(n_draws, n_real, n_pars + n_f)`` array. The result is bitwise identical.

Frequencies
-----------
``freqs=None`` converts ``hc`` at ``f_i = i/T`` (Pandora's choice, and also the
centres of holodeck's bins, whose edges are at ``(i +- 1/2)/T``). An explicit
array of ``n_freqs`` frequencies may be passed instead.

holodeck and h5py are imported lazily, so this module imports without the
``[astro]`` extra.
"""

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ATLAS.experimental.astro_utils import (
    conversion_freqs,
    default_tag,
    draw_indices_on_disk,
    draw_path,
    gwb_bounds as _gwb_bounds,
    metadata_path,
    read_metadata,
    simulate_and_save,
)

__all__ = [
    "PANDORA_PHENOM_DEFAULTS",
    "PANDORA_PHENOM_PARAM_NAMES",
    "pandora_phenom_param_space",
    "gwb_frequencies",
    "hc_to_log10_rho",
    "seeded_holodeck_rng",
    "simulate_log10_rho",
    "generate_training_set",
    "combine_training_set",
    "load_holodeck_library",
    "TrainingSet",
    "train_astro_flow",
    "HDAstroRedModel",
    "AstroInferenceModel",
    "pandora_to_atlas_order",
    "atlas_to_pandora_order",
]

PARAMS_FILE = "test_astro_params.npy"

# Fixed SAM / hardening settings of Pandora's phenom model
# (AstroInferenceUpdated.ipynb, cell 16). The parameters varied in Pandora's
# training sets are listed in PANDORA_PHENOM_PARAM_NAMES; their values here
# are only defaults.
PANDORA_PHENOM_DEFAULTS = dict(
    hard_time=3.0,
    hard_sepa_init=1e4,
    hard_rchar=100.0,
    hard_gamma_inner=-1.0,
    hard_gamma_outer=+2.5,

    gsmf_phi0_log10=-2.77,
    gsmf_phiz=-0.6,
    gsmf_mchar0_log10=11.24,
    gsmf_mcharz=0.11,
    gsmf_alpha0=-1.21,
    gsmf_alphaz=-0.03,

    gpf_frac_norm_allq=0.025,
    gpf_malpha=0.0,
    gpf_qgamma=0.0,
    gpf_zbeta=1.0,
    gpf_max_frac=1.0,

    gmt_norm=0.5,
    gmt_malpha=0.0,
    gmt_qgamma=-1.0,
    gmt_zbeta=-0.5,

    mmb_mamp_log10=8.69,
    mmb_plaw=1.10,
    mmb_scatter_dex=0.3,
)

# Order matches the flow context in Pandora (cell 20).
PANDORA_PHENOM_PARAM_NAMES = (
    "hard_time",
    "gsmf_phi0_log10",
    "gsmf_mchar0_log10",
    "mmb_mamp_log10",
    "mmb_scatter_dex",
    "hard_gamma_inner",
)


########################################################################################
# Parameter spaces
########################################################################################

@lru_cache(maxsize=None)
def _pandora_phenom_class():
    """Define ``PS_Pandora_Phenom`` on first use, so holodeck stays optional."""
    import holodeck as holo
    from holodeck.constants import GYR, PC
    from holodeck.librarian.lib_tools import _Param_Space, PD_Normal, PD_Uniform

    class PS_Pandora_Phenom(_Param_Space):
        """Pandora's 6-parameter phenom SAM (AstroInferenceUpdated.ipynb).

        GSMF Schechter + GPF/GMT power laws + KH2013 M-Mbulge, with
        ``Fixed_Time_2PL_SAM`` hardening. ``defaults`` overrides entries of
        :data:`PANDORA_PHENOM_DEFAULTS`.
        """

        DEFAULTS = dict(PANDORA_PHENOM_DEFAULTS)

        def __init__(self, log=None, nsamples=None, sam_shape=None, seed=None,
                     defaults=None, **kwargs):
            if defaults:
                self.DEFAULTS = {**PANDORA_PHENOM_DEFAULTS, **defaults}
            parameters = [
                PD_Uniform("hard_time", 0.1, 11.0),
                PD_Normal("gsmf_phi0_log10", -2.56, 0.4),
                PD_Normal("gsmf_mchar0_log10", 10.9, 0.4),
                PD_Normal("mmb_mamp_log10", 8.6, 0.2),
                PD_Normal("mmb_scatter_dex", 0.32, 0.15),
                PD_Uniform("hard_gamma_inner", -1.5, 0.0),
            ]
            _Param_Space.__init__(self, parameters, log=log, nsamples=nsamples,
                                  sam_shape=sam_shape, seed=seed, **kwargs)

        def _init_sam(self, sam_shape, params):
            gsmf = holo.sams.GSMF_Schechter(
                phi0=params["gsmf_phi0_log10"],
                phiz=params["gsmf_phiz"],
                mchar0_log10=params["gsmf_mchar0_log10"],
                mcharz=params["gsmf_mcharz"],
                alpha0=params["gsmf_alpha0"],
                alphaz=params["gsmf_alphaz"],
            )
            gpf = holo.sams.GPF_Power_Law(
                frac_norm_allq=params["gpf_frac_norm_allq"],
                malpha=params["gpf_malpha"],
                qgamma=params["gpf_qgamma"],
                zbeta=params["gpf_zbeta"],
                max_frac=params["gpf_max_frac"],
            )
            gmt = holo.sams.GMT_Power_Law(
                time_norm=params["gmt_norm"] * GYR,
                malpha=params["gmt_malpha"],
                qgamma=params["gmt_qgamma"],
                zbeta=params["gmt_zbeta"],
            )
            mmbulge = holo.host_relations.MMBulge_KH2013(
                mamp_log10=params["mmb_mamp_log10"],
                mplaw=params["mmb_plaw"],
                scatter_dex=params["mmb_scatter_dex"],
            )
            return holo.sams.Semi_Analytic_Model(
                gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge, shape=sam_shape,
            )

        def _init_hard(self, sam, params):
            return holo.hardening.Fixed_Time_2PL_SAM(
                sam,
                params["hard_time"] * GYR,
                sepa_init=params["hard_sepa_init"] * PC,
                rchar=params["hard_rchar"] * PC,
                gamma_inner=params["hard_gamma_inner"],
                gamma_outer=params["hard_gamma_outer"],
            )

    # Pickle by reference through the module-level __getattr__ below, so
    # joblib workers can receive instances.
    PS_Pandora_Phenom.__module__ = __name__
    PS_Pandora_Phenom.__qualname__ = "PS_Pandora_Phenom"
    return PS_Pandora_Phenom


def pandora_phenom_param_space(nsamples=None, sam_shape=(30, 30, 30), seed=None,
                               defaults=None):
    """
    Pandora's phenom parameter space as a holodeck ``_Param_Space``.

    Parameters
    ----------
    nsamples : int or None
        Number of Latin-hypercube draws. ``None`` builds the space without
        draws, e.g. to call ``model_for_params`` directly.
    sam_shape : int or (3,) tuple of int
        SAM grid (total mass, mass ratio, redshift). Pandora uses (30, 30, 30).
    seed : int or None
        Seed for ``scipy.stats.qmc.LatinHypercube``. With the same seed,
        ``param_samples`` equals Pandora's cell-11 draws exactly.
    defaults : dict or None
        Overrides for :data:`PANDORA_PHENOM_DEFAULTS`.

    Returns
    -------
    PS_Pandora_Phenom
        Draws are in ``param_samples``, ordered as
        :data:`PANDORA_PHENOM_PARAM_NAMES`.
    """
    cls = _pandora_phenom_class()
    return cls(nsamples=nsamples, sam_shape=sam_shape, seed=seed, defaults=defaults)


########################################################################################
# Spectra
########################################################################################

def gwb_frequencies(tspan, n_freqs, freqs=None):
    """
    Frequencies used to convert ``hc`` to ``log10_rho``, plus holodeck's bin edges.

    Parameters
    ----------
    tspan : float
        Observing span [s].
    n_freqs : int
        Number of GWB frequency bins.
    freqs : None or array_like
        ``None``: ``f_i = i/T``, computed with Pandora's own expression.
        An array: used as given, shape ``(n_freqs,)``.

    Returns
    -------
    f_conv : (n_freqs,) ndarray
    fobs_gw_edges : (n_freqs + 1,) ndarray
        Bin edges passed to ``sam.gwb_new``.
    """
    from holodeck import utils as hutils

    _, edges = hutils.pta_freqs(tspan, n_freqs)
    return conversion_freqs(freqs, tspan, n_freqs), edges



def hc_to_log10_rho(hc, f_conv, tspan):
    """
    ``0.5 * log10(hc^2 / (12 pi^2 f^3 T))``, the enterprise ``log10_rho``.

    ``f_conv`` must broadcast against ``hc``. The operation order matches
    Pandora's, so the result is bitwise identical.
    """
    return 0.5 * np.log10(hc**2 / (12 * np.pi**2 * f_conv**3 * tspan))


@contextmanager
def seeded_holodeck_rng(seed):
    """
    Make holodeck's GWB realizations reproducible inside this context.

    ``holodeck.cyutils`` builds a fresh, unseeded ``numpy.random.PCG64()`` on
    every call to ``sam_poisson_gwb`` (the ``realize=int`` path of
    ``gwb_new``), so ``np.random.seed`` has no effect on it. This temporarily
    replaces that module's ``PCG64`` with one drawing from
    ``SeedSequence(seed)``, and seeds the global numpy RNG (used by holodeck's
    other realization paths). Both are restored on exit. ``seed=None`` does
    nothing.
    """
    if seed is None:
        yield
        return
    import holodeck.cyutils as cy

    seq = np.random.SeedSequence(seed)
    orig_pcg, orig_state = cy.PCG64, np.random.get_state()
    cy.PCG64 = lambda: np.random.PCG64(seq.spawn(1)[0])
    np.random.seed(seed)
    try:
        yield
    finally:
        cy.PCG64 = orig_pcg
        np.random.set_state(orig_state)


def simulate_log10_rho(pspace, params, tspan, n_freqs, n_real, freqs=None,
                       seed=None, sam_shape=None):
    """
    Simulate ``n_real`` GWB realizations for one parameter draw.

    Parameters
    ----------
    pspace : holodeck ``_Param_Space``
        Builds the SAM and hardening model through ``model_for_params``.
    params : dict or array_like
        Parameter values; an array is matched to ``pspace.param_names``.
    tspan, n_freqs, freqs
        See :func:`gwb_frequencies`.
    n_real : int
        Number of Poisson realizations.
    seed : int or None
        Seeds the realizations through :func:`seeded_holodeck_rng`.
        ``None`` leaves them unseeded, as in Pandora.
    sam_shape : optional
        Overrides ``pspace.sam_shape``.

    Returns
    -------
    (n_freqs, n_real) ndarray
        ``log10_rho``. Bins without any sources give ``-inf``.
    """
    if not isinstance(params, dict):
        params = dict(zip(pspace.param_names, np.asarray(params)))
    f_conv, edges = gwb_frequencies(tspan, n_freqs, freqs)
    with seeded_holodeck_rng(seed):
        sam, hard = pspace.model_for_params(params, sam_shape=sam_shape)
        hc = sam.gwb_new(edges, hard=hard, realize=n_real)
    return hc_to_log10_rho(hc, f_conv[:, None], tspan)


########################################################################################
# Training sets
########################################################################################

def generate_training_set(pspace, save_dir, tspan, n_freqs=5, n_real=10_000,
                          freqs=None, params=None, indices=None, n_jobs=1,
                          seed=None, sam_shape=None, overwrite=False, tag=None,
                          combine=True):
    """
    Simulate GWB spectra for each parameter draw and save them in Pandora's layout.

    Parameters
    ----------
    pspace : holodeck ``_Param_Space``
        E.g. :func:`pandora_phenom_param_space` or any ``holodeck.librarian``
        space. Supplies the draws (``param_samples``) unless ``params`` is given.
    save_dir : str
        Output directory, created if needed.
    tspan : float
        Observing span [s]. Pandora uses ``20 * YR``.
    n_freqs, n_real : int
        Frequency bins and Poisson realizations per draw.
    freqs : None or array_like
        See :func:`gwb_frequencies`.
    params : (n_draws, n_pars) array_like, optional
        Parameter draws to use instead of ``pspace.param_samples``, e.g. a
        Pandora ``test_astro_params.npy``.
    indices : iterable of int, optional
        Draws to run (default: all). Use this to split a run across jobs.
    n_jobs : int
        joblib workers.
    seed : int or None
        Draw ``i`` is seeded with ``seed + i`` (:func:`seeded_holodeck_rng`).
    sam_shape : optional
        Overrides ``pspace.sam_shape``.
    overwrite : bool
        Rerun draws whose output already exists (or that were already found
        to be non-finite). Otherwise they are skipped, so runs can resume.
    tag : str, optional
        File suffix, default ``f"{tspan/YR:g}yrs"`` as in Pandora.
    combine : bool
        Also run :func:`combine_training_set` and return its result.

    Returns
    -------
    TrainingSet or None
        ``None`` if ``combine=False``.

    Notes
    -----
    A draw with a non-finite spectrum (a bin with no sources) is not saved,
    as in Pandora. It is recorded under ``bad_indices`` in the metadata file.
    """
    from joblib import Parallel, delayed
    from tqdm_joblib import tqdm_joblib

    os.makedirs(save_dir, exist_ok=True)
    tag = tag or default_tag(tspan)
    params = np.asarray(pspace.param_samples if params is None else params)
    if params.ndim != 2 or params.shape[1] != len(pspace.param_names):
        raise ValueError(f"params has shape {params.shape}, expected "
                         f"(n_draws, {len(pspace.param_names)})")
    sam_shape = pspace.sam_shape if sam_shape is None else sam_shape
    f_conv, _ = gwb_frequencies(tspan, n_freqs, freqs)

    meta = read_metadata(save_dir, tag)
    bad = set(meta.get("bad_indices", []))
    params_path = os.path.join(save_dir, PARAMS_FILE)
    if os.path.exists(params_path) and not overwrite:
        if not np.array_equal(np.load(params_path), params):
            raise ValueError(f"{params_path} holds different draws; pass overwrite=True "
                             "or use a new save_dir")
    else:
        np.save(params_path, params)

    indices = range(len(params)) if indices is None else indices
    todo = [int(i) for i in indices
            if overwrite or (not os.path.exists(draw_path(save_dir, i, tag)) and i not in bad)]

    def draw_seed(i):
        return None if seed is None else seed + i

    jobs = (delayed(simulate_and_save)(pspace, i, params[i], draw_path(save_dir, i, tag),
                                        tspan, n_freqs, n_real, freqs, draw_seed(i), sam_shape)
            for i in todo)
    with tqdm_joblib(desc="GWB draws", total=len(todo)):
        results = Parallel(n_jobs=n_jobs)(jobs)

    for i, ok in results:
        (bad.discard if ok else bad.add)(i)

    from holodeck import __version__ as holo_version

    meta.update(
        tag=tag,
        param_names=list(pspace.param_names),
        param_space=pspace.name,
        tspan=float(tspan),
        n_freqs=int(n_freqs),
        n_real=int(n_real),
        freqs="default" if freqs is None else "array",
        f_conv=f_conv.tolist(),
        sam_shape=np.asarray(sam_shape).tolist() if sam_shape is not None else None,
        seed=seed,
        holodeck_version=holo_version,
        bad_indices=sorted(bad),
    )
    with open(metadata_path(save_dir, tag), "w") as fh:
        json.dump(meta, fh, indent=2)

    return combine_training_set(save_dir, tag) if combine else None


def combine_training_set(save_dir, tag):
    """
    Combine per-draw spectra into ``gwb_spectrum_samples_{tag}.npy`` (cell 25).

    The combined file is a ``(n_kept, n_real, n_f)`` memmap in draw-index
    order. Unlike Pandora, parameters are matched to spectra by draw index,
    so dropped draws do not shift the pairing.

    Returns
    -------
    TrainingSet
    """
    indices = draw_indices_on_disk(save_dir, tag)
    if indices.size == 0:
        raise FileNotFoundError(f"no '*_{tag}.npy' draws in {save_dir}")
    first = np.load(draw_path(save_dir, indices[0], tag))
    n_f, n_real = first.shape
    out = np.lib.format.open_memmap(
        os.path.join(save_dir, f"gwb_spectrum_samples_{tag}.npy"),
        mode="w+", dtype="float64", shape=(len(indices), n_real, n_f),
        fortran_order=False)
    for row, idx in enumerate(indices):
        out[row] = np.load(draw_path(save_dir, idx, tag)).T
    out.flush()
    del out
    return TrainingSet.from_pandora_files(save_dir, tag, indices=indices)


@dataclass
class TrainingSet:
    """
    Astrophysical parameter draws paired with their simulated ``log10_rho``.

    Attributes
    ----------
    params : (n_draws, n_pars) ndarray
    log10_rho : (n_draws, n_real, n_f) ndarray
        May be a read-only memmap.
    param_names : list of str
    f_conv : (n_f,) ndarray or None
        Frequencies used in the ``hc -> log10_rho`` conversion.
    tspan : float or None
    draw_indices : (n_draws,) ndarray
        Row of each draw in the full parameter file.
    metadata : dict
    """

    params: np.ndarray
    log10_rho: np.ndarray
    param_names: list
    f_conv: np.ndarray = None
    tspan: float = None
    draw_indices: np.ndarray = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.params.shape[0] != self.log10_rho.shape[0]:
            raise ValueError(f"{self.params.shape[0]} parameter draws but "
                             f"{self.log10_rho.shape[0]} spectra")
        if self.draw_indices is None:
            self.draw_indices = np.arange(self.params.shape[0])

    @property
    def n_draws(self):
        return self.log10_rho.shape[0]

    @property
    def n_real(self):
        return self.log10_rho.shape[1]

    @property
    def n_freqs(self):
        return self.log10_rho.shape[2]

    @property
    def n_params(self):
        return self.params.shape[1]

    @classmethod
    def from_pandora_files(cls, save_dir, tag, params_file=PARAMS_FILE, indices=None):
        """
        Load ``gwb_spectrum_samples_{tag}.npy`` and its parameter file.

        Works on directories written by Pandora's notebook or by
        :func:`generate_training_set`. The draw index of each spectrum is
        taken from ``indices`` if given, else from the per-draw
        ``{i}_{tag}.npy`` files if present, else assumed to be
        ``0..n-1`` (Pandora's assumption).
        """
        rho = np.load(os.path.join(save_dir, f"gwb_spectrum_samples_{tag}.npy"), mmap_mode="r")
        all_params = np.load(os.path.join(save_dir, params_file))
        if indices is None:
            indices = draw_indices_on_disk(save_dir, tag)
            if indices.size != rho.shape[0]:
                indices = np.arange(rho.shape[0])
        indices = np.asarray(indices, dtype=int)
        meta = read_metadata(save_dir, tag)
        names = meta.get("param_names", [f"param_{i}" for i in range(all_params.shape[1])])
        f_conv = np.asarray(meta["f_conv"]) if "f_conv" in meta else None
        return cls(params=all_params[indices], log10_rho=rho, param_names=list(names),
                   f_conv=f_conv, tspan=meta.get("tspan"), draw_indices=indices,
                   metadata=meta)

    # ---- normalization (cells 28-31) -----------------------------------------

    def normalization(self, B=5):
        """
        Pandora's affine map to ``[-B, B]``, per column of ``[params, log10_rho]``.

        Returns
        -------
        dict
            ``B``, and ``mean = (max+min)/2`` and ``half_range = (max-min)/2``,
            each of shape ``(n_pars + n_f,)``.
        """
        # Pandora asserts the concatenated chain has no zeros (cell 28).
        if not (np.all(self.params) and np.all(self.log10_rho)):
            raise ValueError("training set contains exact zeros")
        min_x = np.concatenate([np.min(self.params, axis=0),
                                np.min(self.log10_rho, axis=(0, 1))])
        max_x = np.concatenate([np.max(self.params, axis=0),
                                np.max(self.log10_rho, axis=(0, 1))])
        mean = (max_x + min_x) / 2
        half_range = (max_x - min_x) / 2
        return dict(B=B, mean=mean, half_range=half_range)

    def normalize(self, B=5, mapping=None):
        """
        Map to ``[-B, B]`` in memory.

        Parameters
        ----------
        B : float
        mapping : dict, optional
            Use this mapping (e.g. the full set's, for a split) instead of
            computing one from this set; ``B`` is then taken from it.

        Returns
        -------
        rho_norm : (n_draws, n_real, n_f) ndarray
        ast_norm : (n_draws, n_pars) ndarray
        mapping : dict
            See :meth:`normalization`.
        """
        mapping = self.normalization(B) if mapping is None else mapping
        B = mapping["B"]
        p = self.n_params
        mean, half = mapping["mean"], mapping["half_range"]
        rho_norm = B * (np.asarray(self.log10_rho) - mean[p:]) / half[p:]
        ast_norm = B * (self.params - mean[:p]) / half[:p]
        return rho_norm, ast_norm, mapping

    def save_normalized(self, save_dir, tag, B=5, chunk=256):
        """
        Write Pandora's normalized files (cell 31), streaming ``chunk`` draws at a time.

        Writes ``gwb_spectrum_samples_{tag}_normalized.npy``,
        ``ast_spectrum_samples_{tag}_normalized.npy`` and
        ``gwb_spectrum_samples_{tag}_mapping_data.npy.npz``.

        Returns
        -------
        dict
            The mapping (see :meth:`normalization`).
        """
        mapping = self.normalization(B)
        p = self.n_params
        mean, half = mapping["mean"], mapping["half_range"]
        rho_out = np.lib.format.open_memmap(
            os.path.join(save_dir, f"gwb_spectrum_samples_{tag}_normalized.npy"),
            mode="w+", dtype="float64", shape=self.log10_rho.shape)
        for start in range(0, self.n_draws, chunk):
            sl = slice(start, start + chunk)
            rho_out[sl] = B * (self.log10_rho[sl] - mean[p:]) / half[p:]
        rho_out.flush()
        del rho_out
        np.save(os.path.join(save_dir, f"ast_spectrum_samples_{tag}_normalized.npy"),
                B * (self.params - mean[:p]) / half[:p])
        np.savez_compressed(os.path.join(save_dir, f"gwb_spectrum_samples_{tag}_mapping_data.npy"),
                            B=B, mean=mean, half_range=half)
        return mapping

    # ---- train / validation split (DataSplitter) ---------------------------------

    def split(self, n_val_draws, n_val_real, seeds=(0, 1)):
        """
        Hold out a validation set, as Pandora's ``DataSplitter`` does.

        ``n_val_draws`` parameter draws and ``n_val_real`` realizations are
        chosen at random; the validation set is their cross product, and the
        training set is everything outside both (so the two share neither
        draws nor realizations). Reproduces ``DataSplitter.splitter_mesh`` on
        ``log10_rho`` and ``DataSplitter.splitter`` on ``params`` given the
        same seeds.

        Parameters
        ----------
        n_val_draws, n_val_real : int
        seeds : (int, int)
            Seeds for the draw and realization choices.

        Returns
        -------
        train, val : TrainingSet
            Validation rows follow the random draw order, as in Pandora.
        """
        import random

        v0 = random.Random(seeds[0]).sample(range(self.n_draws), k=n_val_draws)
        v1 = random.Random(seeds[1]).sample(range(self.n_real), k=n_val_real)
        keep0 = np.ones(self.n_draws, dtype=bool)
        keep0[v0] = False
        keep1 = np.ones(self.n_real, dtype=bool)
        keep1[v1] = False

        def subset(rows, rho):
            return TrainingSet(params=self.params[rows], log10_rho=rho,
                               param_names=list(self.param_names), f_conv=self.f_conv,
                               tspan=self.tspan, draw_indices=self.draw_indices[rows],
                               metadata=dict(self.metadata))

        rho = self.log10_rho
        train = subset(keep0, rho[np.ix_(keep0, keep1)])
        val = subset(v0, rho[np.ix_(v0, v1)])
        return train, val


def train_astro_flow(training_set, save_dir, nf_type="rho|theta", B=5, val=None,
                     split_seeds=(0, 1), steps=10_000, batch_size=512, save_freq=1_000,
                     mode="diagonal", device=None, spline_bins=8, hidden_dims=(512, 512),
                     transforms=3, train_kwargs=None, backend="zuko", architecture="zuko_nsf",
                     arch_kwargs=None, seed=0):
    """
    Train a conditional flow on a training set (Pandora's full training path).

    Normalizes ``training_set`` to ``[-B, B]`` (the same map Pandora saves in
    ``*_mapping_data.npy.npz``), optionally holds out a validation set for
    Hellinger-distance early stopping, trains a flow, and saves it.

    Parameters
    ----------
    training_set : TrainingSet
    save_dir : str
        Receives checkpoints, ``flow_config.json`` and the final weights;
        reload with ``ZukoAstroFlow.load(save_dir)`` or
        ``FlaxAstroFlow.load(save_dir)``.
    nf_type : {'rho|theta', 'theta|rho', 'rho'}
    B : float
    val : (n_val_draws, n_val_real) or None
        Validation split (see :meth:`TrainingSet.split`). Not available for
        ``'theta|rho'``.
    split_seeds : (int, int)
    steps, batch_size, save_freq, mode
        See ``zuko_flows.FlowTrainer.train``. ``mode='flat'`` reproduces the
        loop in ``AstroInferenceUpdated.ipynb``.
    device, spline_bins, hidden_dims, transforms
        See ``zuko_flows.FlowTrainer``. The notebook uses
        ``hidden_dims=[512] * 8``. With ``backend='flax'`` and
        ``architecture='zuko_nsf'`` they define the same flow in JAX (and
        ``device`` is ignored: JAX uses its default device).
    train_kwargs : dict, optional
        Further ``train`` arguments (``learning_rate``, ``seed``,
        ``patience``, ...).
    backend : {'zuko', 'flax'}
        Train with torch (``zuko_flows.FlowTrainer``) or JAX
        (``flax_flows.FlaxFlowTrainer``).
    architecture : {'zuko_nsf', 'coupling_rqs'}
        Flax architecture (``backend='flax'`` only).
    arch_kwargs : dict, optional
        Flax architecture arguments; for ``'zuko_nsf'`` they default to
        ``spline_bins``, ``hidden_dims`` and ``transforms``.
    seed : int
        Weight initialization for ``backend='flax'`` (the zuko backend
        follows ``torch.manual_seed``).

    Returns
    -------
    flow : zuko_flows.ZukoAstroFlow or flax_flows.FlaxAstroFlow
    history : dict
    """
    from ATLAS.experimental.flow_utils import NF_TYPES

    if backend not in ("zuko", "flax"):
        raise ValueError(f"backend must be 'zuko' or 'flax', got {backend!r}")
    if nf_type not in NF_TYPES:
        raise ValueError(f"nf_type must be one of {NF_TYPES}, got {nf_type!r}")
    mapping = training_set.normalization(B)
    if val is not None:
        if nf_type == "theta|rho":
            raise ValueError("validation is only available for 'rho|theta' and 'rho' flows")
        train_set, val_set = training_set.split(*val, seeds=split_seeds)
    else:
        train_set, val_set = training_set, None

    rho_n, ast_n, _ = train_set.normalize(mapping=mapping)
    inputs, context = {"rho|theta": (rho_n, ast_n), "theta|rho": (ast_n, rho_n),
                       "rho": (rho_n, None)}[nf_type]

    validation = None
    if val_set is not None:
        v_rho, v_ast, _ = val_set.normalize(mapping=mapping)
        if nf_type == "rho|theta":
            # (n_ctx, n_val_real, n_f) -> (n_ctx * n_f, n_val_real), Pandora's layout
            validation = (v_rho.transpose(0, 2, 1).reshape(-1, v_rho.shape[1]), v_ast)
        else:
            # unconditional: all held-out samples of each frequency, (n_f, n)
            validation = (v_rho.reshape(-1, v_rho.shape[-1]).T, None)

    metadata = dict(nf_type=nf_type, n_params=training_set.n_params, mapping=mapping,
                    param_names=list(training_set.param_names),
                    f_conv=None if training_set.f_conv is None else list(training_set.f_conv),
                    tspan=training_set.tspan)
    flow_meta = {k: metadata[k] for k in ("param_names", "f_conv", "tspan")}
    if backend == "flax":
        from ATLAS.experimental import flax_flows as ff

        if arch_kwargs is None and architecture == "zuko_nsf":
            arch_kwargs = dict(bins=spline_bins, hidden_features=list(hidden_dims),
                               transforms=transforms)
        trainer = ff.FlaxFlowTrainer(inputs, context, save_dir, B, architecture=architecture,
                                     arch_kwargs=arch_kwargs, metadata=metadata, seed=seed)
        history = trainer.train(steps, batch_size, save_freq, mode=mode, validation=validation,
                                **(train_kwargs or {}))
        flow = ff.FlaxAstroFlow(trainer.model, nf_type, mapping, training_set.n_params,
                                metadata=flow_meta)
    else:
        from ATLAS.experimental import zuko_flows as zf

        trainer = zf.FlowTrainer(inputs, context, save_dir, B, device=device,
                                 spline_bins=spline_bins, hidden_dims=hidden_dims,
                                 transforms=transforms, metadata=metadata)
        history = trainer.train(steps, batch_size, save_freq, mode=mode, validation=validation,
                                **(train_kwargs or {}))
        flow = zf.ZukoAstroFlow(trainer.flow, nf_type, mapping, training_set.n_params,
                                device=trainer.device, metadata=flow_meta)
    flow.save(save_dir)
    return flow, history


def load_holodeck_library(path, n_freqs, tspan=None, freqs=None, hc_floor=1e-20):
    """
    Load a holodeck librarian HDF5 library (e.g. ``sam_lib.hdf5``) as a TrainingSet.

    Follows Pandora's legacy loader (NormalizingFlowTrainDEMO.ipynb): keep the
    first ``n_freqs`` bins of ``gwb``, raise ``hc`` below ``hc_floor`` to
    ``hc_floor``, and convert to ``log10_rho``.

    Parameters
    ----------
    path : str
    n_freqs : int
    tspan : float, optional
        Observing span [s]. Defaults to the inverse bin width of the
        library's ``fobs_edges``.
    freqs : None or array_like
        See :func:`gwb_frequencies`.
    hc_floor : float or None
        ``None`` disables the floor (empty bins then give ``-inf``).

    Returns
    -------
    TrainingSet
    """
    import h5py

    with h5py.File(path, "r") as h5:
        if "gwb" not in h5:
            raise KeyError(f"{path} has no 'gwb' dataset")
        hc = h5["gwb"][:, :n_freqs, :]
        params = h5["sample_params"][()]
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in h5.attrs["param_names"]]
        edges = h5["fobs_edges"][()] if "fobs_edges" in h5 else None

    if tspan is None:
        if edges is None:
            raise ValueError("library has no 'fobs_edges'; pass tspan")
        tspan = 1.0 / (edges[1] - edges[0])
    f_conv = conversion_freqs(freqs, tspan, n_freqs)

    if hc_floor is not None:
        hc[hc < hc_floor] = hc_floor
    # Pandora floors in the stored dtype, then upcasts by concatenating with
    # the float64 parameters before taking the log.
    hc = hc.astype(np.float64)
    rho = hc_to_log10_rho(hc.transpose((0, 2, 1)), f_conv[None, None, :], tspan)
    meta = dict(source=os.path.abspath(path), tspan=float(tspan), n_freqs=int(n_freqs),
                freqs="default" if freqs is None else "array",
                f_conv=f_conv.tolist(), hc_floor=hc_floor, param_names=names)
    return TrainingSet(params=params, log10_rho=rho, param_names=names, f_conv=f_conv,
                       tspan=float(tspan), metadata=meta)


########################################################################################
# Hierarchical inference
########################################################################################

class HDAstroRedModel:
    """
    Red noise for hierarchical astro inference: intrinsic power laws plus an HD GWB.

    The likelihood is marginalized over all Fourier coefficients with a dense
    cross-pulsar covariance, as in Pandora's ``AstroInferenceModel``::

        ln L = 1/2 (TNr^T Sigma^-1 TNr - ln|Sigma| - ln|phi|),  Sigma = TNT + phi^-1

    with the timing model marginalized and the white noise fixed inside TNT
    and TNr. phi, phi^-1 and the prior bounds come from Atlas's
    ``CorrelatedPulsarRedNoise``; build the model with :meth:`from_pta_data`
    (Atlas computes TNT/TNr) or :meth:`from_matrices` (precomputed ones).

    The parameter vector is Atlas's: ``[log10_A, gamma]`` per pulsar, then the
    GWB PSD parameters (``halflog10_rho`` per bin for a free spectrum), then
    any free ORF parameters. Pandora orders each pulsar's pair
    ``[gamma, log10_A]``; see :func:`pandora_to_atlas_order`.

    Attributes
    ----------
    lower, upper : ndarray
        Uniform prior bounds.
    param_names : list of str
    n_irn_params : int
        Length of the intrinsic red-noise block at the front of ``xs``.
    gwb_slice : slice
        Position of the GWB PSD parameters in ``xs``.
    gwb_is_free_spectrum : bool
    crn_bins, int_bins : int
    tspan : float
    f_common : ndarray
        GWB frequencies, ``i / tspan``.
    """

    def __init__(self, TNT, TNr, model, tspan, rNr=0.0, logdet_N=0.0):
        import jax
        import jax.numpy as jnp
        from ATLAS.psd_functions import gwb_free_spectrum

        self.model = model
        self.TNT = jnp.asarray(TNT)
        self.TNr = jnp.asarray(TNr)
        self.npsr = int(model.Npulsars)
        n = self.TNT.shape[0]
        if self.TNT.shape != (n, n) or self.TNr.shape != (n,) or n % self.npsr:
            raise ValueError(f"TNT {self.TNT.shape} / TNr {self.TNr.shape} do not match "
                             f"{self.npsr} pulsars")
        self.kmax = n // self.npsr
        self.int_bins = int(model.irn_bins)
        if self.kmax != 2 * self.int_bins:
            raise ValueError(f"TNT has {self.kmax} columns per pulsar but the red-noise "
                             f"model has {self.int_bins} bins (expected {2 * self.int_bins})")
        self.crn_bins = int(model.crn_bins)
        self.tspan = float(tspan)
        self.f_common = np.asarray(model.f_common)[:, 0]
        self.lower = np.asarray(model.lower_prior_lim_all, dtype=np.float64)
        self.upper = np.asarray(model.upper_prior_lim_all, dtype=np.float64)
        self.param_names = list(model.get_param_names())
        self.n_irn_params = int(model.irn_end_idx)
        self.gwb_slice = slice(int(model.gtm_end_idx), int(model.gwb_psd_end_idx))
        self.gwb_is_free_spectrum = model.gwb_psd_func is gwb_free_spectrum
        self.orf_fixed = bool(model.orf_fixed)
        self.rNr, self.logdet_N = float(rNr), float(logdet_N)
        self._lnl = jax.jit(self._lnlikelihood)

    @classmethod
    def from_pta_data(cls, data, white_noise_params, irn_log10_A_bounds=(-18.0, -11.0),
                      irn_gamma_bounds=(1.0, 7.0), rho_bounds=(-12.0, -4.0),
                      gwb_psd_function=None, orf_function=None, gwb_bounds=None,
                      include_ecorr=None, tspan=None):
        """
        Build TNT/TNr and the red-noise model with Atlas.

        Parameters
        ----------
        data : ATLAS.data.PTA_Data
            Built with ``marg_timing=True``, ``linear_timing=False``,
            ``num_gwb_bins`` (``crn_bins``) and ``num_irn_bins >= num_gwb_bins``.
        white_noise_params : dict or array_like
            Fixed white noise: a noise dictionary, or Atlas's flat vector.
        irn_log10_A_bounds, irn_gamma_bounds : (float, float)
            Intrinsic red-noise priors (Pandora's ``UniformPrior`` defaults).
        rho_bounds : (float, float)
            Prior on each ``halflog10_rho`` of a free-spectrum GWB.
        gwb_psd_function, orf_function : callable, optional
            Default: ``psd_functions.gwb_free_spectrum`` and ``hd_orf``.
        gwb_bounds : (lower, upper), optional
            GWB PSD bounds; required unless the PSD is a free spectrum.
        include_ecorr : bool or None
            Passed to ``WhiteCov`` (``None`` auto-detects).
        tspan : float, optional
            Overrides ``data.pta_tspan`` before the Fourier bases are built,
            e.g. to match the span a flow was trained on.
        """
        import jax.numpy as jnp
        from ATLAS.model_builder import ModelBuilder
        from ATLAS.nMatrix.base import WhiteCov
        from ATLAS.psd_functions import gwb_free_spectrum, hd_orf, powerlaw
        from ATLAS.signals.signals_utils import blockPsrs2sigma, blockVec2sigmaVec

        if not data.marg or data.linear_timing:
            raise ValueError("HDAstroRedModel needs PTA_Data(marg_timing=True, linear_timing=False)")
        gwb_psd_function = gwb_free_spectrum if gwb_psd_function is None else gwb_psd_function
        orf_function = hd_orf if orf_function is None else orf_function
        gwb_lo, gwb_hi = _gwb_bounds(gwb_psd_function, gwb_bounds, rho_bounds,
                                     data.num_gwb_bins)
        if tspan is not None:
            data.pta_tspan = float(tspan)

        wn = WhiteCov(data=data, stabilize_TNT=False, include_ecorr=include_ecorr)
        if isinstance(white_noise_params, dict):
            white_noise_params = wn.params_dict_to_vector(white_noise_params)
        rn = ModelBuilder(data=data).make_red_noise(
            "unc+cor->unc", use_pulsar_tspan=False,
            irn_psd_function=powerlaw, gwb_psd_function=gwb_psd_function,
            orf_function=orf_function,
            irn_lower_bound_psd=jnp.array([irn_log10_A_bounds[0], irn_gamma_bounds[0]]),
            irn_upper_bound_psd=jnp.array([irn_log10_A_bounds[1], irn_gamma_bounds[1]]),
            gwb_lower_bound_psd=gwb_lo, gwb_upper_bound_psd=gwb_hi,
        )
        TNT, TNr, rNr, logdet_N = rn.get_helpers(reff=jnp.concat(data.raw_residuals),
                                                 white_noise_params=jnp.asarray(white_noise_params))
        return cls(blockPsrs2sigma(TNT), blockVec2sigmaVec(TNr), rn.model, data.pta_tspan,
                   rNr=rNr, logdet_N=logdet_N)

    @classmethod
    def from_matrices(cls, TNT, TNr, psr_pos, tspan, crn_bins, int_bins=None,
                      irn_log10_A_bounds=(-18.0, -11.0), irn_gamma_bounds=(1.0, 7.0),
                      rho_bounds=(-12.0, -4.0), gwb_psd_function=None, orf_function=None,
                      gwb_bounds=None, stabilize_delta=None, psr_names=None):
        """
        Use precomputed TNT/TNr (e.g. Pandora's ``data/15yr_TNT_and_TNr_*.npz``).

        Parameters
        ----------
        TNT : (Np * 2 * int_bins, Np * 2 * int_bins) array_like
            Dense, pulsar-major, columns ``[sin f1, cos f1, sin f2, ...]``.
        TNr : (Np * 2 * int_bins,) array_like
        psr_pos : (Np, 3) array_like
            Unit vectors to the pulsars.
        psr_names : list of str, optional
            Used in parameter names (default ``psr0``, ``psr1``, ...).
        tspan : float
            Span defining the Fourier frequencies ``i / tspan``.
        crn_bins, int_bins : int
            GWB and intrinsic-noise bins (``int_bins`` defaults to ``crn_bins``).
        stabilize_delta : float, optional
            Pandora's ``matrix_stabilization``: add ``delta`` to the diagonal
            of TNT's correlation matrix and rescale by ``1 / (1 + delta)``.
        Other parameters
            As in :meth:`from_pta_data`.
        """
        import jax.numpy as jnp
        from ATLAS.parameterized import CorrelatedPulsarRedNoise
        from ATLAS.psd_functions import gwb_free_spectrum, hd_orf, powerlaw
        from ATLAS.signals.correlated.utils import make_gwb_model
        from ATLAS.signals.factorized.utils import make_irn_model
        from ATLAS.signals.signals_utils import get_harmonic_frequencies

        int_bins = crn_bins if int_bins is None else int_bins
        gwb_psd_function = gwb_free_spectrum if gwb_psd_function is None else gwb_psd_function
        orf_function = hd_orf if orf_function is None else orf_function
        gwb_lo, gwb_hi = _gwb_bounds(gwb_psd_function, gwb_bounds, rho_bounds, crn_bins)
        irn_func, irn_help = make_irn_model(
            powerlaw, jnp.array([irn_log10_A_bounds[0], irn_gamma_bounds[0]]),
            jnp.array([irn_log10_A_bounds[1], irn_gamma_bounds[1]]))
        gwb_func, orf_func, gwb_help = make_gwb_model(gwb_psd_function, orf_function,
                                                      gwb_lo, gwb_hi)
        psr_pos = np.asarray(psr_pos, dtype=np.float64)
        if psr_names is None:
            psr_names = [f"psr{i}" for i in range(len(psr_pos))]
        model = CorrelatedPulsarRedNoise(
            psr_pos=psr_pos, Npulsars=len(psr_pos),
            signal_indices={"unc": slice(0, 2 * int_bins), "cor": slice(0, 2 * crn_bins)},
            linear_timing_model_size=0,
            gwb_psd_func=gwb_func, orf_func=orf_func, gwb_helper_dictionary=gwb_help,
            crn_bins=crn_bins, f_common=get_harmonic_frequencies(crn_bins, tspan),
            irn_psd_func=irn_func, irn_helper_dictionary=irn_help, irn_bins=int_bins,
            f_irn=get_harmonic_frequencies(int_bins, tspan), pulsar_names=psr_names,
        )
        TNT = jnp.asarray(TNT)
        if stabilize_delta:
            D = jnp.outer(jnp.sqrt(TNT.diagonal()), jnp.sqrt(TNT.diagonal()))
            corr = TNT / D + stabilize_delta * jnp.eye(TNT.shape[0])
            TNT = D * corr / (1 + stabilize_delta)
        return cls(TNT, TNr, model, tspan)

    def _lnlikelihood(self, xs, TNT, TNr):
        import jax.numpy as jnp
        import jax.scipy as jsp
        from ATLAS.signals.signals_utils import blockModes2sigma

        phi = self.model.get_phi_mat(xs)
        psd_common = self.model.get_phi_diag(xs)[1]
        phiinv, logdet_phi = self.model.get_phi_mat_inv(phi)
        cf = jsp.linalg.cho_factor(TNT + blockModes2sigma(phiinv), lower=False)
        mu = jsp.linalg.cho_solve(cf, TNr)
        logdet_sigma = 2 * jnp.log(cf[0].diagonal()).sum()
        return 0.5 * (jnp.dot(TNr, mu) - logdet_sigma - logdet_phi), psd_common

    def lnlikelihood(self, xs, include_constant=False):
        """
        Marginal log likelihood and the common-process PSD.

        Parameters
        ----------
        xs : array_like
            Red-noise parameters (see the class docstring).
        include_constant : bool
            Add ``-1/2 (r^T N^-1 r + ln|N|)`` (available from
            :meth:`from_pta_data`). Pandora omits it.

        Returns
        -------
        lnl : jax scalar
        psd_common : (crn_bins, 1) jax array
            ``rho^2`` per GWB bin; ``0.5 * log10(psd_common)`` is ``log10_rho``.
        """
        import jax.numpy as jnp

        lnl, psd = self._lnl(jnp.asarray(xs, dtype=jnp.float64), self.TNT, self.TNr)
        if include_constant:
            lnl = lnl - 0.5 * (self.rNr + self.logdet_N)
        return lnl, psd


def pandora_to_atlas_order(xs, n_psr):
    """Swap each pulsar's ``[gamma, log10_A]`` (Pandora) to ``[log10_A, gamma]`` (Atlas)."""
    out = np.array(xs, dtype=np.float64, copy=True)
    pairs = out[..., :2 * n_psr].reshape(out.shape[:-1] + (n_psr, 2))
    out[..., :2 * n_psr] = pairs[..., ::-1].reshape(out.shape[:-1] + (2 * n_psr,))
    return out


def atlas_to_pandora_order(xs, n_psr):
    """Inverse of :func:`pandora_to_atlas_order` (the swap is its own inverse)."""
    return pandora_to_atlas_order(xs, n_psr)


class AstroInferenceModel:
    """
    Joint posterior of red noise and astrophysical parameters (Pandora's
    ``AstroInferenceModel``), sampled with PTMCMCSampler::

        p(xs, theta | data) ∝ L_HD(xs) p_flow(log10_rho(xs) | theta) p(theta)

    ``log10_rho = 0.5 * log10(psd_common)`` is the GWB spectrum implied by the
    red-noise parameters; the flow scores it given the astrophysical
    parameters ``theta``. The sampled vector is ``[red-noise params | varied
    theta]``.

    Parameters
    ----------
    red_model : HDAstroRedModel
    flow : zuko_flows.ZukoAstroFlow
        A ``'rho|theta'`` flow trained at the same frequencies as ``red_model``
        (checked against the flow's ``f_conv``).
    astro_lower, astro_upper : array_like
        Uniform prior bounds of the varied astrophysical parameters (full
        length is also accepted when some are fixed).
    astro_log_prior : callable, optional
        Extra log prior on the varied parameters, e.g. Gaussians; Pandora's
        ``astro_additional_prior_func``.
    fixed_astro : dict, optional
        ``{index: value}`` of astrophysical parameters held fixed, indexed in
        the flow's context order.
    seed : int, optional
        Seeds the jump proposals' generator.

    Differences from Pandora
    ------------------------
    * The prior is the normalized uniform density (Pandora returns -8.01),
      and the flow density includes its Jacobian, so log posteriors are
      proper densities.
    * The flow-draw proposal conditions on the full ``theta`` (Pandora passed
      only the varied parameters, which breaks with fixed parameters), and it
      is used only when the GWB is a free spectrum (Pandora wrote ``rho`` into
      whatever the GWB parameters were).
    * Proposals use one seeded generator; ``sample(seed=...)`` reproduces a
      chain exactly. ``seed=0`` is honoured.
    * The flow's frequencies must match the red-noise model's.
    * Atlas does not rescale TNT/TNr (Pandora's ``renorm_const``), so the
      flow-draw proposal needs no unit offset; Pandora omitted it.
    """

    def __init__(self, red_model, flow, astro_lower, astro_upper, astro_log_prior=None,
                 fixed_astro=None, seed=None, freq_rtol=1e-10):
        if flow.nf_type != "rho|theta":
            raise ValueError(f"need a 'rho|theta' flow, got {flow.nf_type!r}")
        self.red, self.flow = red_model, flow
        self.n_astro = int(flow.n_params)
        fixed_astro = dict(fixed_astro or {})
        self.astro_container = np.zeros(self.n_astro)
        for i, v in fixed_astro.items():
            self.astro_container[int(i)] = v
        self.varied = np.array([i for i in range(self.n_astro) if i not in fixed_astro], dtype=int)
        self.n_varied = len(self.varied)

        astro_lower, astro_upper = (np.asarray(a, dtype=np.float64) for a in (astro_lower, astro_upper))
        if astro_lower.shape == (self.n_astro,) and self.n_varied != self.n_astro:
            astro_lower, astro_upper = astro_lower[self.varied], astro_upper[self.varied]
        if astro_lower.shape != (self.n_varied,) or astro_upper.shape != (self.n_varied,):
            raise ValueError(f"astro bounds must have length {self.n_varied} (varied parameters)")
        self.n_red = len(red_model.lower)
        self.lower = np.concatenate([red_model.lower, astro_lower])
        self.upper = np.concatenate([red_model.upper, astro_upper])
        self.ln_prior_value = -float(np.sum(np.log(self.upper - self.lower)))
        names = flow.metadata.get("param_names") or [f"astro_{i}" for i in range(self.n_astro)]
        self.param_names = red_model.param_names + [names[i] for i in self.varied]
        self.astro_log_prior = astro_log_prior if astro_log_prior is not None else (lambda x: 0.0)

        n_f = len(flow.mean_gwb)
        if n_f != red_model.crn_bins:
            raise ValueError(f"flow models {n_f} frequency bins but the red-noise model has "
                             f"{red_model.crn_bins} GWB bins")
        f_conv = flow.metadata.get("f_conv")
        if f_conv is not None and not np.allclose(f_conv, red_model.f_common, rtol=freq_rtol, atol=0):
            flow_tspan = flow.metadata.get("tspan")
            raise ValueError(
                f"flow was trained at frequencies {np.asarray(f_conv)} (Tspan {flow_tspan}) but the "
                f"red-noise model uses {red_model.f_common} (Tspan {red_model.tspan}); "
                "rebuild the red-noise model with a matching tspan")

        self.rng = np.random.default_rng(seed)

    # ---- posterior ------------------------------------------------------------

    def full_astro(self, varied):
        """The flow's context: fixed values with ``varied`` filled in."""
        theta = self.astro_container.copy()
        theta[self.varied] = varied
        return theta

    def lnlikelihood(self, xs):
        """``ln L_HD + ln p_flow(log10_rho | theta) + astro_log_prior(theta)`` (Pandora's ``get_lnliklihood``)."""
        xs = np.asarray(xs, dtype=np.float64)
        theta_v = xs[self.n_red:]
        lnl, psd = self.red.lnlikelihood(xs[:self.n_red])
        log10_rho = np.asarray(0.5 * np.log10(np.asarray(psd).T))
        try:
            lp = self.flow.log_prob(log10_rho, self.full_astro(theta_v)[None])
        except AssertionError:
            return -np.inf
        return float(np.asarray(lnl) + lp[0] + self.astro_log_prior(theta_v))

    def lnprior(self, xs):
        """Normalized uniform prior on the box (strict inequalities, as in Pandora)."""
        xs = np.asarray(xs)
        inside = np.all((xs > self.lower) & (xs < self.upper))
        return self.ln_prior_value if inside else -np.inf

    def make_initial_guess(self, seed=None):
        """A uniform draw from the prior box (``seed`` gives a fresh generator)."""
        rng = self.rng if seed is None else np.random.default_rng(seed)
        return rng.uniform(self.lower, self.upper)

    # ---- jump proposals (PTMCMCSampler signature) ------------------------------

    def _redraw(self, x, lo, hi):
        q = np.array(x, copy=True)
        i = self.rng.integers(lo, hi)
        q[i] = self.rng.uniform(self.lower[i], self.upper[i])
        return q, 0.0

    def draw_from_prior(self, x, iter, beta):
        """Redraw one parameter from its prior."""
        return self._redraw(x, 0, len(x))

    def draw_from_red_prior(self, x, iter, beta):
        """Redraw one intrinsic red-noise parameter."""
        return self._redraw(x, 0, self.red.n_irn_params)

    def draw_from_nonIR_prior(self, x, iter, beta):
        """Redraw one parameter outside the intrinsic red-noise block."""
        return self._redraw(x, self.red.n_irn_params, len(x))

    def draw_from_astro_prior(self, x, iter, beta):
        """Redraw one astrophysical parameter."""
        return self._redraw(x, self.n_red, len(x))

    def draw_from_gwb_prior(self, x, iter, beta):
        """Independence proposal: ``log10_rho ~ p_flow(. | theta)``."""
        return self._gwb_flow_draw(x, int(self.rng.integers(2**62)))

    def _gwb_flow_draw(self, x, torch_seed):
        q = np.array(x, copy=True)
        theta = self.full_astro(q[self.n_red:])
        sl = self.red.gwb_slice
        current = q[sl][None]
        new = self.flow.sample(1, theta, seed=torch_seed)
        lqxy = (self.flow.log_prob(current, theta[None], jacobian=False)
                - self.flow.log_prob(new, theta[None], jacobian=False))
        q[sl] = new[0]
        return q, float(lqxy[0])

    # ---- sampling --------------------------------------------------------------

    def _write_pars(self, outdir, names):
        with open(os.path.join(outdir, "pars.txt"), "w") as fh:
            fh.write("\n".join(names) + "\n")

    def sample(self, niter, outdir, x0=None, seed=None, resume=True, load_cov=False,
               include_groups=True, include_IRN_groups=False, **sample_kwargs):
        """
        Run PTMCMCSampler (Pandora's ``sample``).

        Parameters
        ----------
        niter : int
        outdir : str
            Chains (``chain_1.txt``), ``pars.txt`` and, with ``load_cov``,
            the starting ``cov.npy``.
        x0 : array_like, optional
            Start; default a prior draw.
        seed : int, optional
            Seeds PTMCMCSampler and the proposals, so the chain is reproducible.
        resume : bool
            Continue an existing chain in ``outdir`` (Pandora's default).
        load_cov : bool
            Start from ``outdir/cov.npy`` instead of ``0.01**2 * I``.
        include_groups, include_IRN_groups : bool
            Pandora's extra jump groups: the non-intrinsic block (and the ORF
            parameters if free), and the intrinsic block.
        sample_kwargs
            Passed to ``PTSampler.sample`` (Pandora's jump weights SCAM 30,
            AM 15, DE 50 are the defaults here).
        """
        from PTMCMCSampler.PTMCMCSampler import PTSampler

        os.makedirs(outdir, exist_ok=True)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        x0 = self.make_initial_guess() if x0 is None else np.asarray(x0, dtype=np.float64)
        ndim = len(x0)
        cov = (np.load(os.path.join(outdir, "cov.npy")) if load_cov
               else np.diag(np.ones(ndim) * 0.01**2))
        n_irn = self.red.n_irn_params
        groups = [list(range(ndim))]
        if include_IRN_groups and n_irn:
            groups += [list(range(n_irn))] * 2
        if include_groups:
            groups += [list(range(n_irn, ndim))] * 2
            if not self.red.orf_fixed:
                groups += [list(range(self.red.gwb_slice.stop, ndim))] * 2

        sampler = PTSampler(ndim, self.lnlikelihood, self.lnprior, cov, groups=groups,
                            outDir=outdir, resume=resume, seed=seed)
        sampler.addProposalToCycle(self.draw_from_prior, 10)
        if n_irn:
            sampler.addProposalToCycle(self.draw_from_red_prior, 10)
        sampler.addProposalToCycle(self.draw_from_nonIR_prior, 10)
        if self.n_varied:
            sampler.addProposalToCycle(self.draw_from_astro_prior, 10)
        if self.red.gwb_is_free_spectrum:
            sampler.addProposalToCycle(self.draw_from_gwb_prior, 10)
        self._write_pars(outdir, self.param_names)
        kwargs = dict(SCAMweight=30, AMweight=15, DEweight=50)
        kwargs.update(sample_kwargs)
        sampler.sample(x0, niter, **kwargs)
        return sampler



#############################################
##            Helper functions             ##
#############################################

def __getattr__(name):
    # Lazy class lookup; must live in this module so PS_Pandora_Phenom
    # instances pickle by reference (joblib workers).
    if name == "PS_Pandora_Phenom":
        return _pandora_phenom_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
