"""Shared construction for the identity tests.

Builds an ATLAS model and the matching independent reference ingredients from
the same fixture, so a test body is a one-line comparison.

Imports only ``ATLAS.data``, ``ATLAS.model_builder``, ``ATLAS.nMatrix``,
``ATLAS.psd_functions`` and the CW waveform in
``ATLAS.signals.deterministic.det_signals`` -- none of which need ``jug``,
``enterprise``, ``libstempo`` or ``$TEMPO2``.  ``ATLAS.pulsar`` is deliberately
never touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from scipy.linalg import block_diag

from . import reference as ref
from .fixtures.pulsar import load_fixture
from .fixtures.synthetic import make_adaptus_basis, make_synth_pta

DATA_DIR = Path(__file__).resolve().parent / "fixtures" / "data"

from ATLAS.data import PTA_Data
from ATLAS.model_builder import ModelBuilder
from ATLAS.nMatrix.base import WhiteCov
from ATLAS.psd_functions import powerlaw, hd_orf
from ATLAS.signals.deterministic.det_signals import cw_delay_evolve_float64

# One white-noise point, used everywhere so the reference and ATLAS agree by
# construction on the parameters rather than by accident.
EFAC = 1.1
LOG10_EQUAD = float(np.log10(3e-7))
LOG10_ECORR = float(np.log10(2e-7))
TOAS_PER_SESSION = 3

# DM reference radio frequency [MHz], shared the same way: `build` hands it to
# `PTA_Data`, and `design_blocks` applies the chromatic scaling with it.
DM_REF_FREQ = 1400.0

ZERO_ORF = lambda angle: jnp.zeros_like(jnp.asarray(angle))
ORFS = {"hd": hd_orf, "zero": ZERO_ORF}

# One continuous-wave source for a `det` block, in the order
# `cw_delay_evolve_float64` unpacks: log10 chirp mass, log10 f_gw, cos i, psi,
# log10 h, cos theta, phi, phase0.
CW_BOUNDS = np.array([[7.2, -8.7, -1.0, 0.0, -18.0, -1.0, 0.0, 0.0],
                      [9.0, -8.2, 1.0, np.pi, -12.0, 1.0, 2 * np.pi, 2 * np.pi]]).T
CW_PARAMS = (8.5, -8.4, 0.3, 1.0, -13.5, 0.2, 2.0, 1.0)


def cw_point(npsr, cw=CW_PARAMS):
    """`D_params` for one CW source: (det_params, psr_phases, psr_dists)."""
    return (jnp.array(cw),
            jnp.linspace(0.1, 6.0, npsr),
            jnp.linspace(0.8, 1.5, npsr))


@dataclass
class Model:
    psrs: list
    data: object
    wn: object
    rn: object
    helpers: tuple
    wn_vec: jnp.ndarray
    residuals: np.ndarray
    n_gwb: int
    n_irn: int
    orf_name: str
    include_ecorr: bool
    n_dm: int = 0
    n_gtm: int = 0
    adaptus_basis: object = None
    gtm_psd: object = None
    n_det: int = 0

    @property
    def npsr(self):
        return len(self.psrs)

    @property
    def n_tm(self):
        return self.rn.linear_timing_model_size

    @property
    def tm_widths(self):
        return [p.Mmat.shape[1] for p in self.psrs]

    @property
    def ncol(self):
        """Width of the stochastic (non-det) basis: what `z` whitens and what
        `lnposterior_reparam` returns coefficients for."""
        return self.n_tm + 2 * self.n_irn + 2 * self.n_dm + self.n_gtm

    def det_params(self, cw=CW_PARAMS):
        """`D_params` for a `det` block, or None without one -- which the
        likelihoods accept either way."""
        return cw_point(self.npsr, cw) if self.n_det else None

    def red_params(self, irn=(-15.0, 3.5), gwb=(-14.8, 13 / 3), dm=(-15.5, 2.5),
                   irn_overrides=None):
        """Flat red-noise vector, in the order `get_param_names` reports:
        [irn per pulsar] + [dm per pulsar, if any] + [gwb].

        Adaptus contributes no free parameters: a `gtm_psd` supplied directly
        pins the free spectrum, so `gtm` adds columns but not parameters.
        """
        vals = []
        for i in range(self.npsr):
            a, g = irn_overrides[i] if (irn_overrides and i in irn_overrides) else irn
            vals += [a, g]
        if self.n_dm:
            for i in range(self.npsr):
                vals += list(dm)
        return jnp.array(vals + list(gwb))


def fixture_available(name):
    return name == "synth" or (DATA_DIR / f"{name}.npz").exists()


def load_psrs(fixture, npsr):
    if fixture == "synth":
        return make_synth_pta(npsr)
    psrs, _ = load_fixture(DATA_DIR / f"{fixture}.npz")
    return psrs[:npsr] if npsr else psrs


def white_noise_vector(psrs, include_ecorr=True):
    """Flat WN vector: per pulsar, [EFACs | log10_equads | log10_ecorrs].

    Backend counts differ per pulsar in real data, so this cannot be a tile.
    """
    out = []
    for p in psrs:
        nb = np.unique(np.asarray(p.backend_flags)).size
        out += [EFAC] * nb + [LOG10_EQUAD] * nb
        if include_ecorr:
            out += [LOG10_ECORR] * nb
    return jnp.array(out)


@lru_cache(maxsize=32)
def build(npsr=2, model_string="ltm|unc+cor->unc", linear_timing=True,
          marg_timing=False, orf_name="hd", n_gwb=4, n_irn=6, include_ecorr=True,
          fixture="synth", n_dm=0, n_gtm=0, n_det=0):
    psrs = load_psrs(fixture, npsr)
    npsr = len(psrs)
    adaptus_basis, gtm_psd = (make_adaptus_basis(psrs, n_gtm) if n_gtm else (None, None))
    data = PTA_Data(
        psrs, num_gwb_bins=n_gwb, num_irn_bins=n_irn,
        num_dm_bins=(n_dm or None), num_det_bins=(n_det or None),
        adaptus_basis=adaptus_basis, adaptus_size=(n_gtm or None),
        fixed_white_noise_params=None,
        linear_timing=linear_timing, marg_timing=marg_timing,
        diag_white_cov=False, fixed_res=False,
        timfiles=None, parfiles=None, noise_dict=None, dm_ref_freq=DM_REF_FREQ,
    )
    wn = WhiteCov(data=data, stabilize_TNT=False, include_ecorr=include_ecorr)
    rn = ModelBuilder(data=data).make_red_noise(
        model_string, use_pulsar_tspan=False,
        irn_psd_function=powerlaw, gwb_psd_function=powerlaw,
        orf_function=ORFS[orf_name],
        dm_psd_function=(powerlaw if n_dm else None),
        det_delay_function=(cw_delay_evolve_float64 if n_det else None),
        det_parameter_bounds=(CW_BOUNDS if n_det else None),
        gt_psd_val=(jnp.asarray(gtm_psd) if n_gtm else None),
        irn_lower_bound_psd=jnp.array([-20.0, 0.0]),
        irn_upper_bound_psd=jnp.array([-11.0, 7.0]),
        gwb_lower_bound_psd=jnp.array([-18.0, 0.0]),
        gwb_upper_bound_psd=jnp.array([-11.0, 7.0]),
        dm_lower_bound_psd=(jnp.array([-20.0, 0.0]) if n_dm else None),
        dm_upper_bound_psd=(jnp.array([-11.0, 7.0]) if n_dm else None),
        upper_bound_orf=None, lower_bound_orf=None,
    )
    wn_vec = white_noise_vector(psrs, include_ecorr)
    residuals = np.concatenate([np.asarray(p.residuals) for p in psrs])
    helpers = rn.get_helpers(reff=jnp.concat(data.raw_residuals),
                             white_noise_params=wn_vec)
    return Model(psrs, data, wn, rn, helpers, wn_vec, residuals,
                 n_gwb, n_irn, orf_name, include_ecorr,
                 n_dm, n_gtm, adaptus_basis, gtm_psd, n_det)


# --------------------------------------------------------------------------- #
#  Independent reference ingredients for the same model
# --------------------------------------------------------------------------- #

def svd_basis(M):
    """``signals_utils._timing_model_svd``, reimplemented from its docstring."""
    U, _, _ = np.linalg.svd(np.asarray(M, dtype=np.float64), full_matrices=False)
    norm = np.sqrt((U ** 2).sum(axis=0))
    out = U / norm
    out[:, norm == 0] = 0.0
    return out


def freq_grid(model):
    tspan = model.data.pta_tspan
    return (np.arange(1, model.n_irn + 1) / tspan,
            np.arange(1, model.n_gwb + 1) / tspan,
            1.0 / tspan)


def design_blocks(model, include_timing=None):
    """Per-pulsar ``T = [M_padded | F_unc | F_dm | U_gtm]``, built independently.

    Column order follows `SuperSignal.build_basis`: the timing prefix, then the
    shared block (with `cor` nested at its head), then each separate block in
    the order the model string lists them.  ``F_dm`` is chromatic: each TOA's
    row is scaled by ``(DM_REF_FREQ / f_radio)**2``.
    """
    if include_timing is None:
        include_timing = model.n_tm > 0
    f_irn, _, _ = freq_grid(model)
    f_dm = (np.arange(1, model.n_dm + 1) / model.data.pta_tspan) if model.n_dm else None
    blocks = []
    for i, p in enumerate(model.psrs):
        parts = []
        if include_timing:
            Ms = svd_basis(p.Mmat)
            Mp = np.zeros((p.ntoa, model.n_tm))
            Mp[:, :Ms.shape[1]] = Ms
            parts.append(Mp)
        parts.append(ref.fourier_basis(p.toas, f_irn))
        if model.n_dm:
            chromatic = (DM_REF_FREQ / np.asarray(p.freqs, dtype=np.float64)) ** 2
            parts.append(ref.fourier_basis(p.toas, f_dm) * chromatic[:, None])
        if model.n_gtm:
            parts.append(np.asarray(model.adaptus_basis[i], dtype=np.float64))
        blocks.append(np.hstack(parts))
    return blocks


def noise_blocks(model, include_ecorr=None):
    if include_ecorr is None:
        include_ecorr = model.include_ecorr
    blocks = []
    for p in model.psrs:
        backends = np.unique(np.asarray(p.backend_flags))
        bidx = np.searchsorted(backends, np.asarray(p.backend_flags))
        nvec = ref.white_nvec(p.toaerrs, bidx,
                              np.full(backends.size, EFAC),
                              np.full(backends.size, LOG10_EQUAD))
        if include_ecorr:
            epochs, _ = ref.epochs_from_toas(p.toas, p.backend_flags)
            jvec = np.full(len(epochs), 10.0 ** (2 * LOG10_ECORR))
            blocks.append(ref.dense_N(nvec, epochs, jvec))
        else:
            blocks.append(ref.dense_N(nvec))
    return blocks


def psd_pieces(model, red_params):
    """IRN, DM (both per pulsar, bin resolution) and GWB PSDs, from the flat
    vector, unpacked in `get_param_names` order."""
    f_irn, f_gwb, df = freq_grid(model)
    xs = np.asarray(red_params, dtype=np.float64)
    irn = np.column_stack([ref.powerlaw_psd(f_irn, df, xs[2 * i], xs[2 * i + 1])
                           for i in range(model.npsr)])
    dm = None
    if model.n_dm:
        f_dm = np.arange(1, model.n_dm + 1) / model.data.pta_tspan
        off = 2 * model.npsr
        dm = np.column_stack([
            ref.powerlaw_psd(f_dm, df, xs[off + 2 * i], xs[off + 2 * i + 1])
            for i in range(model.npsr)])
    gwb = ref.powerlaw_psd(f_gwb, df, xs[-2], xs[-1])
    return irn, dm, gwb


def orf_matrix(model):
    n = model.npsr
    pos = np.array([np.asarray(p.pos) for p in model.psrs])
    out = np.eye(n)
    if model.orf_name == "zero":
        return out
    for i in range(n):
        for j in range(i + 1, n):
            zeta = np.arccos(np.clip(pos[i] @ pos[j], -1.0, 1.0))
            out[i, j] = out[j, i] = ref.hd_orf(zeta)
    return out


def global_phi(model, red_params):
    irn, dm, gwb = psd_pieces(model, red_params)
    return ref.build_phi(model.npsr, model.ncol, model.n_tm, model.tm_widths,
                         irn, gwb, orf_matrix(model),
                         dm_psd=dm, gtm_psd=model.gtm_psd)


def phiinv_diag(model, red_params):
    """The diagonal precision ATLAS uses to build its standardising transform."""
    irn, dm, gwb = psd_pieces(model, red_params)
    pad = np.asarray(model.rn._pad_mask) if model.n_tm else None
    out = []
    for i in range(model.npsr):
        d = np.empty(model.ncol)
        if model.n_tm:
            d[:model.n_tm] = np.where(pad[i] > 0, 1.0, 1e-40)
        tot = irn[:, i].copy()
        tot[:model.n_gwb] += gwb
        pieces = [np.repeat(tot, 2)]
        if model.n_dm:
            pieces.append(np.repeat(dm[:, i], 2))
        if model.n_gtm:
            pieces.append(np.asarray(model.gtm_psd)[:, i])
        d[model.n_tm:] = 1.0 / np.concatenate(pieces)
        out.append(d)
    return out


def dense_bundle(model, include_ecorr=None):
    T = design_blocks(model)
    N = noise_blocks(model, include_ecorr)
    return T, N, block_diag(*N), model.residuals


# --------------------------------------------------------------------------- #
#  The model-string corpus
# --------------------------------------------------------------------------- #
#
# Every layout that currently works, with the kwargs that make it work. The
# identity suite runs over this, and the Stage 1 regression harness uses the
# same list -- so "every currently-working model string" has one definition.
#
# `"ltm|unc+cor->unc;gtm"` is the production string in
# notebooks/easiest_way_to_use_atlas.py, so it is the one that matters most.
#
# A `det` case needs `D_params` (`Model.det_params()`) in every likelihood call,
# and its `z` is `Model.ncol` wide: the deterministic columns are not whitened.
#
CORPUS = {
    "curn": dict(model_string="unc+cor->unc", linear_timing=False),
    "curn-margtm": dict(model_string="unc+cor->unc", linear_timing=False,
                        marg_timing=True),
    "ltm": dict(model_string="ltm|unc+cor->unc", linear_timing=True),
    "ltm-dm": dict(model_string="ltm|unc+cor->unc;dm", linear_timing=True, n_dm=3),
    "ltm-gtm": dict(model_string="ltm|unc+cor->unc;gtm", linear_timing=True, n_gtm=8),
    "ltm-dm-gtm": dict(model_string="ltm|unc+cor->unc;dm,gtm", linear_timing=True,
                       n_dm=3, n_gtm=8),
    "curn-margtm-det": dict(model_string="unc+cor->unc;det", linear_timing=False,
                            marg_timing=True, n_det=6),
    "ltm-det": dict(model_string="ltm|unc+cor->unc;det", linear_timing=True, n_det=6),
}
