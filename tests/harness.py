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


# --------------------------------------------------------------------------- #
#  NG20 v2.0 vanilla noise: chrom, cquad, sw
# --------------------------------------------------------------------------- #
#
# `build_ng20` mirrors `build` for the chromatic and solar-wind blocks.  The
# reference side (`ng20_dense`) assembles T and phi per pulsar from
# `reference`, in the column order `SuperSignal.build_basis` produces:
# [timing | unc (cor nested at its head) | separate blocks in string order].
# Zero padding columns ATLAS adds (cquad's 4th column, sw up to the widest
# pulsar) are left out of the reference: a zero column of T changes nothing
# in C = N + T phi T^T.

NG20_CHROM_BINS = 7
NG20_SW_DT_DAYS = 60.0
NG20_DMCONST = 1.0 / 2.41e-4        # PINT's DMconst, s MHz^2 cm^3 / pc
NG20_MATERN = dict(kernel="MATERN", log10_ell=2.0, nu=1.5)

NG20_CORPUS = {
    "curn-all":   dict(model_string="unc+cor->unc;dm,chrom,cquad,sw"),
    "irn-all":    dict(model_string="unc;dm,chrom,cquad,sw"),
    "ltm-all":    dict(model_string="ltm|unc+cor->unc;dm,chrom,cquad,sw"),
    "curn-chrom": dict(model_string="unc+cor->unc;dm,chrom"),
    "curn-sw":    dict(model_string="unc+cor->unc;dm,sw"),
    "curn-matern": dict(model_string="unc+cor->unc;dm,chrom,cquad,sw", sw_kernel="MATERN"),
}


def sw_geometry_synth(psr):
    """A smooth, positive, annually modulated stand-in for the solar-wind
    geometry [pc].  The real one comes from PINT; the algebra does not care."""
    t = np.asarray(psr.toas, dtype=np.float64)
    return 1e-5 * (1.5 + np.cos(2 * np.pi * t / 3.15581e7 + 0.3 * t.size))


@dataclass
class NG20Model:
    psrs: list
    data: object
    rn: object
    helpers: tuple
    wn_vec: jnp.ndarray
    residuals: np.ndarray
    blocks: tuple            # ordered stochastic blocks after unc, e.g. ('dm', 'chrom', ...)
    has_cor: bool
    n_tm: int
    chrom_index: float
    cquad_project: bool
    sw_kernel: str
    n_irn: int
    n_gwb: int
    n_dm: int
    n_chrom: int

    @property
    def npsr(self):
        return len(self.psrs)

    def theta(self, seed=0):
        """One draw of every sampled hyperparameter, as a dict of lists."""
        rng = np.random.default_rng(seed)
        n = self.npsr
        return dict(
            irn=[(rng.uniform(-15.5, -14.0), rng.uniform(2, 5)) for _ in range(n)],
            dm=[(rng.uniform(-15.5, -14.0), rng.uniform(1, 4)) for _ in range(n)],
            chrom=[(rng.uniform(-16.0, -14.0), rng.uniform(1, 4)) for _ in range(n)],
            sw=[rng.uniform(-0.5, 0.8) for _ in range(n)],
            gwb=(rng.uniform(-15.5, -14.5), 13 / 3))

    def red_params(self, theta):
        """``theta`` -> the flat vector, by name, so this cannot drift from
        ``get_param_names``."""
        vals = {}
        for i, psr in enumerate(self.data.psr_names):
            psr = str(psr)
            vals[f"{psr}_irn_log10_A"], vals[f"{psr}_irn_gamma"] = theta["irn"][i]
            vals[f"{psr}_dm_log10_A"], vals[f"{psr}_dm_gamma"] = theta["dm"][i]
            vals[f"{psr}_chrom_log10_A"], vals[f"{psr}_chrom_gamma"] = theta["chrom"][i]
            vals[f"{psr}_sw_log10_amp"] = theta["sw"][i]
        vals["gwb_log10_A"], vals["gwb_gamma"] = theta["gwb"]
        return jnp.array([vals[k] for k in self.rn.model.get_param_names()])


@lru_cache(maxsize=16)
def build_ng20(model_string="unc+cor->unc;dm,chrom,cquad,sw", npsr=2,
               n_irn=6, n_gwb=4, n_dm=5, n_chrom=NG20_CHROM_BINS,
               chrom_index=4.0, cquad_project=True, sw_kernel="RIDGE"):
    from ATLAS.signals.signals_utils import parse_basis_string
    psrs = load_psrs("synth", npsr)
    linear_timing = model_string.startswith("ltm|")
    data = PTA_Data(
        psrs, num_gwb_bins=n_gwb, num_irn_bins=n_irn, num_dm_bins=n_dm,
        num_det_bins=None, adaptus_basis=None, adaptus_size=None,
        fixed_white_noise_params=None, linear_timing=linear_timing,
        marg_timing=False, diag_white_cov=False, fixed_res=False,
        timfiles=None, parfiles=None, noise_dict=None, dm_ref_freq=DM_REF_FREQ)
    WhiteCov(data=data, stabilize_TNT=False, include_ecorr=True)
    lo, hi = jnp.array([-20.0, 0.0]), jnp.array([-11.0, 7.0])
    sw_kwargs = dict(dt_days=NG20_SW_DT_DAYS)
    if sw_kernel != "RIDGE":
        sw_kwargs.update(NG20_MATERN)
    rn = ModelBuilder(data=data).make_red_noise(
        model_string,
        irn_psd_function=powerlaw, gwb_psd_function=powerlaw, orf_function=ZERO_ORF,
        dm_psd_function=powerlaw, chrom_psd_function=powerlaw,
        irn_lower_bound_psd=lo, irn_upper_bound_psd=hi,
        gwb_lower_bound_psd=lo, gwb_upper_bound_psd=hi,
        dm_lower_bound_psd=lo, dm_upper_bound_psd=hi,
        chrom_lower_bound_psd=lo, chrom_upper_bound_psd=hi,
        num_chrom_bins=n_chrom, chrom_index=chrom_index,
        cquad_project_timing=cquad_project,
        sw_geometry=[sw_geometry_synth(p) for p in psrs], sw_kwargs=sw_kwargs)
    wn_vec = white_noise_vector(psrs, True)
    helpers = rn.get_helpers(reff=jnp.concat(data.raw_residuals), white_noise_params=wn_vec)
    cfg = parse_basis_string(model_string)
    return NG20Model(psrs, data, rn, helpers, wn_vec,
                     np.concatenate([np.asarray(p.residuals) for p in psrs]),
                     tuple(n for n in cfg["separate_names"] if n != "unc"),
                     "cor" in cfg["shared_names"],
                     rn.linear_timing_model_size, chrom_index, cquad_project,
                     sw_kernel, n_irn, n_gwb, n_dm, n_chrom)


def ng20_dense(model, theta, chi=None):
    """Per-pulsar reference ``T_p`` and the global ``phi`` for ``theta``.

    ``chi`` (per pulsar) is the chromatic index of chrom and cquad; defaults
    to the index the model was built with.
    """
    chi = [model.chrom_index] * model.npsr if chi is None else list(chi)
    tspan = float(model.data.pta_tspan)
    df = 1.0 / tspan
    f = lambda n: np.arange(1, n + 1) / tspan
    T_blocks, phis = [], []
    for i, p in enumerate(model.psrs):
        t, nu = np.asarray(p.toas, float), np.asarray(p.freqs, float)
        cols, ph = [], []
        if model.n_tm:
            Ms = svd_basis(p.Mmat)
            Mp = np.zeros((t.size, model.n_tm)); Mp[:, :Ms.shape[1]] = Ms
            cols.append(Mp)
            ph.append(np.where(np.arange(model.n_tm) < Ms.shape[1], 1e40, 1.0))
        irn = ref.powerlaw_psd(f(model.n_irn), df, *theta["irn"][i])
        if model.has_cor:
            irn[:model.n_gwb] += ref.powerlaw_psd(f(model.n_gwb), df, *theta["gwb"])
        cols.append(ref.fourier_basis(t, f(model.n_irn))); ph.append(np.repeat(irn, 2))
        for blk in model.blocks:
            if blk == "dm":
                cols.append(ref.fourier_basis(t, f(model.n_dm))
                            * ref.chromatic_weight(nu, 2.0, DM_REF_FREQ)[:, None])
                ph.append(np.repeat(ref.powerlaw_psd(f(model.n_dm), df, *theta["dm"][i]), 2))
            elif blk == "chrom":
                cols.append(ref.fourier_basis(t, f(model.n_chrom))
                            * ref.chromatic_weight(nu, chi[i], DM_REF_FREQ)[:, None])
                ph.append(np.repeat(ref.powerlaw_psd(f(model.n_chrom), df, *theta["chrom"][i]), 2))
            elif blk == "cquad":
                U = svd_basis(p.Mmat) if model.cquad_project else None
                cols.append(ref.chrom_quad_basis(t, nu, chi[i], U, DM_REF_FREQ))
                ph.append(np.ones(3))                        # cquad_log10_amp = 0
            elif blk == "sw":
                L, nodes = ref.interp_basis(t, dt_days=NG20_SW_DT_DAYS)
                cols.append(L * (sw_geometry_synth(p) * NG20_DMCONST / nu ** 2)[:, None])
                if model.sw_kernel == "RIDGE":
                    ph.append(ref.sw_kernel(nodes, "RIDGE", theta["sw"][i]))
                else:
                    kw = {k: v for k, v in NG20_MATERN.items() if k != "kernel"}
                    ph.append(ref.sw_kernel(nodes, "MATERN", theta["sw"][i], **kw))
            else:
                raise ValueError(f"no reference for block {blk!r}")
        T_blocks.append(np.hstack(cols))
        phis.append(block_diag(*[np.diag(x) if x.ndim == 1 else x for x in ph]))
    return T_blocks, block_diag(*phis)