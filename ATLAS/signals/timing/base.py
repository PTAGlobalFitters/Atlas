from __future__ import annotations

import math
import re
import warnings
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.linalg as jsl

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from ATLAS.signals.timing.routing import route
from ATLAS.signals.timing import preconditioning as P
from ATLAS.samplers.canetoadracing import MultiHMCGibbs
from ATLAS.signals.signals_utils import _timing_model_svd
from ATLAS.utils import jit_method

from jug.engine.session import TimingSession
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY, C_KM_S, T_SUN_SEC
from jug.delays.combined import combined_delays
from jug.delays.barycentric_jax import (
    pulsar_direction_jax, roemer_delay_jax, shapiro_delay_jax,
)
from jug.model.dmx import parse_dmx_ranges
from jug.fitting.derivatives_jump import (
    parse_jump_from_par_line,
    create_jump_mask_from_flags,
    create_jump_mask_from_mjd_range,
)

jax.config.update('jax_enable_x64', True)

# NOTE: tqdm_joblib / joblib are only needed for the optional parallel par/tim
# loader in build_multi_psr_timing_model; they are imported lazily there so this
# module (and the single-pulsar path) imports even when those deps are absent.


_MAS_YR_TO_RAD_DAY = (np.pi / 180.0 / 3.6e6) / 365.25

_BINARY_KIND = {
    '': 'none', 'NONE': 'none',
    'ELL1': 'ell1', 'ELL1H': 'ell1',
    'DD': 'dd', 'DDH': 'dd', 'DDGR': 'dd',
    'DDK': 'ddk',
    'BT': 'bt', 'BTX': 'bt',
}

# jug.delays.combined.combined_delays binary_model_id switch (jax.lax.switch order):
#   0=None, 1=ELL1, 2=DD, 3=T2, 4=BT, 5=DDK.  We route ELL1/DD/BT/DDK here; the
#   hand-rolled binary/astrometry kernels are retired in favour of
#   combined_delays (binary/DM/FD/SW) and barycentric_jax (astrometry).  DDK
#   selects the Kopeikin (KIN/KOM) branch; DDGR still reduces to DD (MTOT-derived
#   PK params are not yet solved for — MTOT is rejected in the setup guard).
_BINARY_MODEL_ID = {'none': 0, 'ell1': 1, 'dd': 2, 'bt': 4, 'ddk': 5}

# Parameters wired into delay kernels below.
# Anything else is accepted (warned) but contributes zero delay.
_KERNEL_PARAMS = frozenset({
    'F0', 'F1', 'DM',
    'RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX',
    # Ecliptic coordinate aliases — remapped to equatorial internally by JUG.
    # They are accepted here and transparently forwarded to the same astrometry
    # kernel via the _make_ecl_to_eq_fn() rotation.  Sampling ELONG samples the
    # equatorial RAJ/DECJ the kernel actually consumes.
    'ELONG', 'ELAT', 'PMELONG', 'PMELAT',
    'PB', 'A1',
    'TASC', 'EPS1', 'EPS2',
    'T0', 'ECC', 'OM',
    'M2', 'SINI',
    # Secondary binary params, all wired through combined_delays as traced args.
    # Which ones are physically active depends on the binary model (e.g. EPS*DOT
    # for ELL1, OMDOT/EDOT for DD, KIN/KOM for DDK, H3/H4/STIG orthometric
    # Shapiro for ELL1H); inactive slots evaluate to zero delay, which the
    # zero-Jacobian guard reports if such a param is sampled under a model that
    # ignores it.  Fn / FDn / DMn / FBn / DMX / JUMP are handled by pattern below.
    'XDOT', 'OMDOT', 'EDOT', 'GAMMA', 'PBDOT',
    'EPS1DOT', 'EPS2DOT',
    'H3', 'H4', 'STIG', 'KIN', 'KOM', 'DR', 'DTH',
    # Solar-wind amplitude — returned by combined_delays as sw_sec(ne_sw), the
    # same delay family as dm_sec(DM); traced like any other kernel param.
    'NE_SW',
})

# Binary params JUG exposes as fittable but ATLAS's combined_delays wiring does
# NOT model — sampling any of these would silently produce a zero/NaN Jacobian
# column, so they are rejected up front with a clear error (see setup guard).
# These genuinely have no combined_delays slot (unlike NE_SW, which does).
#   A0/B0        : first-order aberration (no combined_delays slot)
#   MTOT         : DDGR total mass (DDGR reduces to DD here; PK params not derived)
#   SHAPMAX      : DDS Shapiro reparam (no slot)
#   XOMDOT/XPBDOT: DDGR excess secular terms (no slot)
_UNSUPPORTED_PARAMS = frozenset({
    'A0', 'B0', 'MTOT', 'SHAPMAX', 'XOMDOT', 'XPBDOT',
})

# Patterns for parameter families handled generically.
_RE_FN   = re.compile(r'^F(\d+)$')          # F2, F3, …
_RE_DM   = re.compile(r'^DM(\d+)$')         # DM1, DM2, … (DM Taylor; NOT DMX_/DM)
_RE_DMX  = re.compile(r'^DMX_\d+$', re.IGNORECASE)
_RE_JUMP = re.compile(r'^JUMP(\d+)$', re.IGNORECASE)
_RE_FD   = re.compile(r'^FD(\d+)$', re.IGNORECASE)   # FD1, FD2, FD3, …
_RE_FB   = re.compile(r'^FB(\d+)$', re.IGNORECASE)   # FB0, FB1, … (orbital freq)

# Ecliptic public names → JUG internal equatorial working keys
_ECL_TO_EQ: dict[str, str] = {
    'ELONG':   '_ecliptic_lon_deg',   # degrees; converted to _raj_rad by JUG
    'ELAT':    '_ecliptic_lat_deg',   # degrees; converted to _decj_rad by JUG
    'PMELONG': '_ecliptic_pm_lon',    # mas/yr along ecliptic lon
    'PMELAT':  '_ecliptic_pm_lat',    # mas/yr along ecliptic lat
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_param(params: dict, name: str) -> float:
    """Read any par-file parameter as float, handling coordinate aliases.

    RAJ/DECJ  → JUG internal radian keys (_raj_rad/_decj_rad).
    ELONG/ELAT/PMELONG/PMELAT → JUG internal ecliptic keys (_ecliptic_*).
      The ecliptic values are in degrees (lon/lat) or mas/yr (proper motion).
      The astrometry kernel receives the equatorial equivalents computed by the
      _make_ecl_to_eq_fn() rotation, not these raw values, so theta_0_full
      stores the ecliptic values purely for computing deltas.
    """
    _aliases = {
        'RAJ':  '_raj_rad',
        'DECJ': '_decj_rad',
        **_ECL_TO_EQ,   # ELONG → _ecliptic_lon_deg, etc.
    }
    key = _aliases.get(name, name)
    val = params.get(key, 0.0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _classify_param(name: str) -> str:
    """Return the class of a parameter.

    One of: 'kernel', 'fn', 'dm', 'fb', 'dmx', 'jump', 'fd', 'unknown'.
    """
    if name in _KERNEL_PARAMS:
        return 'kernel'
    if _RE_FN.match(name):
        return 'fn'
    if _RE_DMX.match(name):      # DMX_0001 — check before _RE_DM (DM\d+)
        return 'dmx'
    if _RE_DM.match(name):       # DM1, DM2, … Taylor coefficients
        return 'dm'
    if _RE_FB.match(name):       # FB0, FB1, … orbital-frequency Taylor
        return 'fb'
    if _RE_JUMP.match(name):
        return 'jump'
    if _RE_FD.match(name):
        return 'fd'
    return 'unknown'


def _make_ecl_to_eq_fn(params_base: dict):
    """Return a pure-JAX function that rotates ecliptic → equatorial.

    The obliquity rotation matrix is computed once from the par file (a Python
    float) and baked in as a compile-time constant.  The returned function
    accepts JAX tracers for lon_deg, lat_deg, pm_lon, pm_lat and is safe to
    call inside @jax.jit.

    Proper-motion convention: pm_lon is the *great-circle* rate μ_λ·cos β
    (as stored by TEMPO/PINT for PMELONG), matching the returned PMRA = μ_α·cos δ.

    Parameters
    ----------
    params_base : dict
        The JUG params dict; used only to read the ECL frame string at setup time.

    Returns
    -------
    ecl_to_eq : callable(lon_deg, lat_deg, pm_lon, pm_lat)
                → (ra_rad, dec_rad, pmra_masyr, pmdec_masyr)
        All arguments and return values are JAX scalars (traceable).
    """
    from jug.io.par_reader import OBLIQUITY_ARCSEC
    ecl_frame  = str(params_base.get('_ecliptic_frame',
                                     params_base.get('ECL', 'IERS2010'))).upper()
    obl_arcsec = OBLIQUITY_ARCSEC.get(ecl_frame, OBLIQUITY_ARCSEC['IERS2010'])
    obl_rad    = obl_arcsec * np.pi / (180.0 * 3600.0)
    # Frozen Python floats — become compile-time constants inside jit.
    _cos_obl = float(np.cos(obl_rad))
    _sin_obl = float(np.sin(obl_rad))
    _deg2rad = float(np.pi / 180.0)
    _twopi   = float(2.0 * np.pi)

    def ecl_to_eq(lon_deg, lat_deg, pm_lon, pm_lat):
        """Pure-JAX ecliptic → equatorial rotation.  Safe inside @jax.jit."""
        lon_rad = lon_deg * _deg2rad
        lat_rad = lat_deg * _deg2rad

        cos_lon = jnp.cos(lon_rad)
        sin_lon = jnp.sin(lon_rad)
        cos_lat = jnp.cos(lat_rad)
        sin_lat = jnp.sin(lat_rad)

        # Ecliptic → ICRS Cartesian rotation (obliquity about x-axis)
        x = cos_lon * cos_lat
        y = sin_lon * cos_lat * _cos_obl - sin_lat * _sin_obl
        z = sin_lon * cos_lat * _sin_obl + sin_lat * _cos_obl

        ra_rad  = jnp.arctan2(y, x) % _twopi
        dec_rad = jnp.arctan2(z, jnp.sqrt(x ** 2 + y ** 2))

        # Proper-motion rotation (same obliquity matrix, Jacobian)
        cos_ra  = jnp.cos(ra_rad)
        sin_ra  = jnp.sin(ra_rad)
        cos_dec = jnp.cos(dec_rad)
        sin_dec = jnp.sin(dec_rad)

        dx    = -sin_lon * pm_lon - cos_lon * sin_lat * pm_lat
        dy    =  cos_lon * pm_lon - sin_lon * sin_lat * pm_lat
        dz    =  cos_lat * pm_lat
        dx_eq =  dx
        dy_eq =  dy * _cos_obl - dz * _sin_obl
        dz_eq =  dy * _sin_obl + dz * _cos_obl

        pmra_masyr  = -sin_ra * dx_eq + cos_ra * dy_eq
        pmdec_masyr = (-cos_ra * sin_dec * dx_eq
                       - sin_ra * sin_dec * dy_eq
                       + cos_dec * dz_eq)

        return ra_rad, dec_rad, pmra_masyr, pmdec_masyr

    return ecl_to_eq


def _build_dmx_masks(params: dict, mjd_utc_np: np.ndarray) -> dict[str, jnp.ndarray]:
    """Precompute one boolean JAX mask per DMX window from the par file.

    The key matches the canonical label returned by parse_dmx_ranges
    (e.g. 'DMX_0001').  Mask is True for every TOA in the window.

    DMX window edges (DMXR1_/DMXR2_) are par-file MJDs, so they are compared
    against the *observatory* MJD (mjd_utc), not the barycentred TDB MJD.
    """
    dmx_ranges = parse_dmx_ranges(params)
    masks = {}
    for rng in dmx_ranges:
        mask_np = (mjd_utc_np >= rng.r1_mjd) & (mjd_utc_np <= rng.r2_mjd)
        masks[rng.label] = jnp.array(mask_np, dtype=jnp.bool_)
    return masks


def _build_jump_masks(params: dict, toas_data: list,
                      mjd_for_ranges_np: np.ndarray) -> dict[str, jnp.ndarray]:
    """Precompute one boolean JAX mask per JUMP line from the par file.

    Keys are 'JUMP1', 'JUMP2', … matching the numbered convention used
    throughout the JUG engine (session.py writes JUMP{n} into params).

    ``mjd_for_ranges_np`` is only consulted for MJD-range jumps.  See the call
    site in setup_timing_model for which time scale is passed.
    """
    jump_lines = params.get('_jump_lines', [])
    masks = {}
    toa_flags = [getattr(t, 'flags', {}) for t in toas_data]

    for idx, jline in enumerate(jump_lines, start=1):
        key = f'JUMP{idx}'
        jinfo = parse_jump_from_par_line(jline)
        if jinfo['type'] == 'flag':
            mask_np = create_jump_mask_from_flags(
                toa_flags, jinfo['flag_name'], jinfo['flag_value'])
        elif jinfo['type'] == 'mjd':
            mask_np = np.array(
                create_jump_mask_from_mjd_range(
                    jnp.array(mjd_for_ranges_np),
                    jinfo['mjd_start'], jinfo['mjd_end']),
                dtype=bool)
        else:
            mask_np = np.zeros(len(mjd_for_ranges_np), dtype=bool)
        masks[key] = jnp.array(mask_np, dtype=jnp.bool_)

    return masks


# ---------------------------------------------------------------------------
# Physical reparametrisations (NaN-gradient guard for the JUG kernels)
# ---------------------------------------------------------------------------
# combined_delays evaluates  log(1 - SINI·sinΦ)  and  sqrt(1 - ECC²).  Because
# jnp.where evaluates BOTH branches, a SINI>1 or ECC≥1 proposal produces a NaN
# that poisons the gradient even when the runtime guard would have selected 0 —
# fatal for NUTS.  We therefore map the SAMPLED (unconstrained, real) coordinate
# u = z·σ to a PHYSICAL value that is *always* in range, for every u ∈ ℝ, with a
# finite derivative everywhere.  This is stronger than a bounded prior: even an
# out-of-support proposal can never reach the kernel.
#
#   SINI : sample cos i.  cosi = tanh(u + atanh(cosi₀)) ∈ (-1,1)
#          → SINI = sqrt(1 - cosi²) ∈ (0,1].  Uses cos i (not sin(angle)) so the
#          sampler sits away from the SINI=1 edge where log(1 - SINI·sinΦ) → log(0).
#   ECC  : ecc = ½(1 + tanh(u + atanh(2ecc₀-1))) ∈ (0,1).  Never reaches 1.
#   M2,PX: positive via log-space, M2 = M2₀·exp(u) (or softplus if par value ≤0).
#   else : plain affine  x₀ + u   (F0, F1, DM, OM, EPS1/2, … unchanged).
#
# u = z·σ is centred so that u=0 reproduces the par-file value x₀.
#
# NOTE: this transform carries NO prior Jacobian and is used ONLY by the legacy
# z_concat path (z_to_theta / residuals / residuals_from_z_dict).  The numpyro
# blocks below instead use proper bounded priors on the physical coordinate.
_REPARAM_PARAMS = frozenset({'SINI', 'ECC', 'M2', 'PX'})

_EPS_BND = 1e-9  # keep atanh arguments strictly inside (-1, 1)


def reparametrise(name: str, x0: float, u):
    """Map a sampled unconstrained coordinate ``u`` → physical parameter value.

    ``u`` is the scaled, mean-zero sampler coordinate (u = z·σ); ``x0`` is the
    par-file value reproduced at u=0.  Returns a JAX-traceable value guaranteed
    to lie in the parameter's physical domain for every real ``u``, with finite
    gradient everywhere — so the downstream JUG delay kernels never see a
    log/sqrt of a negative.
    """
    if name == 'SINI':
        # Monotonic inclination map: sample cos i = tanh(u+c) ∈ (-1,1), one-to-one
        # in the sampler coordinate, then SINI = sqrt(1 - cosi²).  Do NOT use the
        # algebraic shortcut sech(u+c): cosh is even, so ±(u+c) alias to the same
        # SINI — a two-to-one map that plants a mirror mode in the posterior.
        # cosi is clipped to ±(1-ε) so 1-cosi² ≥ ε(2-ε) > 0 and the sqrt gradient
        # never reaches the singular endpoint (tanh only hits ±1 at infinite u;
        # clip guards float64 saturation anyway).
        # CAVEAT: cosi₀ is taken positive, i.e. the i<90° branch is assumed.
        cosi0 = jnp.sqrt(jnp.clip(1.0 - x0 ** 2, 0.0, 1.0))
        c     = jnp.arctanh(jnp.clip(cosi0, -1.0 + _EPS_BND, 1.0 - _EPS_BND))
        cosi  = jnp.clip(jnp.tanh(u + c), -1.0 + _EPS_BND, 1.0 - _EPS_BND)
        return jnp.sqrt(1.0 - cosi ** 2)                  # ∈ (0, 1)
    if name == 'ECC':
        c = jnp.arctanh(jnp.clip(2.0 * x0 - 1.0, -1.0 + _EPS_BND, 1.0 - _EPS_BND))
        return 0.5 * (1.0 + jnp.tanh(u + c))              # ∈ (0, 1)
    if name in ('M2', 'PX'):
        # log-space positivity.  If the par value is ≤0 (param absent), fall back
        # to softplus so u=0 still gives a small positive number with finite grad.
        return jnp.where(
            x0 > 0.0,
            x0 * jnp.exp(u),
            jnp.logaddexp(0.0, u),                        # softplus(u) > 0
        )
    return x0 + u                                         # plain affine (unchanged)


# ---------------------------------------------------------------------------
# Canonical bounded-prior sampling — THE single prior mechanism (production+tests)
# ---------------------------------------------------------------------------
# Per-parameter PROPER bounded priors on the PHYSICAL coordinate.  Established as
# the robust path in the weak-data prereq: improper-z + log|dθ/dz| Jacobian blew
# up (NUTS escaping to |z|→∞ at the SINI→0 boundary); proper bounded priors give
# nominal coverage.  No Jacobian factor is used anywhere — the prior lives
# entirely in the bounded distribution on the physical coordinate.
#
#   SINI : cos i ~ Uniform(-1, 1)  →  SINI = sqrt(1 - cos²i)   (isotropic incl.)
#   ECC  : Uniform(0, 1-1e-6)
#   M2   : Uniform over _M2_BOUNDS   (solar masses)
#   PX   : Uniform(0, PX_MAX) mas    (flat, inert — no scale preference)
#   else : affine  theta0 + z·σ_JUG,  z ~ Normal(0, AFFINE_Z_SD)  (well-constrained)
#
# M2 and PX use FLAT (uniform) physical-range priors, deliberately inert for
# recovery: log-uniform pulls toward low mass / small parallax (scale-informative)
# and would bias an informative-data param.  These are VALIDATION priors — the
# production science-run prior choice is a separate, later decision.
#
# reparametrise() above is NOT used here — that is the in-domain transform for the
# unconstrained z_concat path only, and carries no prior.
_M2_BOUNDS = (0.0, 3.0)        # M2 ∈ [0, 3] Msun  (J1909 truth ≈0.21 → 7% of range)
_PX_BOUNDS = (0.0, 10.0)       # PX ∈ [0, 10] mas  (J1909 truth ≈1.0 → 10% of range)
_ECC_HI    = 1.0 - 1e-6
_BOUND_SPECS = {"M2": ("uniform", *_M2_BOUNDS), "ECC": ("uniform", 0.0, _ECC_HI)}

# Bound families implemented by prepare_dense_metric / build_gibbs_model.
_GIBBS_BOUND_SUPPORTED = ("M2", "SINI", "ECC")
DEFAULT_BOUND = _GIBBS_BOUND_SUPPORTED


# Physical domain of each bounded param, used when perturb_bounded=True.
# Matches the clips in sample_timing_theta_batched: M2 and PX are bounded below
# only, SINI and ECC on both sides.
_PERTURB_DOMAIN = {
    'SINI': (1e-6, 1.0),
    'ECC':  (0.0, _ECC_HI),
    'M2':   (1e-12, None),
    'PX':   (1e-12, None),
}


def sample_timing_theta(sample_list, theta_0, sigma, prefix='', affine_z_sd=100.0,
                        affine_z_sd_bounded = None,
                        perturb_bounded=False):
    """numpyro: draw each sampled timing param from its proper bounded prior.

    THE single prior mechanism — called by both the production MultiPsrTimingModel
    block and the validation harnesses, so every timing param is sampled the same
    way.  Returns ``{param_name: value}`` (physical units), and registers each as
    a numpyro.deterministic for post-processing.

    Parameters
    ----------
    sample_list : list[str]     params to sample (physical names)
    theta_0     : dict          par-file values (affine centre / init)
    sigma       : dict          JUG formal σ per param (affine scale = aux['sigJUG'])
    prefix      : str           name prefix (e.g. per-pulsar 'p0_') to avoid clashes
    affine_z_sd : float         SD of the unit-scale Normal prior on z
    perturb_bounded : bool, default False
        False: SINI/ECC/M2/PX are drawn from their full bounded priors
        (cos i ~ U(-1,1), ECC ~ U(0,1), M2 ~ U(*_M2_BOUNDS), PX ~ U(*_PX_BOUNDS)),
        ignoring ``affine_z_sd``.  This is the production prior.
        True: they are perturbed around ``theta_0`` by ``affine_z_sd * sigma[k]``
        like the affine branch, matching ``sample_timing_theta_batched``.  Use it
        for a controlled N-sigma displacement rather than a prior-predictive draw.

        The perturbation is a TruncatedNormal in the z coordinate rather than a
        Normal clipped to the domain.  Clipping (what the batched version does,
        where it is only ever used to generate draws) would give NUTS a flat
        region with zero gradient outside the bounds and pile posterior mass on
        the boundary; the truncated distribution has the same support with a
        correct normalisation and a live gradient everywhere inside it.
    """
    _PERTURB_DOMAIN = {
    'SINI': (1e-6, 1.0),
    'ECC':  (0.0, _ECC_HI),
    'M2':   (1e-12, None),
    'PX':   (1e-12, None),
                        }
    theta = {}
    for k in sample_list:
        if perturb_bounded and k in _PERTURB_DOMAIN:
            # Affine perturbation, but confined to the physical domain.  The
            # site stays `z_{k}` so the name matches both the affine branch and
            # sample_timing_theta_batched.
            s = max(float(sigma[k]), 1e-30)
            lo, hi = _PERTURB_DOMAIN[k]
            z_lo = (lo - theta_0[k]) / s if lo is not None else None
            z_hi = (hi - theta_0[k]) / s if hi is not None else None
            z = numpyro.sample(
                f'{prefix}z_{k}',
                dist.TruncatedNormal(0.0, affine_z_sd_bounded, low=z_lo, high=z_hi))
            theta[k] = theta_0[k] + z * s
            numpyro.deterministic(f'{prefix}{k}', theta[k])
        elif k == 'SINI':
            cosi = numpyro.sample(f'{prefix}cosi_{k}', dist.Uniform(-1.0, 1.0))
            theta[k] = jnp.sqrt(jnp.clip(1.0 - cosi ** 2, 1e-12, 1.0))
            numpyro.deterministic(f'{prefix}{k}', theta[k])   # transformed → emit
        elif k == 'ECC':
            theta[k] = numpyro.sample(f'{prefix}{k}', dist.Uniform(0.0, _ECC_HI))
        elif k == 'M2':
            theta[k] = numpyro.sample(f'{prefix}{k}', dist.Uniform(*_M2_BOUNDS))
        elif k == 'PX':
            theta[k] = numpyro.sample(f'{prefix}{k}', dist.Uniform(*_PX_BOUNDS))
        else:                                          # affine, wide Normal in z
            s = max(float(sigma[k]), 1e-30)
            z = numpyro.sample(f'{prefix}z_{k}', dist.Normal(0.0, affine_z_sd))
            theta[k] = theta_0[k] + z * s
            numpyro.deterministic(f'{prefix}{k}', theta[k])   # transformed → emit
    return theta


def sample_timing_theta_batched(key, sample_list, theta_0, sigma, num_samples,
                                prefix='', affine_z_sd=100.0, affine_z_sd_bounded = 1,
                                perturb_bounded=False,
                                override = None):
    """...
    perturb_bounded : bool
        False (default): SINI/ECC/M2/PX are drawn from their full bounded
        priors, ignoring affine_z_sd.  Correct for prior-predictive training
        sets (the PCA basis in GaussianTiming), wrong for a 1-sigma injection.
        True: they are perturbed around theta_0 by affine_z_sd * sigJUG[k],
        matching the affine branch.
    """
    keys = jrandom.split(key, len(sample_list))
    theta = {}
    for k, subkey in zip(sample_list, keys):
        if k in ('SINI', 'ECC', 'M2', 'PX') and perturb_bounded:
            s = max(float(sigma[k]), 1e-30)
            z = jrandom.normal(subkey, (num_samples,)) * affine_z_sd_bounded
            val = theta_0[k] + z * s
            # clip into the physical domain
            if k == 'SINI':  val = jnp.clip(val, 1e-6, 1.0)
            elif k == 'ECC': val = jnp.clip(val, 0.0, _ECC_HI)
            else:            val = jnp.clip(val, 1e-12, None)
            theta[f'{prefix}z_{k}'] = z
            theta[f'{prefix}{k}'] = val
        elif k == 'SINI':
            cosi = jrandom.uniform(subkey, (num_samples,), minval=-1.0, maxval=1.0)
            theta[f'{prefix}cosi_{k}'] = cosi
            theta[f'{prefix}{k}'] = jnp.sqrt(jnp.clip(1.0 - cosi ** 2, 1e-12, 1.0))
        elif k == 'ECC':
            theta[f'{prefix}{k}'] = jrandom.uniform(
                subkey, (num_samples,), minval=0.0, maxval=_ECC_HI)
        elif k == 'M2':
            lo, hi = _M2_BOUNDS
            theta[f'{prefix}{k}'] = jrandom.uniform(
                subkey, (num_samples,), minval=lo, maxval=hi)
        elif k == 'PX':
            lo, hi = _PX_BOUNDS
            theta[f'{prefix}{k}'] = jrandom.uniform(
                subkey, (num_samples,), minval=lo, maxval=hi)
        else:
            s = max(float(sigma[k]), 1e-30)
            z = jrandom.normal(subkey, (num_samples,)) * affine_z_sd
            theta[f'{prefix}z_{k}'] = z
            theta[f'{prefix}{k}'] = theta_0[k] + z * s

    if override is not None:
        for k, v in override.items():
            theta[k] = jnp.broadcast_to(v, (num_samples, ))
            
    return theta


def timing_init_values(sample_list, theta_0, prefix=''):
    """Matching init dict (init_to_value) for ``sample_timing_theta`` priors."""
    iv = {}
    for k in sample_list:
        if k == 'SINI':
            iv[f'{prefix}cosi_{k}'] = float(np.sqrt(max(1.0 - theta_0[k] ** 2, 0.0)))
        elif k == 'ECC':
            iv[f'{prefix}{k}'] = float(min(max(theta_0[k], 0.0), _ECC_HI))
        elif k == 'M2':
            iv[f'{prefix}{k}'] = float(min(max(theta_0[k], _M2_BOUNDS[0] + 1e-6),
                                           _M2_BOUNDS[1] - 1e-6))
        elif k == 'PX':
            iv[f'{prefix}{k}'] = float(min(max(theta_0[k], _PX_BOUNDS[0] + 1e-6),
                                           _PX_BOUNDS[1] - 1e-6))
        else:
            iv[f'{prefix}z_{k}'] = 0.0
    return iv


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_timing_model(par_path: str, tim_path: str,
                       sample_list: list[str] | None = None,
                       theta_0_overrides=None,
                       marginalise_list: list[str] | None = None,
                       *,
                       marginalise_all: bool = True,
                       modes: dict[str, str] | None = None):
    """Build a JIT'd JAX delta-m function with an explicit per-parameter split.

    Each fittable JUG parameter is placed in exactly one of three modes:
      • 'sample'      — varied nonlinearly through ``delta_m_us`` (sampled z).
      • 'marginalise' — carried as a column of the analytic linear basis ``M``.
      • 'fix'         — held at its par-file value (no column, not sampled).

    Parameters
    ----------
    par_path, tim_path : str
    sample_list : list[str], optional
        Params to sample nonlinearly.
    theta_0_overrides : dict, optional
        Manually override the par-file value of any parameter used as the
        expansion point θ₀.  e.g. {'SINI': 0.99, 'M2': 0.25}.
        Applied after theta_0_full is built from the par file, so it
        overrides JUG's value without touching the session or par file.
        Only affects θ₀ (the expansion point and prior centre) — JUG's
        own fit, design matrix, and sigJUG are computed at the par values.
    marginalise_list : list[str], optional
        Params to marginalise linearly.  If given, ONLY these (plus OFFSET) are
        marginalised and every other non-sampled label is fixed.
    marginalise_all : bool, default True
        When ``marginalise_list`` is None: marginalise every fittable JUG label
        not in ``sample_list`` (the historical behaviour).  When False, nothing
        is marginalised except OFFSET (all non-sampled params fixed).
    modes : dict[str, str], optional
        Direct per-param mapping {param: 'sample'|'marginalise'|'fix'}.  When
        given it is authoritative and ``sample_list`` / ``marginalise_list`` /
        ``marginalise_all`` are ignored.  Params absent from the dict are fixed.

    OFFSET is always marginalised (kept in M); it is never sampled or fixed.

    Returns
    -------
    delta_m_us : jit-compiled callable(theta_dict) → jnp.ndarray  [µs]
    aux : dict
        theta_0, errors_us, r_obs_us, tdb_mjd, mle, sigJUG, sample_list,
        marginalise_list, fixed_list, modes, binary_model, n_toa, dmx_masks,
        jump_masks, Mmat, Mmat_param_names, Cjug, fittable_order
    """
    # ---- 1. Run JUG --------------------------------------------------------
    session = TimingSession(par_path, tim_path, verbose=False)
    result  = session.compute_residuals(subtract_tzr=False)

    # Snapshot the par values BEFORE fitting.  `session.params` is a live dict;
    # if JUG's fitter writes updated values back into it, θ₀ would silently
    # become the *fitted* point while `result` (and hence r_obs_us) still refers
    # to the par values — an inconsistent linearisation centre.  A shallow copy
    # pins the scalar entries taken at the same point as `result`.
    params = dict(session.params)

    result_dummy = session.fit_parameters(max_iter=5)
    # Raw design matrix — PINT-matching sign/units; columns == design_matrix_labels
    # (OFFSET first, then every free fit param).  JUG returns design_matrix=None
    # when its fit accepts zero steps (an already-optimal or very stiff fit like
    # B1937+21 where every trial step marginally worsens the RMS); guard against
    # the silent np.asarray(None) that would otherwise corrupt the linear basis.
    if result_dummy.get('design_matrix') is None:
        raise RuntimeError(
            "JUG fit_parameters returned design_matrix=None (no fit step was "
            "accepted).  ATLAS needs the design matrix at the par values as its "
            "linear basis.  Update JUG to seed the design matrix on the first "
            "iteration (optimized_fitter saves it only on step-accept).")
    _design_matrix_raw = np.asarray(result_dummy['design_matrix'])    # (n_toa, n_col)
    _design_labels_raw = list(result_dummy['design_matrix_labels'])   # incl. 'OFFSET'
    _fittable = [lab for lab in _design_labels_raw if lab != 'OFFSET']
    _fittable_set = set(_fittable)
    _Cjug = np.asarray(result_dummy["covariance"], float)

    # ---- Resolve the three-way mode assignment ----------------------------
    if modes is not None:
        if sample_list or marginalise_list:
            raise ValueError("Pass either `modes` OR (`sample_list`/"
                             "`marginalise_list`), not both — they would conflict.")
        bad_mode = {k: v for k, v in modes.items()
                    if v not in ('sample', 'marginalise', 'fix')}
        if bad_mode:
            raise ValueError(f"modes: invalid mode(s) {bad_mode}; "
                             "use 'sample' | 'marginalise' | 'fix'.")
        sample_set = {k for k, v in modes.items() if v == 'sample'}
        marg_set   = {k for k, v in modes.items() if v == 'marginalise'}
    else:
        sample_set = set(sample_list or [])
        if marginalise_list is not None:
            marg_set = set(marginalise_list)
        elif marginalise_all:
            marg_set = _fittable_set - sample_set          # historical default
        else:
            marg_set = set()

    # ---- Validation: real labels, exactly one mode ------------------------
    # Every requested param must match a real JUG design label — this catches
    # the ECC-vs-E / XDOT-vs-A1DOT silent-mismatch case.
    requested = sample_set | marg_set
    unmatched = sorted(requested - _fittable_set)
    if unmatched:
        import difflib
        hints = {p: difflib.get_close_matches(p, _fittable, n=3) for p in unmatched}
        raise ValueError(
            f"Requested timing params not found among JUG design labels: {unmatched}. "
            f"Closest real labels: {hints}.  Fittable labels: {sorted(_fittable)}")
    overlap = sorted(sample_set & marg_set)
    if overlap:
        raise ValueError(f"Params {overlap} are in BOTH sample and marginalise "
                         "modes; each param must be in exactly one mode.")

    sample_list = [k for k in _fittable if k in sample_set]   # canonical order
    fixed_set   = _fittable_set - sample_set - marg_set
    mode_map = ({k: 'sample' for k in sample_set}
                | {k: 'marginalise' for k in marg_set}
                | {k: 'fix' for k in fixed_set})

    # ---- Guard: sampled params must be wired into m_total_sec -------------
    # A SAMPLED param is recovered by autodiff of ATLAS's m_total_sec, so it
    # must have a delay kernel here; a MARGINALISED param instead rides JUG's
    # own design-matrix column and needs no ATLAS wiring.  So this guard is
    # scoped to sample_list only.
    #
    # (1) Known-fittable-in-JUG but deliberately-not-modelled here → hard error,
    #     otherwise they would silently yield a zero/NaN Jacobian column and a
    #     NaN MLE (the failure mode this guard exists to prevent).  Marginalise
    #     or fix them instead.
    unsupported = [k for k in sample_list if k in _UNSUPPORTED_PARAMS]
    if unsupported:
        raise NotImplementedError(
            f"Sampled params {unsupported} are fittable in JUG but not modelled "
            "by ATLAS's combined_delays wiring, so they cannot be recovered by "
            "sampling (they would give zero/NaN Jacobian columns).  Move them to "
            "marginalise_list (they ride JUG's linear design matrix) or fix them "
            "at their par value.")
    # (2) Anything else with no recognised class is an unforeseen gap → error
    #     rather than warn, so a new JUG parameter family can never slip through
    #     as a silent zero column.
    unknown = [k for k in sample_list if _classify_param(k) == 'unknown']
    if unknown:
        raise NotImplementedError(
            f"Sampled params {unknown} have no delay kernel in m_total_sec and no "
            "recognised parameter family (kernel/Fn/DMn/FBn/FDn/DMX/JUMP).  This "
            "usually means JUG gained a new fittable parameter that ATLAS has not "
            "wired yet; add a kernel branch (and _KERNEL_PARAMS/pattern entry) or, "
            "if it cannot be modelled, add it to _UNSUPPORTED_PARAMS.  Marginalise "
            "or fix it in the meantime.")

    # ---- Build M from EXACTLY the marginalised set (strip BEFORE the SVD) --
    # A sampled param must never appear in M (double-counting); a fixed param is
    # dropped from M too (held at par).  OFFSET is always retained.  The strip
    # acts on the RAW design matrix — its columns ARE the parameters; after the
    # SVD, U's columns mix every parameter and can no longer be split per-param.
    _linear_keep_idx = [i for i, lab in enumerate(_design_labels_raw)
                        if lab == 'OFFSET' or lab in marg_set]
    Mmat_param_names = [_design_labels_raw[i] for i in _linear_keep_idx]
    Mmat = jnp.asarray(_design_matrix_raw[:, _linear_keep_idx])  # (n,0) if none
    if not _linear_keep_idx and marg_set:
        raise ValueError(f"marginalise_list {sorted(marg_set)} matched no design "
                         "columns — internal error.")
    # Empty M (fully-nonlinear fit, no OFFSET column) is allowed: the constant
    # phase offset is absorbed by delta_m_us's weighted-mean subtraction, so no
    # linear basis is required.  _timing_model_svd handles the (n_toa, 0) case.

    binary_model = params.get('BINARY', '').upper()
    binary_kind  = _BINARY_KIND.get(binary_model)
    if binary_kind is None:
        raise NotImplementedError(
            f"Binary model {binary_model!r} not supported.  "
            f"Supported: {sorted(set(_BINARY_KIND) - {''})} or no binary."
        )
    has_binary = binary_kind != 'none'

    # ---- 2. Cache fixed arrays --------------------------------------------
    n_toa       = result['n_toas']
    tdb_mjd_ld  = np.asarray(result['tdb_mjd'], dtype=np.longdouble)
    mjd_utc_np  = np.asarray(result.get('mjd_utc', result['tdb_mjd']), dtype=np.float64)
    PEPOCH      = np.longdouble(params['PEPOCH'])
    dt_sec      = jnp.asarray(np.asarray(
        (tdb_mjd_ld - PEPOCH) * np.longdouble(SECS_PER_DAY), dtype=np.float64))
    tdb_mjd     = jnp.asarray(np.asarray(tdb_mjd_ld, dtype=np.float64))
    freq_mhz    = jnp.asarray(np.asarray(result['freq_bary_mhz'], dtype=np.float64))
    errors_us   = jnp.asarray(np.asarray(result['errors_us'], dtype=np.float64))
    weights     = 1.0 / errors_us ** 2
    ssb_obs_km  = jnp.asarray(np.asarray(result['ssb_obs_pos_ls'], dtype=np.float64) * C_KM_S)
    obs_sun_km  = jnp.asarray(np.asarray(result['obs_sun_pos_ls'], dtype=np.float64) * C_KM_S)
    # Solar wind is now computed inside combined_delays from ne_sw (a traceable,
    # fittable param); the frozen result['sw_delay_sec'] precompute is no longer
    # folded in (it double-counted with the ne_sw path).
    tropo_delay = jnp.asarray(np.asarray(
        result.get('tropo_delay_sec', np.zeros(n_toa)), dtype=np.float64))
    r_obs_us    = jnp.asarray(np.asarray(result['residuals_us'], dtype=np.float64))

    POSEPOCH_0 = float(params.get('POSEPOCH', params['PEPOCH']))
    # Proper-motion baseline (t - POSEPOCH) reduced in longdouble on the host,
    # then float64 — fed to pulsar_direction_jax so the JAX trace never forms the
    # years-long subtraction itself (matches the NumPy barycentric precision path).
    dt_pos_days = jnp.asarray(np.asarray(
        tdb_mjd_ld - np.longdouble(POSEPOCH_0), dtype=np.float64))

    # Secondary binary params are traced via _g(theta, …) (see _KERNEL_PARAMS),
    # so their frozen baselines come from the theta_0_full fallback pool below.
    # Two need an explicit read here because the par label differs from / is not
    # the JUG design label: XDOT (par stores A1DOT) and OMDOT (override baseline).
    OMDOT_0 = float(params.get('OMDOT', 0.0))
    XDOT_0  = float(params.get('XDOT', params.get('A1DOT', 0.0)))

    # ---- combined_delays wiring (binary + DM + FD + solar wind) ------------
    # combined_delays carries the binary epoch (TASC for ELL1, T0 for DD/BT)
    # baked into the precomputed tt_binary_sec array, NOT as a traced argument.
    # We rebuild tt_binary_sec inside the JAX trace from the (possibly sampled)
    # epoch param so its physics is unchanged from the hand-rolled path.
    binary_model_id = _BINARY_MODEL_ID[binary_kind]
    BINARY_EPOCH_NAME = {'ell1': 'TASC', 'dd': 'T0', 'bt': 'T0',
                         'ddk': 'T0'}.get(binary_kind)
    DMEPOCH_0 = float(params.get('DMEPOCH', params['PEPOCH']))
    # DM Taylor: dm_eff = Σ_n DMn·dtⁿ/n! with dt in years from DMEPOCH
    # (combined_delays forms dt_years = (tdb − dm_epoch)/365.25 internally and
    # indexes dm_coeffs by power 0,1,2,….).  We build a DENSE coefficient vector
    # DM0(≡DM), DM1, …, DM_max covering every DMn present in the par (missing
    # orders zero-filled), with factorials [0!,1!,…,max!] matching PINT's DM(t)
    # convention.  DMn are frozen at par values unless sampled, so when frozen
    # they contribute the SAME per-TOA term to m(θ) and m(θ₀) and cancel in the
    # θ₀-referenced delta — the pass-1 regression stays bit-identical.
    dm_orders_full: list[int] = []
    k = 1
    while f'DM{k}' in params:
        dm_orders_full.append(k)
        k += 1
    max_dm_order = max(dm_orders_full) if dm_orders_full else 0
    dm_0_full = {n: float(params.get(f'DM{n}', 0.0)) for n in dm_orders_full}
    _DM_FACTORIALS = jnp.asarray(
        [float(math.factorial(n)) for n in range(max_dm_order + 1)],
        dtype=jnp.float64)
    # Orbital-frequency (FB) parametrisation: when the par uses FB0,FB1,… instead
    # of PB, combined_delays' use_fb branch integrates the orbital phase from the
    # FB Taylor (referenced to the binary epoch baked into tt_binary_sec).  Build
    # a DENSE FB0..FB_max vector; use_fb is a static Python bool so the PB slot
    # can be forced non-zero below (the dead PB branch of jnp.where must not form
    # 1/PB=inf and poison the reverse-mode gradient).
    fb_orders_full: list[int] = []
    k = 0
    while f'FB{k}' in params:
        fb_orders_full.append(k)
        k += 1
    max_fb_order = max(fb_orders_full) if fb_orders_full else -1
    fb_0_full = {n: float(params.get(f'FB{n}', 0.0)) for n in fb_orders_full}
    use_fb = bool(has_binary and 'FB0' in params)
    _FB_FACTORIALS = (jnp.asarray(
        [float(math.factorial(n)) for n in range(max_fb_order + 1)],
        dtype=jnp.float64) if use_fb else jnp.asarray([], dtype=jnp.float64))
    _EMPTY_VEC = jnp.asarray([], dtype=jnp.float64)

    # ---- 3. Higher-order spindown: collect all Fn from par -----------------
    fn_orders_full: list[int] = [0, 1]  # F0, F1 always present
    k = 2
    while f'F{k}' in params:
        fn_orders_full.append(k)
        k += 1
    fn_0_full = {n: float(params.get(f'F{n}', 0.0)) for n in fn_orders_full}

    # ---- 3b. FD profile-evolution coefficients ----------------------------
    # FD delay: Σ_n FDn × log(ν_GHz)^n  [seconds]
    fd_orders_full: list[int] = []
    k = 1
    while f'FD{k}' in params:
        fd_orders_full.append(k)
        k += 1
    fd_0_full = {n: float(params.get(f'FD{n}', 0.0)) for n in fd_orders_full}

    # ---- 3c. Ecliptic coordinate mode -------------------------------------
    is_ecliptic = bool(params.get('_ecliptic_coords', False))
    _ECL_SAMPLE_KEYS = frozenset({'ELONG', 'ELAT', 'PMELONG', 'PMELAT'})
    has_ecliptic_sample = is_ecliptic and bool(_ECL_SAMPLE_KEYS & set(sample_list))
    # Build the pure-JAX rotation function once (obliquity baked in as constant).
    ecl_to_eq_jax = _make_ecl_to_eq_fn(params) if is_ecliptic else None

    # ---- 4. DMX masks (frozen, precomputed outside JAX) --------------------
    freq_mhz_np = np.asarray(result['freq_bary_mhz'], dtype=np.float64)
    tdb_mjd_np  = np.asarray(tdb_mjd_ld, dtype=np.float64)
    dmx_masks   = _build_dmx_masks(params, mjd_utc_np)

    # DM derivative: K_DM / ν² — same for every DMX window, frozen
    dm_deriv = jnp.asarray(K_DM_SEC / freq_mhz_np ** 2, dtype=jnp.float64)

    # Frozen DMX baseline values from par
    dmx_0 = {label: float(params.get(label, 0.0)) for label in dmx_masks}

    # ---- 5. JUMP masks (frozen, precomputed from _jump_lines + TOA flags) --
    # NOTE: MJD-range JUMPs are cut on the TDB MJD here, while DMX windows are
    # cut on the observatory (UTC) MJD.  Both are par-file MJD ranges, so this is
    # an inconsistency worth resolving upstream; it is left as-is to preserve
    # existing behaviour (the two scales differ by ≲ a few hundred seconds, so it
    # only matters for a TOA within ~0.01 d of a window edge).
    jump_masks = _build_jump_masks(params, session.toas_data, tdb_mjd_np)

    # Frozen JUMP baseline values from params (JUG stores them as JUMP1, JUMP2, …)
    jump_0 = {key: float(params.get(key, 0.0)) for key in jump_masks}

    # ---- 6. Build θ₀ dynamically for every requested parameter -------------
    theta_0_full: dict[str, float] = {}
    for k_name in sample_list:
        cls = _classify_param(k_name)
        if cls == 'dmx':
            theta_0_full[k_name] = dmx_0.get(k_name, 0.0)
        elif cls == 'jump':
            theta_0_full[k_name] = jump_0.get(k_name, 0.0)
        elif cls == 'fn':
            order = int(_RE_FN.match(k_name).group(1))
            theta_0_full[k_name] = fn_0_full.get(order, 0.0)
        elif cls == 'fd':
            order = int(_RE_FD.match(k_name).group(1))
            theta_0_full[k_name] = fd_0_full.get(order, 0.0)
        elif cls == 'dm':
            order = int(_RE_DM.match(k_name).group(1))
            theta_0_full[k_name] = dm_0_full.get(order, 0.0)
        elif cls == 'fb':
            order = int(_RE_FB.match(k_name).group(1))
            theta_0_full[k_name] = fb_0_full.get(order, 0.0)
        else:
            theta_0_full[k_name] = _read_param(params, k_name)
    # Fallback pool for _g() — every kernel param and every Fn/FDn/DMn/FBn.  The
    # dense DM/FB coefficient vectors read EVERY order 0..max via _g, so every
    # order (including gaps not in the par) must have a θ₀ entry.
    for k_name in _KERNEL_PARAMS:
        theta_0_full.setdefault(k_name, _read_param(params, k_name))
    for order in fn_orders_full:
        theta_0_full.setdefault(f'F{order}', fn_0_full[order])
    for order in fd_orders_full:
        theta_0_full.setdefault(f'FD{order}', fd_0_full[order])
    for order in range(1, max_dm_order + 1):
        theta_0_full.setdefault(f'DM{order}', dm_0_full.get(order, 0.0))
    for order in range(0, max_fb_order + 1):
        theta_0_full.setdefault(f'FB{order}', fb_0_full.get(order, 0.0))
    # Frozen baselines for the secondary binary params.  XDOT is the JUG design
    # label for A1DOT, so read it via the XDOT_0 fallback (par stores A1DOT);
    # OMDOT is par-native.  These override any _read_param zero.
    theta_0_full['XDOT']  = XDOT_0
    theta_0_full['OMDOT'] = OMDOT_0

    # ---- Apply manual θ₀ overrides ----------------------------------------
    if theta_0_overrides:
        for k, v in theta_0_overrides.items():
            if k not in theta_0_full:
                warnings.warn(
                    f"theta_0_overrides: {k!r} not found in theta_0_full and "
                    f"will be ignored.  Known keys: {sorted(theta_0_full)}",
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"theta_0 is being overridden.",
                    stacklevel=2,
                )
                theta_0_full[k] = v

    sampled_names = set(sample_list)   # canonical, post-validation sample set

    def _g(theta: dict, k_name: str):
        """Return sampled value (traced) or frozen Python float constant."""
        return theta[k_name] if k_name in sampled_names else theta_0_full[k_name]

    # ---- 7. Total timing model in seconds ----------------------------------
    @jax.jit
    def m_total_sec(theta: dict) -> jnp.ndarray:
        """Total model delay per TOA, in seconds, for one parameter dict.

        ``theta`` carries ONLY the sampled params (keys == ``sample_list``);
        every other value is the frozen par-file constant closed over by
        ``_g``.  The sum is astrometry (Roemer + solar Shapiro) + the
        combined_delays block (binary + DM + FD + solar wind) + the
        theta_0-referenced spindown, DMX and JUMP terms.
        """
        # --- Astrometry ---
        # When ecliptic params are sampled we rotate the traced ecliptic values
        # to equatorial inside JAX.  ecl_to_eq_jax is a closure whose obliquity
        # constants were baked in at setup time, so it is fully traceable.
        if has_ecliptic_sample:
            lon_deg = _g(theta, 'ELONG')
            lat_deg = _g(theta, 'ELAT')
            pm_lon  = _g(theta, 'PMELONG')
            pm_lat  = _g(theta, 'PMELAT')
            RA, DEC, PMRA, PMDEC = ecl_to_eq_jax(lon_deg, lat_deg, pm_lon, pm_lat)
        else:
            RA   = _g(theta, 'RAJ')
            DEC  = _g(theta, 'DECJ')
            PMRA = _g(theta, 'PMRA')
            PMDEC = _g(theta, 'PMDEC')

        # NOTE: F0/F1 are not bound here — the spindown block below reads every
        # Fn (including F0, which also sets the phase→time scale) through _g.
        DM    = _g(theta, 'DM')
        PX    = _g(theta, 'PX')
        PB    = _g(theta, 'PB')
        A1    = _g(theta, 'A1')
        TASC  = _g(theta, 'TASC')
        EPS1  = _g(theta, 'EPS1')
        EPS2  = _g(theta, 'EPS2')
        T0    = _g(theta, 'T0')
        ECC   = _g(theta, 'ECC')
        OM    = _g(theta, 'OM')
        M2    = _g(theta, 'M2')
        SINI  = _g(theta, 'SINI')
        XDOT  = _g(theta, 'XDOT')
        OMDOT = _g(theta, 'OMDOT')
        EDOT  = _g(theta, 'EDOT')
        GAMMA = _g(theta, 'GAMMA')
        PBDOT = _g(theta, 'PBDOT')
        EPS1DOT = _g(theta, 'EPS1DOT')
        EPS2DOT = _g(theta, 'EPS2DOT')
        H3   = _g(theta, 'H3')
        H4   = _g(theta, 'H4')
        STIG = _g(theta, 'STIG')
        KIN  = _g(theta, 'KIN')
        KOM  = _g(theta, 'KOM')
        DR   = _g(theta, 'DR')
        DTH  = _g(theta, 'DTH')
        NE_SW = _g(theta, 'NE_SW')
        # Dense DM Taylor vector [DM0≡DM, DM1, …, DM_max]; missing orders read a
        # 0.0 θ₀ baseline so they contribute nothing.
        dm_coeffs = jnp.stack(
            [DM] + [_g(theta, f'DM{n}') for n in range(1, max_dm_order + 1)])
        # Dense FB Taylor vector [FB0, …, FB_max] (empty when the par uses PB).
        fb_coeffs = (jnp.stack(
            [_g(theta, f'FB{n}') for n in range(0, max_fb_order + 1)])
            if use_fb else _EMPTY_VEC)
        # Force the (dead) PB branch's 1/PB away from 1/0 when FB-parametrised,
        # so jnp.where's unused side cannot poison the reverse-mode gradient.
        PB_arg = jnp.asarray(1.0, dtype=jnp.float64) if use_fb else PB

        # Astrometry sourced from the JUG JAX twins (jug.delays.barycentric_jax),
        # mirroring JUG's NumPy barycentric path.  dt_pos_days is the host-side
        # longdouble-reduced (t - POSEPOCH).  NOTE: the twin uses RIGOROUS
        # great-circle proper-motion propagation (matching PINT/ERFA), where the
        # retired hand-rolled _pulsar_direction used the linear tangent-plane
        # update — they differ at O((PM·dt)²).  For frozen astrometry (the recovery
        # configs) L_hat is identical in m(θ) and m(θ₀), so roemer/shapiro cancel
        # in the θ₀-referenced delta and the change is correctness-neutral there.
        L_hat   = pulsar_direction_jax(dt_pos_days, RA, DEC,
                                       PMRA * _MAS_YR_TO_RAD_DAY,
                                       PMDEC * _MAS_YR_TO_RAD_DAY)
        roemer  = roemer_delay_jax(ssb_obs_km, L_hat, PX)
        shapiro = shapiro_delay_jax(obs_sun_km, L_hat, T_SUN_SEC)

        # --- Binary + DM + FD + solar wind via JUG combined_delays ----------
        # combined_delays returns (dm_sec + sw_sec + fd_sec + binary_sec); it does
        # NOT include roemer_shapiro in its output (that is added in the return).
        # roemer_shapiro is consumed only inside the kernel, to form the pre-binary
        # delay that shifts the binary time.  The binary epoch (TASC/T0) is carried
        # by the precomputed tt_binary_sec, rebuilt here from the (possibly sampled)
        # epoch so the physics matches the retired hand-rolled kernels exactly.
        #
        # Neutrality wiring (reproduces the hand-rolled path):
        #   • DM      : dense Taylor [DM, DM1, …, DM_max] → Σ K_DM·DMn·dtⁿ/(n!·ν²);
        #               frozen orders cancel in the θ₀-referenced delta.
        #   • FD      : fd_coeffs = [FD1, FD2, …] → Σ FDn·log(ν/GHz)ⁿ.  Absolute
        #               vs the old (FDn−FDn₀) form, but identical in delta_m_us
        #               (the θ₀ subtraction removes the constant per-TOA offset).
        #   • sol.W   : ne_sw traced → combined_delays returns sw_sec(ne_sw) as a
        #               dispersive delay, the SAME family as dm_sec(DM).  SW is NOT
        #               folded into the prebinary slot (that legacy placement made
        #               NE_SW un-fittable and treated SW inconsistently with DM,
        #               which is likewise returned rather than prebinary).
        #   • DMX     : dmx_sec = None — the hand-rolled binary pre-delay excluded
        #               DMX; the DMX offsets are added separately below.
        #   • ELL1    : eps1dot/eps2dot/xdot/pbdot/gamma + orthometric Shapiro
        #               (h3/h4/stig) are traced through; the eccentric slots
        #               (ecc/om/t0/omdot/edot) stay zero (ELL1 uses eps1/eps2/tasc).
        #   • DD/BT/DDK: full secondaries (gamma/pbdot/omdot/xdot/edot) plus
        #               kin/kom (DDK Kopeikin, model_id 5) traced through.
        #   • FB      : when use_fb, fb_coeffs drive the orbital phase and the PB
        #               slot is forced to 1.0 (dead branch; see PB_arg above).
        fd_coeffs = (jnp.stack([_g(theta, f'FD{order}') for order in fd_orders_full])
                     if fd_orders_full else _EMPTY_VEC)
        has_fd = len(fd_orders_full) > 0
        # Prebinary slot carries only the (frozen) troposphere now; solar wind is
        # returned via ne_sw below, matching DM's returned-delay treatment.
        tropo_prebin = tropo_delay

        if has_binary:
            epoch = _g(theta, BINARY_EPOCH_NAME)
            tt_binary_sec = (tdb_mjd - epoch) * SECS_PER_DAY
        else:
            tt_binary_sec = jnp.zeros_like(tdb_mjd)

        if binary_kind == 'ell1':
            r_shap = T_SUN_SEC * M2
            combined = combined_delays(
                tdb_mjd, freq_mhz, obs_sun_km, L_hat,
                dm_coeffs, _DM_FACTORIALS, DMEPOCH_0,
                NE_SW, fd_coeffs, has_fd,
                roemer + shapiro, has_binary, binary_model_id,
                PB_arg, A1, TASC, EPS1, EPS2, EPS1DOT, EPS2DOT, PBDOT, XDOT,
                GAMMA, r_shap, SINI,
                0.0, 0.0, 0.0, 0.0, 0.0, M2, SINI, KIN, KOM, H3, H4, STIG,
                fb_coeffs, _FB_FACTORIALS, 0.0, use_fb,
                dr=DR, dth=DTH,
                tropo_sec=tropo_prebin, tt_binary_sec=tt_binary_sec,
            )
        else:   # dd / bt / ddk — full secondaries + Kopeikin (kin/kom) forwarded
            combined = combined_delays(
                tdb_mjd, freq_mhz, obs_sun_km, L_hat,
                dm_coeffs, _DM_FACTORIALS, DMEPOCH_0,
                NE_SW, fd_coeffs, has_fd,
                roemer + shapiro, has_binary, binary_model_id,
                PB_arg, A1, 0.0, EPS1, EPS2, EPS1DOT, EPS2DOT, PBDOT, XDOT,
                GAMMA, 0.0, 0.0,
                ECC, OM, T0, OMDOT, EDOT, M2, SINI, KIN, KOM, H3, H4, STIG,
                fb_coeffs, _FB_FACTORIALS, 0.0, use_fb,
                dr=DR, dth=DTH,
                tropo_sec=tropo_prebin, tt_binary_sec=tt_binary_sec,
            )

        # --- Spindown Taylor series (θ₀-referenced) ---
        # Δφ(t) = Σ_n (Fn − Fn₀)·dt^(n+1)/(n+1)!  is the phase perturbation at
        # fixed time; the predicted arrival time shifts by Δt = −Δφ/ν so the
        # same pulse number is recovered.
        #
        # The divisor is the SAMPLED F0, not the frozen F0₀.  With F0₀ the whole
        # block is exactly linear in every Fn, so a sampled F0 displaces the
        # phase but never rescales its own phase→time conversion — a systematic
        # (ΔF0/F0) error on the entire spin delta, and a model that is not the
        # nonlinear F0 delay it claims to be.  _g(theta,'F0') reduces to F0₀
        # exactly at θ₀, so m0_sec, every Jacobian column and the linearised MLE
        # are bit-unchanged there; only the behaviour away from θ₀ differs.
        # When F0 is marginalised or fixed, _g returns the frozen float and this
        # is identical to the old expression everywhere.
        #
        # ν is taken as F0 rather than the instantaneous ν(t) = Σ_n Fn·dtⁿ/n!,
        # matching PINT/JUG's d_delay = −d_phase/F0 convention so the sampled
        # and marginalised (design-matrix) spin paths stay mutually consistent.
        # The neglected term is a relative F1·dt/F0 ~ 1e-8 OF the spin delta.
        #
        # F0 cannot reach the pole at 0: σ_F0 ~ 1e-13 Hz against F0 ~ 1e2 Hz, so
        # the sampler would need ~1e15 σ to cross it.
        F0 = _g(theta, 'F0')
        dphase = jnp.zeros_like(dt_sec)
        for order in fn_orders_full:
            fname  = f'F{order}'
            Fn     = _g(theta, fname)
            Fn_0   = theta_0_full[fname]
            coeff  = dt_sec ** (order + 1) / math.factorial(order + 1)
            dphase = dphase + (Fn - Fn_0) * coeff
        spin = -dphase / F0

        # --- DMX chromatic offsets ---
        dmx_delay = jnp.zeros_like(tdb_mjd)
        for label, mask in dmx_masks.items():
            if label in sampled_names:
                dDMX      = _g(theta, label) - theta_0_full[label]
                dmx_delay = dmx_delay + jnp.where(mask, dDMX * dm_deriv, 0.0)

        # --- JUMP phase offsets ---
        jump_delay = jnp.zeros_like(tdb_mjd)
        for jkey, mask in jump_masks.items():
            if jkey in sampled_names:
                dJUMP      = _g(theta, jkey) - theta_0_full[jkey]
                jump_delay = jump_delay + jnp.where(mask, dJUMP, 0.0)

        # ----------------------------------------------------------------
        # Extension point: add new delay kernels here.
        # ----------------------------------------------------------------

        return roemer + shapiro + combined + spin + dmx_delay + jump_delay

    # ---- 8. delta_m_us at θ₀ ----------------------------------------------
    theta_0_sampled = {k_name: theta_0_full[k_name] for k_name in sample_list}
    m0_sec = m_total_sec(theta_0_sampled)

    # Fail fast if the model itself is non-finite at θ₀ — otherwise every Jacobian
    # column below inherits the NaN and the whole pulsar silently returns a NaN
    # MLE (the classic FB-vs-PB / PB=0 collapse).  A clear error beats silent NaN.
    if not bool(np.isfinite(np.asarray(m0_sec)).all()):
        raise FloatingPointError(
            "m_total_sec is non-finite at the par values (θ₀).  Common cause: a "
            "binary parametrisation whose required parameter is zero/absent (e.g. "
            "an FB-parametrised orbit read with PB=0, or a missing epoch).  Check "
            f"the par binary model {binary_model!r} and its parameters.")

    @jax.jit
    def delta_m_us(theta: dict) -> jnp.ndarray:
        """Mean-subtracted Δm in µs.  Autodiff-able."""
        dm_us = (m_total_sec(theta) - m0_sec) * 1.0e6
        wmean = jnp.sum(weights * dm_us) / jnp.sum(weights)
        return dm_us - wmean

    # ---- 9. Linearised MLE (Jacobian via JVP at θ₀) -----------------------
    # Cast the primal to explicit float64 arrays so primal/tangent dtypes match
    # exactly (Python floats are weakly typed and can trip jax.jvp).
    theta_0_jax = {k_name: jnp.asarray(v, dtype=jnp.float64)
                   for k_name, v in theta_0_sampled.items()}
    X_cols = []
    col_norms = []
    for k_name in sample_list:
        v = {kk: jnp.zeros((), dtype=jnp.float64) for kk in sample_list}
        v[k_name] = jnp.ones((), dtype=jnp.float64)
        col = np.asarray(jax.jvp(delta_m_us, (theta_0_jax,), (v,))[1])
        X_cols.append(col)
        col_norms.append(float(np.sqrt(np.dot(col, col))))

    # Guard: any parameter whose Jacobian column is numerically zero OR non-finite
    # (e.g. an unmodelled param that still reached sample_list, or a NaN from a
    # degenerate kernel) would make XtWX singular.  Identify these, warn, and
    # exclude them from the solve.  With the setup guard above rejecting unwired
    # params, a surviving zero column now signals a param that IS wired but is
    # physically inert under this par's binary model (e.g. XDOT under a model
    # that ignores it, or an EPS*DOT under DD) — still worth surfacing loudly.
    _ZERO_THRESH = 1e-30
    col_norms_arr = np.array(col_norms)
    active_mask = np.isfinite(col_norms_arr) & (col_norms_arr > _ZERO_THRESH)
    zero_params = [k for k, active in zip(sample_list, active_mask) if not active]
    if zero_params:
        warnings.warn(
            f"MLE: parameters {zero_params} have zero or non-finite Jacobian "
            "columns and will be excluded from the linear solve (their MLE "
            "delta/sigma will be NaN).  The param is wired but contributes no "
            "delay under this par's binary model, or produced a NaN derivative.",
            stacklevel=2,
        )

    active_idx = np.where(active_mask)[0]
    X = (np.column_stack([X_cols[i] for i in active_idx])
         if active_idx.size else np.empty((n_toa, 0)))
    W = np.asarray(weights)
    r_obs_np = np.asarray(r_obs_us)

    if active_idx.size > 0:
        XtWX = X.T @ (W[:, None] * X)
        XtWr = X.T @ (W * r_obs_np)
        # SVD-based pseudoinverse for robustness against near-singular cases
        # (e.g. highly covariant DMX bins or JUMP/astrometry degeneracies).
        # WARNING: a single global rcond on a ~20-order-heterogeneous XtWX
        # truncates almost every singular value, so mle['sigma'] can be many
        # orders of magnitude too small.  Use aux['sigJUG'] for prior scales.
        U, s, Vt = np.linalg.svd(XtWX)
        rcond = np.finfo(float).eps * max(XtWX.shape) * s[0]
        s_inv = np.where(s > rcond, 1.0 / s, 0.0)
        Sigma_active = (Vt.T * s_inv) @ U.T
        theta_hat_active = Sigma_active @ XtWr
        sigmas_active = np.sqrt(np.maximum(np.diag(Sigma_active), 0.0))
    else:
        Sigma_active = np.empty((0, 0))
        theta_hat_active = np.empty(0)
        sigmas_active = np.empty(0)

    # Reconstruct full-length arrays (NaN for zero/inactive params)
    n_sampled = len(sample_list)
    theta_hat_full = np.full(n_sampled, np.nan)
    sigmas_full    = np.full(n_sampled, np.nan)
    cov_full       = np.full((n_sampled, n_sampled), np.nan)
    for out_i, in_i in enumerate(active_idx):
        theta_hat_full[in_i] = theta_hat_active[out_i]
        sigmas_full[in_i]    = sigmas_active[out_i]
        for out_j, in_j in enumerate(active_idx):
            cov_full[in_i, in_j] = Sigma_active[out_i, out_j]

    mle = {
        'delta':  dict(zip(sample_list, theta_hat_full)),
        'sigma':  dict(zip(sample_list, sigmas_full)),
        'array':  theta_hat_full,
        'sigmas': sigmas_full,
        'cov':    cov_full,
        'zero_params': zero_params,
    }

    # JUG's formal per-parameter σ (authoritative fit uncertainties) for the
    # sampled params — the prior/coordinate SCALE (theta = theta_0 + z·scale).
    # JUG computes these via its stable augmented-SVD solve.  Do NOT scale with
    # mle['sigma']: its naive pinv of the raw, ~20-order-heterogeneous XᵀWX
    # collapses (a single global rcond truncates all but ~1 of the singular
    # values), returning σ ~1e-37× too small — smaller than the float64 ULP of
    # theta_0, which freezes every well-measured param at the fit value.  The
    # setup guard above ensures every sampled param is a JUG design label, so a
    # finite σ is always available here.
    _jug_unc = result_dummy['uncertainties']
    sigJUG = {}
    for k in sample_list:
        v = float(_jug_unc[k]) if k in _jug_unc else float('nan')
        if not np.isfinite(v) or v == 0.0:
            raise ValueError(
                f"JUG formal uncertainty for sampled param {k!r} is {v!r}; cannot "
                "set its prior/coordinate scale.  A JUG design label should always "
                "carry a finite, non-zero uncertainty — a NaN/0 signals a "
                "degenerate JUG fit for this par/tim.")
        sigJUG[k] = v

    aux = {
        'theta_0':          theta_0_sampled,
        'errors_us':        errors_us,
        'r_obs_us':         r_obs_us,
        'tdb_mjd':          tdb_mjd,
        'mle':              mle,
        'sigJUG':           sigJUG,   # JUG formal σ per sampled param (prior scale)
        'sample_list':      list(sample_list),
        'marginalise_list': sorted(marg_set),       # params carried in M
        'fixed_list':       sorted(fixed_set),      # params held at par value
        'modes':            mode_map,               # {param: sample|marginalise|fix}
        'binary_model':     binary_model or 'NONE',
        'n_toa':            n_toa,
        'dmx_masks':        dmx_masks,   # {label: bool array}   — for diagnostics
        'jump_masks':       jump_masks,  # {JUMP{n}: bool array}  — for diagnostics
        'Mmat':             _timing_model_svd(Mmat),  # SVD basis of stripped M
        'Mmat_param_names': Mmat_param_names,  # retained linear labels (pre-SVD)
        'Cjug':             _Cjug,
        'fittable_order':   _fittable,   # JUG design-label order Cjug is indexed by
    }
    return delta_m_us, aux


class MultiPsrTimingModel:
    """Multi-pulsar timing model residual calculator.

    Wraps a list of per-pulsar ``delta_m_us`` callables from
    ``setup_timing_model`` and maps a concatenated parameter vector →
    concatenated timing residuals.

    The parameter layout is:
        z_concat = [z_0_F0, z_0_F1, ...,   # pulsar 0 params in its sample order
                    z_1_F0, z_1_F1, ...,   # pulsar 1 params in its sample order
                    ...]
    i.e. pulsar p contributes ``len(sample_list[p])`` entries.

    Attributes
    ----------
    delta_m_list : list of callable
        Per-pulsar JIT'd delta_m_us(theta_dict) -> jnp.ndarray [n_toa_p], µs.
    aux_list : list of dict
        Per-pulsar aux dicts from setup_timing_model.
    sample_list : list of list of str
        PER-PULSAR parameter-name lists (one list per pulsar; they need not be
        identical across pulsars).
    scales : list of dict
        Per-pulsar {param: sigma} dicts used to un-normalise z → theta.
    nparams : list of int
        Number of sampled parameters for each pulsar.
    nparams_total : int
        Sum of ``nparams``.
    npulsars : int
    ntoas_per_psr, toa_starts, toa_ends : tuple of int
        TOA block layout in the concatenated residual vector.
    neps_per_psr, eps_starts, eps_ends : tuple of int
        Linear-basis (epsilon) block layout.

    NOTE: the jitted methods use ``static_argnums=0``, so ``self`` is hashed by
    identity.  Mutating the attributes of an instance after a jitted method has
    been traced will NOT invalidate the compilation cache — build a new instance
    instead.
    """

    def __init__(self, delta_m_list, aux_list, sample_list, scales, data):
        """
        Parameters
        ----------
        delta_m_list : list of callable
            Output of setup_timing_model(...)[0] for each pulsar.
        aux_list : list of dict
            Output of setup_timing_model(...)[1] for each pulsar.
        sample_list : list of list of str
            Per-pulsar parameter-name lists (NOT a flat list of names).
        scales : list of dict
            Per-pulsar {param: scale} dicts.  The z-vector is un-normalised as
                theta[k] = reparametrise(k, theta_0[k], z[k] * scales[p][k])
        data : object with ``raw_residuals`` — list of per-pulsar residual
            arrays (seconds).  This, not aux['r_obs_us'], is the observed-data
            source for every residual path, so all paths subtract the model from
            the identical data vector.
        """
        n = len(delta_m_list)
        if not (len(aux_list) == len(sample_list) == len(scales) == n):
            raise ValueError(
                "delta_m_list, aux_list, sample_list and scales must all have "
                f"one entry per pulsar; got {n}, {len(aux_list)}, "
                f"{len(sample_list)}, {len(scales)}.")
        if any(isinstance(sl, str) for sl in sample_list):
            raise TypeError(
                "sample_list must be a list of PER-PULSAR name lists, e.g. "
                "[['F0','F1'], ['F0','M2']] — a flat list of strings would be "
                "silently misread (len('F0') == 2 params).")

        self.raw_residuals = data.raw_residuals

        self.delta_m_list  = delta_m_list
        self.aux_list      = aux_list
        self.sample_list   = [list(sl) for sl in sample_list]
        self.scales        = scales
        self.nparams       = [len(sl) for sl in self.sample_list]
        self.nparams_total = sum(self.nparams)
        self.npulsars      = n

        ntoas = tuple(int(aux['n_toa']) for aux in aux_list)
        self.ntoas_per_psr = ntoas
        self.total_ntoas = sum(self.ntoas_per_psr)
        cumulative = np.cumsum([0] + list(ntoas))
        self.toa_starts = tuple(int(c) for c in cumulative[:-1])
        self.toa_ends   = tuple(int(c) for c in cumulative[1:])

        self.Mmats = [x['Mmat'] for x in aux_list]
        self.neps_per_psr = tuple(x.shape[1] for x in self.Mmats)
        self.total_neps = sum(self.neps_per_psr)
        # Precompute static slice boundaries for each pulsar in the global array.
        cumulative_eps = np.cumsum([0] + list(self.neps_per_psr))
        self.eps_starts = tuple(int(c) for c in cumulative_eps[:-1])
        self.eps_ends   = tuple(int(c) for c in cumulative_eps[1:])

        self._prepare_family_index()

        # Populated by prepare_dense_metric() — left None so build_gibbs_model /
        # run_gibbs can fail with a clear message rather than a bare AttributeError.
        self.bound_names = None
        self.OFFSET = self.AFFINE = self.BOUND = None
        self.metric_cols = None
        self.timing_metric_list = self.metric_order_list = self.metric_sites = None

    # ------------------------------------------------------------------
    # Linear (marginalised) basis helpers
    # ------------------------------------------------------------------

    @jit_method
    def linear_residuals(self, epsilons):
        """Project per-pulsar linear-basis coefficients onto the TOA vector.

        Parameters
        ----------
        epsilons : array [total_neps]
            Concatenated linear-basis coefficients, ordered by pulsar
            (eps_starts/eps_ends).

        Returns
        -------
        array [total_ntoas]   M_p @ eps_p for each pulsar, concatenated.
        """
        timing_residuals = jnp.zeros(self.total_ntoas)
        for Mmat, start_eps, end_eps, start_toa, end_toa in zip(
                self.Mmats, self.eps_starts, self.eps_ends,
                self.toa_starts, self.toa_ends):
            timing_residuals = timing_residuals.at[start_toa:end_toa].set(
                Mmat @ epsilons[start_eps:end_eps])
        return timing_residuals

    @jit_method
    def get_epsilon(self, helpers, key):
        """Draw one realisation of the linear-basis coefficients per pulsar.

        The draw is from N(mean, Sigma) with
            Sigma = (MNM)^-1,   mean = Sigma @ MNr,
        implemented via the Cholesky factor of MNM (upper R, MNM = RᵀR), so
            eps = mean + R^-1 u,   u ~ N(0, I)   ⇒   Cov(eps) = (RᵀR)^-1.

        Parameters
        ----------
        helpers : tuple (MNMs, MNrs)
            Per-pulsar [neps, neps] normal matrices and [neps] data vectors.
        key : jax.random.PRNGKey
            Split once per pulsar, so the pulsars draw independent noise.

        Returns
        -------
        array [total_neps]
        """
        MNMs, MNrs = helpers
        # Split the key per pulsar.  Reusing a single key across pulsars gives
        # identical standard normals for every equally-sized block, i.e. a
        # perfectly correlated (and therefore wrong) joint draw.
        keys = jrandom.split(key, self.npulsars)
        epsilons = jnp.zeros(self.total_neps)
        for MNM, MNr, start, end, subkey in zip(
                MNMs, MNrs, self.eps_starts, self.eps_ends, keys):
            cf   = jsl.cho_factor(MNM)
            mean = jsl.cho_solve(cf, MNr[..., None])[..., 0]
            u    = jrandom.normal(subkey, shape=(mean.shape[0],))
            epsilons = epsilons.at[start:end].set(
                mean + jsl.solve_triangular(cf[0], u))
        return epsilons

    @jit_method
    def get_mean(self, helpers):
        """Posterior mean of the linear-basis coefficients, per pulsar.

        Returns a list of [neps_p] arrays (NOT concatenated), one per pulsar.
        """
        MNMs, MNrs = helpers
        means = []
        for MNM, MNr in zip(MNMs, MNrs):
            cf = jsl.cho_factor(MNM)
            means.append(jsl.cho_solve(cf, MNr[..., None])[..., 0])
        return means

    @jit_method
    def get_epsilon_from_z(self, helpers, z):
        """As ``get_epsilon``, but with the unit normals supplied externally.

        Parameters
        ----------
        helpers : tuple (MNMs, MNrs)
        z : array [total_neps]
            Standard-normal draws, laid out like ``epsilons``.

        Returns
        -------
        array [total_neps]
        """
        MNMs, MNrs = helpers
        epsilons = jnp.zeros(self.total_neps)
        for MNM, MNr, start, end in zip(MNMs, MNrs, self.eps_starts, self.eps_ends):
            cf   = jsl.cho_factor(MNM)
            mean = jsl.cho_solve(cf, MNr[..., None])[..., 0]
            epsilons = epsilons.at[start:end].set(
                mean + jsl.solve_triangular(cf[0], z[start:end]))
        return epsilons

    # ------------------------------------------------------------------
    # Legacy z_concat path
    # ------------------------------------------------------------------

    def z_to_theta(self, pidx, z_p):
        """Un-normalise a per-pulsar z-vector → theta dict.

        Parameters
        ----------
        pidx : int
            Pulsar index.
        z_p : array [nparams[pidx]]
            Normalised parameter vector for this pulsar.

        Returns
        -------
        theta : dict  {param_name: scalar}
        """
        theta_0 = self.aux_list[pidx]['theta_0']
        sc      = self.scales[pidx]
        names   = self.sample_list[pidx]
        # Shape is static under jit, so this Python check is free and traced-safe.
        # It matters: JAX CLAMPS out-of-bounds indices instead of raising, so a
        # z_p shorter than the name list would silently give every trailing
        # parameter the value z_p[-1] — an injection/recovery mismatch that
        # produces no error and no warning.
        if getattr(z_p, 'shape', None) != (len(names),):
            raise ValueError(
                f"z_to_theta(pidx={pidx}): expected a z-vector of shape "
                f"({len(names)},) to match sample_list[{pidx}]={names}, got "
                f"shape {getattr(z_p, 'shape', None)}.  Build z in "
                "tm.sample_list[pidx] order — setup_timing_model re-sorts the "
                "requested params into JUG's design-label order, so it need not "
                "match the order you passed in.")
        # u = z·σ is the scaled, mean-zero sampler coordinate; reparametrise()
        # maps it to a physical value that stays in-domain for every real z,
        # so SINI≤1 / 0≤ECC<1 / M2,PX>0 are guaranteed before the JUG kernels.
        return {k: reparametrise(k, theta_0[k], z_p[i] * sc[k])
                for i, k in enumerate(names)}

    @partial(jax.jit, static_argnums=0)
    def timing_model_residuals_per_pulsar(self, z_concat):
        """Per-pulsar model delay Δm(θ(z)) in SECONDS (list, not concatenated)."""
        if getattr(z_concat, 'shape', None) != (self.nparams_total,):
            raise ValueError(
                f"z_concat must have shape ({self.nparams_total},) = "
                f"{self.nparams} per pulsar; got "
                f"{getattr(z_concat, 'shape', None)}.  A short vector would be "
                "silently clamped by JAX indexing, not rejected.")
        parts = []
        start_index = 0
        for pidx in range(self.npulsars):
            end_index = start_index + self.nparams[pidx]
            z_p     = z_concat[start_index:end_index]
            theta_p = self.z_to_theta(pidx, z_p)
            parts.append(self.delta_m_list[pidx](theta_p) * 1e-6)   # µs → s
            start_index = end_index
        return parts

    @partial(jax.jit, static_argnums=0)
    def residuals(self, z_concat):
        """Concatenated stochastic residuals for all pulsars.

        Parameters
        ----------
        z_concat : array [nparams_total]
            Flat, normalised parameter vector, laid out pulsar by pulsar.

        Returns
        -------
        res : array [total_ntoas]
            ``r_obs_p - delta_m_p(theta_p)`` in seconds, ordered by pulsar
            (matching toa_starts/toa_ends).
        """
        tm_parts = self.timing_model_residuals_per_pulsar(z_concat)
        return jnp.concatenate([r_obs - tm for r_obs, tm
                                in zip(self.raw_residuals, tm_parts)])

    @partial(jax.jit, static_argnums=0)
    def residuals_from_z_dict(self, z_dict):
        """Same as ``residuals()`` but accepts a numpyro-style z_dict.

        Parameters
        ----------
        z_dict : dict  {f'z_{k};{pidx}': scalar}

        Returns
        -------
        res : array [total_ntoas]
        """
        parts = []
        for pidx in range(self.npulsars):
            theta_0 = self.aux_list[pidx]['theta_0']
            sc      = self.scales[pidx]
            # Per-pulsar name list — self.sample_list is a list of lists.
            theta_p = {k: reparametrise(k, theta_0[k], z_dict[f'z_{k};{pidx}'] * sc[k])
                       for k in self.sample_list[pidx]}
            tm_res = self.delta_m_list[pidx](theta_p) * 1e-6
            parts.append(self.raw_residuals[pidx] - tm_res)
        return jnp.concatenate(parts)

    # ------------------------------------------------------------------
    # Production numpyro timing block — bounded proper priors
    # ------------------------------------------------------------------

    def _prepare_family_index(self):
        """Static (Python-level) grouping of every (pulsar, param) into its
        bounded-prior family.  Called once at construction — no JAX tracing here.
        """
        families = {'SINI': [], 'ECC': [], 'M2': [], 'PX': [], 'affine': []}
        slot_map = []  # slot_map[pidx] = [(family, idx_in_family), …] in sample order

        for pidx, sl in enumerate(self.sample_list):
            theta_0 = self.aux_list[pidx]['theta_0']
            sig     = self.aux_list[pidx]['sigJUG']
            pulsar_slots = []
            for k in sl:
                fam = k if k in ('SINI', 'ECC', 'M2', 'PX') else 'affine'
                idx = len(families[fam])
                families[fam].append((pidx, k, theta_0[k],
                                      sig[k] if fam == 'affine' else None))
                pulsar_slots.append((fam, idx))
            slot_map.append(pulsar_slots)

        self._fam_entries = families   # raw (pidx, k, theta0, sigma) — for init_values
        self._fam_size    = {f: len(v) for f, v in families.items()}
        self._fam_theta0  = {f: jnp.asarray([e[2] for e in v])
                             for f, v in families.items() if v}
        self._fam_sigma   = ({'affine': jnp.asarray([e[3] for e in families['affine']])}
                             if families['affine'] else {})
        self._slot_map    = slot_map

    # ------------------------------------------------------------------
    # Dense-mass-matrix preconditioning (ported from the single-pulsar
    # JointProblem) — per-pulsar OFFSET/AFFINE/BOUND routing + a frozen
    # dense NUTS mass matrix derived from JUG's own covariance.  This is
    # what fixes NUTS step-size collapse: instead of letting warmup LEARN
    # the parameter scale/orientation from scratch, it starts from JUG's
    # actual covariance.  Cross-pulsar timing correlations are zero
    # (independent TOA sets), so a per-pulsar dense block is the correct
    # structure, not an approximation — and much cheaper than one dense
    # matrix over every pulsar's params combined.
    # ------------------------------------------------------------------

    def prepare_dense_metric(self, bound=DEFAULT_BOUND):
        """Route params into OFFSET/AFFINE/BOUND and build the dense metric.

        Call once (after __init__) if you want to use build_gibbs_model /
        run_gibbs instead of the family-batched sample_residuals path.

        ``bound`` names the params given bounded physical priors in
        build_gibbs_model; only M2/SINI/ECC are implemented there, so anything
        else is rejected rather than silently frozen at its par value.
        """
        bad = [k for k in bound if k not in _GIBBS_BOUND_SUPPORTED]
        if bad:
            raise ValueError(
                f"bound={tuple(bound)} contains {bad}, which build_gibbs_model "
                f"does not implement (supported: {_GIBBS_BOUND_SUPPORTED}).  A "
                "bound param with no branch there would be silently held at its "
                "par value.")

        self.bound_names = tuple(bound)
        self.OFFSET, self.AFFINE, self.BOUND = [], [], []
        self.metric_cols = []
        self.timing_metric_list, self.metric_order_list, self.metric_sites = [], [], []

        for pidx in range(self.npulsars):
            delta_m = self.delta_m_list[pidx]
            aux     = self.aux_list[pidx]
            sl      = self.sample_list[pidx]
            th0     = aux['theta_0']
            sigJUG  = aux['sigJUG']
            Cjug    = aux['Cjug']
            fittable_order = aux['fittable_order']
            base_sampled = {k: jnp.asarray(float(th0[k])) for k in sl}

            bound_p  = [k for k in self.bound_names if k in sl]
            offset_p = [k for k in sl if k not in bound_p and
                        route(k, float(th0[k]), sigJUG[k], delta_m,
                              base_sampled)["bucket"] == "offset"]
            affine_p = [k for k in sl if k not in offset_p + bound_p]
            cols_p   = P.offset_columns(delta_m, base_sampled, offset_p) if offset_p else {}

            # Metric order must be exactly the union of the three buckets, with
            # no duplicates.  Using a hard-coded ("M2","SINI","ECC") tail here
            # duplicates any of those params that is NOT bound (it is then also
            # in affine_p), which corrupts Cphys and emits a repeated site name.
            order = offset_p + affine_p + bound_p
            idx   = [fittable_order.index(k) for k in order]
            Cphys = Cjug[np.ix_(idx, idx)]

            specs = {k: _BOUND_SPECS[k] for k in bound_p if k in _BOUND_SPECS}
            if "SINI" in bound_p:
                SINI_0 = float(th0["SINI"])
                specs["SINI"] = ("sini",
                                 float(np.sqrt(max(1.0 - SINI_0 ** 2, 0.0))),
                                 SINI_0)

            D = P.coordinate_jacobian(order, {k: float(th0[k]) for k in order},
                                      sigJUG, specs)
            metric = P.jug_metric(Cphys, D) if order else None
            sites  = [f"p{pidx}_{self._site_of(k, offset_p, bound_p)}" for k in order]

            self.OFFSET.append(offset_p)
            self.AFFINE.append(affine_p)
            self.BOUND.append(bound_p)
            self.metric_cols.append(cols_p)
            self.timing_metric_list.append(metric)
            self.metric_order_list.append(order)
            self.metric_sites.append(sites)

    @staticmethod
    def _site_of(k, offset_p, bound_p):
        """Site name for param ``k`` given its bucket (must match build_gibbs_model)."""
        if k in offset_p:
            return f"u_{k}"
        if k in bound_p:
            return "cosi" if k == "SINI" else k
        return f"z_{k}"

    def _require_dense_metric(self, caller):
        """Raise unless prepare_dense_metric() has been called."""
        if self.OFFSET is None:
            raise RuntimeError(
                f"{caller} requires prepare_dense_metric() to have been called "
                "first (it builds the OFFSET/AFFINE/BOUND routing and the frozen "
                "dense mass matrix).")

    def build_gibbs_model(self, noise_loglik, sample_noise=True,
                          fixed_noise_sites=None, z_prior_sd=1.0):
        """One NumPyro model: per-pulsar OFFSET/AFFINE/BOUND timing sites
        (preconditioned by prepare_dense_metric's dense blocks) → concatenated
        residual → ``noise_loglik(res_parts, sample_noise, fixed_noise_sites)``.
        """
        self._require_dense_metric("build_gibbs_model")

        def model():
            """NumPyro model: timing sites -> residuals -> noise log-likelihood."""
            res_parts = []
            for pidx in range(self.npulsars):
                th0    = self.aux_list[pidx]['theta_0']
                sigJUG = self.aux_list[pidx]['sigJUG']
                offset_p = self.OFFSET[pidx]
                affine_p = self.AFFINE[pidx]
                bound_p  = self.BOUND[pidx]
                cols, pfx = self.metric_cols[pidx], f"p{pidx}_"

                theta, off = {}, 0.0
                for k in offset_p:
                    u = numpyro.sample(f"{pfx}u_{k}", dist.Normal(0., z_prior_sd))
                    off = off + u * sigJUG[k] * cols[k]
                    numpyro.deterministic(f"{pfx}d{k}", u * sigJUG[k])
                for k in affine_p:
                    z = numpyro.sample(f"{pfx}z_{k}", dist.Normal(0., z_prior_sd))
                    theta[k] = float(th0[k]) + z * sigJUG[k]
                    numpyro.deterministic(f"{pfx}{k}", theta[k])
                for k in bound_p:   # prepare_dense_metric guarantees these 3 only
                    if k == "SINI":
                        cosi = numpyro.sample(f"{pfx}cosi", dist.Uniform(-1., 1.))
                        theta["SINI"] = jnp.sqrt(jnp.clip(1 - cosi ** 2, 1e-12, 1.))
                        numpyro.deterministic(f"{pfx}SINI", theta["SINI"])
                    elif k == "M2":
                        theta["M2"] = numpyro.sample(f"{pfx}M2",
                                                     dist.Uniform(*_M2_BOUNDS))
                    elif k == "ECC":
                        theta["ECC"] = numpyro.sample(f"{pfx}ECC",
                                                      dist.Uniform(0., _ECC_HI))

                th = {k: jnp.asarray(float(th0[k])) for k in self.sample_list[pidx]}
                th.update(theta)
                tm_res = (self.delta_m_list[pidx](th) + off) * 1e-6
                res_parts.append(self.raw_residuals[pidx] - tm_res)

            numpyro.factor("lnpost",
                           noise_loglik(res_parts, sample_noise, fixed_noise_sites))

        return model

    def gibbs_init_values(self):
        """init_to_value dict matching build_gibbs_model's sites."""
        self._require_dense_metric("gibbs_init_values")
        iv = {}
        for pidx in range(self.npulsars):
            th0, pfx = self.aux_list[pidx]['theta_0'], f"p{pidx}_"
            iv.update({f"{pfx}u_{k}": 0.0 for k in self.OFFSET[pidx]})
            iv.update({f"{pfx}z_{k}": 0.0 for k in self.AFFINE[pidx]})
            if "SINI" in self.BOUND[pidx]:
                SINI_0 = float(th0["SINI"])
                iv[f"{pfx}cosi"] = float(np.sqrt(max(1 - SINI_0 ** 2, 0.0)))
            if "M2" in self.BOUND[pidx]:
                iv[f"{pfx}M2"] = float(th0["M2"])
            if "ECC" in self.BOUND[pidx]:
                iv[f"{pfx}ECC"] = float(th0["ECC"])
        return iv

    def run_gibbs(self, model, noise_site_group=None, key=1, num_warmup=700,
                  num_samples=1500, num_chains=2, target_accept_prob=0.9,
                  progress_bar=False):
        """2-block MultiHMCGibbs: [all pulsars' timing sites, block-diagonal
        frozen dense mass | noise sites, adaptive].
        """
        self._require_dense_metric("run_gibbs")

        iv = self.gibbs_init_values()
        dense_mass_groups = [tuple(s) for s in self.metric_sites if s]
        inv_mass = {tuple(s): jnp.asarray(m)
                    for s, m in zip(self.metric_sites, self.timing_metric_list)
                    if s and m is not None}
        all_timing_sites = [s for group in self.metric_sites for s in group]

        kernels, groups = [], []
        if all_timing_sites:
            kernels.append(NUTS(model, target_accept_prob=target_accept_prob,
                                max_tree_depth=9,
                                dense_mass=dense_mass_groups,
                                inverse_mass_matrix=inv_mass,
                                adapt_mass_matrix=False,
                                init_strategy=init_to_value(values=iv)))
            groups.append(all_timing_sites)
        if noise_site_group:
            kernels.append(NUTS(model, target_accept_prob=target_accept_prob,
                                max_tree_depth=8, dense_mass=False,
                                init_strategy=init_to_value(values=iv)))
            groups.append(list(noise_site_group))
        if not kernels:
            raise ValueError("run_gibbs: no timing sites and no noise_site_group "
                             "— nothing to sample.")

        mc = MCMC(MultiHMCGibbs(kernels, groups), num_warmup=num_warmup,
                  num_samples=num_samples, num_chains=num_chains,
                  chain_method="sequential", progress_bar=progress_bar)
        mc.run(jrandom.PRNGKey(key))
        return mc

    # ------------------------------------------------------------------
    # Family-batched / per-site numpyro blocks
    # ------------------------------------------------------------------

    def residuals_hybrid(self, affine_z_sd=100.0):
        """Bounded no-Jacobian priors for SINI/ECC/M2/PX (avoids the tanh/log
        saturation that pathologically shrinks NUTS' step size in the
        multi-pulsar case); a single batched affine Normal for everything else.
        Priors match sample_timing_theta exactly, just batched by family
        instead of by (pulsar, param).

        Returns
        -------
        res : array [total_ntoas]
        """
        fam_vals = {}

        # --- bounded families: no saturating transform, no Jacobian needed ---
        if self._fam_size['SINI']:
            cosi = numpyro.sample(
                'cosi_batch', dist.Uniform(-1.0, 1.0).expand([self._fam_size['SINI']]))
            fam_vals['SINI'] = jnp.sqrt(jnp.clip(1.0 - cosi ** 2, 1e-12, 1.0))
            numpyro.deterministic('SINI_batch', fam_vals['SINI'])
        if self._fam_size['ECC']:
            fam_vals['ECC'] = numpyro.sample(
                'ECC_batch', dist.Uniform(0.0, _ECC_HI).expand([self._fam_size['ECC']]))
        if self._fam_size['M2']:
            fam_vals['M2'] = numpyro.sample(
                'M2_batch', dist.Uniform(*_M2_BOUNDS).expand([self._fam_size['M2']]))
        if self._fam_size['PX']:
            fam_vals['PX'] = numpyro.sample(
                'PX_batch', dist.Uniform(*_PX_BOUNDS).expand([self._fam_size['PX']]))

        # --- affine class: single flat vector, cheap to trace/compile ---
        if self._fam_size['affine']:
            z = numpyro.sample(
                'z_affine_batch',
                dist.Normal(0.0, affine_z_sd).expand([self._fam_size['affine']]))
            fam_vals['affine'] = self._fam_theta0['affine'] + z * self._fam_sigma['affine']
            numpyro.deterministic('theta_affine_batch', fam_vals['affine'])

        parts = []
        for pidx in range(self.npulsars):
            theta_p = {k: fam_vals[fam][idx]
                       for k, (fam, idx) in zip(self.sample_list[pidx],
                                                self._slot_map[pidx])}
            tm_res = self.delta_m_list[pidx](theta_p) * 1e-6
            parts.append(self.raw_residuals[pidx] - tm_res)
        return jnp.concatenate(parts)

    def sample_residuals(self):
        """numpyro block: sample every pulsar's timing params from the canonical
        bounded proper priors and return the concatenated stochastic residuals
        (seconds).  This is the production timing Gibbs group; it imposes the
        SAME bounded priors as the validation harnesses via ``sample_timing_theta``
        and uses NO Jacobian factor.  Call inside a numpyro model.

        Returns
        -------
        res : array [total_ntoas]   r_obs_p - delta_m_p(theta_p), per pulsar.
        """
        parts = []
        for pidx in range(self.npulsars):
            theta_p = sample_timing_theta(
                self.sample_list[pidx],
                self.aux_list[pidx]['theta_0'],
                self.aux_list[pidx]['sigJUG'],   # NOT mle['sigma'] — that collapses
                prefix=f'p{pidx}_',
            )
            tm_res = self.delta_m_list[pidx](theta_p) * 1e-6              # µs → s
            parts.append(self.raw_residuals[pidx] - tm_res)
        return jnp.concatenate(parts)

    def sample_training_residuals(self, key, num_samples, pulsar_index,
                                  z_scale=100.0, affine_z_sd_bounded= 1, return_draws=False, perturb_bounded=False,
                                 override = None):
        """Pure-JAX prior-predictive draws of ONE pulsar's timing-model delay.

        Draws ``num_samples`` parameter vectors from the same bounded priors as
        ``sample_timing_theta`` and evaluates delta_m for each.

        Returns
        -------
        tm_res : array [num_samples, n_toa_p]   model delay in seconds
        draws  : dict, only if ``return_draws`` — the sampled sites, including
                 the underlying z_/cosi_ coordinates.
        """
        draws = sample_timing_theta_batched(
            key,
            self.sample_list[pulsar_index],
            self.aux_list[pulsar_index]['theta_0'],
            self.aux_list[pulsar_index]['sigJUG'],
            num_samples=num_samples,
            affine_z_sd=z_scale,
            affine_z_sd_bounded = affine_z_sd_bounded,
            perturb_bounded=perturb_bounded,
            override = override
        )
        # Keep ONLY the physical param names: the batched sampler also returns
        # the 'z_*'/'cosi_*' sampler coordinates, and feeding those extra keys
        # into the jitted delta_m changes its input pytree (forcing a retrace,
        # and breaking outright if it ever validates its keys).
        theta_draws = {k: draws[k] for k in self.sample_list[pulsar_index]}
        tm_res = jax.vmap(self.delta_m_list[pulsar_index])(theta_draws) * 1e-6
        return (tm_res, draws) if return_draws else tm_res

    def init_values(self):
        """init_to_value dict matching the batched residuals_hybrid priors.
        Same clipping/fallback logic as timing_init_values, just vectorised
        per family instead of per (pulsar, param).
        """
        iv = {}
        if self._fam_size['SINI']:
            iv['cosi_batch'] = jnp.asarray([
                float(np.sqrt(max(1.0 - theta0 ** 2, 0.0)))
                for (_, _, theta0, _) in self._fam_entries['SINI']
            ])
        if self._fam_size['ECC']:
            iv['ECC_batch'] = jnp.asarray([
                float(min(max(theta0, 0.0), _ECC_HI))
                for (_, _, theta0, _) in self._fam_entries['ECC']
            ])
        if self._fam_size['M2']:
            iv['M2_batch'] = jnp.asarray([
                float(min(max(theta0, _M2_BOUNDS[0] + 1e-6), _M2_BOUNDS[1] - 1e-6))
                for (_, _, theta0, _) in self._fam_entries['M2']
            ])
        if self._fam_size['PX']:
            iv['PX_batch'] = jnp.asarray([
                float(min(max(theta0, _PX_BOUNDS[0] + 1e-6), _PX_BOUNDS[1] - 1e-6))
                for (_, _, theta0, _) in self._fam_entries['PX']
            ])
        if self._fam_size['affine']:
            iv['z_affine_batch'] = jnp.zeros(self._fam_size['affine'])
        return iv


# ---------------------------------------------------------------------------
# Constructor helper
# ---------------------------------------------------------------------------

def build_multi_psr_timing_model(parfiles, timfiles, sample_list, data,
                                 load_how_many_in_parallel=1,
                                 theta_0_overrides=None):
    """Set up a MultiPsrTimingModel from par/tim file lists.

    Drop-in replacement for the unoptimised setup loop:

        delta_m_us, aux = [], []
        for parfile, timfile, sl in zip(parfiles, timfiles, sample_list):
            dm, a = setup_timing_model(parfile, timfile, sl)
            delta_m_us.append(dm); aux.append(a)

    Parameters
    ----------
    parfiles, timfiles : list of str
    sample_list : list of list of str, or list of str
        Per-pulsar parameter lists, e.g. ``[['F0','F1'], ['F0','M2']]``.  A flat
        list of names (``['F0','F1','DM']``) is broadcast to every pulsar.
    data : object with ``raw_residuals``
    load_how_many_in_parallel : int
        >1 uses joblib.  NOTE: setup_timing_model returns jit-compiled closures,
        which process-based joblib backends cannot pickle; use a threading
        backend or leave this at 1 if you hit a pickling error.

    Returns
    -------
    model : MultiPsrTimingModel
    """
    # Normalise so both branches can always zip over theta_0_overrides.
    if theta_0_overrides is None:
        theta_0_overrides = [None] * len(parfiles)

    npsr = len(parfiles)
    if len(timfiles) != npsr:
        raise ValueError(f"parfiles/timfiles length mismatch: {npsr} vs {len(timfiles)}")
    # Broadcast a flat name list to every pulsar — passing a flat list straight
    # through would be silently misread as one "list" per character position.
    if sample_list and all(isinstance(s, str) for s in sample_list):
        sample_list = [list(sample_list) for _ in range(npsr)]
    else:
        sample_list = [list(sl) for sl in sample_list]
    if len(sample_list) != npsr:
        raise ValueError(f"sample_list must have one entry per pulsar "
                         f"({npsr}); got {len(sample_list)}.")

    njobs = int(load_how_many_in_parallel)

    if njobs > 1:
        from tqdm_joblib import ParallelPbar
        from joblib import delayed
        results = ParallelPbar("Loading the par and tim files...")(n_jobs=njobs)(
            delayed(setup_timing_model)(par, tim, sl, theta0)
            for par, tim, sl, theta0 in zip(parfiles, timfiles, sample_list, theta_0_overrides)
        )
        # Plain unzip — np.array(..., dtype=object) on a list of
        # (callable, dict) pairs is shape-fragile and can raise or silently
        # produce a 1-D object array.
        delta_m_list = [r[0] for r in results]
        aux_list     = [r[1] for r in results]
    else:
        delta_m_list, aux_list = [], []
        for par, tim, sl, theta0 in zip(parfiles, timfiles, sample_list, theta_0_overrides):
            dm, a = setup_timing_model(par, tim, sl, theta0)
            delta_m_list.append(dm)
            aux_list.append(a)

    # Per-pulsar scales: JUG formal σ for affine params (physical step per unit
    # z).  Sourced from aux['sigJUG'] (JUG's stable uncertainties), NOT
    # aux['mle']['sigma'] whose naive pinv collapses to ~1e-37× the true σ and
    # freezes the param at θ₀.  Reparametrised params (SINI/ECC/M2/PX) are sampled
    # in an UNCONSTRAINED coordinate u = z·scale that reparametrise() maps to the
    # physical value; their physical σ (often ~1e-3) would make u·scale tiny and
    # freeze the param at θ₀, so use a unit scale → O(1) exploration of the full
    # physical domain.
    scales = [
        {k: (1.0 if k in _REPARAM_PARAMS else float(aux['sigJUG'][k]))
         for k in sl}
        for aux, sl in zip(aux_list, sample_list)
    ]

    return MultiPsrTimingModel(delta_m_list, aux_list, sample_list, scales, data=data)