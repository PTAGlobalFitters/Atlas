"""
Utilities for constructing a stochastic gravitational-wave background (GWB)
from a superposition of individual continuous-wave (CW) sources in a pulsar
timing array (PTA) setting.

Notes
-----
- Time and distance conventions:
  * `toas` are assumed to be in seconds.
  * pulsar distances are drawn in kpc and converted to light-travel time (seconds).
- Frequencies:
  * `log10_fgw` is log10 of the *observer-frame* GW frequency in Hz.
  * Internally we use the orbital angular frequency w0 = pi * fgw, because
    fgw = 2 * f_orb and omega_orb = 2*pi*f_orb = pi*fgw.
- The FFT grid:
  * CW residuals are evaluated on a sparse uniform grid per pulsar with
    (2*CW_bins + 2) samples, then FFT is taken and converted to sine/cos
    coefficients consistent with a real Fourier series representation.
- JAX:
  * Several methods are `jax.jit` compiled. Inputs should be JAX arrays when
    calling those compiled methods.
"""

import numpy as np
from tqdm.auto import trange
from functools import partial
from torch.quasirandom import SobolEngine
import jax
import jax.numpy as jnp
import jax.random as jr
import torch
import math
import random
from ATLAS.experimental.flows import ConditionalFlow, Flow

# -----------------------------
# Astronomical / physical constants
# -----------------------------
c = 299792458.0  # speed of light [m/s]
Tsun = 4.9254909476412675e-06  # G*M_sun/c^3 [s]
kpc = 3.085677581491367e19  # kiloparsec [m]
Mpc = 3.085677581491367e22  # megaparsec [m]

# Reference time used to shift TOAs in CW residual evaluation.
# Shifting can improve numerical stability when times are large.
tref = 1e9  # [s]


def shape_maker(x, feature_size, shape):
    """
    Broadcast an array into a common batch shape.

    Parameters
    ----------
    x : array-like
        Input array to broadcast.

    feature_size : int
        Number of feature dimensions appended to the broadcasted tensor.
        A value of 1 is treated as a scalar feature.

    shape : sequence of int
        Desired batch dimensions.

    Returns
    -------
    jax.Array
        Broadcasted tensor with trailing feature dimension(s).
    """
    if feature_size == 1:
        return jnp.broadcast_to(x, shape)[..., None]
    else:
        return jnp.broadcast_to(x, [*shape] + [feature_size])

def make_right_shape(
    arr,
    n_draws_compact,
    NUM_FREQS,
    SAM_SHAPE,
    n_draws_poisson,
    non_zero_sources):
    """
    Broadcast, reorder, and compact an array of shape (N_Mtot, N_Mratio, N_redshift, Nfreq) into 
    a 1-D array of non-zero values.

    Parameters
    ----------
    arr : array_like
        Input array defined on the masked sampling grid with trailing frequency axis.
        Expected shape is compatible with `masked_shape`:

            masked_shape = (SAM_SHAPE[0]-1, SAM_SHAPE[1]-1, SAM_SHAPE[2]-1, NUM_FREQS)

        i.e. arr should be broadcastable to that shape.
    n_draws_compact : int
        Total number of compacted grid points (sources) after flattening the masked
        3D grid. In most usages:

            n_draws_compact == (SAM_SHAPE[0]-1) * (SAM_SHAPE[1]-1) * (SAM_SHAPE[2]-1)

    NUM_FREQS : int
        Number of frequency bins.
    SAM_SHAPE : tuple of int
        Original sampling grid shape, typically 3D (e.g., (Nx, Ny, Nz)).
        This function uses a masked version of that grid with size reduced by 1
        along each of the first three axes.
    n_draws_poisson : int
        Number of Poisson realizations/draws to replicate across.
    non_zero_sources : array_like (bool or int)
        Mask or index array selecting which compacted sources are "active".
        - If boolean mask: shape should be (n_draws_compact,) and True keeps a source.
        - If integer indices: selects specific source rows.

    Returns
    -------
    out : jax.numpy.ndarray, shape (Nactive, NUM_FREQS, n_draws_poisson)
        Compacted array filtered to active sources, where:
          - Nactive = sum(non_zero_sources) if boolean, else len(non_zero_sources)
          - axis 0 indexes sources (after compaction + filtering)
          - axis 1 indexes frequency bins
          - axis 2 indexes Poisson draws

    Steps performed
    ---------------
    1) Build the masked grid shape:
         (SAM_SHAPE[0]-1, SAM_SHAPE[1]-1, SAM_SHAPE[2]-1, NUM_FREQS)
    2) Broadcast `arr` to include a leading Poisson-draw axis:
         (n_draws_poisson, *masked_shape)
    3) Transpose so Poisson axis is last:
         (SAM_SHAPE[0]-1, SAM_SHAPE[1]-1, SAM_SHAPE[2]-1, NUM_FREQS, n_draws_poisson)
    4) Flatten masked grid axes into one "compact source" axis:
         (n_draws_compact, NUM_FREQS, n_draws_poisson)
    5) Filter to `non_zero_sources` and convert to JAX array.
    """
    masked_shape = (SAM_SHAPE[0] - 1, SAM_SHAPE[1] - 1, SAM_SHAPE[2] - 1, NUM_FREQS)

    # Replicate across Poisson draws (leading axis)
    mod_arr = np.broadcast_to(arr, (n_draws_poisson, *masked_shape))

    # Move Poisson draw axis to the end to match final desired layout
    mod_arr = mod_arr.transpose((1, 2, 3, 4, 0))

    # Flatten masked 3D grid to a compact "source" axis
    mod_arr = mod_arr.reshape(n_draws_compact, NUM_FREQS, n_draws_poisson)

    # Keep only active/non-zero sources
    mod_arr = mod_arr[non_zero_sources]

    return jnp.array(mod_arr)

class AstroGWB(object):
    """
    Construct a GWB-like signal in a PTA by summing many individual CW sources.

    The main entrypoint is `create_gwb`, which returns accumulated Fourier
    coefficients for the PTA across frequency bins, built by summing the
    coefficients from individual sources.

    Parameters
    ----------
    Npulsars : int
        Number of pulsars in the PTA.
    toas : sequence of array_like
        Per-pulsar time-of-arrival arrays [seconds]. Shape is (Npulsars,) with
        variable-length arrays per pulsar.
    N : sequence
        (Project-specific) list/array of per-pulsar design/observation metadata.
        This code casts each element to a JAX array and stores it.
    CW_bins : int
        Number of positive-frequency bins used for CW Fourier representation.
        Internally kmax_CW = 2*CW_bins corresponds to sine+cos coefficients.
    psr_pos : array_like, shape (Npulsars, 3)
        Pulsar unit vectors in Cartesian coordinates (x, y, z).
    psr_dist_mean : array_like, shape (Npulsars,)
        Mean pulsar distances in kpc (used as Gaussian mean in draws).
    psr_dist_sigma : array_like, shape (Npulsars,)
        Pulsar distance uncertainties in kpc (used as Gaussian sigma in draws).
    Tspan : float
        Total PTA timespan [seconds].
    """

    def __init__(
        self,
        Npulsars,
        toas,
        CW_bins,
        psr_pos,
        psr_dist_mean,
        psr_dist_sigma,
        Tspan,
        n_ast_pars,
        thethree_batch_size = 2**3,
        geo_batch_size = 2**4,
        smallest_cw_coeff_amplitude_allowed = 1e-30,
        pytorch_device = 'cuda'):

        self.cw_eps = smallest_cw_coeff_amplitude_allowed
        self.n_ast_pars = n_ast_pars
        # -----------------------------------------------------------------------------
        # Batch sizes
        # -----------------------------------------------------------------------------
        # Independent Monte Carlo dimensions.
        self.piosson_batch_size = 1
        self.thethree_batch_size = thethree_batch_size
        self.geo_batch_size = geo_batch_size
        self.pulsar_batch_size = self.geo_batch_size
        # The right ordering: (Poisson, Geometry, Pulsar, Astrophysical realization)
        self.S = [
            self.piosson_batch_size,
            self.geo_batch_size,
            self.pulsar_batch_size,
            self.thethree_batch_size,
        ]
        self.total_sample_size = np.prod(self.S)

        # Alternative layouts used before transposing into the right ordering.
        self.S_geo = [
            self.piosson_batch_size,
            self.pulsar_batch_size,
            self.thethree_batch_size,
            self.geo_batch_size,
        ]

        self.S_pulsar = [
            self.piosson_batch_size,
            self.thethree_batch_size,
            self.geo_batch_size,
            self.pulsar_batch_size,
        ]

        self.S_piosson = [
            self.geo_batch_size,
            self.pulsar_batch_size,
            self.thethree_batch_size,
            self.piosson_batch_size,
        ]
        # Pulsar distance prior parameters (kpc)
        self.psr_dist_mean = psr_dist_mean[None, None, None, None, None, :, None]
        self.psr_dist_sigma = psr_dist_sigma[None, None, None, None, None, :, None]

        # Time-of-arrival arrays [s]
        self.toas = toas

        # Total PTA baseline [s]
        self.Tspan = Tspan

        # Number of pulsars
        self.Npulsars = Npulsars

        # Pulsar sky positions as Cartesian unit vectors
        self.psr_pos = jnp.array(psr_pos)[None, None, None, None, None, :, None, :]

        # Fourier-bin settings
        self.CW_bins = CW_bins
        self.kmax_CW = (
            2 * self.CW_bins
        )  # number of sine+cos coefficients per pulsar (excluding DC)

        # Individual pulsar spans [s]
        self.ind_Tspan = jnp.array(
            [self.toas[idx][-1] - self.toas[idx][0] for idx in range(self.Npulsars)]
        )

        # Sparse uniform TOA grid per pulsar for FFT extraction
        # Nsparse = 2*CW_bins + 2 so that (Nsparse//2) = CW_bins + 1 includes Nyquist handling
        self.sparse_toas_CW = jnp.array(
            [
                np.linspace(
                    self.toas[idx][0],
                    self.toas[idx][-1],
                    2 * self.CW_bins + 2,
                    endpoint=False,
                )
                for idx in range(self.Npulsars)
            ]
        )[None, None, None, None, None,:, :]
        self.Nsparse = self.sparse_toas_CW.shape[-1]

        # FFT frequency arrays (cycles / second) per pulsar based on each pulsar's span
        self.freqs_forFFT = jnp.array(
            [
                jnp.fft.fftfreq(self.Nsparse, self.ind_Tspan[idx] / self.Nsparse)
                for idx in range(self.Npulsars)
            ]
        )
        
        self.dtype = torch.float64
        self.device = pytorch_device
        # -------------------------
        # Priors / bounds for geometric parameters
        # -------------------------
        # The "geometric" parameters drawn are (the order is important!):
        #   cos_gwtheta     : cosine of inclination, in [-1, 1]
        #   psi             : polarization angle, in [-pi/2, pi/2]
        #   cos_inc         : cosine of GW source colatitude, in [-1, 1]
        #   gwphi           : GW source longitude, in [0, 2pi]
        #   phase0          : initial GW phase (here treated as orbital-phase *2), in [-pi/2, pi/2]
        #
        # Additionally, we draw per-pulsar "pulsar phases" uniformly in [0, 2pi],
        # and per-pulsar distances from a Gaussian.
        self.pmin = jnp.array([-1.0, -np.pi / 2, -1.0, 0.0, -np.pi / 2])
        self.pmax = jnp.array([1.0, np.pi / 2, 1.0, np.pi * 2, np.pi / 2])
        self.pmin_torch = torch.tensor([-1.0, -np.pi / 2, -1.0, 0.0, -np.pi / 2], dtype = self.dtype, device = self.device)
        self.pmax_torch = torch.tensor([1.0, np.pi / 2, 1.0, np.pi * 2, np.pi / 2], dtype = self.dtype, device = self.device)

        self.psr_dist_mean_torch = torch.tensor(np.array(self.psr_dist_mean), dtype = self.dtype, device = self.device)
        self.psr_dist_sigma_torch = torch.tensor(np.array(self.psr_dist_sigma), dtype = self.dtype, device = self.device)
        
        self._num_geo = self.pmin.shape[0] 

        # -------------------------
        # QMC / SOBOL 
        # -------------------------
        self._sobol = SobolEngine(
            dimension=self._num_geo + 2 * self.Npulsars,
            scramble=True,
        )
        # Clamp away from 0/1 to keep icdf finite
        self.eps = torch.finfo(self.dtype).eps
        
    def draw_from_astro_uninformed_params_torch(self, number_of_copies):
        """
        Draw geometric source parameters, pulsar phases, and pulsar distances
        all from a scrambled Sobol QMC sequence.
    
        The Sobol engine must be initialised with
        ``dimension = _num_geo + 2 * Npulsars`` dimensions:
          - [:_num_geo]                     → geometric parameters
          - [_num_geo : _num_geo+Npulsars]  → pulsar phases
          - [_num_geo+Npulsars : ...]       → pulsar distance quantiles
    
        Parameters
        ----------
        number_of_copies : int
            Number of source copies.
    
        Returns
        -------
        geo   : JAX array, shape (number_of_copies, 1, 1, 1, 1, 1, 1, _num_geo)
        dist  : JAX array, shape (number_of_copies, 1, 1, 1, 1, Npulsars, 1)
        phase : JAX array, shape (number_of_copies, 1, 1, 1, 1, Npulsars, 1)
        """
        # unit: (number_of_copies, total_dims) in [0, 1)
        unit = self._sobol.draw(number_of_copies, dtype=self.dtype).to(self.device)
    
        # ------------------------------------------------------------------
        # Geometric parameters  [0 : _num_geo]
        # ------------------------------------------------------------------
        geo_flat = self.pmin_torch + unit[:, :self._num_geo] * (self.pmax_torch - self.pmin_torch)
        geo = geo_flat.reshape(number_of_copies, 1, 1, 1, 1, 1, 1, self._num_geo)
    
        # ------------------------------------------------------------------
        # Pulsar phases  [_num_geo : _num_geo + Npulsars]  →  Uniform[0, 2π)
        # ------------------------------------------------------------------
        i0 = self._num_geo
        i1 = i0 + self.Npulsars
        phase = (2.0 * math.pi * unit[:, i0:i1]).reshape(
            number_of_copies, 1, 1, 1, 1, self.Npulsars, 1
        )
    
        # ------------------------------------------------------------------
        # Pulsar distances  [_num_geo + Npulsars : ...]  →  Gaussian via icdf
        # ------------------------------------------------------------------
        dist_u = unit[:, i1 : i1 + self.Npulsars]
        dist_u = unit[:, i1 : i1 + self.Npulsars].clamp(self.eps, 1.0 - self.eps)
        z = (torch.erfinv(2.0 * dist_u - 1.0) * math.sqrt(2.0))[:, None, None, None, None, :, None]
        dist = (
            self.psr_dist_mean_torch + z * self.psr_dist_sigma_torch
        )
    
        return (
            jax.dlpack.from_dlpack(geo,   copy=False),
            jax.dlpack.from_dlpack(dist,  copy=False),
            jax.dlpack.from_dlpack(phase, copy=False),
        )

        
    def draw_from_astro_uninformed_params_given_number_of_sources(self, rng_keys, number_of_sources):
        """
        Fast, GPU-friendly draw of geometric params + pulsar distances with *static* shapes.

        This avoids recompiles when `n_sources` varies by always drawing `max_sources`
        and masking out the unused rows.

        Parameters
        ----------
        rng_keys : jax.random.PRNGKey
            Base PRNG key (we split internally).
        number_of_sources : int
            Number of active sources for this draw.
        """

        # Draw geo params (incl. pulsar phases) and pulsar distances
        geo = jr.uniform(
            rng_keys[0],
            minval=self.pmin,
            maxval=self.pmax,
            shape=(number_of_sources, 1, 1, 1, 1, 1, 1, self._num_geo),
        )

        # Uniform pulsar phase draws
        phase = jr.uniform(
            rng_keys[1],
            minval=0,
            maxval=2 * jnp.pi,
            shape=(1, 1, 1, 1, 1, self.Npulsars, 1),
        )

        # Gaussian pulsar distance draws around mean/sigma (kpc)
        eps = jr.normal(
            rng_keys[2],
            shape=(1, 1, 1, 1, 1, self.Npulsars, 1),
        )
        dist = self.psr_dist_mean + eps * self.psr_dist_sigma

        return geo, dist, phase


    @partial(jax.jit, static_argnums=(0,))
    def create_gw_antenna_pattern(self, gwtheta, gwphi):
        """
        Compute PTA antenna pattern factors for a GW source at (gwtheta, gwphi).

        Parameters
        ----------
        gwtheta : float
            Source colatitude (theta) [radians], in [0, pi].
        gwphi : float
            Source longitude (phi) [radians], in [0, 2*pi).

        Returns
        -------
        fplus : jax.numpy.ndarray, shape (Npulsars,)
            Plus-polarization antenna factor for each pulsar.
        fcross : jax.numpy.ndarray, shape (Npulsars,)
            Cross-polarization antenna factor for each pulsar.
        cosMu : jax.numpy.ndarray, shape (Npulsars,)
            cos(mu) where mu is the angle between GW propagation direction and pulsar direction.
            Used in the pulsar term time shift: tp = t - L(1-cosMu).

        Notes
        -----
        Uses conventions consistent with Sesana et al. (2010) and Ellis et al. (2012).
        """
        sgwphi = jnp.sin(gwphi)
        cgwphi = jnp.cos(gwphi)
        sgwtheta = jnp.sin(gwtheta)
        cgwtheta = jnp.cos(gwtheta)

        mdotpos = sgwphi * self.psr_pos[..., 0] - cgwphi * self.psr_pos[..., 1]
        ndotpos = (
            -cgwtheta * cgwphi * self.psr_pos[..., 0]
            - cgwtheta * sgwphi * self.psr_pos[..., 1]
            + sgwtheta * self.psr_pos[..., 2]
        )
        omhatdotpos = (
            -sgwtheta * cgwphi * self.psr_pos[..., 0]
            - sgwtheta * sgwphi * self.psr_pos[..., 1]
            - cgwtheta * self.psr_pos[..., 2]
        )

        fplus = 0.5 * (mdotpos**2 - ndotpos**2) / (1 + omhatdotpos)
        fcross = (mdotpos * ndotpos) / (1 + omhatdotpos)
        cosMu = -omhatdotpos

        return fplus, fcross, cosMu

    @partial(jax.jit, static_argnums=(0,))
    def cw_delay(sself,             
                log10_mchirp,
                log10_freq,
                log10_dc,
                geo,
                pdists,
                p_phases
            ):
        """
        Computes the CW signal in the sparse time domain.
    
        Parameters
        ----------
        log10_mchirp : float array
            log10 of the chirp mass in units of solar mass.
        log10_freq : float array
            log10 of the frequency in units of Hz.
        log10_dc : float array
            log10 of the co-moving distance in units of Mpc
        geo : float array
            The geometrical source params of feature size 5
        pdists: float array
            The pulsar distances
        p_phases: float array
            The pulsar phases
        """
    
        # Convert parameters to physical values
        fgw = 10.0**log10_freq 
    
        dist = 10 ** log10_dc * Mpc / c
    
        # Angles
        gwtheta = jnp.arccos(geo[..., 0:1])  # [0,pi]
        inc = jnp.arccos(geo[..., 2:3])  # [0,pi]
    
        # Pulsar distances converted to light travel time [s]
        p_dists = pdists * kpc / c
    
        fplus, fcross, cosMu = sself.create_gw_antenna_pattern(gwtheta, geo[..., 3:4])
    
        # Time grids relative to reference time
        toas_copy = sself.sparse_toas_CW - tref  # shape (Npulsars, Nsparse)
        tp = (
            toas_copy - (p_dists * (1.0 - cosMu))
        )  # retarded times for pulsar term
    
        # Redshifted chirp mass in seconds (GM/c^3 units)
        mc = 10.0**log10_mchirp * Tsun
    
        # Orbital angular frequency (since fgw = 2 f_orb => omega_orb = pi fgw)
        w0 = jnp.pi * fgw
        phase0 = (geo[..., -2:-1] / 2.0) # interpret input as GW phase; convert to orbital phase
    
        # Chirping evolution
        mc53 = mc ** (5.0 / 3.0)
        w083 = w0 ** (8.0 / 3.0)
        fac1 = (256.0 / 5.0) * mc53 * w083
    
        omega = w0 * (1.0 - fac1 * toas_copy) ** (-3.0 / 8.0)
        omega_p = w0 * (1.0 - fac1 * tp) ** (-3.0 / 8.0)
    
        # omega at "pulsar emission time zero" used for phase reference
        omega_p0 = (w0 * (1.0 + fac1 * p_dists * (1.0 - cosMu)) ** (-3.0 / 8.0))
    
        # Orbital phase evolution
        phase = phase0 + (1.0 / (32.0 * mc53)) * (
            w0 ** (-5.0 / 3.0) - omega ** (-5.0 / 3.0)
        )
    
        phase_p = (
            phase0
            + p_phases
            + (1.0 / (32.0 * mc53))
            * (omega_p0 ** (-5.0 / 3.0) - omega_p ** (-5.0 / 3.0))
        )
    
        # Geometry factors for plus/cross contributions
        inc_factor = (-0.5 * (3.0 + jnp.cos(2.0 * inc)))
        At = jnp.sin(2.0 * phase) * inc_factor
        Bt = 2.0 * jnp.cos(2.0 * phase) * (geo[..., 2:3])
        At_p = jnp.sin(2.0 * phase_p) * inc_factor
        Bt_p = 2.0 * jnp.cos(2.0 * phase_p) * (geo[..., 2:3])
    
        alpha = mc**(5./3.)/(dist*omega**(1./3.))
        alpha_p = mc**(5./3.)/(dist*omega_p**(1./3.))
    
        c2psi = jnp.cos(2.0 * geo[..., 1:2])
        s2psi = jnp.sin(2.0 * geo[..., 1:2])
    
        rplus = alpha * (-At * c2psi + Bt * s2psi)
        rcross = alpha * (At * s2psi + Bt * c2psi)
        rplus_p = alpha_p * (-At_p * c2psi + Bt_p * s2psi)
        rcross_p = alpha_p * (At_p * s2psi + Bt_p * c2psi)
        
        # Residuals: project polarization residuals onto pulsars and take (pulsar - earth)
        res = fplus * (rplus_p - rplus) + fcross * (rcross_p - rcross)

        return res

    @partial(jax.jit, static_argnums=(0,))
    def get_CW_coefficients(self,
                            log10_mchirp,
                            log10_freq,
                            log10_dc,
                            geo,
                            pdists,
                            p_phases):
        """
        Computes the Fourier coefficients.

        Parameters
        ----------
        log10_mchirp : float array
            log10 of the chirp mass in units of solar mass.
        log10_freq : float array
            log10 of the frequency in units of Hz.
        log10_dc : float array
            log10 of the co-moving distance in units of Mpc
        geo : float array
            The geometrical source params of feature size 5
        pdists: float array
            The pulsar distances
        p_phases: float array
            The pulsar phases
        """
        cw_residuals = self.cw_delay(log10_mchirp,
                                    log10_freq,
                                    log10_dc,
                                    geo,
                                    pdists,
                                    p_phases)
        cw_fft = jnp.fft.fft(cw_residuals, n=None, axis=-1, norm=None)

        # Shift FFT to treat sparse_toas_CW[:,0] as the effective "start"
        cw_fft *= jnp.exp(
            -1.0j * 2.0 * jnp.pi * self.freqs_forFFT * self.sparse_toas_CW[..., 0:1]
        )

        # Extract positive-frequency half (includes DC at index 0)
        a_n = jnp.imag(cw_fft[..., : self.Nsparse // 2]) * (-2.0 / self.Nsparse)
        b_n = jnp.real(cw_fft[..., : self.Nsparse // 2]) * (2.0 / self.Nsparse)

        # Pack as (Npulsars, 2*(CW_bins+1)) then drop DC (first sine/cos slot)
        coeff = (
            jnp.concatenate((a_n, b_n), axis=-1)
            .reshape((*a_n.shape[:-1], 2, self.CW_bins + 1))
            .mT
            .reshape((*a_n.shape[:-1], 2 * self.CW_bins + 2))
        )
        return coeff[..., 2:]  # remove DC terms (a_0, b_0)

    def make_conditional_flow(self,
                cw_coeff,
                context,
                normalizer_dict=jnp.array([False]),
                context_min=None,
                context_max=None,
                paths_to_training_set=None,
                path_to_load_flow_param_file=None,
                coeff_max_absolute_value=1e-4,
                flow_num_layers=4,
                hidden_size=128,
                mlp_num_layers=2,
                num_bins=8,
                p=.1,
                B=5,
                learning_rate=1e-4):
        """
        Construct and optionally initialize a conditional normalizing flow model.

        This method creates a :class:`ConditionalFlow` object configured to model
        the distribution of continuous-wave (CW) Fourier coefficients conditioned
        on a supplied context vector. The data range for the CW coefficients is
        assumed to be bounded by ``coeff_max_absolute_value`` for every feature.

        If ``path_to_load_flow_param_file`` is provided, the flow parameters are
        loaded from disk after construction.

        Parameters
        ----------
        cw_coeff : array-like
            Training data containing the CW coefficients. Each sample is expected
            to contain ``2 * self.Npulsars * self.CW_bins`` coefficients
            corresponding to the real and imaginary components for each pulsar.

        context : array-like
            Conditioning variables associated with each coefficient sample.
            Typically, just the astrophysical params

        normalizer_dict : dict or array-like, optional
            Normalization specification passed directly to
            :class:`ConditionalFlow`. The default disables custom normalization.

        context_min : array-like, optional
            Minimum values used for context normalization.

        context_max : array-like, optional
            Maximum values used for context normalization.

        paths_to_training_set : sequence of str, optional
            Paths to memory-mapped or on-disk training datasets. Passed directly
            to the ``ConditionalFlow`` constructor.

        path_to_load_flow_param_file : str, optional
            Path to a saved flow parameter file (NPZ). If provided, the parameters are
            loaded before returning the flow object.

        coeff_max_absolute_value : float, optional
            Absolute bound used to define the minimum and maximum values for every
            CW coefficient feature. Default is ``1e-4``.

        flow_num_layers : int, optional
            Number of coupling layers in the normalizing flow. Default is ``4``.

        hidden_size : int, optional
            Width of the hidden layers in the coupling network. Default is ``128``.

        mlp_num_layers : int, optional
            Number of hidden layers in each coupling-network MLP. Default is ``2``.

        num_bins : int, optional
            Number of spline bins used by the spline transformations. Default is
            ``8``.

        p : float, optional
            Normalization parameter passed directly to ``ConditionalFlow``.
            Default is ``0.1``.

        B : int, optional
            Normalization range parameter. Internally, ``B - 1`` is passed to the
            flow constructor. Default is ``5``.

        learning_rate : float, optional
            Optimizer learning rate used during training. Default is ``1e-4``.

        Returns
        -------
        ConditionalFlow
            A configured conditional normalizing flow instance. If
            ``path_to_load_flow_param_file`` is specified, the returned object
            contains the loaded model parameters.

        Notes
        -----
        The data normalization bounds are constructed internally as

        - ``data_min = -coeff_max_absolute_value``
        - ``data_max = +coeff_max_absolute_value``

        for every coefficient feature, resulting in arrays of length
        ``2 * self.Npulsars * self.CW_bins``.
        """

        data_min = np.full(
            shape=self.Npulsars * 2 * self.CW_bins,
            fill_value=-coeff_max_absolute_value,
        )
        data_max = np.full(
            shape=self.Npulsars * 2 * self.CW_bins,
            fill_value=coeff_max_absolute_value,
        )

        flow_object = ConditionalFlow(
            # ── data / context (memmapped or in-memory) ──────────────────────
            cw_coeff,
            context,
            # ── normalisation ranges (per-feature 1-D arrays) ────────────────
            normalizer_dict=normalizer_dict,
            data_min=data_min,
            data_max=data_max,
            context_min=context_min,
            context_max=context_max,
            tset_paths=paths_to_training_set,
            last_feature_index=self.Npulsars * 2 * self.CW_bins,
            last_feature_index_for_context = self.n_ast_pars + self.Npulsars * 2 * self.CW_bins,
            # ── shared normalisation hyperparameters ─────────────────────────
            p=p,
            B=B - 1, #-1 ensures no boundry problems
            # ── flow architecture ─────────────────────────────────────────────
            flow_num_layers=flow_num_layers,
            hidden_size=hidden_size,
            mlp_num_layers=mlp_num_layers,
            num_bins=num_bins,
            # ── optimisation ─────────────────────────────────────────────────
            learning_rate=learning_rate,
            seed=0,
        )
        if path_to_load_flow_param_file:
            flow_object.load_params(path_to_load_flow_param_file)

        return flow_object

    def make_flow(self,
                data,
                data_min = jnp.array([False]),
                data_max = jnp.array([False]),
                paths_to_training_set = None,
                path_to_load_flow_param_file = None,
                coeff_max_absolute_value=1e-4,
                flow_num_layers=4,
                hidden_size=128,
                mlp_num_layers=2,
                num_bins=8,
                p=.1,
                B=5,
                learning_rate=1e-4):
        """
        Construct and optionally initialize a conditional normalizing flow model.

        This method creates a :class:`ConditionalFlow` object configured to model
        the distribution of continuous-wave (CW) Fourier coefficients conditioned
        on a supplied context vector. The data range for the CW coefficients is
        assumed to be bounded by ``coeff_max_absolute_value`` for every feature.

        If ``path_to_load_flow_param_file`` is provided, the flow parameters are
        loaded from disk after construction.

        Parameters
        ----------
        data : array-like
            Training data containing with the shapoe (N_samps, N_features).

        flow_num_layers : int, optional
            Number of coupling layers in the normalizing flow. Default is ``4``.

        hidden_size : int, optional
            Width of the hidden layers in the coupling network. Default is ``128``.

        mlp_num_layers : int, optional
            Number of hidden layers in each coupling-network MLP. Default is ``2``.

        num_bins : int, optional
            Number of spline bins used by the spline transformations. Default is
            ``8``.

        p : float, optional
            Normalization parameter passed directly to ``ConditionalFlow``.
            Default is ``0.1``.

        B : int, optional
            Normalization range parameter. Internally, ``B - 1`` is passed to the
            flow constructor. Default is ``5``.

        learning_rate : float, optional
            Optimizer learning rate used during training. Default is ``1e-4``.

        Returns
        -------
        Flow
            A configured conditional normalizing flow instance. If
            ``path_to_load_flow_param_file`` is specified, the returned object
            contains the loaded model parameters.
        """
        if not data_min.any() and not data_max.any():
            data_min = np.full(
                shape=self.Npulsars * 2 * self.CW_bins,
                fill_value=-coeff_max_absolute_value,
            )
            data_max = np.full(
                shape=self.Npulsars * 2 * self.CW_bins,
                fill_value=coeff_max_absolute_value,
            )

        flow_object = Flow(
            # ── data / context (memmapped or in-memory) ──────────────────────
            data,
            data_min = data_min,
            data_max = data_max,
            tset_paths = paths_to_training_set,
            # ── shared normalisation hyperparameters ─────────────────────────
            p=p,
            B=B - 1, #-1 ensures no boundry problems
            # ── flow architecture ─────────────────────────────────────────────
            flow_num_layers=flow_num_layers,
            hidden_size=hidden_size,
            mlp_num_layers=mlp_num_layers,
            num_bins=num_bins,
            # ── optimisation ─────────────────────────────────────────────────
            learning_rate=learning_rate,
            seed=0,
        )
        if path_to_load_flow_param_file:
            flow_object.load_params(path_to_load_flow_param_file)

        return flow_object

    @partial(jax.jit, static_argnums=(0, 4, 5, 6))
    def get_gwb_coeff_clt(self,
                        rng_key,
                        num_sources,
                        clt_context,
                        gen_samp_size_clt,
                        context_feature_size,
                        flow):
        """
        Generate a gravitational-wave background (GWB) coefficient
        realization using a Central Limit Theorem (CLT) approximation.

        This method estimates the mean and covariance of the conditional flow
        distribution by drawing samples from the provided normalizing flow. The
        coefficient vector for ``num_sources`` statistically independent
        sources is then approximated as a multivariate Gaussian with

        - mean = ``num_sources * μ``
        - covariance = ``num_sources * Σ``

        where ``μ`` and ``Σ`` are estimated from the generated flow samples.

        The covariance matrix is symmetrized and regularized before a Cholesky
        decomposition is computed to ensure numerical stability.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            JAX random number generator key.

        num_sources : float or int
            Number of statistically independent GWB sources contributing to the
            aggregate realization.

        clt_context : array-like
            Conditioning vector for the flow. This context is broadcast so that
            every generated flow sample uses the same conditioning information.

        gen_samp_size_clt : int
            Number of flow samples used to estimate the conditional mean and
            covariance.

        context_feature_size : int
            Length of the conditioning vector.

        flow : ConditionalFlow
            Trained conditional normalizing flow used to generate conditional
            coefficient samples.

        Returns
        -------
        jax.Array
            A single realization of the summed GWB coefficient vector with shape
            ``(2 * self.Npulsars * self.CW_bins,)`` generated using the CLT
            approximation.

        Notes
        -----
        The returned sample is computed as

        .. math::

            N μ + sqrt{N} L z,

        where

        - ``N`` is ``num_sources``,
        - ``μ`` is the sample mean of the flow realizations,
        - ``Σ`` is the sample covariance,
        - ``L`` is the Cholesky factor of the regularized covariance matrix,
        - ``z`` is a standard multivariate normal random vector.

        A small diagonal regularization,

        ``eps = 1e-4 * mean(diag(Σ))``,

        is added before the Cholesky factorization to improve numerical stability.
        """

        context = jnp.broadcast_to(
            clt_context,
            (gen_samp_size_clt, context_feature_size)
        )

        z = jax.random.normal(
            rng_key,
            (gen_samp_size_clt + 1, self.Npulsars * 2 * self.CW_bins)
        )

        flow_samps = flow.forward_pass(c=context, z=z[1:])

        mean = jnp.mean(flow_samps, axis=0)
        sigma = jnp.cov(flow_samps.T)
        sigma = 0.5 * (sigma + sigma.mT)

        eps = 1e-4 * jnp.mean(jnp.diag(sigma))
        L = jnp.linalg.cholesky(
            sigma + eps * jnp.eye(sigma.shape[0])
        )

        return num_sources * mean + jnp.sqrt(num_sources) * L @ z[0].T

    @partial(jax.jit, static_argnums=(0, 4, 5, 6))
    def get_gwb_coeff_nonclt(self,
                            rng_key,
                            num_sources,
                            clt_context,
                            gen_samp_size_nonclt,
                            context_feature_size,
                            flow):
        """
        Generate a gravitational-wave background (GWB) coefficient
        realization by explicitly summing individual source realizations.

        Unlike :meth:`get_gwb_coeff_clt`, this method does not rely on a Central
        Limit Theorem approximation. Instead, it generates independent conditional
        samples from the normalizing flow and directly sums the first
        ``num_sources`` realizations to produce the aggregate coefficient vector.

        Parameters
        ----------
        rng_key : jax.random.PRNGKey
            JAX random number generator key.

        num_sources : int
            Number of individual source realizations to include in the summed GWB
            coefficient vector. It is assumed that
            ``num_sources <= gen_samp_size_nonclt``.

        clt_context : array-like
            Conditioning vector for the flow. This context is broadcast so that
            every generated source realization uses identical conditioning
            information.

        gen_samp_size_nonclt : int
            Number of individual flow samples to generate. This should be at least
            as large as ``num_sources``.

        context_feature_size : int
            Length of the conditioning vector.

        flow : ConditionalFlow
            Trained conditional normalizing flow used to generate conditional
            coefficient samples.

        Returns
        -------
        jax.Array
            A single realization of the summed GWB coefficient vector with shape
            ``(2 * self.Npulsars * self.CW_bins,)`` obtained by explicitly summing
            ``num_sources`` independent flow realizations.

        Notes
        -----
        A boolean mask is constructed to select only the first ``num_sources``
        generated samples,

        .. math::

            sum_{i=1}^{N} x_i,

        where each :math:`x_i` is an independent sample drawn from the conditional
        normalizing flow. Samples beyond ``num_sources`` are multiplied by zero and
        therefore do not contribute to the returned coefficient vector.
        """

        context = jnp.broadcast_to(
            clt_context,
            (gen_samp_size_nonclt, context_feature_size)
        )

        z = jax.random.normal(
            rng_key,
            (gen_samp_size_nonclt, self.Npulsars * 2 * self.CW_bins)
        )

        flow_samps = flow.forward_pass(c=context, z=z)

        mask = (jnp.arange(gen_samp_size_nonclt) < num_sources)[:, None]

        return (flow_samps * mask).sum(axis=0)

    def get_key(self, seed = None):
        if seed:
            return jr.key(int(seed))
        else:
            return jr.key(random.randint(0, 91862156))

    def gen_z(self, key, shape):
        """Generates random numbers from
        a standard normal distibution

        Args:
            key (jax.random.PRNGKey): Random number generator key
            shape (tuple): the shape of the requested random numbers
        """       
        return jr.normal(key, shape)

    def make_gwb_coeff(self,
                    key1,
                    key2,
                    ast_params_start_index,
                    lambda_index,
                    astro_lambda_flow,
                    cw_flow):
        """
        Generate a realization of gravitational-wave background (GWB) Fourier
        coefficients using a hierarchical model.

        This method performs three sequential sampling steps:

        1. Draw astrophysical parameters and the expected number of sources
        (``lambda``) from a trained astrophysical normalizing flow.
        2. Sample the actual number of sources from a Poisson distribution with
        mean ``lambda``.
        3. Generate one set of CW Fourier coefficients for each source using a
        conditional CW normalizing flow conditioned on the sampled
        astrophysical parameters.

        The resulting CW coefficients are reshaped so that each row corresponds to
        one independently generated source.

        Parameters
        ----------
        key1 : jax.random.PRNGKey
            Random number generator key used for latent-variable sampling from the
            normalizing flows.

        key2 : jax.random.PRNGKey
            Random number generator key used to sample the Poisson-distributed
            number of sources.

        ast_params_start_index : int
            Index of the first astrophysical parameter within the output vector of
            ``astro_lambda_flow``.

        lambda_index : int
            Index of the Poisson rate parameter (``lambda``) within the output
            vector of ``astro_lambda_flow``.

        astro_lambda_flow : Flow
            Trained normalizing flow that generates astrophysical parameters and
            the expected source count.

        cw_flow : ConditionalFlow
            Trained conditional normalizing flow that generates CW Fourier
            coefficients conditioned on the sampled astrophysical parameters.

        Returns
        -------
        coeff_cw.sum(axis = 0)
        """

        # Step 1: Sample from astro-lambda flow
        z1 = self.gen_z(key1, shape=(self.n_ast_pars + 1,))
        samps1 = astro_lambda_flow.forward_pass(z=z1)
        ast = samps1[ast_params_start_index:self.n_ast_pars]
        lam = 10**(samps1[lambda_index]) - 1

        # Step 2: Draw from a Poisson distribution
        N_s = jr.poisson(key2, lam)

        # Step 3: Draw from CW flow
        z2 = self.gen_z(key1, shape=(N_s * self.Npulsars * self.CW_bins * 2,))
        context = jnp.broadcast_to(ast, (N_s, self.n_ast_pars))
        coeff_cw = cw_flow.forward_pass(
            z=z2,
            c=context
        ).reshape(N_s, self.Npulsars * self.CW_bins * 2)

        return coeff_cw.sum(axis = 0)

    @partial(jax.jit, static_argnums=(0))
    def cw_training_set(self,
                        key1,
                        # key2,
                        # ast_theta,
                        holo_data,
                        geo, 
                        dist, 
                        phase):
        """
        SMBHB-induced Fourier coefficient generation for pulsar timing array (PTA)
        gravitational-wave background (GWB) analysis.

        This script prepares a batched set of supermassive black hole binary (SMBHB)
        source parameters and pulsar geometry realizations before evaluating the
        continuous-wave (CW) induced Fourier coefficients used in PTA likelihood
        calculations.

        Overview
        --------
        The batching strategy marginalizes over several independent stochastic
        quantities simultaneously:

            - Poisson realizations of the source population.
            - Random binary sky/orientation (geometric) parameters.
            - Pulsar distance and pulsar phase realizations.
            - Randomly selected astrophysical population realizations.

        The resulting tensors are broadcast into a common shape and flattened into a
        single batch before being passed to
        `correct_gwb.best_gwb_ever.get_CW_coefficients()`.

        Tensor batch ordering
        ---------------------
        Throughout this script the canonical batch ordering is

            (Poisson, Geometry, Pulsar, Astrophysical realization)

        which is stored in `S`.

        Different intermediate tensors require different axis orderings before being
        transposed into this convention.

        Outputs
        -------
        `out` contains the induced CW Fourier coefficients for every pulsar and every
        batched realization.
        """
        chosen_batch = jr.choice(key1, 
                                holo_data, 
                                shape = (self.thethree_batch_size, ), 
                                replace=True, 
                                p=None, 
                                axis=1, 
                                mode=None)
        # -----------------------------------------------------------------------------
        # Source parameters
        # -----------------------------------------------------------------------------
        log10_mchirp = chosen_batch[0]
        log10_dc = chosen_batch[1]
        log10_freq = chosen_batch[2]
        # lambda_val = chosen_batch[-1]

        # -----------------------------------------------------------------------------
        # Poisson draws for source counts
        # -----------------------------------------------------------------------------
        # piosson_draws = jr.poisson(
        #     key2,
        #     lam=lambda_val,
        #     shape=(self.piosson_batch_size, lambda_val.shape[0]),
        # ).T

        # -----------------------------------------------------------------------------
        # Sample geometric and pulsar parameters
        # -----------------------------------------------------------------------------
        # geo, dist, phase = self.draw_from_astro_uninformed_params_torch(self.geo_batch_size)

        # -----------------------------------------------------------------------------
        # Broadcast source and astrophysical parameters
        # -----------------------------------------------------------------------------
        log10_mchirp = shape_maker(log10_mchirp, 1, self.S)
        log10_dc = shape_maker(log10_dc, 1, self.S)
        log10_freq = shape_maker(log10_freq, 1, self.S)
        # lambda_val = shape_maker(lambda_val, 1, self.S)
        # theta_ast = shape_maker(ast_theta, self.n_ast_pars, self.S)
        # -----------------------------------------------------------------------------
        # Broadcast geometry and pulsar realizations
        # -----------------------------------------------------------------------------
        geo = shape_maker(
            geo[:, 0, 0, 0, 0, 0, 0, :],
            5,
            self.S_geo,
        ).transpose((0, 1, 3, 2, 4))

        dist = shape_maker(
            dist[:, 0, 0, 0, 0, :, 0],
            self.Npulsars,
            self.S_pulsar,
        ).transpose((0, 2, 3, 1, 4))

        phase = shape_maker(
            phase[:, 0, 0, 0, 0, :, 0],
            self.Npulsars,
            self.S_pulsar,
        ).transpose((0, 2, 3, 1, 4))

        # piosson_draws = shape_maker(
        #     piosson_draws,
        #     1,
        #     self.S_piosson,
        # ).transpose((3, 0, 1, 2, 4))

        # context = theta_ast.reshape(-1, self.n_ast_pars) 
        # jnp.concat((lambda_val.reshape(-1, 1), 
        #                       theta_ast.reshape(-1, self.n_ast_pars)
        #                       ), 
        #                       axis = -1)
        # -----------------------------------------------------------------------------
        # Evaluate induced CW Fourier coefficients
        # -----------------------------------------------------------------------------
        # All batch dimensions are flattened into a single leading axis before calling
        # the CW response model.
        coeff = self.get_CW_coefficients(
            log10_mchirp=log10_mchirp.reshape(
                (self.total_sample_size, 1, 1, 1, 1, 1, 1, 1)
            ),
            log10_freq=log10_freq.reshape(
                (self.total_sample_size, 1, 1, 1, 1, 1, 1, 1)
            ),
            log10_dc=log10_dc.reshape(
                (self.total_sample_size, 1, 1, 1, 1, 1, 1, 1)
            ),
            geo=geo.reshape(
                (self.total_sample_size, 1, 1, 1, 1, 1, 1, 5)
            ),
            pdists=dist.reshape(
                (self.total_sample_size, 1, 1, 1, 1, 1, self.Npulsars, 1)
            ),
            p_phases=phase.reshape(
                (self.total_sample_size, 1, 1, 1, 1, 1, self.Npulsars, 1)
            ),
        )[:, 0, 0, 0, 0, 0, :, :].reshape(-1, self.Npulsars * 2*self.CW_bins)

        return jnp.where(jnp.logical_and(coeff < self.cw_eps, coeff > -self.cw_eps), self.cw_eps, coeff)