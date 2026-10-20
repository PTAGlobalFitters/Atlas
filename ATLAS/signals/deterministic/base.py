"""
Holds the class for general deterministic signals.
"""

import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import vmap
import numpy as np
from scipy.signal.windows import tukey

from ATLAS.utils import jit
from ATLAS.signals import signals_utils as sutils


class Deterministic:

    """
    Class for determinist signals. Contains methods to get
    the induced residuals from a deterministic signal over
    the observed TOAs using interpolation from a Fourier
    representation. Used for resiual subtraction.

    Required Attributes
    -------------------
    name : str
        The name of the deterministic signal.
    data : Data
        An instance of the Data class from the `data` module.
    get_delays_func : Callable
        A JAX friendly function which takes in parameters of the
        deterministic model and outputs the induced timing delays
        across all pulsars. See det_signals.py for example functions.
    det_parameter_bounds : array
        The minima and maxima allowed values of the deterministic parameters,
        with shape (nparams, 2) where nparams is the length of the parameters
        supplied to 'get_delays_func'. The parameter minima are at [:, 0] and
        the maxima at [:, 1].
    nfreqs_det : int
        Number of frequency bins used to reprsentent deterministic signals.
        Defaults to 60.
    with_psr_params : bool
        Deterministic signals from individual binaries depend on a set of (npsr) pulsar
        phase parameters and (npsr) pulsar distance parameters, where npsr is the number
        of pulsars in the array. If False, the pipeline will wrap 'get_delays_func'
        to accept these parameters anyway and feed it NoneType in those parameter slots.
        Defaults to True.
    window_ext_factor : float
        Factor by which to extend Tspan for frequency bins representing deterministic signal,
        aids with Gibbs phenomena. Defaults to 2.
    get_coeffs_func : Callable
        A JAX friendly function which maps the parameters of the deterministic
        model to the frequency representation of the signal. If None, defaults
        to the FFT method below. Defaults to None.
    """

    def __init__(self,
                 name,
                 data,
                 get_delays_func,
                 det_parameter_bounds,
                 nfreqs_det = 60,
                 with_psr_params = True,
                 window_ext_factor = 2.,
                 get_coeffs_func = None,
                 ):

        self.name = name
        self.data = data
        self.get_delays_func = get_delays_func
        self.det_parameter_bounds = jnp.array(det_parameter_bounds)
        self.det_param_mins = self.det_parameter_bounds[:, 0]
        self.det_param_maxs = self.det_parameter_bounds[:, 1]
        self.nparams_det = self.det_param_mins.shape[0]
        self.nfreqs_det = nfreqs_det
        self.num_coeff_det = 2 * self.nfreqs_det
        self.with_psr_params = with_psr_params

        # if pulsar parameters not needed for model, wrap input function
        self.get_delays_func = jit(self._wrap_delays_func(self.get_delays_func,
                                                          self.with_psr_params))

        # if no get_coeffs_func specified, use FFT method
        self._get_coeffs_func = get_coeffs_func or self.get_coeffs_via_FFT
        self.get_coeffs_func = jit(self._get_coeffs_func)

        # sparse TOAs for deterministic signal's FFT
        window_ext = self.data.pta_tspan * window_ext_factor
        Tspan_ext = self.data.pta_tspan + 2. * window_ext
        first_toa = np.min([np.min(psr.toas) for psr in self.data.psrs])
        last_toa = np.max([np.max(psr.toas) for psr in self.data.psrs])
        sparse_toas_det = np.array([np.linspace(first_toa - window_ext, last_toa + window_ext,
                                                self.num_coeff_det + 2, endpoint=False)
                                    for _ in range(self.data.npsrs)])
        self.sparse_toas_det_jax = jnp.array(sparse_toas_det)
        self.Nsparse = sparse_toas_det.shape[1]

        # frequency bins used in FFT
        self.freqs_forFFT = jnp.array([np.fft.fftfreq(self.Nsparse, Tspan_ext / self.Nsparse)
                                       for _ in range(self.data.npsrs)])
        # frequencies of the Fourier design matrix that interpolates onto the observed TOAs.
        # The design matrix itself is NOT stored: at full-array scale it is a multi-GB
        # [n_toas, 2 * nfreqs_det] array, and SuperSignal keeps its own copy inside the
        # unified T-matrix. `get_basis` builds it on demand.
        self.freqs_for_Fmat = jnp.array([self.freqs_forFFT[0, j + 1] for j in range(self.nfreqs_det)])

        # Tukey window for FFT
        self.Tukey_det = jnp.array(tukey(self.Nsparse, alpha=(Tspan_ext - self.data.pta_tspan) / Tspan_ext))

        # pulsar distances
        psr_dists_mean = []
        psr_dists_std = []
        psr_dists_method = []
        for psr in self.data.psrs:
            if psr.name in list(self.data.psr_dists_dict.keys()):
                dist, std, method = self.data.psr_dists_dict[psr.name]
                psr_dists_mean.append(dist)
                psr_dists_std.append(std)
                psr_dists_method.append(method)
            else:
                psr_dists_mean.append(psr.pdist[0])
                psr_dists_std.append(psr.pdist[1])
                psr_dists_method.append('other')
        self.psr_dists_mean = jnp.array(psr_dists_mean)
        self.psr_dists_std = jnp.array(psr_dists_std)
        self.psr_dists_method = np.array(psr_dists_method)
        self.where_PX = jnp.where(self.psr_dists_method == 'PX')[0]
        self.where_DM = jnp.where(self.psr_dists_method == 'DM')[0]
        self.where_other = jnp.where(self.psr_dists_method == 'other')[0]
        self.where_not_other = jnp.ones(self.data.npsrs, dtype=bool)
        self.where_not_other = self.where_not_other.at[self.where_other].set(False)


    def get_basis(self):
        """
        Get Fourier design matrix used to interpolate determinsitic signal onto observed TOAs.

        Returns
        -------
        Fs_det : list
            List of Fourier design matrices for every pulsar.
        """
        
        return [self._get_psr_basis(psr) for psr in self.data.psrs]


    def _get_psr_basis(self, psr):
        """Fourier design matrix of one pulsar, [npsr_toas, 2 * nfreqs_det]."""
        return sutils.get_fourier_design_matrix(psr.toas, self.freqs_for_Fmat)


    def get_coeffs_via_FFT(self, det_params, psr_phases, psr_dists):
        """
        Mapping from deterministic signal parameters to Fourier space.
        This is simply a FFT.

        Note the frequency bins used here are generally different than those used
        by stochastic models. To avoid Gibbs phenomena from non-periodic deterministic
        signals over Tspan, we use an "extended" basis where we FFT the signal over a period
        extended either side of Tspan after applying a Tukey window. The Fourier design matrix,
        however, maps the Fourier coefficients to TOAs **within** the PTA Tspan. This introduces
        cross terms in the posterior. See Gundersen & Cornish 2025.

        Parameters
        ----------
        det_params : array
            Parmeters values of the deterministic model.
        psr_phases : array or None
            The phase of the gravitational wave at each pulsar. This can be None if
            `Deterministic.with_psr_params = False`.
        psr_dists : array or None
            The distance [kpc] to each pulsar. This can be None if
            `Deterministic.with_psr_params = False`.

        Returns
        -------
        coeff : array
            A (npsrs, 2*nfreq) array where npsrs is the number of pulsars in the array and
            nfreq is the number of frequency bins used to represent to the deterministic
            model in Fourier space.
        """

        # get timing delays induced by the deterministic signal over "sparse" (evenly-spaced) TOAs
        det_residuals = self.get_delays_func(self.sparse_toas_det_jax, self.data.psr_pos,
                                             det_params, psr_phases, psr_dists)

        # window residuals over extended observation
        det_residuals_windowed = self.Tukey_det * det_residuals

        # do FFT
        det_fft = jnp.fft.fft(det_residuals_windowed, n=None, axis=-1, norm=None)  # dim (Np, 2 * Nf + 2)

        # apply time shift to set initial time
        det_fft *= jnp.exp(-1.j * 2 * jnp.pi * self.freqs_forFFT * self.sparse_toas_det_jax[:, 0:1])

        # extract sine and cosine coefficients
        a_n = jnp.imag(det_fft[:, :self.Nsparse // 2]) * (-2 / self.Nsparse)  # (Np, Nf + 1)
        b_n = jnp.real(det_fft[:, :self.Nsparse // 2]) * (2 / self.Nsparse)  # (Np, Nf + 1)

        # interweave sine/cosine coefficients and reshape to (Np, 2 * Nf)
        coeff = jnp.concatenate((a_n, b_n), axis=1).reshape((self.data.npsrs, 2, self.nfreqs_det + 1))\
                        .transpose((0, 2, 1)).reshape((self.data.npsrs, self.num_coeff_det + 2))

        return coeff[:, 2:]  # remove DC

    
    def get_det_residuals(self, det_params, psr_phases, psr_dists):
        """
        Get the residuals induced over the observed TOAs via
        interpolation with the Fourier basis.

        Parameters
        ----------
        det_params : array
            Parameters of deterministic signal.
        psr_phases : array
            Phase at each pulsar [npsr].
        psr_dists : array
            Distance to each pulsar [kpc], [npsr].

        Returns
        -------
        det_residuals : array
            Residuals induced by deterministic signal
            over the observed TOAs, [ntoas_all].
        """

        # get Fourier representation
        a_det = self.get_coeffs_func(det_params, psr_phases, psr_dists)  # [npsr, 2 * nfreqs_det]

        # interpolate onto the TOAs one pulsar at a time: building the design matrix per
        # pulsar keeps the peak footprint at one pulsar's [npsr_toas, 2 * nfreqs_det] block
        # rather than the [n_toas, 2 * nfreqs_det] repeat-and-multiply over the whole array
        det_residuals = [self._get_psr_basis(psr) @ a_det[pidx]
                         for pidx, psr in enumerate(self.data.psrs)]
        return jnp.concat(det_residuals)


    def ln_p_PX(self, value, dist, err):
        """
        Parallax-based prior on pulsar distance (Arzoumanian+ 2023 Eq. 20)
        p(d) ∝ N(1/d | 1/dist, err/dist^2) * 1/d^2
        """
        pi = 1.0 / dist
        pi_err = err / dist**2
        inv_value = 1.0 / value
        z = (inv_value - pi) / pi_err
        lnprob = -0.5 * z**2 - jnp.log(jnp.sqrt(2 * jnp.pi) * pi_err * value**2)
        lnprob = jnp.where(value > 0, lnprob, -jnp.inf)
        return lnprob


    def ln_p_DM(self, value, dist, err):
        """
        DM-based prior on pulsar distance (Arzoumanian+ 2023 Eq. 21)
        Flat between dist±err, Gaussian tails outside that range.
        """
        sigma = 0.25 * err
        boxheight = 1.0 / (2.0 * err)
        gaussheight = 1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma)
        scale = boxheight / gaussheight
        area = 1.0 + scale  # normalization factor

        left  = norm.pdf(value, loc=dist - err, scale=sigma) * scale
        mid   = boxheight
        right = norm.pdf(value, loc=dist + err, scale=sigma) * scale

        y = jnp.where(
            value <= (dist - err), left,
            jnp.where(value < (dist + err), mid, right)
        )
        lnprob = jnp.log(y / area + 1e-12)
        lnprob = jnp.where(value > 0, lnprob, -jnp.inf)
        return lnprob


    # the additional prior weighting take the determinisitic model parameters as input
    def psr_dists_lnprior(self, det_params, psr_phases, psr_dists):

        # prior on pulsar distance when measured with parallax
        lnpriors_PX_dist = vmap(lambda x, y, z : self.ln_p_PX(x, y, z),
                                    in_axes=(0, 0, 0))(psr_dists, self.psr_dists_mean, self.psr_dists_std)
        lnprior_PX_summed = jnp.sum(lnpriors_PX_dist[self.where_PX])

        # prior on pulsar distance when measured with DM
        lnpriors_DM_dist = vmap(lambda x, y, z : self.ln_p_DM(x, y, z),
                                    in_axes=(0, 0, 0))(psr_dists, self.psr_dists_mean, self.psr_dists_std)
        lnprior_DM_summed = jnp.sum(lnpriors_DM_dist[self.where_DM])

        # Prometheus still samples pulsar distance using a normal prior,
        # so we must subtract that log-density here
        ln_normal_corrections = 0.5 * (psr_dists - self.psr_dists_mean)**2 / self.psr_dists_std**2
        ln_normal_correction = jnp.sum(ln_normal_corrections[self.where_not_other])

        # new "advanced" pulsar distance prior
        return lnprior_PX_summed + lnprior_DM_summed + ln_normal_correction


    def _wrap_delays_func(self, func, with_psr_params):
        """
        If a deterministic model is supplied that does **not**
        take pulsar distance and phase as model parameters,
        then the 'get_delay_func' is wrapped to accept these
        parameters anyway and the sampler will supply NoneTypes
        for these inputs.
        """
        if with_psr_params:
            return func
        else:
            def wrapped(toas, psrpos, params, psr_phases, psr_dists):
                return func(toas, psrpos, params)
            return wrapped

