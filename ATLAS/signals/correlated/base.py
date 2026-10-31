
from ATLAS.utils import jit, jit_method
from ATLAS.signals import signals_utils as sutils
from ATLAS.signals import orf_functions as orf_funcs

from functools import cached_property, partial
from ATLAS.signals.correlated.utils import make_gwb_model
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.scipy as jsp
import jax.random as jrandom

class Correlated:
    """A signal class for a common-correlated GWB modeled as a free spectrum.

    This signal models the gravitational wave background as a pulsar-correlated 
    common signal in all pulsars. The power spectrum is modeled as a free spectrum 
    with nfreqs frequency bins. The parameters are given as halflog10_rho, which 
    is defined as
    - <a^T a> = rho^2 -> halflog10_rho = 0.5 * log10(<a^T a>)
    This means that the units are log(seconds).

    The frequencies are the first `nfreqs` harmonics of 1/Tspan.

    Attributes
    ----------
    name : str
        The name of the signal.
    init_params : dict
        The parameters used for initialization, which can be useful for re-initialization.
    parameter_names : list of str
        The list of parameter names for this signal.
    n_parameters : int
        The number of parameters for this signal.
    parameter_range : array
        The range for each parameter, shape (n_parameters, 2).
    allow_posterior_draw : bool
        Whether to allow posterior draws for this signal.
    sampling_method : str
        The sampling method to use for the signal.
    initialized : bool
        Whether the signal is fully initialized with data.
    psr_toas : list of arrays
        The list of TOA arrays for each pulsar. [npsr, npsr_toas]
    nfreqs : int
        The number of frequencies in the free spectrum.
    nmodes : int
        The number of modes, which is 2*nfreqs (sine and cosine for each frequency).
    orf : str
        The string for the overlap reduction function, e.g. 'hd'.
    npsrs : int
        The number of pulsars in the PTA.
    npairs : int
        The number of pulsar pairs in the PTA.
    tspan : float
        The time span of the PTA data in seconds.
    freqs : array
        The frequencies for the free spectrum, which are the first nfreqs harmonics of 1/Tspan. [nfreqs]
    log_prior_volume : float
        The log of the prior volume for the parameters.
    fixed_wn : bool
        Whether the white noise is fixed, which can be used to optimize computations.
    Gamma : array
        The overlap reduction function matrix for the pulsars. [npsr, npsr]
    Gamma_inv : array
        The inverse of the overlap reduction function matrix. [npsr, npsr]
    logdet_Gamma : float
        The log-determinant of the overlap reduction function matrix.
    """

    def __init__(self,
                 data,
                 name,
                 psd_function,
                 orf_function,
                 upper_bound_psd,
                 lower_bound_psd,
                 upper_bound_orf = None,
                 lower_bound_orf = None,
                 orf='hd',
                 nfreqs=10, 
                 halflog10_rho_range=(-9,-2), 
                 posterior_draw_ndraws = 1,
                 user_freqs=None
                 ):
        """The constructor for the GWB_Freespectrum signal class.

        This signal models the GWB as a pulsar-correlated common signal in all
        pulsars. The power spectrum is modeled as a free spectrum with nfreqs 
        frequencies. The parameters are given as halflog10_rho, which is defined as
        - <a^T a> = rho^2 -> halflog10_rho = 0.5 * log10(<a^T a>)
        This means that the units are log(seconds). 

        The frequencies are the first `nfreqs` harmonics of 1/Tspan.

        The overlap reduction function (ORF) can be specified by the orf parameter, 
        which must be a string that is recognized by the `orf_funcs.get_orf_matrix(orf)`.

        If data is not provided, the signal use a simple initialization (see 
        Atlas.signals.base.Signal_Base).

        Parameters
        ----------
        name : str, optional
            The name of the signal, by default 'GWB'
        orf : str, optional
            The overlap reduction function, by default 'hd'
        nfreqs : int, optional
            The number of frequencies, by default 10
        halflog10_rho_range : tuple, optional
            The range for the halflog10_rho parameters, by default (-9,-2)
        sampling_method : str, optional
            The sampling method to use for the signal. Should be one of the allowed methods
            in SAMPLING_METHODS from Atlas.samplers.CaneToadRacing.
            Default = 'posterior_draws'
        posterior_draw_ndraws : int, optional
            The number of posterior draws to run in parallel when sampling_method is 'posterior_draws'.
        reparam : bool, optional
            Whether to use the reparameterized version of the signal, which can improve sampling efficiency.
        cache_TNT:
            do you want to cache TNT?
        data : Atlas.data.Data.PTA_Data, optional
            The PTA data, by default None
        """
        # PSD and ORF reparameterization
        self.psd_function, self.orf_function, self.psd_reparam_helper = make_gwb_model(psd_function,
                                        orf_function, 
                                        lower_bound_array = lower_bound_psd, 
                                        upper_bound_array = upper_bound_psd,
                                        lower_bound_orf = lower_bound_orf,
                                        upper_bound_orf = upper_bound_orf)

        # Simple initialization-------------------------------------------------
        self.name = name
        self.data = data
        self.ndraws = posterior_draw_ndraws # Number of posterior draws to run in parallel when sampling_method is 'posterior_draws'
        # Required attributes:      parameter_names, n_parameters, 
        # parameter_range, allow_posterior_draw, sampling_methods, initialized

        # Helper attributes needed for the rest of the class--------------------
        self.psr_toas = self.data.toas # Reference to a list of TOA arrays for each pulsar [npsr, npsr_toas]
        self.nfreqs = nfreqs
        self.nmodes = 2*nfreqs # Sine and cosine modes per frequency
        self.orf = orf # string for orf function, e.g. 'hd'
        self.npsrs = self.data.npsrs
        self.npairs = self.data.npairs
        self.tspan = self.data.pta_tspan
        
        if user_freqs is not None:
            self.freqs = user_freqs
            self.nfreqs = len(self.freqs)
            self.nmodes = 2 * self.nfreqs
        else:    
            self.tspans = data.pta_tspan
            # Frequencies for this signal (Same frequencies for all pulsars)
            self.freqs = sutils.get_harmonic_frequencies(self.nfreqs, self.tspans) # [nfreqs]  

        # Number of parameters and range
        self.n_parameters = len(self.freqs)
        par_range = jnp.ones((self.n_parameters, 2)) * jnp.array(halflog10_rho_range)[None,:]
        self.parameter_range = par_range # Range for each parameter, shape (n_parameters, 2)

        # No need to pre-compute the basis matrices, compute them on the fly instead.
        #self.basis = None # List of basis matrices for each pulsar [npsr, npsr_toas, nmodes]
        # Uniform prior volume
        diff = par_range[:,1] - par_range[:,0] # high - low
        self.log_prior_volume = jnp.sum( jnp.log(diff) ) 

        # Correlation stuff-----------------------------------------------------
        self.Gamma = orf_funcs.get_orf_matrix(data.psr_pos, self.orf) # [npsr, npsr]
        self.Gamma_inv = jnp.linalg.inv(self.Gamma) # [npsr, npsr]
        # Precompute log-determinant of Gamma
        cf = jsl.cholesky(self.Gamma)
        self.logdet_Gamma = 2*jnp.sum(jnp.log(jnp.diag(cf)))
        # Ensure no NaNs in Gamma_inv or logdet_Gamma
        assert ~jnp.isnan(self.Gamma_inv).any(), "Gamma_inv contains NaNs"
        assert ~jnp.isnan(self.logdet_Gamma), "logdet_Gamma is NaN"
        
        self.fixed_wn = self.data.fixed_wn
        self.fixed_res = self.data.fixed_res

        # get helper arrays for likelihood evaluation
        self.get_helpers = self._get_helpers()

    # Helper methods------------------------------------------------------------
    
    def _get_helpers(self):
        """This helper method returns a jit-ed function to calculate TNT, TNr, rNr, logdet_N objects
        needed for likelihood evaluation. Data analysis settings are extracted from the
        data object.

        Returns
        -------
        new_func: callable
            Function which return TNT, TNr, rNr, logdet_N, etc. helper arrays.
        """

        if self.fixed_wn and not self.fixed_res:
            new_func = partial(self.update_white_matrix_products_unjitted,
                               N_list = self.data.Nmat,
                               white_noise_params = self.data.fixed_white_noise_params,
                               )
            return jit(new_func)

        elif not self.fixed_wn and not self.fixed_res:
            new_func = partial(self.update_white_matrix_products_unjitted,
                               N_list = self.data.Nmat,
                               )
            return jit(new_func)

        elif not self.fixed_wn and self.fixed_res:
            new_func = partial(self.update_white_matrix_products_unjitted,
                               N_list = self.data.Nmat,
                               reff = jnp.concat(self.data.raw_residuals)[:, None],
                               )
            return jit(new_func)

        elif self.fixed_wn and self.fixed_res:
            new_func = partial(self.update_white_matrix_products_unjitted,
                               N_list = self.data.Nmat,
                               white_noise_params = self.data.fixed_white_noise_params,
                               reff = jnp.concat(self.data.raw_residuals)[:, None],)
            return jit(new_func)

    @jit_method
    def get_basis(self):
        """Get the list Fourier design matrix for all pulsars. [npsr, npsr_toas, nmodes]

        This helper method computes the Fourier design matrix for each pulsar based 
        on the TOAs and the frequencies. Each Fourier design matrix has dimensions
        [npsr_toas, nmodes]. Ordered as sine and cosine for each frequency:
        i.e. columns are ordered as [sin(2pi f1 t), cos(2pi f1 t), sin(2pi f2 t), cos(2pi f2 t), ...]

        Returns
        -------
        List of arrays
             The list of Fourier design matrices for each pulsar. [npsr, npsr_toas, nmodes]
        """
        T = [sutils.get_fourier_design_matrix(t, self.freqs, microseconds=False) 
             for t in self.psr_toas] # List of [npsr_toas, nmodes]
        return T # [npsr, npsr_toas, nmodes]

    # Helper methods------------------------------------------------------------

    def update_white_matrix_products_unjitted(self, N_list, white_noise_params, reff):
        """Get the helper objects for likelihood evaluation.

        This method computes the helper objects TNT, TNr, rNr, and logdet_N for each pulsar, which are
        needed for the likelihood evaluation and posterior drawing. The TNT, TNr, rNr, and logdet_N
        are computed as:
        - TNT = T^T N^{-1} T
        - TNr = T^T N^{-1} r
        where T is the Fourier design matrix for each pulsar, N is the white noise
        covariance matrix for each pulsar, and r is the effective residuals.

        Parameters
        ----------
        N_list : list of Atlas.nMatrix.base.Base_TOA_cov
            The white noise covariance matrices for each pulsar.
        reff : list of arrays
            The effective residuals for each pulsar. [npsr, npsr_toas]

        Returns
        -------
        tuple
            The helper objects (TNT, TNr, rNr, and logdet_N) for each pulsar. 
            [npsr, nmode, nmode], [npsr, nmode]
        """
        return N_list.get_red_helpers(red_noise_basis = self.get_Fmat_concat, 
                                      residuals = reff, 
                                      white_noise_params = white_noise_params) # [FNF, FNr, rNr, logdetN]
                                      
    @jit_method
    def get_phi_diag(self, params):
        """Get the diagonal of the phi matrix from the parameters. [nmodes]

        This helper method transforms the input parameters (halflog10_rhos) 
        into the diagonal of the phi matrix (fourier coefficients covariance).
        Note that this only returns the diagonal, meaning assuming a stationary
        process.

        Parameters
        ----------
        params : array
             The input parameters, which are halflog10_rhos for each frequency. [nfreqs]

        Returns
        -------
        array
             The diagonal of the phi matrix. [nmodes]
        """
        # params is halflog10_rhos [nfreq]
        # From halflog10_rho to linear rho, then repeat for sine and cosine modes
        phi_diag = jnp.repeat( 10**(2*params), 2) # [nmode]
        return phi_diag # [nmodes]

    @jit_method
    def get_phi(self, phi_diag):
        """
        Get red noise covariance matrix for Fourier
        coefficients from diagonal elements.
        """
        return phi_diag[:,None,None] * self.Gamma[None,:,:]

    @jit_method
    def get_phiinv(self, phi_diag):
        """
        Get inverse red noise covariance matrix for Fourier
        coefficients from diagonal elements.
        """
        return 1/phi_diag[:,None,None] * self.Gamma_inv[None,:,:]

    @jit_method
    def get_sigma(self, TNT, phi_diag):
        """Get the full pulsar fourier coefficient covariance matrix Sigma. [npsr*nmodes, npsr*nmodes]

        This helper method computes the full covariance matrix Sigma for the
        pulsar fourier coefficients. This is given by Sigma = TNT + phi^{-1}, where
        TNT is the contribution from the white noise and phi^{-1} is the contribution
        from the GWB signal.

        Parameters
        ----------
        TNT : array
            The TNT matrices for each pulsar [npsr, nmode, nmode]
        phi_diag : array
            The diagonal of the phi matrix [nmode]

        Returns
        -------
        array
            The full pulsar fourier coefficient covariance matrix Sigma. [npsr*nmodes, npsr*nmodes]
        """
        # Full TNT block is simple!
        full_TNT = sutils.blockPsrs2sigma(TNT) # [npsr*nmodes, npsr*nmodes]

        # Need to add phi^{-1}. Use self.Gamma_inv and 1/phi
        phiinv = 1/phi_diag[:,None,None] * self.Gamma_inv[None,:,:] # [nmode, npsr, npsr]
        full_phiinv = sutils.blockModes2sigma(phiinv) # [npsr*nmodes, npsr*nmodes]
        sigma = full_TNT + full_phiinv # [npsr*nmodes, npsr*nmodes]
        return sigma # [npsr*nmodes, npsr*nmodes]

    @jit_method
    def get_sigma_from_phiinv(self, TNT, phiinv):
        """Get the full pulsar fourier coefficient covariance matrix Sigma. [npsr*nmodes, npsr*nmodes]

        This helper method computes the full covariance matrix Sigma for the
        pulsar fourier coefficients. This is given by Sigma = TNT + phi^{-1}, where
        TNT is the contribution from the white noise and phi^{-1} is the contribution
        from the GWB signal.

        Parameters
        ----------
        TNT : array
            The TNT matrices for each pulsar [npsr, nmode, nmode]
        phi_diag : array
            The diagonal of the phi matrix [nmode]

        Returns
        -------
        array
            The full pulsar fourier coefficient covariance matrix Sigma. [npsr*nmodes, npsr*nmodes]
        """
        # Full TNT block is simple!
        full_TNT = sutils.blockPsrs2sigma(TNT) # [npsr*nmodes, npsr*nmodes]
        full_phiinv = sutils.blockModes2sigma(phiinv) # [npsr*nmodes, npsr*nmodes]
        sigma = full_TNT + full_phiinv # [npsr*nmodes, npsr*nmodes]
        return sigma # [npsr*nmodes, npsr*nmodes]
    
    @jit_method
    def _get_coefficient_realization(self, helpers, params, key):
        """Get multiple realizations of all pulsar fourier coefficients. [npsr, nmodes]

        This helper method generates a random realization of the pulsar fourier 
        coefficients given the helpers (TNT and TNr), the parameters (halflog10_rhos),
        and a random key. The realization is drawn from a multivariate normal
        distribution such that:
        - covariance -> Sigma = TNT + phi^{-1} 
        - mean = Sigma^{-1} TNr

        Parameters
        ----------
        helpers : tuple
            The helpers (TNT and TNr) [npsr, nmode, nmode], [npsr, nmode]
        params : array
            The input parameters, which are halflog10_rhos for each frequency. [nfreqs]
        key : jax.random.PRNGKey
            The random key for generating the realization

        Returns
        -------
        array
            Multiple (self.ndraws) realizations of the pulsar fourier coefficients. [npsr, nmodes]
        """
        TNT, TNr = helpers # Unpack helpers [npsr, nmode, nmode], [npsr, nmode]
        phi_diag = self.get_phi_diag(params) # Get phi diagonal from params [nmode]
        Sigma = self.get_sigma(TNT, phi_diag) # [npsr*nmodes, npsr*nmodes]
        cf = jsl.cho_factor(Sigma) # [npsr*nmodes, npsr*nmodes]
        TNr_flat = sutils.blockVec2sigmaVec(TNr)[:,None] # [npsr, nmode] -> [npsr*nmodes, 1]

        # Get the mean (covariance is Sigma)
        mean = jsl.cho_solve(cf, TNr_flat) # [npsr*nmodes, 1]

        # Transform a unit mean Gaussian random variable to desired distribution
        U = jrandom.normal(key, shape=(self.npsrs*self.nmodes, self.ndraws)) # [npsr*nmodes, ndraws]
        # Project U into the desired distribution
        Up = (mean + jsl.solve_triangular(cf[0], U)) # [npsr*nmodes, ndraws]
        coef = Up.reshape(self.npsrs, self.nmodes, self.ndraws)# [npsr*nmodes, ndraws] -> [npsr, nmodes, ndraws]
        return coef # [npsr, nmodes, ndraws]

    @jit_method
    def _get_coefficient_realization_from_phiinv(self, helpers, phiinv, key):
        """Get multiple realizations of all pulsar fourier coefficients. [npsr, nmodes]

        This helper method generates a random realization of the pulsar fourier 
        coefficients given the helpers (TNT and TNr), the parameters (halflog10_rhos),
        and a random key. The realization is drawn from a multivariate normal
        distribution such that:
        - covariance -> Sigma = TNT + phi^{-1} 
        - mean = Sigma^{-1} TNr

        Parameters
        ----------
        helpers : tuple
            The helpers (TNT and TNr) [npsr, nmode, nmode], [npsr, nmode]
        params : array
            The input parameters, which are halflog10_rhos for each frequency. [nfreqs]
        key : jax.random.PRNGKey
            The random key for generating the realization

        Returns
        -------
        array
            Multiple (self.ndraws) realizations of the pulsar fourier coefficients. [npsr, nmodes]
        """
        TNT, TNr = helpers # Unpack helpers [npsr, nmode, nmode], [npsr, nmode]
        Sigma = self.get_sigma_from_phiinv(TNT, phiinv) # [npsr*nmodes, npsr*nmodes]
        cf = jsl.cho_factor(Sigma) # [npsr*nmodes, npsr*nmodes]
        TNr_flat = sutils.blockVec2sigmaVec(TNr)[:,None] # [npsr, nmode] -> [npsr*nmodes, 1]

        # Get the mean (covariance is Sigma)
        mean = jsl.cho_solve(cf, TNr_flat) # [npsr*nmodes, 1]

        # Transform a unit mean Gaussian random variable to desired distribution
        U = jrandom.normal(key, shape=(self.npsrs*self.nmodes, self.ndraws)) # [npsr*nmodes, ndraws]
        # Project U into the desired distribution
        Up = (mean + jsl.solve_triangular(cf[0], U)) # [npsr*nmodes, ndraws]
        coef = Up.reshape(self.npsrs, self.nmodes, self.ndraws)# [npsr*nmodes, ndraws] -> [npsr, nmodes, ndraws]
        return coef # [npsr, nmodes, ndraws]
    
    @jit_method
    def ln_prior(self, params):
        """Compute the log-prior for the GWB signal parameters.

        This method computes the log of the uniform prior on the GWB parameters.
        All parameters are uniform priors with the range specified by 
        self.parameter_range.

        Parameters
        ----------
        params : array
             The input parameters, which are halflog10_rhos for each frequency. [nfreqs]

        Returns
        -------
        float
            The log-prior for the GWB signal parameters. [1]
        """
        state = jnp.logical_and(params >= self.parameter_range[:,0],
                                params <= self.parameter_range[:,1]).all()
        lnprior = jnp.where(state, -self.log_prior_volume, -jnp.inf)
        return lnprior # scalar
    
    
    @jit_method
    def prior_draw(self, key):
        """Draw a set of parameters from the prior distribution for the GWB signal.

        This method generates a random draw from the uniform prior distribution on the
        GWB parameters. The parameters are drawn uniformly from the range specified by
        self.parameter_range.

        Parameters
        ----------
        key : jax.random.PRNGKey
             The random key for generating the draw.

        Returns
        -------
        array
            The drawn parameters. [n_parameters]
        """
        params = jrandom.uniform(key, shape=(self.n_parameters,),
                                 minval=self.parameter_range[:,0],
                                 maxval=self.parameter_range[:,1]) # [n_parameters]
        return params # [n_parameters]
    
    @jit_method
    def posterior_draw(self, helpers, params, key):
        """Draw a batched (as many as self.draws) set of parameters from 
        the posterior distribution for the GWB signal.

        This method generates a random draw from the posterior distribution on the 
        GWB parameters. This method is often called "gibbs sampling" in the literature,
        but since we are using blocked-gibbs sampling, we opt to label it as "drawing
        from the posterior" instead. Since the fourier coefficients distribution for
        a set of PSDs is analytically known and the distribution of PSDs is known
        for a set of fourier coefficients, we can directly draw from the posterior
        using this method.

        Parameters
        ----------
        helpers : tuple
            The helper objects (TNT and TNr) for each pulsar. 
            [npsr, nmode, nmode], [npsr, nmode]
        params : array
             The current parameters, which are halflog10_rhos for each frequency. [nfreqs]
        key : jax.random.PRNGKey
            The random key for generating the draw.

        Returns
        -------
        array            
            The drawn parameters from the posterior distribution. [nfreqs]
        """
        key1, key2 = jrandom.split(key)
        coef = self._get_coefficient_realization(helpers, params, key1) # [npsr, nmodes, ndraws]
            
        a_sin = coef[:,0::2].transpose((2, 0, 1)) # [ndraws, npsr, nfreqs]
        a_cos = coef[:,1::2].transpose((2, 0, 1)) # [ndraws, npsr, nfreqs]
        gamma_inv = self.Gamma_inv[None] # [1, npsr, npsr]

        # Compute the products a^T Gamma^-1 a for sine and cosine coefficients
        # (gamma_inv @ a_sin) is [npsr, nfreqs]. Then dot with a_sin to get [nfreqs]
        aGa_sin = jnp.sum(a_sin * (gamma_inv @ a_sin), axis=1) # [ndraws, nfreqs]
        aGa_cos = jnp.sum(a_cos * (gamma_inv @ a_cos), axis=1) # [ndraws, nfreqs]

        # Add the sine and cosine contributions together
        aGa = aGa_sin + aGa_cos # [ndraws, nfreqs]

        # Compute the parameters for the inverse-gamma distribution
        alpha = self.npsrs
        beta = 0.5 * aGa # [ndraws, nfreqs]

        # Draw from the inverse-gamma distribution for each frequency
        new_psd = beta / jrandom.gamma(key2, alpha, shape=beta.shape)

        halflog10_rho = 0.5 * jnp.log10(new_psd) # [ndraws, nfreqs]

        # Check if the drawn parameters are within the prior range
        valid = jnp.logical_and(halflog10_rho > self.parameter_range[:,0],
                                halflog10_rho < self.parameter_range[:,1]).all()
        
        return jnp.where(valid, halflog10_rho, params) # [ndraws, nfreqs]

    @jit_method
    def posterior_draw_from_coeff(self, coef, key):
        """Draw a batched (as many as self.draws) set of parameters from 
        the posterior distribution for the GWB signal.

        This method generates a random draw from the posterior distribution on the 
        GWB parameters. This method is often called "gibbs sampling" in the literature,
        but since we are using blocked-gibbs sampling, we opt to label it as "drawing
        from the posterior" instead. Since the fourier coefficients distribution for
        a set of PSDs is analytically known and the distribution of PSDs is known
        for a set of fourier coefficients, we can directly draw from the posterior
        using this method.

        Parameters
        ----------
        helpers : tuple
            The helper objects (TNT and TNr) for each pulsar. 
            [npsr, nmode, nmode], [npsr, nmode]
        params : array
             The current parameters, which are halflog10_rhos for each frequency. [nfreqs]
        key : jax.random.PRNGKey
            The random key for generating the draw.

        Returns
        -------
        array            
            The drawn parameters from the posterior distribution. [nfreqs]
        """
        key1, key2 = jrandom.split(key)
        # Get a realization of the coefficients
            
        a_sin = coef[:,0::2].transpose((2, 0, 1)) # [ndraws, npsr, nfreqs]
        a_cos = coef[:,1::2].transpose((2, 0, 1)) # [ndraws, npsr, nfreqs]
        gamma_inv = self.Gamma_inv[None] # [1, npsr, npsr]

        # Compute the products a^T Gamma^-1 a for sine and cosine coefficients
        # (gamma_inv @ a_sin) is [npsr, nfreqs]. Then dot with a_sin to get [nfreqs]
        aGa_sin = jnp.sum(a_sin * (gamma_inv @ a_sin), axis=1) # [ndraws, nfreqs]
        aGa_cos = jnp.sum(a_cos * (gamma_inv @ a_cos), axis=1) # [ndraws, nfreqs]

        # Add the sine and cosine contributions together
        aGa = aGa_sin + aGa_cos # [ndraws, nfreqs]

        # Compute the parameters for the inverse-gamma distribution
        alpha = self.npsrs
        beta = 0.5 * aGa # [ndraws, nfreqs]

        # Draw from the inverse-gamma distribution for each frequency
        new_psd = beta / jrandom.gamma(key2, alpha, shape=beta.shape)

        halflog10_rho = 0.5 * jnp.log10(new_psd) # [ndraws, nfreqs]

        return halflog10_rho # [ndraws, nfreqs]