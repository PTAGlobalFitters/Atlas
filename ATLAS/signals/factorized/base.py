from ATLAS.utils import jit_method, jit
from ATLAS.signals import signals_utils as sutils
from ATLAS.signals.factorized.utils import make_irn_model
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jax.random as jrandom
from ATLAS import parameterized
from ATLAS.psd_functions import free_spectrum, make_free_spectrum
from tqdm import trange
from functools import partial
import numpy as np
import random
from sklearn.decomposition import PCA
from functools import cached_property

# Dispersion delays scale as f^-2: the chromatic index a `dm` block is built with.
DM_CHROM_INDEX = 2.0

class Red:
    """Per-pulsar red-noise signal on a Fourier (sine/cosine) basis.

    Builds the Fourier design matrix for every pulsar, holds the PSD model and
    its prior bounds, and supplies the white-noise helper products
    (TNT, TNr, rNr, logdet_N) the likelihoods consume.  The
    ``*_freespec`` methods implement the blocked-Gibbs conditional draws of
    Laal et al. and are valid only for a FREE-SPECTRUM ``psd_function``.

    Parameter layout: a flat ``[npsr * nfreqs]`` vector of ``halflog10_rho``,
    pulsar index varying slowest, matching ``sutils.sigmaVec2blockVec``.
    """

    def __init__(self,
                 name,
                 data,
                 psd_function,
                 lower_bound_psd,
                 upper_bound_psd, 
                 nfreqs=10, 
                 halflog10_rho_range=(-9,-2), 
                 use_pulsar_tspan=False,
                 posterior_draw_ndraws = 1,
                 user_freqs = None,
                ):
        """
        The ATLAS red noise signal for uncorrelated processes.
        This signal models the uncorelated red noise.

        Parameters
        ----------
        name : str
            The name of the signal.
        data : Atlas.data.Data.PTA_Data
            Atlas data object.
        psd_function: callable
            a functools.partial function capable of modeling red noise PSD
        lower_bound_psd: jax.array
            the lower prior bound ordered exacly as how the `psd_function` accepts parameters
        upper_bound_psd: jax.array
            the upper prior bound ordered exacly as how the `psd_function` accepts parameters

        nfreqs : int, optional
            The number of frequency bins in the free spectrum, by default 10.
        halflog10_rho_range : tuple, optional
            The range of the halflog10_rho parameters, by default (-9,-2).
        use_pulsar_tspan : bool, optional
            Whether to use individual pulsar timespans for frequency calculation, 
            or the PTA timespan, by default False.
        user_freqs: arr, optional
            Frequency bins provided by user to use instead of i/T harmonics.
        """
        # PSD reparameterization
        self.psd_function, self.psd_reparam_helper = make_irn_model(psd_function, 
                                        lower_bound_array = lower_bound_psd, 
                                        upper_bound_array = upper_bound_psd)
        
        self.name = name
        self.data = data
        self.ndraws = posterior_draw_ndraws # Number of posterior draws to run in parallel when sampling_method is 'posterior_draws'

        # Helper attributes needed for the rest of the class--------------------
        self.psr_toas = self.data.toas # Reference to a list of TOA arrays for each pulsar [npsr, npsr_toas]
        self.nfreqs = nfreqs
        self.nmodes = 2*nfreqs # Sine and cosine modes per frequency
        self.npsrs = self.data.npsrs 

        self.use_pulsar_tspan = use_pulsar_tspan # Use pulsar tspan, or PTA tspan?
        # `user_freqs` used to default to jnp.array([False]) and be tested with
        # .any(); that silently ignored a user array of all-zero frequencies and
        # raised AttributeError on a plain Python list.  None is the sentinel now,
        # and the old sentinel is still accepted so existing callers keep working.
        if user_freqs is not None and not (
                getattr(user_freqs, 'dtype', None) == jnp.bool_ and not jnp.asarray(user_freqs).any()):
            self.freqs = jnp.asarray(user_freqs)
            self.nfreqs = len(self.freqs)
            self.nmodes = 2 * self.nfreqs
            # Set even on this branch: get_basis()/diagnostics read self.tspans,
            # which was previously left undefined whenever user_freqs was given.
            self.tspans = data.psr_tspans if use_pulsar_tspan else data.pta_tspan
        else:    
            if use_pulsar_tspan:
                self.tspans = data.psr_tspans # Array of individual pulsar timespans [npsrs]
                # Frequencies for this signal (Different frequencies for each pulsar)
                self.freqs = []
                for i in range(self.npsrs):
                    f = sutils.get_harmonic_frequencies(self.nfreqs, self.tspans[i]) # [nfreqs]
                    self.freqs.append(f)
                self.freqs = jnp.array(self.freqs) # [npsrs, nfreqs]
            else:
                self.tspans = data.pta_tspan
                # Frequencies for this signal (Same frequencies for all pulsars)
                self.freqs = sutils.get_harmonic_frequencies(self.nfreqs, self.tspans) # [nfreqs]  

        # Length of the parameter vector.  This was len(self.freqs), which is
        # nfreqs when the frequencies are shared and npsrs when they are
        # per-pulsar -- neither matches the [npsr*nfreqs] layout that
        # get_phi_diag / ln_prior_freespec / posterior_draw_freespec all assume.
        # With npsrs > 1 the old value made ln_prior_freespec's comparison
        # broadcast-incompatible and prior_draw_freespec return the wrong length.
        self.n_parameters = self.npsrs * self.nfreqs
        par_range = jnp.ones((self.n_parameters, 2)) * jnp.array(halflog10_rho_range)[None,:]
        self.parameter_range = par_range # Range for each parameter, shape (n_parameters, 2)

        # Uniform prior volume
        diff = par_range[:,1] - par_range[:,0] # high - low
        self.log_prior_volume = jnp.sum( jnp.log(diff) ) 
        
        # extract data analysis settings from data object
        self.fixed_wn = data.fixed_wn
        self.fixed_res = data.fixed_res
        self.linear_timing = data.linear_timing

        # Hidden attributes if needed-------------------------------------------
        # PSD range in linear space for computations.
        self._psd_range = 10**(2*jnp.array(halflog10_rho_range, dtype=float)) 
        self._diag_idx = jnp.arange(self.nmodes)

        # Built BEFORE _get_helpers() so the helper wrapper can hand it over as a
        # traced argument.  (Despite the name, this is an ARRAY, not a callable.)
        self.get_red_basis = jnp.concat(self.get_basis()) # [n_toas, nmodes]

        # function to get helper objects for likelihood
        self.get_helpers = self._get_helpers()

    # Helper methods------------------------------------------------------------
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
        if self.use_pulsar_tspan:
            # self.freqs is [npsr, nfreqs]
            T = [sutils.get_fourier_design_matrix(t, f, microseconds=False)
                 for t,f in zip(self.psr_toas, self.freqs)] # List of [npsr_toas, nmodes]
        else:
            # self.freqs is [nfreqs], same for all pulsars
            T = [sutils.get_fourier_design_matrix(t, self.freqs, microseconds=False)
                 for t in self.psr_toas] # List of [npsr_toas, nmodes]
        return T # [npsr, npsr_toas, nmodes]
    

    def _get_helpers(self):
        """This helper method returns a jit-ed function to calculate TNT, TNr, rNr, logdet_N objects
        needed for likelihood evaluation. Data analysis settings are extracted from the
        data object.

        Returns
        -------
        new_func: callable
            Function which return TNT, TNr, rNr, logdet_N, etc. helper arrays.
        """

        # Four near-identical branches collapsed into one bound-kwargs dict --
        # same semantics, and the fixed_wn/fixed_res combinations can no longer
        # drift apart.  The basis is passed as a TRACED ARGUMENT rather than
        # captured from self: capturing it makes it a compile-time constant that
        # XLA writes into the executable (see SuperSignal._get_helpers for the
        # 9.5 GB incident on NANOGrav 15yr).
        bound = dict(N_list = self.data.Nmat)
        if self.fixed_wn:
            bound['white_noise_params'] = self.data.fixed_white_noise_params
        if self.fixed_res:
            bound['reff'] = jnp.concat(self.data.raw_residuals)[:, None]

        core = jit(partial(self.update_white_matrix_products_unjitted, **bound))

        def get_helpers(red_noise_basis = None, **kwargs):
            """TNT / TNr / rNr / logdet_N for the current white-noise params.

            ``red_noise_basis`` defaults to this signal's own T-matrix; pass one
            explicitly to override it (e.g. the chromatic-index path, which
            rescales the DM columns).  Any argument not bound at construction
            (``white_noise_params`` when the white noise is free, ``reff`` when
            the residuals are not fixed) must be supplied as a keyword.
            """
            if red_noise_basis is None:
                red_noise_basis = self.get_red_basis
            return core(red_noise_basis = red_noise_basis, **kwargs)

        get_helpers.core = core
        return get_helpers

    @jit_method
    def get_phi_diag(self, params):
        """Get the diagonal of the phi matrix from the parameters. [npsr, nmodes]

        This helper method transforms the input parameters (npsr*halflog10_rhos) 
        into the diagonal of the phi matrix (fourier coefficients covariance) for 
        each pulsar. Note that this only returns the diagonal, meaning assuming 
        a stationary process.

        Parameters
        ----------
        params : array
            The input parameters for the signal [npsr*nfreqs]

        Returns
        -------
        array
            The diagonal of the phi matrix for each pulsar. [npsr, nmodes]
        """
        # params is [nfreqs*npsrs], convert to [npsrs, nfreqs]
        halflog10_rho = sutils.sigmaVec2blockVec(params, self.npsrs) # [npsrs, nfreqs]
        # Convert to PSD values and diagonal matrices
        phi_diag = jnp.repeat( 10**(2*halflog10_rho), 2, axis=1 ) # [npsrs, 2*nfreqs]
        return phi_diag # [npsrs, 2*nfreqs]

    @jit_method
    def get_sigma(self, TNT, phi_diag):
        """Get the per-pulsar fourier coefficient covariance matrix Sigma. [npsr, nmodes, nmodes]

        This helper method computes the per-pulsar covariance matrix Sigma for the
        fourier coefficients. This is given by Sigma = TNT + phi^{-1}, where
        TNT is the contribution from the white noise and phi^{-1} is the contribution
        from the IRN signal.

        Note that this implicitly assumes that the each pulsar is independent.

        Parameters
        ----------
        TNT : array
            The contribution from the white noise for each pulsar. [npsr, nmodes, nmodes]
        phi_diag : array
            The diagonal of the phi matrix for each pulsar. [npsrs, nmodes]

        Returns
        -------
        array
            The covariance matrix Sigma for each pulsar. [npsr, nmodes, nmodes]
        """
        # Since each pulsar is independent, full Sigma is block diagonal, so we can
        # just make each individual pulsar's Sigma matrix instead
        phiinv = 1/phi_diag # [npsrs, 2*nfreqs]
        Sigma = TNT.at[:, self._diag_idx, self._diag_idx].add(phiinv) # [npsr, nmodes, nmodes]
        return Sigma # List of arrays [npsr, nmodes, nmodes]

    @jit_method
    def get_sigma_from_phiinv(self, TNT, phiinv):
        """Get the per-pulsar fourier coefficient covariance matrix Sigma. [npsr, nmodes, nmodes]

        This helper method computes the per-pulsar covariance matrix Sigma for the
        fourier coefficients. This is given by Sigma = TNT + phi^{-1}, where
        TNT is the contribution from the white noise and phi^{-1} is the contribution
        from the IRN signal. phi^{-1} is supplied by the user in this function.

        Note that this implicitly assumes that the each pulsar is independent.

        Parameters
        ----------
        TNT : array
            The contribution from the white noise for each pulsar. [npsr, nmodes, nmodes]
        phiinv : array
            The inverse of the red noise covaraince matrix. [npsrs, nmodes]

        Returns
        -------
        array
            The covariance matrix Sigma for each pulsar. [npsr, nmodes, nmodes]
        """
        # Since each pulsar is independent, full Sigma is block diagonal, so we can
        # just make each individual pulsar's Sigma matrix instead
        Sigma = TNT.at[:, self._diag_idx, self._diag_idx].add(phiinv) # [npsr, nmodes, nmodes]
        return Sigma # List of arrays [npsr, nmodes, nmodes]

    @jit_method
    def _get_coefficient_realization(self, helpers, params, key):
        """Get a realization of all pulsar fourier coefficients. [npsr, nmodes]

        This helper method generates a random realization of the pulsar fourier 
        coefficients given the helpers (TNT, TNr, rNr, and logdet_N), the parameters (halflog10_rhos),
        and a random key. The realization is drawn from a multivariate normal
        distribution such that:
        - covariance -> Sigma = TNT + phi^{-1} 
        - mean = Sigma^{-1} TNr

        Parameters
        ----------
        helpers : tuple
            The helpers (TNT, TNr, rNr, and logdet_N). [npsr, nmodes, nmodes], [npsr, nmodes]
        params : array
            The input parameters for the signal [npsr*nfreqs]
        key : jax.random.PRNGKey
            The random key for generating the realization.

        Returns
        -------
        array
            A realization of the fourier coefficients for each pulsar. [npsr, nmodes]
        """
        TNT, TNr, rNr, logdet_N = helpers # Unpack helpers [npsr, nmode, nmode], [npsr, nmode]
        phi_diag = self.get_phi_diag(params) # Get phi diagonal from params [npsr, nfreqs]
        
        Sigma = self.get_sigma(TNT, phi_diag) # [npsr, nmode, nmode]
        cfs = jsl.cho_factor(Sigma) # Cholesky for each psr [npsr, nmodes, nmodes]

        # Get the mean (covariance is Sigma)
        mean = jsl.cho_solve(cfs, TNr[:,:,None]) # [npsr, nmodes, 1] 

        # Transform a unit mean Gaussian random variable to desired distribution
        U = jrandom.normal(key, shape=(self.npsrs, self.nmodes, self.ndraws)) # [npsr, nmodes, ndraws]
        # Project U into the desired distribution
        coef = (mean + jsl.solve_triangular(cfs[0], U)) # [npsr, nmodes, ndraws]
        return coef # [npsr, nmodes]
    
    # Required methods----------------------------------------------------------
    def update_white_matrix_products_unjitted(self, red_noise_basis, N_list,
                                              white_noise_params, reff):
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
        red_noise_basis : array, [n_toas, nmodes]
            The T-matrix, passed as a traced argument rather than captured.
        N_list : Atlas.nMatrix.base.Base_TOA_cov
            The white-noise covariance model for all pulsars.
        white_noise_params : array
            Current white-noise parameters (bound at construction when fixed).
        reff : array, [n_toas, 1]
            The effective residuals (bound at construction when fixed).

        Returns
        -------
        tuple
            (TNT, TNr, rNr, logdet_N): [npsr, nmode, nmode], [npsr, nmode],
            scalar, scalar.  ``rNr`` and ``logdet_N`` are totals over the
            array, not per pulsar.
        """
        return N_list.get_red_helpers(red_noise_basis = red_noise_basis, 
                                      residuals = reff, 
                                      white_noise_params = white_noise_params) # [FNF, FNr, rNr, logdetN]

    # Required methods----------------------------------------------------------
    @jit_method
    def ln_prior_freespec(self, params):
        """Compute the log-prior for the IRN signal parameters.

        Uniform over ``self.parameter_range``; returns ``-log_prior_volume``
        inside the bounds and -inf outside.  ``params`` is the full
        ``[npsr*nfreqs]`` vector, matching ``n_parameters``.

        Parameters
        ----------
        params : array
             The input parameters, which are halflog10_rhos for each frequency
             for each pulsar. [npsr*nfreqs]

        Returns
        -------
        float
            The log-prior for the  signal parameters. [1]
        """
        state = jnp.logical_and(params >= self.parameter_range[:,0],
                                params <= self.parameter_range[:,1]).all()
        lnprior = jnp.where(state, -self.log_prior_volume, -jnp.inf)
        return lnprior # scalar


    @jit_method
    def prior_draw_freespec(self, key):
        """Draw a set of parameters from the prior distribution for the IRN signal.

        This method generates a random draw from the uniform prior distribution on the
        IRN parameters. The parameters are drawn uniformly from the range specified by
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
    def posterior_draw_freespec(self, helpers, params, key):
        """Draw a set of parameters from the posterior distribution for the IRN signal.

        This method generates a random draw from the posterior distribution on the 
        IRN parameters. This method is often called "gibbs sampling" in the literature,
        but since we are using blocked-gibbs sampling, we opt to label it as "drawing
        from the posterior" instead. Since the fourier coefficients distribution for
        a set of PSDs is analytically known and the distribution of PSDs is known
        for a set of fourier coefficients, we can directly draw from the posterior
        using this method.

        Parameters
        ----------
        helpers : tuple
            The helper objects (TNT, TNr, rNr, and logdet_N) for each pulsar. 
            [npsr, nmode, nmode], [npsr, nmode]
        params : array
             The current parameters, which are halflog10_rhos for each frequency
             for each pulsar. [npsr*nfreqs]
        key : jax.random.PRNGKey
            The random key for generating the draw.

        Returns
        -------
        array            
            The drawn parameters from the posterior distribution. [npsr*nfreqs]
        """
        key1, key2 = jrandom.split(key)
        # Get a realization of the coefficients
        coef = self._get_coefficient_realization(helpers, params, key1) # [npsr, nmodes, ndraws]
        
        # Sum sine and cosine elements
        beta = 0.5*(coef[:,::2]**2 + coef[:,1::2]**2) # [npsr, nfreqs, ndraws]

        # Get exponential terms
        low = jnp.exp(-beta/self._psd_range[0]) # [npsr, nfreqs, ndraws]
        high = jnp.exp(-beta/self._psd_range[1]) # [npsr, nfreqs, ndraws]

        # Uniform values
        U = jrandom.uniform(key2, shape=low.shape, 
                            minval=low, 
                            maxval=high) # [npsr, nfreqs, ndraws]
        
        # Compute equation B3 from Laal et al. 
        new_params = -beta / jnp.log(U) # [npsr, nfreqs, ndraws]

        # Convert back to halflog10_rho parameters
        halflog10_rho = 0.5*jnp.log10(new_params).reshape(-1, self.ndraws) # [npsr*nfreqs, ndraws]
        return halflog10_rho.T # [ndraws, nfreqs*npsrs]

    @jit_method
    def posterior_draw_from_coeff(self, coef, key):
        """As ``posterior_draw_freespec``, but from coefficients already drawn.

        Same conditional (Laal et al. eq. B3); this variant skips
        ``_get_coefficient_realization`` and takes the Fourier coefficients
        directly, for callers that already have a realization in hand.

        NOTE: valid only for a FREE-SPECTRUM psd_function.

        Parameters
        ----------
        coef : array
            A realization of the Fourier coefficients. [npsr, nmodes, ndraws]
        key : jax.random.PRNGKey
            The random key for generating the draw.

        Returns
        -------
        array            
            The drawn parameters from the posterior. [ndraws, npsr*nfreqs]
        """
        # split-and-discard, exactly as before, so the RNG stream is unchanged
        _, key2 = jrandom.split(key)
        
        # Sum sine and cosine elements
        beta = 0.5*(coef[:,::2]**2 + coef[:,1::2]**2) # [npsr, nfreqs, ndraws]

        # Get exponential terms
        low = jnp.exp(-beta/self._psd_range[0]) # [npsr, nfreqs, ndraws]
        high = jnp.exp(-beta/self._psd_range[1]) # [npsr, nfreqs, ndraws]

        # Uniform values
        U = jrandom.uniform(key2, shape=low.shape, 
                            minval=low, 
                            maxval=high) # [npsr, nfreqs, ndraws]
        
        # Compute equation B3 from Laal et al. 
        new_params = -beta / jnp.log(U) # [npsr, nfreqs, ndraws]

        # Convert back to halflog10_rho parameters
        halflog10_rho = 0.5*jnp.log10(new_params).reshape(-1, self.ndraws) # [npsr*nfreqs, ndraws]
        return halflog10_rho.T # [ndraws, nfreqs*npsrs]

class GaussianTiming:
    """A data-driven Gaussian basis for the nonlinear timing model.

    Instead of the analytic design matrix, this signal builds a reduced basis
    from the timing model's own PRIOR PREDICTIVE: it draws ``num_samples``
    parameter vectors from the timing priors, evaluates the model delay for
    each, and takes the leading ``nmodes`` principal components of the
    resulting residual ensemble.  The basis then enters the T-matrix like any
    other Fourier block, with a free-spectrum (or fixed, unit) PSD on it.

    Attributes
    ----------
    name : str
        The name of the signal.
    nmodes : int
        Number of basis columns retained (must be even).  This is also the
        number of PSD parameters per pulsar: the covariance classes are built
        with ``gtm_mode_resolved=True``, so every column gets its own variance
        rather than sharing one with its neighbour.
    nfreqs : int
        nmodes // 2.  Passed on as ``gtm_bins`` because the shared signal
        interface counts basis columns in sin/cos pairs; these columns are
        PCs, not quadrature pairs.
    freqs : array
        Placeholder ones([nmodes]), one per basis column -- the PCs have no
        associated frequency.  A PSD function that actually consumes
        frequencies must NOT be used here.
    U : list of arrays
        Per-pulsar basis, [npsr][n_toa_p, nmodes].
    explained_variance_ratio : list of float
        Fraction of the prior-predictive variance captured per pulsar.
    z_scale : float
        Prior width (in units of the JUG formal sigma) used to draw the
        training ensemble.
    tm_model : MultiPsrTimingModel or None
        Source of the prior draws; None when an explicit ``basis`` was given.
    """
    def __init__(self,
                 name,
                 data,
                 nmodes,
                 gtm_psd = None,
                 lower_bound_psd = None,
                 upper_bound_psd = None,
                 timing_model = None,
                 basis = None,
                 z_scale = 100, 
                 num_trials_for_pca = 10,
                 pca_seed_per_pulsar = None
                ):
        """
        Parameters
        ----------
        name : str
            The name of the signal.
        data : Atlas.data.Data.PTA_Data
            Atlas data object (used for ``npsrs`` and ``psr_names``).
        nmodes : int
            Number of basis columns to keep; must be even.  Also the number of
            PSD parameters per pulsar (one per column).
        gtm_psd : array, optional
            Precomputed phi for the basis, ``(nmodes, npsrs)``.  Forwarded to
            the covariance class, which then samples no GTM parameters.
        lower_bound_psd, upper_bound_psd : array, optional
            Free-spectrum prior bounds, broadcast to ``nmodes`` entries.  Give
            both or neither; neither holds the PSD fixed at unit variance.
        timing_model : MultiPsrTimingModel, optional
            Source of the prior-predictive draws.  Required unless ``basis`` is
            given, and mutually exclusive with it.
        basis : list of array, optional
            Precomputed per-pulsar basis, ``[npsr][n_toa_p, nmodes]``; skips
            the sampling and PCA entirely.
        z_scale : float, default 100
            Width of the affine timing priors used to draw the training set,
            in units of the JUG formal sigma.  Shrunk by 10x per retry when a
            pulsar yields too few finite draws.
        num_trials_for_pca : int, default 10
            Retries allowed per pulsar before giving up.
        pca_seed_per_pulsar : array of PRNGKey, optional
            One key per pulsar; drawn from the global RNG (and recorded in
            ``self.pca_seed``) when not supplied.

        Raises
        ------
        ValueError
            On an odd ``nmodes``, a half-specified bound pair, neither or both
            of ``basis``/``timing_model``, or a pulsar whose prior draws never
            yield enough finite samples.
        """
        if nmodes % 2 != 0:
            raise ValueError(f"Expected an even integer for `nmodes`, got {nmodes}.")
        if (lower_bound_psd is None) != (upper_bound_psd is None):
            raise ValueError(
                "lower_bound_psd and upper_bound_psd must be given together: "
                "pass both to sample a free spectrum over the PCA basis, or "
                "neither to hold it fixed at unit variance.")
        if basis is None and timing_model is None:
            raise ValueError(
                "GaussianTiming needs either an explicit `basis` or a "
                "`timing_model` to draw the prior-predictive training set from.")
        if basis is not None and timing_model is not None:
            raise ValueError(
                "Pass `basis` OR `timing_model`, not both -- an explicit basis "
                "makes the timing model unused, which silently hides a stale basis.")
        self.nmodes = nmodes
        self.nfreqs = int(nmodes/2)
        self.gtm_psd = gtm_psd
        
        # PSD reparameterization
        # One PSD parameter per BASIS COLUMN (nmodes), not per sin/cos pair:
        # these columns are principal components, so two consecutive columns are
        # unrelated and must not be forced to share a variance.  The covariance
        # classes are constructed with gtm_mode_resolved=True to match.
        if lower_bound_psd is None and upper_bound_psd is None:
            # No bounds given -> hold every column at halflog10_rho = 0, i.e.
            # unit prior variance on the PCA coefficients.
            psd_function = make_free_spectrum(self.nmodes)
            param_names = [f"halflog10_rho_{i}" for i in range(self.nmodes)]
            fixed_kwargs = {name: 0.0 for name in param_names}
            self.psd_function, self.psd_reparam_helper = make_irn_model(partial(psd_function, **fixed_kwargs), 
                                            lower_bound_array = jnp.array([]), 
                                            upper_bound_array = jnp.array([]))
        else:
            lower_bound_psd = jnp.broadcast_to(jnp.asarray(lower_bound_psd), (self.nmodes,))
            upper_bound_psd = jnp.broadcast_to(jnp.asarray(upper_bound_psd), (self.nmodes,))
            self.psd_function, self.psd_reparam_helper = make_irn_model(free_spectrum, 
                                            lower_bound_array = lower_bound_psd, 
                                            upper_bound_array = upper_bound_psd)
                                            
        self.name = name
        self.data = data
        self.z_scale = z_scale
        self.num_trials_for_pca = num_trials_for_pca

        if pca_seed_per_pulsar is None:
            # Was seeded from the global `random` module, so a run could not be
            # reproduced even with everything else fixed.  Record the seed we
            # actually used so it can be passed back in.
            self.pca_seed = random.randint(0, 81982)
            self.pca_seeds = jrandom.split(jrandom.key(self.pca_seed), data.npsrs)
        else:
            self.pca_seed = None
            self.pca_seeds = pca_seed_per_pulsar

        self.tm_model = timing_model
        if basis is None:
            (self.U, self.explained_variance_ratio,
             self.timing_residual_training_set) = self._get_basis()
        else:
            self.U = basis
            # Defined unconditionally: downstream code (and any diagnostic) that
            # touches these on the explicit-basis path used to hit AttributeError.
            self.explained_variance_ratio = None
            self.timing_residual_training_set = None

        # Placeholder, ONE PER BASIS COLUMN: the PCA columns are not harmonics,
        # so there is no meaningful frequency to report, but the covariance
        # classes evaluate the GTM PSD once per column when mode-resolved.
        # Kept because SuperSignal.model_maker forwards `.freqs` as `f_gtm`; any
        # PSD that actually uses frequencies would be evaluated at 1 Hz for
        # every column.
        self.freqs = jnp.ones(self.nmodes)

    
    def get_basis(self):
        """Per-pulsar basis matrices, ``[npsr][n_toa_p, nmodes]``."""
        return self.U

    def _fit_pulsar(self, tm_residuals_prior, random_state=0):
        """Leading `nmodes` PCs of a prior-predictive residual ensemble.

        Returns ``(U, explained_variance_ratio)`` or None when too few finite
        draws survive to fit `nmodes` components.
        """
        tm_residuals_prior = np.asarray(tm_residuals_prior)
        mask = np.isfinite(tm_residuals_prior).all(axis=-1)
        tm_residuals_prior = tm_residuals_prior[mask]

        if tm_residuals_prior.shape[0] < self.nmodes:
            return None

        # random_state was drawn from the global RNG, which made the basis
        # irreproducible; it is now derived from the caller's pulsar seed.
        svd = PCA(n_components=self.nmodes, random_state=random_state)
        svd.fit(tm_residuals_prior)
        # whiten=True only affects .transform(), not .components_, so it was a
        # no-op here; the explicit sqrt(explained_variance_) scaling below is
        # what actually sets the column norms.
        U = svd.components_.T * np.sqrt(svd.explained_variance_)[None, :]

        return U, svd.explained_variance_ratio_.sum()

    def _get_basis(self, num_samples=int(1e4)):
        """Per-pulsar PCA basis from prior-predictive timing residuals.

        Returns (bases, explained_variance_ratios, training_sets), each a list
        with one entry per pulsar.
        """
        timing_bases = []
        explained_variance_ratio = []
        training_sets = []
        self.z_scale_used = []

        pbar = trange(self.data.npsrs)
        for pidx in pbar:
            pbar.set_description(
                f"Generating prior samples and performing PCA for {self.data.psr_names[pidx]}")

            # Per-pulsar working copy.  `self.z_scale` was shrunk in place, so
            # one difficult pulsar permanently narrowed the prior for every
            # pulsar after it -- and the recorded z_scale no longer described
            # the bases already built.
            z_scale = self.z_scale
            for num_iters in range(self.num_trials_for_pca):
                # Fold the attempt index into the key: re-drawing with the same
                # key only changed the affine params (z_scale multiplies them);
                # the bounded params (SINI/ECC/M2/PX) ignore z_scale entirely
                # and so were bit-identical on every retry.
                key = (self.pca_seeds[pidx] if num_iters == 0
                       else jrandom.fold_in(self.pca_seeds[pidx], num_iters))
                training_set = self.tm_model.sample_training_residuals(
                    key = key,
                    num_samples = num_samples,
                    z_scale = z_scale,
                    pulsar_index = pidx)
                ans = self._fit_pulsar(training_set, random_state=pidx)
                if ans is not None:
                    timing_bases.append(ans[0])
                    explained_variance_ratio.append(ans[1])
                    training_sets.append(training_set)
                    self.z_scale_used.append(z_scale)
                    break
                print(f'Not enough finite prior samples for '
                      f'{self.data.psr_names[pidx]}; shrinking the prior '
                      f'({z_scale} -> {z_scale/10}).')
                z_scale = z_scale/10
            else:
                raise ValueError(
                    f"Cannot find prior samples for {self.data.psr_names[pidx]} "
                    f"after {self.num_trials_for_pca} attempts (final z_scale "
                    f"{z_scale}).")

        return timing_bases, explained_variance_ratio, training_sets

class SuperSignal:
    """Assembles several signals into one likelihood over a shared T-matrix.

    Takes a list of component signals (``'unc'`` intrinsic red noise, ``'cor'``
    correlated GWB, ``'dm'`` DM noise, ``'gtm'`` Gaussian timing model,
    ``'det'`` deterministic) plus a combination string describing which of them
    share basis columns, concatenates their bases (optionally prefixed by the
    linear timing-model design matrix), builds the matching
    ``parameterized`` covariance model, and exposes the log-posteriors that
    consume them.

    The combination string is parsed by ``sutils.parse_basis_string``; e.g.
    ``"[T]:unc+cor->unc | dm"`` prefixes the timing columns, lets ``unc`` and
    ``cor`` share one block represented by ``unc``'s basis, and gives ``dm``
    its own.

    Attributes
    ----------
    signal_map : dict[str, signal]
        Component signals by name; ``has_unc`` / ``has_cor`` / ``has_dm`` /
        ``has_gtm`` / ``has_det`` mirror its keys.
    get_Fmat_concat : array, [n_toas, nmodes]
        The assembled T-matrix, ``[timing | shared block | separate blocks]``.
        A ``dm`` block carries the DM chromatic index: each TOA's row is
        scaled by ``(f_ref / f_radio)**DM_CHROM_INDEX``.
    chrom_idxs : slice or None
        Column slice of the ``dm`` block, or None without one.
    signal_comb_idxs : dict[str, slice]
        Column slice of each signal (plus ``'timing'``) within that matrix.
    nmodes : int
        Total column count of the T-matrix.
    nfreqs : int
        ``(nmodes - linear_timing_model_size) // 2``; only meaningful when the
        basis is [timing | one Fourier block].
    linear_timing_model_size : int
        Width of the padded timing block (0 when the timing model is
        marginalized or absent).
    _pad_mask : array, [npsr, linear_timing_model_size]
        1 where a timing column is padding, 0 where it is real.
    model : parameterized.PerPulsarRedNoise | CorrelatedPulsarRedNoise
        The phi model, built by ``model_maker``.
    _phi_is_mode_resolved : bool
        True when ``model`` returns phi at basis-column rather than
        frequency-bin resolution.
    get_helpers : callable
        TNT / TNr / rNr / logdet_N for the assembled basis.
    """
    def __init__(self,
                signal_list,
                signal_combination_string,
                data,
                ):
        """
        Parameters
        ----------
        signal_list : list
            Component signal objects.  Their ``.name`` attributes must be
            unique and are what the combination string refers to.
        signal_combination_string : str
            Which components share basis columns, e.g.
            ``"[T]:unc+cor->unc | dm"``.  A leading ``[T]`` prefixes the linear
            timing-model design matrix.
        data : Atlas.data.Data.PTA_Data
            Supplies the residuals, the design matrices and the analysis flags
            (``fixed_wn``, ``fixed_res``, ``linear_timing``, ``marg``).

        Raises
        ------
        ValueError
            On duplicate signal names, or a combination string naming a signal
            that is not in ``signal_list``.
        """
        self.signal_combination_string = signal_combination_string
        self.data = data
        # extract data anlaysis settings from data object
        self.fixed_wn = self.data.fixed_wn
        self.fixed_res = self.data.fixed_res
        self.linear_timing = self.data.linear_timing
        self.marg_tm = self.data.marg
        self.Mmat = self.data.Mmat
        
        # linear timing model attributes
        if self.marg_tm:
            self.linear_timing_model_size = 0
        else:
            self.linear_timing_model_size = max([x.shape[-1] for x in self.Mmat]) if self.linear_timing else 0
        self.lowest_value_eq_to_zero = 1e-40

        self.signal_map = {s.name: s for s in signal_list}
        if len(self.signal_map) != len(signal_list):
            raise ValueError(
                "signal_list contains duplicate names; the later signal would "
                f"silently replace the earlier one: {[s.name for s in signal_list]}")
        for flag in ('unc', 'cor', 'dm', 'gtm', 'det'):
            setattr(self, f'has_{flag}', flag in self.signal_map)
        self.det_signal = self.signal_map['det'] if self.has_det else None

        self.get_Fmat_concat, self.signal_comb_idxs = self.build_basis(self.signal_combination_string, self.signal_map)
        self.chrom_idxs = self.signal_comb_idxs['dm'] if 'dm' in self.signal_comb_idxs.keys() else None
        if self.chrom_idxs is not None:
            # build_basis returns the dm block achromatic; give it the DM index.
            self.get_Fmat_concat = self._scale_chromatic(
                self.get_Fmat_concat, jnp.ones(self.data.npsrs) * DM_CHROM_INDEX)

        # Total column count of the assembled T-matrix: timing + every Fourier
        # block + any deterministic block.  NOT 2*nfreqs of a single signal.
        self.nmodes = self.get_Fmat_concat.shape[-1]
        self.npsrs = self.data.npsrs

        # get helper arrays for likelihood
        self.get_helpers = self._get_helpers()

        # Number of frequency bins for the NumPyro interface.  Only meaningful
        # when the basis is [timing | one Fourier block]; with separate blocks
        # (or a det block) this counts columns that are not sine/cosine pairs.
        self.nfreqs = int((self.nmodes - self.linear_timing_model_size)/2)

        # Hidden attributes if needed-------------------------------------------
        self._diag_idx = jnp.arange(self.nmodes)
        
        # (npsr, linear_timing_model_size) with ones where padded, zero else
        self._pad_mask_list = []
        for M in self.Mmat:
            mask_per_psr = jnp.ones((self.linear_timing_model_size,))
            mask_per_psr = mask_per_psr.at[:M.shape[1]].set(0.)
            self._pad_mask_list.append(mask_per_psr)
        self._pad_mask = jnp.array(self._pad_mask_list)
        
        self.eps_diag_idx = jnp.arange(self.linear_timing_model_size)

        self.model_maker()
        
    @jit_method
    def update_red_basis(self, chrom_index, red_noise_basis=None):
        """
        Return the red noise basis with the dm block's chromatic index set
        per pulsar.

        Args:
            chrom_index: The chromatic index per pulsar.
            The shape of the array must be (npsrs)
            red_noise_basis: The T-matrix to rescale; defaults to
            ``get_Fmat_concat``.  Its dm block must carry the DM index, as
            ``get_Fmat_concat``'s does, so pass the basis as built rather
            than the output of an earlier call.

        Returns:
            the updated red noise basis
        """
        if red_noise_basis is None:
            red_noise_basis = self.get_Fmat_concat
        # Relative to the DM index the basis already carries, so no second,
        # achromatic copy of the dm block has to be stored or traced.
        return self._scale_chromatic(red_noise_basis, chrom_index - DM_CHROM_INDEX)

    def _scale_chromatic(self, basis, chrom_index):
        """Scale each TOA's row of the dm block by ``(f_ref / f_radio)**chrom_index``."""
        index = chrom_index[self.data.dm_exploder_idxs]
        DM = self.data.ref_over_radio_freqs ** index
        return basis.at[:, self.chrom_idxs].multiply(DM[:, None])

    def padd_tm_design_matrix(self):
        """        
        Padd zeros to the timing model design matrix in places where 
        there is no parameter.

        Returns:
            list: padded timing model design matrix in the shape of
            (n_psr, linear_timing_model_size)
        """        
        padded_list = []
        for M in self.Mmat:
            padded = jnp.zeros((M.shape[0], self.linear_timing_model_size))
            padded = padded.at[:, :M.shape[1]].set(M)
            padded_list.append(padded)
        return padded_list
    
    def Tmaker(self, 
                Fmats, 
                padd_tm_design_matrix = True):
        """
        Build the T-matrix as the concatenated basis of 
        timing and red noise (in that order!). There is an
        option to pad to the timing basis of each pulsar with
        zeros so that all timing design matricies have the same size.
        This is more GPU friendly!

        Args:
            Fmats array: the red noise basis matrix (F-mat) for all pulsars
            padd_tm_design_matrix (bool, optional): Do you want to pad 
            the timing design matrix with zeros to extend its size to a 
            common size across pulsar? Defaults to True.

        Returns:
            aray: The T-matrix as[M, F]
        """                
        if padd_tm_design_matrix:
            Mmats  = self.padd_tm_design_matrix()
        else:
            Mmats = self.Mmat
        T = []
        for F, M in zip(Fmats, Mmats):
            T.append(jnp.concat((M, F), axis = -1))
        return T

    def build_basis(self, basis_string, signal_map):
        """
        Build the combined basis matrix and index slices for each signal.

        The timing model columns (if [T] prefix used) are prepended to the
        shared block only. Signal indices are offset accordingly so they
        correctly index into the full T-matrix columns.

        Parameters
        ----------
        basis_string : str
            e.g. "[T]:unc+cor->unc | cw"
        signal_map : dict
            Maps signal names to signal objects,
            e.g. {'unc': sig_unc, 'cor': sig_cor, 'cw': sig_cw}

        Returns
        -------
        Fmat : jnp.ndarray, shape (n_toas, total_cols)
            Full basis matrix [M | F_shared | F_sep1 | ...]
        signal_indices : dict[str, slice]
            Maps each signal name to its column slice in Fmat.
            For shared signals, slices index into the Fourier columns only
            (i.e. offset by linear_timing_model_size, width = sig.nmodes).
            For separate signals, same convention.
        """
        cfg = sutils.parse_basis_string(basis_string)

        include_timing = cfg['include_timing']
        shared_names   = cfg['shared_names']
        rep            = cfg['representative']
        separate_names = cfg['separate_names']
        order          = cfg['order']

        # Validate all names are in signal_map
        missing = set(order) - set(signal_map)
        if missing:
            raise ValueError(f"Signals {missing} not found in signal_map")

        tm_offset = self.linear_timing_model_size if include_timing else 0

        # --- Shared block ---
        if rep is not None:
            raw_basis = signal_map[rep].get_basis()
            shared_Fmat = (
                jnp.concat(self.Tmaker(raw_basis, padd_tm_design_matrix=True))
                if include_timing
                else jnp.concat(raw_basis)
            )
        else:
            shared_Fmat = None

        # --- Separate blocks (never include timing model) ---
        separate_Fmats = {
            name: jnp.concat(signal_map[name].get_basis())
            for name in separate_names
        }

        # --- Assemble in order ---
        Fmats          = []
        signal_indices = {}
        col            = tm_offset   # start after M columns

        # Track timing model indices
        if include_timing:
            signal_indices['timing'] = slice(0, tm_offset)

        shared_block_placed = False
        shared_start        = None

        # With '[T]' but no shared signal the timing columns were never appended
        # to Fmat, while tm_offset still shifted every separate signal's slice --
        # so signal_indices['timing'] pointed at the first separate block and all
        # other slices were off by linear_timing_model_size.  Emit the M block
        # on its own in that case.
        if include_timing and rep is None:
            Fmats.append(jnp.concat(self.padd_tm_design_matrix()))

        for name in order:
            if name in shared_names:
                if not shared_block_placed:
                    Fmats.append(shared_Fmat)
                    shared_start        = col
                    shared_block_placed = True
                    col += shared_Fmat.shape[1] - tm_offset  # advance by F cols only
                # Each shared signal gets a slice of width nmodes within the shared block
                sig_nmodes = signal_map[name].nmodes
                signal_indices[name] = slice(shared_start, shared_start + sig_nmodes)
            else:
                F      = separate_Fmats[name]
                n_cols = F.shape[1]
                Fmats.append(F)
                signal_indices[name] = slice(col, col + n_cols)
                col += n_cols

        Fmat = jnp.concat(Fmats, axis=1)
        return Fmat, signal_indices

    def _get_helpers(self):
        """This helper method returns a jit-ed function to calculate TNT, TNr, rNr, logdet_N objects
        needed for likelihood evaluation. Data analysis settings are extracted from the
        data object.

        The T-matrix is passed to the jitted function as a *traced argument*, never
        bound into the partial. Binding it makes it a compile-time constant, and XLA
        then writes it into the executable as a literal -- twice over, since it also
        keeps a pre-tiled copy for the GEMM, and again for every epoch-gather that
        cannot be constant-folded once the white noise is a runtime value. On the
        NANOGrav 15-year set that inflated the compiled helper build to 9.5 GB, past
        what a 24 GB card will load. As an argument it is the buffer we already own.

        Returns
        -------
        callable
            Function which returns TNT, TNr, rNr, logdet_N, etc. helper arrays. It
            accepts an optional ``red_noise_basis`` to override the default T-matrix
            (used by the chromatic-index path, which rescales the DM columns).
        """
        bound = dict(N_list = self.data.Nmat)
        if self.fixed_wn:
            bound['white_noise_params'] = self.data.fixed_white_noise_params
        if self.fixed_res:
            bound['reff'] = jnp.concat(self.data.raw_residuals)[:, None]

        core = jit(partial(self.update_white_matrix_products_unjitted, **bound))

        def get_helpers(red_noise_basis = None, **kwargs):
            """TNT / TNr / rNr / logdet_N for the current white-noise params.

            ``red_noise_basis`` defaults to the assembled T-matrix; pass one
            explicitly to override it (the chromatic-index path rescales the DM
            columns this way).  Arguments not bound at construction
            (``white_noise_params`` when the white noise is free, ``reff`` when
            the residuals are not fixed) must be supplied as keywords.
            """
            if red_noise_basis is None:
                red_noise_basis = self.get_Fmat_concat
            return core(red_noise_basis = red_noise_basis, **kwargs)

        get_helpers.core = core

        return get_helpers

    def model_maker(self):
        """Build ``self.model``, the phi model for the assembled basis.

        Collects each present component's PSD function, helper dictionary, bin
        count and frequencies, then instantiates
        ``parameterized.CorrelatedPulsarRedNoise`` when a ``'cor'`` signal is
        present and ``parameterized.PerPulsarRedNoise`` otherwise.  Also sets
        ``self._phi_is_mode_resolved``, which tells the likelihoods whether phi
        rows are basis columns or frequency bins.
        """
        model_kwargs = {'pulsar_names':self.data.psr_names}

        if self.has_unc:
            model_kwargs.update(
                irn_psd_func          = self.signal_map['unc'].psd_function,
                irn_helper_dictionary = self.signal_map['unc'].psd_reparam_helper,
                irn_bins              = self.signal_map['unc'].nfreqs,
                f_irn                 = self.signal_map['unc'].freqs,
            )

        if self.has_gtm:
            model_kwargs.update(
                gtm_psd_func          = self.signal_map['gtm'].psd_function,
                gtm_helper_dictionary = self.signal_map['gtm'].psd_reparam_helper,
                gtm_bins              = self.signal_map['gtm'].nfreqs,
                f_gtm                 = self.signal_map['gtm'].freqs,
                gtm_psd               = self.signal_map['gtm'].gtm_psd,
                # The GTM basis columns are principal components, not sin/cos
                # pairs: give each its own PSD value/parameter.
                gtm_mode_resolved     = True,
            )

        if self.has_dm:
            model_kwargs.update(
                dm_psd_func           = self.signal_map['dm'].psd_function,
                dm_helper_dictionary  = self.signal_map['dm'].psd_reparam_helper,
                dm_bins               = self.signal_map['dm'].nfreqs,
                f_dm                  = self.signal_map['dm'].freqs,
            )

        if not self.has_cor:
            model_kwargs.update(dict(
                Npulsars                 = self.data.npsrs,
                signal_indices           = self.signal_comb_idxs,
                linear_timing_model_size = self.linear_timing_model_size,
            )
            )
            self.model = partial(parameterized.PerPulsarRedNoise, **model_kwargs)()
            
        else:
            model_kwargs.update(dict(
                psr_pos                  = self.data.psr_pos,
                Npulsars                 = self.data.npsrs,
                signal_indices           = self.signal_comb_idxs,
                linear_timing_model_size = self.linear_timing_model_size,
                gwb_psd_func             = self.signal_map['cor'].psd_function,
                orf_func                 = self.signal_map['cor'].orf_function,
                gwb_helper_dictionary    = self.signal_map['cor'].psd_reparam_helper,
                crn_bins                 = self.signal_map['cor'].nfreqs,
                f_common                 = self.signal_map['cor'].freqs,
            )
            )
            self.model = partial(parameterized.CorrelatedPulsarRedNoise, **model_kwargs)()

        # phi comes back at basis-column (mode) resolution whenever the GTM
        # block is mode-resolved; otherwise it is at frequency-bin resolution
        # and the consumer has to repeat each bin across its two modes.
        self._phi_is_mode_resolved = bool(
            getattr(self.model, 'gtm_mode_resolved', False)
            and getattr(self.model, 'has_gtm', False))

    def update_white_matrix_products_unjitted(self, red_noise_basis, N_list, white_noise_params, reff):
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
        red_noise_basis : array, [n_toas, nmodes]
            The T-matrix, passed as a traced argument rather than captured.
        N_list : Atlas.nMatrix.base.Base_TOA_cov
            The white-noise covariance model for all pulsars.
        white_noise_params : array
            Current white-noise parameters (bound at construction when fixed).
        reff : array, [n_toas, 1]
            The effective residuals (bound at construction when fixed).

        Returns
        -------
        tuple
            (TNT, TNr, rNr, logdet_N): [npsr, nmode, nmode], [npsr, nmode],
            scalar, scalar.  ``rNr`` and ``logdet_N`` are totals over the
            array, not per pulsar.
        """
        return N_list.get_red_helpers(red_noise_basis = red_noise_basis, 
                                      residuals = reff, 
                                      white_noise_params = white_noise_params) # [FNF, FNr, rNr, logdetN]
                                     
    @jit_method
    def get_sigma(self, TNT, phi_diag):
        """Get the per-pulsar fourier coefficient covariance matrix Sigma. [npsr, nmodes, nmodes]

        This helper method computes the per-pulsar covariance matrix Sigma for the
        fourier coefficients. This is given by Sigma = TNT + phi^{-1}, where
        TNT is the contribution from the white noise and phi^{-1} is the contribution
        from the IRN signal.

        Note that this implicitly assumes that the each pulsar is independent.

        Parameters
        ----------
        TNT : array
            The contribution from the white noise for each pulsar. [npsr, nmodes, nmodes]
        phi_diag : array
            The diagonal of the phi matrix for each pulsar. [npsrs, nmodes]

        Returns
        -------
        array
            The covariance matrix Sigma for each pulsar. [npsr, nmodes, nmodes]
        """
        # Since each pulsar is independent, full Sigma is block diagonal, so we can
        # just make each individual pulsar's Sigma matrix instead
        phiinv = 1/phi_diag # [npsrs, 2*nfreqs]
        Sigma = TNT.at[:, self._diag_idx, self._diag_idx].add(phiinv) # [npsr, nmodes, nmodes]
        return Sigma # List of arrays [npsr, nmodes, nmodes]

    @jit_method
    def get_sigma_from_phiinv(self, TNT, phiinv):
        """Get the per-pulsar fourier coefficient covariance matrix Sigma. [npsr, nmodes, nmodes]

        This helper method computes the per-pulsar covariance matrix Sigma for the
        fourier coefficients. This is given by Sigma = TNT + phi^{-1}, where
        TNT is the contribution from the white noise and phi^{-1} is the contribution
        from the IRN signal. phi^{-1} is supplied by the user in this function.

        Note that this implicitly assumes that the each pulsar is independent.

        Parameters
        ----------
        TNT : array
            The contribution from the white noise for each pulsar. [npsr, nmodes, nmodes]
        phiinv : array
            The inverse of the red noise covaraince matrix. [npsrs, nmodes]

        Returns
        -------
        array
            The covariance matrix Sigma for each pulsar. [npsr, nmodes, nmodes]
        """
        # Since each pulsar is independent, full Sigma is block diagonal, so we can
        # just make each individual pulsar's Sigma matrix instead
        Sigma = TNT.at[:, self._diag_idx, self._diag_idx].add(phiinv) # [npsr, nmodes, nmodes]
        return Sigma # List of arrays [npsr, nmodes, nmodes]

    def ln_likelihood_curn(self, helpers, params):
        """Get the Fourier coefficient marginalized likelihood function for 
        CURN + IRN (i.e., joint common and uncommon) signal.

        This likelihood is marginalized over the fourier coefficients
        and is given by:
        lnlike = (rN^{-1}T)Sigma^{-1}(T^TN^{-1}r) - logdet(Sigma) - logdet(phi))
        
        Parameters
        ----------
        helpers : tuple
            The helper objects (TNT, TNr, rNr, logdet_N) for each pulsar. 
            [npsr, nmode, nmode], [npsr, nmode], [0], [0]
        params : array
            The input parameters, which describe both IRN and GWB

        Returns
        -------
        float
            The log-likelihood contribution from the IRN signal. [1]
        """
        # NOTE: this path assumes T is PURELY Fourier -- phiinv below covers
        # 2*nfreq modes, so with linear-timing columns in T the diagonal add in
        # get_sigma_from_phiinv is shape-incompatible.  Use
        # lnposterior_reparam / partial_marg_lnposterior for a padded T.
        TNT, TNr, rNr, logdet_N = helpers # [npsr, nmode, nmode], [npsr, nmode], [0], [0]

        phi, psd_common = self.model.get_phi_mat_CURN(params) #[nfreq,npsrs]
        phiinv = jnp.repeat(1/phi.T, 2, axis = 1)
        logdet_phis = 2 * jnp.sum(jnp.log(phi), axis = 0)

        Sigma = self.get_sigma_from_phiinv(TNT, phiinv) # [npsr, nmode, nmode]
        cfs = jsl.cho_factor(Sigma) # Cholesky for each psr [npsr, nmodes, nmodes]

        # Log determinant of phi
        diags = jnp.diagonal(cfs[0], axis1=1, axis2=2) # [npsr, nmodes]
        logdet_Sigma = 2*jnp.sum(jnp.log(diags)) # scalar

        # rNT(Sigma^{-1})TNr
        expvals = jnp.sum(TNr[:,:,None] * jsl.cho_solve(cfs, TNr[:,:,None])) # [npsr, nmodes, 1] -> scalar
        lnlike = 0.5 * (expvals - logdet_Sigma - logdet_phis.sum()) # scalar
        return lnlike - 0.5 * (rNr + logdet_N)

    @cached_property
    def partial_marg_lnposterior_helper(self):
        """Column index sets used by both reparameterised log-posteriors.

        Returns
        -------
        cor_idx : slice or None      the GWB block (None when absent)
        P_idx : index set            every non-GWB stochastic block --
                                     timing, IRN, DM, GTM -- merged in column
                                     order
        det_idx : slice or None      the deterministic block

        Raises
        ------
        ValueError
            If the model string carries no non-GWB stochastic block, leaving
            nothing to marginalize or reparameterise over.
        """
        # 'cor' and 'det' are already single slices, so they stay contiguous
        # by construction — no conversion needed.
        #
        # 'cor' is optional. `partial_marg_lnposterior` genuinely requires it --
        # it keeps the GWB block and marginalizes the rest analytically -- and
        # guards for it at its own entry point. `lnposterior_reparam` does not:
        # it merges 'cor' straight back into one block. Indexing it here
        # unconditionally is what made every GWB-free model unusable from
        # a40d71a onward, single-pulsar noise runs included.
        cor_idx = self.signal_comb_idxs.get('cor')
        # 'P' is every non-GWB stochastic block: the linear timing model, the
        # intrinsic red noise, DM noise and the Adaptus basis, in column order.
        # `dm` was missing here, so with a `dm` block in the model string the
        # reparameterised index set omitted its columns entirely and the phi
        # diagonal no longer matched the sliced TNT -- DM noise could be built
        # but never evaluated.
        # Every key is optional: a model string need not carry an 'ltm' prefix,
        # and need not include every stochastic block. Indexing these
        # unconditionally is what made both reparameterised likelihoods raise
        # KeyError('timing') for any string without 'ltm'.
        P_parts = [self.signal_comb_idxs[k]
                   for k in ('timing', 'unc', 'dm', 'gtm')
                   if k in self.signal_comb_idxs]
        if not P_parts:
            raise ValueError(
                f"model string {self.signal_combination_string!r} has no "
                "non-GWB stochastic block, so there is nothing to marginalize "
                "or reparameterise over"
            )
        P_idx = sutils.merge_slices(*P_parts)
        det_idx = None
        if self.has_det:
            det_idx = self.signal_comb_idxs['det']

        return cor_idx, P_idx, det_idx

    @cached_property
    def _jitted_partial_marg_lnposterior(self):
        """Built once per instance and reused — avoids re-tracing on every call."""
        if self.has_det:
            return jit(self.__partial_marg_lnposterior)
        else:
            return jit(partial(self.__partial_marg_lnposterior, D_params=None))

    @cached_property
    def lnposterior_reparam_helper(self):
        """Index slices for lnposterior_reparam.

        Unlike partial_marg_lnposterior, lnposterior_reparam jointly
        reparameterizes ALL non-det modes (timing [+ gtm] + unc + cor) via z —
        there's no analytic marginalization here, so 'cor' and 'P' don't need
        to stay separate. Merge them into one contiguous 'reparam' block,
        with the merge order matching how partial_marg_lnposterior_helper
        itself is built (P = timing+unc+[gtm] first, cor appended after) so
        that `linear_timing_model_size`-based slicing of the timing-model
        prefix still lines up correctly.

        ASSUMPTION: this assumes the underlying T matrix column order is
        [timing (+unc+gtm) | cor | det] i.e. matches partial_marg's P/cor/det
        layout. If your T matrix actually orders columns differently, adjust
        the merge_slices call below accordingly.
        """
        cor_idx, P_idx, det_idx = self.partial_marg_lnposterior_helper
        # A model string with no 'cor' block is fine here: P is then the whole
        # reparameterised set, and nothing below this line consults cor_idx.
        reparam_idx = P_idx if cor_idx is None else \
            sutils.merge_slices_unique(P_idx, cor_idx)
        return reparam_idx, det_idx

    @cached_property
    def _jitted_lnposterior_reparam(self):
        """Built once per instance and reused — avoids re-tracing on every call."""
        if self.has_det:
            return jit(self.__lnposterior_reparam)
        else:
            return jit(partial(self.__lnposterior_reparam, D_params=None))

    def partial_marg_lnposterior(self, helpers, red_params, z, D_params=None):
        """Public entry point — dispatches to the cached jitted implementation.

        Requires a 'cor' block, unlike `lnposterior_reparam`. Checked here
        rather than in `partial_marg_lnposterior_helper`, which
        `lnposterior_reparam_helper` shares and which must stay usable without
        one. Without this the failure is a `TypeError: 'NoneType' object is not
        subscriptable` from inside `block_slice`, several frames down.
        """
        if self.signal_comb_idxs.get('cor') is None:
            raise ValueError(
                "partial_marg_lnposterior keeps the 'cor' (GWB) block and "
                "marginalizes the rest analytically, so it requires one; model "
                f"string {self.signal_combination_string!r} has none. Use "
                "lnposterior_reparam instead."
            )
        if self.has_det:
            return self._jitted_partial_marg_lnposterior(helpers, red_params, z, D_params)
        else:
            return self._jitted_partial_marg_lnposterior(helpers, red_params, z)
            
    def lnposterior_reparam(self, helpers, red_params, z, D_params=None):
        """Public entry point — dispatches to the cached jitted implementation.

        Mirrors partial_marg_lnposterior: det signal is only touched when
        self.has_det is True, in which case D_params = (det_params,
        psr_phases, psr_dists) must be supplied.
        """
        if self.has_det:
            return self._jitted_lnposterior_reparam(helpers, red_params, z, D_params)
        else:
            return self._jitted_lnposterior_reparam(helpers, red_params, z)

    def __lnposterior_reparam(self, helpers, red_params, z, D_params=None):
        """
        Evaluates the posterior under a reparameterization of the Fourier
        coefficients (Gaussian processes described by phi_cube), jointly with
        an optional deterministic (e.g. CW) signal. Call within gradient-based
        samplers.

        Parameters
        ----------
        helpers : tuple
            (TNT, TNr, rNr, logdet_N) for each pulsar, over the FULL unified
            design matrix (timing [+ gtm] + unc + cor [+ det], whichever are
            present) — the same TNT/TNr used by partial_marg_lnposterior.
            [npsr, nmode_total, nmode_total], [npsr, nmode_total]
        red_params : array
            Spectral model parameters for the red noise / GWB covariance.
        z : array
            "Whitened coefficients" for the reparameterized (non-det) modes,
            [npsr, n_reparam].
        D_params : tuple or None
            (det_params, psr_phases, psr_dists). Required iff self.has_det.

        Returns
        -------
        tuple
            (log_density, coeff) — coeff are the reparameterized (non-det)
            Fourier coefficients, [npsr, n_reparam].
        """
        TNT, TNr, rNr, logdet_N = helpers
        reparam_idx, det_idx = self.lnposterior_reparam_helper

        # slice out the jointly-reparameterized block (timing [+gtm] + unc + cor)
        RR = TNT[:, *sutils.block_slice(reparam_idx)]        # [npsr, nreparam, nreparam]
        Rr = TNr[:, sutils.vec_slice(reparam_idx), None]      # [npsr, nreparam, 1]

        red_noise_cov = self.model.get_phi_mat_full(red_params)
        if self.npsrs == 1:
            if self._phi_is_mode_resolved:
                # Already one value per basis column -- no repeat, and the
                # logdet is not doubled.  (Keyed off the model's resolution
                # rather than merely `has_gtm`: a bin-resolved GTM block still
                # needs the repeat below.)
                phiinvs_diags = 1 / red_noise_cov  # [nmodes, npsrs]
                logdet_phimat = jnp.sum(jnp.log(red_noise_cov))
            else:
                phiinvs_diags = jnp.repeat(1 / red_noise_cov, 2, axis=0)  # [nmodes, npsrs]
                logdet_phimat = 2 * jnp.sum(jnp.log(red_noise_cov))  # 2 accounts for 2*nfreq=nmodes
        else:
            phiinvs, logdet_phimat = self.model.get_phi_mat_inv(red_noise_cov)
            phiinvs_diags = phiinvs.diagonal(axis1=-2, axis2=-1)  # [nmodes, npsrs]

        if self.linear_timing and not self.marg_tm:
            # self.nmodes includes deterministic contributions,
            # so shape GP priors from `RR` which only includes reparameterized modes
            phiinvs_diags_ltm = jnp.full(shape=(RR.shape[-1], self.npsrs),
                                        fill_value=self.lowest_value_eq_to_zero)
            phiinvs_diags = phiinvs_diags_ltm.at[self.linear_timing_model_size:, :].add(phiinvs_diags)
            # set prior variance of padded parameters to one for stable transformation
            phiinvs_diags = phiinvs_diags.at[:self.linear_timing_model_size, :].add(self._pad_mask.T)

        # deterministic signal, sliced directly out of the same unified TNT/TNr
        # (no separately-built design matrix / helper tensors)
        if self.has_det:
            det_params, psr_phases, psr_dists = D_params
            a_det = self.det_signal.get_coeffs_func(det_params, psr_phases, psr_dists)[..., None]  # [npsr, ndet, 1]

            DD = TNT[:, *sutils.block_slice(det_idx)]                    # [npsr, ndet, ndet]
            RD = TNT[:, *sutils.block_slice(reparam_idx, det_idx)]       # [npsr, nreparam, ndet]
            Dr = TNr[:, sutils.vec_slice(det_idx), None]                 # [npsr, ndet, 1]

            RDas = RD @ a_det  # [npsr, nreparam, 1]
        else:
            RDas = 0.

        # Posterior precision Cholesky (cho_factor equivalent), batched over pulsars
        diag_idx = jnp.arange(RR.shape[-1])
        Sigma_inv = RR.at[:, diag_idx, diag_idx].add(phiinvs_diags.T)  # [npsr, nreparam, nreparam]
        Sigma_inv_L = jsl.cho_factor(Sigma_inv, lower=True)

        # MAP coefficients — shifted by the deterministic signal's contribution when present
        a_hat = jsl.cho_solve(Sigma_inv_L, Rr - RDas)

        # Standardizing transform via back substitution
        Lz = jax.lax.linalg.triangular_solve(
            Sigma_inv_L[0], z[..., None], left_side=True, lower=True, transpose_a=True,
        )  # L^T Lz = z

        coeff = a_hat + Lz  # [npsr, nreparam, 1]

        lndet_Jac = -jnp.sum(jnp.log(Sigma_inv_L[0].diagonal(axis1=-2, axis2=-1)))

        # Log-likelihood (non-det part)
        aFNr = jnp.sum(coeff[..., 0] * Rr[..., 0])
        aFNFa = jnp.sum(coeff.mT @ RR @ coeff)
        lnlike_value = aFNr - 0.5 * aFNFa

        if self.npsrs == 1:
            lnprior_value = -0.5 * ((coeff[:, self.linear_timing_model_size:, 0]**2 * phiinvs_diags[self.linear_timing_model_size:, :].T).sum() + logdet_phimat)
        else:
            aG = coeff[:, self.linear_timing_model_size:]  # [npsr, 2*nfreq, 1]
            lnprior_value = -0.5 * ((aG.transpose(1, 2, 0) @ phiinvs @ aG.transpose(1, 0, 2)).sum() + logdet_phimat)

        if self.linear_timing and not self.marg_tm:
            # add probability density for padded (i.e. zero-ed) timing model parameters for HMC sampler
            # these parameters do not impact the likelihood, prior, and are uncorrelated with all other
            # parameters so this should not affect parameter estimation, but merely provides some
            # curvature for HMC to latch onto when sampling
            padded_logpdf = -0.5 * jnp.sum((self._pad_mask * coeff[:, :self.linear_timing_model_size, 0])**2)
        else:
            padded_logpdf = 0.

        # deterministic (e.g. CW) contribution
        if self.has_det:
            lnlike_det_add = jnp.sum(a_det.mT @ Dr) \
                - jnp.sum(coeff.mT @ RDas) \
                - 0.5 * jnp.sum(a_det.mT @ DD @ a_det)
        else:
            lnlike_det_add = 0.

        log_density = lnlike_value + lnprior_value + lndet_Jac - 0.5 * (rNr + logdet_N) \
            + padded_logpdf + lnlike_det_add

        return log_density, coeff[..., 0]

    def __partial_marg_lnposterior(self, helpers, red_params, z, D_params=None):
        """Log-posterior with the GWB coefficients kept and the rest marginalized.

        The non-GWB coefficients (timing, IRN, DM, GTM) are integrated out
        analytically; the GWB coefficients are reparameterised as
        ``g = g_hat + L^-T z`` so a gradient sampler can move in ``z``.

        Parameters
        ----------
        helpers : tuple
            (TNT, TNr, rNr, logdet_N) over the full unified basis.
        red_params : array
            Spectral parameters of the red-noise / GWB covariance.
        z : array, [npsr, n_gwb_modes]
            Whitened GWB coefficients.
        D_params : tuple or None
            (det_params, psr_phases, psr_dists); required iff ``self.has_det``.

        Returns
        -------
        log_density : float
        g : array, [npsr, n_gwb_modes]
            The GWB coefficients implied by ``z``.
        """
        # unpack helper objects
        TNT, TNr, rNr, logdet_N = helpers
        cor_idx, P_idx, det_idx = self.partial_marg_lnposterior_helper
        GG = TNT[:, *sutils.block_slice(cor_idx)]
        PP = TNT[:, *sutils.block_slice(P_idx)]
        GP = TNT[:, *sutils.block_slice(cor_idx, P_idx)]
        Gr = TNr[:, sutils.vec_slice(cor_idx), None]
        Pr = TNr[:, sutils.vec_slice(P_idx), None]

        # only touch deterministic-signal terms if the parent class says to
        if self.has_det:
            DD = TNT[:, *sutils.block_slice(det_idx)]
            GD = TNT[:, *sutils.block_slice(cor_idx, det_idx)]
            PD = TNT[:, *sutils.block_slice(P_idx, det_idx)]
            Dr = TNr[:, sutils.vec_slice(det_idx), None]

            det_params, psr_phases, psr_dists = D_params
            d = self.det_signal.get_coeffs_func(det_params, psr_phases, psr_dists)[..., None]

        # get covariance matrices
        phiinv_G, logdet_phi_G, phiinv_P, logdet_phi_P = self.model.partial_reparm_helper(red_params, self._pad_mask)

        diag_idxs_P = jnp.arange(PP.shape[-1])
        Sigma_inv_P = PP.at[:, diag_idxs_P, diag_idxs_P].add(phiinv_P)

        # natively batched cholesky factor/solve — no vmap needed
        Sigma_inv_P_chol = jsl.cho_factor(Sigma_inv_P, lower=True)
        I_P = jnp.broadcast_to(jnp.identity(Sigma_inv_P.shape[-1]), Sigma_inv_P.shape)
        Sigma_P = jsl.cho_solve(Sigma_inv_P_chol, I_P)
        logdet_Sigma_inv_P = 2 * jnp.sum(jnp.log(jnp.diagonal(Sigma_inv_P_chol[0], axis1=-2, axis2=-1)))

        # Build the quadratic form of the log-posterior.  Every term here stays
        # PER-PULSAR, shape [npsr, 1, 1], and must stay that way until the single
        # `jnp.sum` at the end.  Two traps, both of which were live:
        #   * `rNr` is the ARRAY-WIDE total, accumulated across pulsars inside
        #     `get_red_helpers`.  Folding it into this per-pulsar array subtracts
        #     it once per pulsar.  It is applied exactly once, outside the sum.
        #   * collapsing `U` to a scalar here lets it broadcast back across the
        #     per-pulsar axis when it is added to `g.mT @ V` below, so the final
        #     `jnp.sum` counts it `npsr` times over.
        # Together those scaled the P-block evidence term by `npsr` and `rNr` by
        # `npsr**2` -- and because `Sigma_P` depends on the red-noise parameters,
        # that is a parameter-dependent bias, not a constant offset.
        U = 0.5 * Pr.mT @ Sigma_P @ Pr
        V = -GP @ Sigma_P @ Pr + Gr
        if self.has_det:
            U = U + 0.5 * d.mT @ PD.mT @ Sigma_P @ PD @ d \
                - Pr.mT @ Sigma_P @ PD @ d - 0.5 * d.mT @ DD @ d + d.mT @ Dr
            V = V + GP @ Sigma_P @ PD @ d - GD @ d

        W_inv_without_prior = -GP @ Sigma_P @ GP.mT + GG

        # normalization
        norm = -0.5 * (logdet_N + logdet_phi_G + logdet_phi_P + logdet_Sigma_inv_P)

        # standardizing transformation
        diag_idxs_G = jnp.arange(W_inv_without_prior.shape[-1])
        phiinv_G_diag = jnp.diagonal(phiinv_G, axis1=1, axis2=2).T
        W_inv_curn = W_inv_without_prior.at[:, diag_idxs_G, diag_idxs_G].add(phiinv_G_diag)
        W_inv_curn_L = jsl.cho_factor(W_inv_curn, lower=True)
        g_hat = jsl.cho_solve(W_inv_curn_L, V)
        Lz = jax.lax.linalg.triangular_solve(W_inv_curn_L[0], z[..., None],
                                            left_side=True, lower=True, transpose_a=True)
        g = g_hat + Lz  # [npsr, nmodes, 1]
        lndet_Jac = -jnp.sum(jnp.log(W_inv_curn_L[0].diagonal(axis1=-2, axis2=-1)))

        ln_likelihood = U + g.mT @ V - 0.5 * g.mT @ W_inv_without_prior @ g
        lnprior = -0.5 * (g.transpose(1, 2, 0) @ phiinv_G @ g.transpose(1, 0, 2)).sum()

        # `rNr` enters exactly once, here, outside the per-pulsar sum.
        result = norm + lnprior + lndet_Jac + jnp.sum(ln_likelihood) - 0.5 * rNr
        return result, g