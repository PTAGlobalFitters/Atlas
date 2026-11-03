import itertools
import jax.numpy as jnp
import jax.random as jar
import numpy as np
import random, copy
from tqdm import trange
from tqdm_joblib import ParallelPbar
from joblib import delayed

class Sim(object):
    """
    Simulate pulsar timing array (PTA) residual data.

    This class generates simulated PTA residuals by injecting stochastic red
    noise, white noise, and timing-model contributions. 
    The simulated residuals are produced
    on the same Fourier basis used by the detection pipeline, ensuring
    consistency between simulation and parameter estimation.

    The simulator supports both correlated and uncorrelated red-noise
    processes. Correlated processes (e.g., a stochastic gravitational-wave
    background) are generated using the Cholesky decomposition of the
    cross-pulsar covariance matrix, while uncorrelated processes are generated
    independently for each pulsar.

    Multiple independent realizations can be generated simultaneously using
    JAX's pseudo-random number generator.

    Parameters
    ----------
    data : object
        PTA data object containing the pulsar information. The object is
        expected to provide at least

        - ``psrs`` : list of pulsar objects
        - ``npsrs`` : number of pulsars
        - ``raw_residuals`` : residual arrays used to determine the number
          of TOAs for each pulsar.

    red_noise_basis : ndarray
        Fourier design matrix used to transform sampled Fourier coefficients
        into time-domain red-noise residuals.

    cor_signal : object/bool, optional
        Object describing a correlated red-noise process. If provided,
        correlated Fourier coefficients are generated.

    uncor_signal : object/bool, optional
        Object describing an uncorrelated red-noise process. Used only when
        ``cor_signal`` is not supplied.

    parameterized_model : object, optional
        Model used to construct the Fourier-domain covariance matrix through
        ``get_phi_mat_full()``.

    timing_model : object, optional
        Timing-model object providing deterministic timing residuals through
        ``residuals_per_pulsar()``.

    white_noise_model : object, optional
        White-noise model containing covariance matrices for each pulsar.

    N_realizations : int, optional
        Number of independent simulated realizations to generate.
        Default is 1.

    seed : int, optional
        Random seed used to initialize the JAX random number generator.
        If omitted, a random integer seed is chosen.

    Attributes
    ----------
    Fmat : ndarray
        Fourier basis matrix.

    psrs : list
        Deep copy of the pulsars stored in the input data object.

    Npulsars : int
        Number of pulsars.

    real : int
        Number of realizations generated simultaneously.

    rngkeys : jax.Array
        Collection of JAX random keys, one per pulsar plus one additional key
        used for generating red-noise coefficients.

    has_cor : bool
        Whether correlated red noise is enabled.

    has_uncor : bool
        Whether uncorrelated red noise is enabled.

    has_timing : bool
        Whether deterministic timing-model residuals are included.

    has_white : bool
        Whether white noise is included.
    """

    def __init__(self,
                data,
                red_noise_basis,
                cor_signal=None,
                uncor_signal=None,
                parameterized_model=None,
                timing_model=None,
                white_noise_model=None,
                N_realizations=1,
                seed=None):
        """
        Initialize the simulation object.

        The constructor stores the supplied models, determines which simulation
        components are active, creates a local copy of the pulsars, and
        initializes JAX random-number generator keys.

        Parameters
        ----------
        data : object
            PTA dataset.

        red_noise_basis : ndarray
            Fourier basis matrix.

        cor_signal : object, optional
            Correlated red-noise signal model.

        uncor_signal : object, optional
            Uncorrelated red-noise signal model.

        parameterized_model : object, optional
            Model used to compute the Fourier covariance matrix.

        timing_model : object, optional
            Timing-model residual generator.

        white_noise_model : object, optional
            White-noise covariance model.

        N_realizations : int, optional
            Number of independent realizations.

        seed : int, optional
            Random seed for JAX.
        """

        self.Fmat = red_noise_basis

        self.cor_signal = cor_signal
        self.uncor_signal = uncor_signal
        self.parameterized_model = parameterized_model
        self.timing_model = timing_model
        self.white_noise_model = white_noise_model

        self.has_cor = False
        self.has_uncor = False
        self.has_timing = False
        self.has_white = False

        if self.cor_signal:
            self.has_cor = True
        if self.uncor_signal:
            self.has_uncor = True
        if self.timing_model:
            self.has_timing = True
        if self.white_noise_model:
            self.has_white = True

        self.data = data
        self.Npulsars = data.npsrs
        self.real = N_realizations

        if seed:
            int_key = int(seed)
        else:
            int_key = random.randint(10, 100000)
        self.rngkeys = jar.split(
            jar.key(int_key), num=self.Npulsars + 1
        )

    def get_red_coeff(self, red_params):
        """
        Generate Fourier-domain red-noise coefficients.

        Draw random Gaussian Fourier coefficients and transform them according
        to the supplied red-noise covariance model.

        If a correlated signal model is active, the coefficients are generated
        using the Cholesky factor of the full cross-pulsar covariance matrix.
        Otherwise, independent coefficients are generated for each pulsar.

        Parameters
        ----------
        red_params : array_like
            Parameters defining the red-noise covariance model.

        Returns
        -------
        jax.Array
            Simulated Fourier coefficients with shape

            ``(N_realizations, N_modes, 1, N_pulsars)``.

        Notes
        -----
        The covariance matrix is obtained from

        ``parameterized_model.get_phi_mat_full(red_params)``.

        The returned coefficients are intended to be projected into the
        time domain using the Fourier basis matrix.
        """

        phimat = self.parameterized_model.get_phi_mat_full(red_params)[None] #[1, nmodes, npsrs, npsrs] or #[1, nmodes, npsrs]
        nu = jar.normal(
                key=self.rngkeys[-1], shape=(self.real, 2 * phimat.shape[1], self.Npulsars, 1)
            )

        if self.has_cor:
            L = jnp.linalg.cholesky(phimat) 
            coeff = jnp.repeat(L, 2, axis=1) @ nu #[1, nmodes, npsrs, npsrs] @ [n_real, nmodes, npsrs, 1] -> [n_real, nmodes, npsrs, 1]

        elif self.has_uncor and not self.has_cor:
            L = jnp.repeat(jnp.sqrt(phimat), 2, axis=1)[..., None]
            coeff = L * nu #[1, nmodes, npsrs, 1] * [n_real, nmodes, npsrs, 1] - >[n_real, nmodes, npsrs, 1]

        return coeff[:, :, None, :, 0] #[n_real, nmodes, 1, npsrs]

    def sim(self, red_params, white_params, z_tm_concat):
        """
        Generate simulated PTA residuals.

        This method combines red-noise realizations, white-noise realizations,
        and optional timing-model residuals to produce simulated timing
        residuals for every pulsar.

        Parameters
        ----------
        red_params : array_like
            Parameters controlling the red-noise covariance model.

        white_params : array_like
            Parameters defining the white-noise covariance model.

        z_tm_concat : array_like
            Timing-model parameters passed to the timing model.

        Returns
        -------
        sim_res : list
            List containing simulated residual arrays for each pulsar.

        red_res : jax.Array
            Red-noise contribution computed for the final pulsar processed.

        white_res : jax.Array or float
            White-noise contribution computed for the final pulsar processed.

        timing_res : jax.Array
            Timing-model contribution for the final pulsar processed.
        """

        sim_res = []
        a = self.get_red_coeff(red_params)
        ####################################Timing####################################
        if self.has_timing:
            timing_res = self.timing_model.timing_model_residuals_per_pulsar(z_tm_concat)  #jagged_list [n_psrs, n_toas] 
        else:
            timing_res = jnp.zeros(self.Npulsars)

        start_wn_index = 0
        start_rn_index = 0

        for pidx in trange(self.Npulsars):

            ####################################White Noise####################################
            if self.has_white:
                Nmat = self.white_noise_model.cov_matrices[pidx]
                stop_wn_index = start_wn_index + Nmat.n_params   # 2 or 3 per backend (ECORR optional)
                rand_white = jar.normal(self.rngkeys[pidx], shape=(Nmat.ntoas, 1))
                white_helper = Nmat.get_nvec_jvec(
                    white_params[start_wn_index:stop_wn_index]
                )
                white_res = Nmat.CholN_dot(white_helper, rand_white).T
                start_wn_index = stop_wn_index
            else:
                white_res = 0.

            ####################################Red Noise####################################
            stop_rn_index = start_rn_index + self.data.raw_residuals[pidx].shape[0]
            F = self.Fmat[start_rn_index:stop_rn_index][None]  # [1, n_toas, nmodes]
            red_res = (F @ a[..., pidx])[..., 0] #[1, n_toas, nmodes] @ [n_real, nmodes, 1] - >[n_real, n_toas]
            start_rn_index = stop_rn_index

            sim_res.append(red_res + white_res + timing_res[pidx][None])

        return sim_res

    def write_to_psrs(self, 
                    residual_list, 
                    overwrite=False,
                    load_how_many_in_parallel = 10):
        """
        Write simulated residuals into copies of the pulsar objects.

        A deep copy of every pulsar is created for each realization, allowing
        simulated datasets to be generated without modifying the original PTA
        dataset.

        Parameters
        ----------
        residual_list : list
            Simulated residual arrays returned by :meth:`sim`.

        overwrite : bool, optional
            If True, replace the existing pulsar residuals with the simulated
            residuals.

            If False (default), add the simulated residuals to the existing
            residuals.

        Returns
        -------
        list
            A list of pulsar lists, one for each simulated realization.

        """
        # all_poss = itertools.product(np.arange(self.real), np.arange(self.Npulsars))
        # def doit(id):
        #     rr, pidx = id
        #     psr_copy = copy.deepcopy(self.data.psrs[pidx])
        #     if overwrite:
        #         psr_copy._residuals = np.array(residual_list[pidx][rr])
        #     else:
        #         psr_copy._residuals += np.array(residual_list[pidx][rr])

        #     return psr_copy

        # njobs = int(load_how_many_in_parallel)
        # return ParallelPbar("Modifying the Pickle Files...")(n_jobs=njobs)(
        #     delayed(doit)(id) for id in all_poss)

        ans = []
        for rr in trange(self.real):
            psrs_copy = copy.deepcopy(self.data.psrs)
            for pidx, psr in enumerate(psrs_copy):

                if overwrite:
                    psr._residuals = np.array(residual_list[pidx][rr])
                else:
                    psr._residuals += np.array(residual_list[pidx][rr])

            ans.append(psrs_copy)

        return ans