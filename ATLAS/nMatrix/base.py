from ATLAS.utils import jagged2padded, jit_method
from ATLAS.utils import EPOCH_THRESHOLD
from ATLAS.utils import jit
from ATLAS.signals.signals_utils import _timing_model_svd, stabilize_TNT, stabilize_TDNTD

import numpy as np
import itertools
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from functools import partial

def _logdet_total(cov, white_noise_helpers):
    """log|N| (or log|D| with the timing model marginalised) without a solve."""
    logdet = cov.logdet_unmarg(white_noise_helpers)
    if cov.marg:
        NiM = cov._apply_inv_unmarg(white_noise_helpers, cov.Mmat)
        cf = jsl.cho_factor(cov.Mmat.T @ NiM)
        logdet = logdet + 2 * jnp.sum(jnp.log(cf[0].diagonal())) + cov.Mprior
    return logdet


# Helper functions for finding TOA epochs --------------------------------------
def _get_psr_wn_attributes(psr):
    """Get the helper matrices for computing the white noise covariance matrix

    This function computes several helper quantities for computing the N matrix
    and its inverse solutions for a given pulsar. These include:
    - backends: (N_backends) an ordered list of unique backend names for this pulsar
    - B: (N_toa) an array of backend indices for each toa (i.e. which backend each toa belongs to)
    - U_pad: (N_epoch, max_epoch_size) a padded array of toa indices for each epoch [padded with -1]
    - U_mask: (N_epoch, max_epoch_size) a boolean mask indicating which elements of U_pad are valid
    - V: (N_epoch) an array of backend indices for each epoch (i.e. which backend each epoch belongs to)

    The user supplies a pulsar object which must contain the following attributes:
    - toas: (N_toa) array of time of arrivals for this pulsar
    - backend_flags: (N_toa) array of backend names for each toa

    Within a backend, TOAs are grouped into epochs by chaining: a new epoch starts
    wherever two adjacent (time-sorted) toas are separated by `EPOCH_THRESHOLD` or
    more, so an epoch may span more than `EPOCH_THRESHOLD` in total.
    `EPOCH_THRESHOLD` is given in seconds.

    Parameters
    ----------
    psr : object
        The pulsar object to compute the helper arrays/matrices for [check description]
    
    Returns
    -------
    out : tuple
        A tuple containing the following helper arrays/matrices for computing the white noise covariance matrix:
        - backends: (N_backends) an ordered tuple of unique backend names for this pulsar
        - B: (N_toa) an array of backend indices for each toa (i.e. which backend each toa belongs to)
        - U_pad: (N_epoch, max_epoch_size) a padded array of toa indices for each epoch [padded with -1]
        - U_mask: (N_epoch, max_epoch_size) a boolean mask indicating which elements of U_pad are valid
        - V: (N_epoch) an array of backend indices for each epoch (i.e. which backend each epoch belongs to)
    """
    # The array of backends and an array which indicates which backend each toa belongs to
    backends, B = np.unique(psr.backend_flags, return_inverse=True) # [n_backends], [n_toas]
    backends = tuple(backends) # Convert to tuple

    # The indices of toas for each backend (i.e. which toas belong to each backend)
    backend_idx = [np.where(B == i)[0] for i in range(len(backends))]

    U = [] # We don't know how many toas will be in each epoch!
    V = [] # We don't know how many epochs there will be!
    for i in range(len(backends)):
        # Get this backend's toas
        toas = psr.toas[backend_idx[i]]
        isort = np.argsort(toas)
        sort_toas = toas[isort]

        # Find indices where adjacent toas are separated by >= dt (different epochs)
        breaks = np.where(np.diff(sort_toas) >= EPOCH_THRESHOLD)[0] + 1 # (plus 1 since diff)
        epochs = np.split(isort, breaks) # List of arrays of indices for each epoch (in backend indexing)

        # Convert to sorted jax arrays and drop single-toa epochs (in backend indexing)
        epoch_idx = [np.sort(jnp.array(epoch, dtype=int))
                     for epoch in epochs if len(epoch) > 1]

        # Map backend indices to global indices through B[i]
        epochs = [backend_idx[i][e] for e in epoch_idx] 

        U.extend(epochs) # Add these epochs to U
        V.extend([i]*len(epochs)) # Add the backend index for these epochs to V

    # U is a list of arrays of toa indices for each epoch
    # we need to convert this to a padded array for fast einsum computations
    if len(U) == 0:
        # No epoch holds more than one TOA (singletons are dropped above), so
        # there is no ECORR block structure at all -- e.g. a single-frequency
        # campaign with one TOA per observing session. Empty (0, 1) helpers let
        # the per-epoch einsums in the solvers reduce over an empty axis, which
        # contributes exactly zero rather than raising in jagged2padded.
        U_pad = jnp.zeros((0, 1), dtype=int)
        U_mask = jnp.zeros((0, 1), dtype=bool)
    else:
        U_pad, U_mask = jagged2padded(U, pad_value=-1) # [n_epochs, max_epoch_size], [n_epochs, max_epoch_size]
        U_pad, U_mask = jnp.array(U_pad, dtype=int), jnp.array(U_mask, dtype=bool)

    # V is a list of backend indices for each epoch (i.e. which backend each epoch belongs to)
    V = jnp.array(V, dtype=int)
    # B is an array of backend indices for each toa (i.e. which backend each toa belongs to)
    B = jnp.array(B, dtype=int)

    out = (backends, B, U_pad, U_mask, V)
    return out


class DiagSinglePulsarWhiteCov:
    """A simple TOA covariance matrix with only TOA errors. (no extra noise factors)

    This class implements a simplified white noise covariance matrix for a pulsar
    with a fixed pulsar timing model. This model only includes TOA errors and does 
    not include any EFAC, EQUAD, or ECORR parameters. This class is best used for 
    simulated or whitened data.
    
    The covariance matrix can be written as:
        N_ij = delta_ij * sigma_i^2
        where sigma_i is the TOA error for the i-th TOA.
    """

    def __init__(self, psr, marg = False):
        """The constructor for the simplified white noise covariance matrix.

        This constructor initializes the simplified white noise covariance matrix. It
        takes a pulsar object and extracts the TOA errors to construct the diagonal
        covariance matrix.

        Parameters
        ----------
        psr : object
            The pulsar object to construct the covariance matrix for.
        marg: bool:
            Do you want to marginalize over linear timing model errors?
        """
        # All attributes are static in the simple case
        # Static attributes ----------------------------------------------------
        self.psr_name = psr.name
        self.ntoas = len(psr.toas)
        self.toaerrs = psr.toaerrs
        self.marg = marg
        self.n_params = 0 # no EFAC/EQUAD/ECORR here -- nothing to sample
        # See SinglePulsarWhiteCov: only _solve_marg reads these.
        if self.marg:
            self.Mmat = _timing_model_svd(psr.Mmat)
            self.Mprior = self.Mmat.shape[1] * jnp.log(1e40)
        else:
            self.Mmat = None
            self.Mprior = None

        #jitted function that returns (left.T N right)
        self.solve = self._solve_func_maker(return_logdet = False)
        #unmarg solver to be used within marg solver function later
        self.solve_unmarg = partial(self._solve_unmarg, return_logdet = False) 
        #jitted function that returns (left.T N right, logDetN)
        self.solve_with_logdet = self._solve_func_maker(return_logdet = True)

    @jit_method
    def get_nvec_jvec(self):
        """Compute nvec.

        Returns
        -------
        tuple
            A tuple containing:
            - nvec: The diagonal elements of the covariance matrix (the white noise variances). [ntoas]
            - jvec: None.
        """
        nvec = self.toaerrs**2
        jvec = None
        return nvec, jvec

    def _solve_func_maker(self, return_logdet):
        """Create a solver function for the covariance matrix.

        This method returns a JIT-compiled function that solves the linear system
        left^T N^{-1} right, where N is the covariance matrix. The returned function
        can optionally also return the log-determinant of N.

        Parameters
        ----------
        return_logdet : bool
            If True, the returned solver function will also return the log-determinant of the covariance matrix. If False, only the solution to the linear system is returned.

        Returns
        -------
        function
            A JIT-compiled function that solves the linear system left^T N^{-1} right. 
        """
        if self.marg:
            new_func = partial(self._solve_marg, return_logdet = return_logdet)
            return jit(new_func)
        else:
            new_func = partial(self._solve_unmarg, return_logdet = return_logdet)
            return jit(new_func)


    def _apply_inv_unmarg(self, white_noise_helpers, right):
        """``N^{-1} right`` for the diagonal covariance."""
        return right / white_noise_helpers[0][:, None]

    def apply_inv(self, white_noise_helpers, right):
        """``N^{-1} right`` (or ``D^{-1} right`` when the timing model is
        marginalised), as a matrix the size of ``right``.

        For ``left^T N^{-1} right`` with a small ``right`` and a wide ``left``,
        ``left.T @ apply_inv(right)`` touches ``left`` only through one matrix
        product, while ``solve`` also runs its epoch gathers over ``left``.
        """
        x = self._apply_inv_unmarg(white_noise_helpers, right)
        if not self.marg:
            return x
        NiM = self._apply_inv_unmarg(white_noise_helpers, self.Mmat)       # [n, k]
        cf = jsl.cho_factor(self.Mmat.T @ NiM)
        return x - NiM @ jsl.cho_solve(cf, NiM.T @ right)

    def _solve_unmarg(self, white_noise_helpers, left, right, return_logdet):
        """Solve the linear system left^T N^{-1} right.

        This method implements the solution to the linear system left^T N^{-1} right
        for the simplified covariance matrix, which is just a diagonal matrix with the
        TOA errors squared on the diagonal. This method can be JIT compiled in other functions!

        NOTE: the left matrix will be transposed for you and should be [N_toa, N]

        Parameters
        ----------
        white_noise_helpers: tuple
            This contaains (nvec, jvec). jvec is set to None
        left : array-like
            The left-hand side of the linear system. [N_toa, N]
        right : array-like
            The right-hand side of the linear system. [N_toa, M]
        
        Returns
        -------
        array-like
            The solution to the linear system. [N, M]
        """
        nvec = white_noise_helpers[0]
        # Only the diagonal! (L^T Ainv R)
        LNR = left.T @ (1/nvec[:,None] * right) # (N, M)

        if return_logdet:
            return LNR, self.logdet_unmarg(white_noise_helpers)
        else:
            return LNR


    def _solve_marg(self, white_noise_helpers, left, right, return_logdet):
        """Solve the linear system left^T D^{-1} right. 

        This method solves the linear equation (left).T @ D^{-1} @ right, where
        D is the timing-model marginalized covariance matrix.

        NOTE: the left matrix will be transposed for you and should be [N_toa, N]

        Parameters
        ----------
        white_noise_helpers: tuple
            (nvec, jvec) where jvec = None
        left : array-like
            The left-hand side of the equation. [Ntoa, n]
        right : array-like
            The right-hand side of the equation. [Ntoa, m]

        Returns
        -------
        array-like
            The result of the linear equation. [n, m] or [m]
        """
        # Solve L^T N^{-1} R - L^T N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1} R
        # Term1 = L^T N^{-1} R
        term1 = self.solve_unmarg(white_noise_helpers, left, right)

        # Term2 = L^T N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1} R
        MNM = self.solve_unmarg(white_noise_helpers, self.Mmat, self.Mmat)
        LNM = self.solve_unmarg(white_noise_helpers, left, self.Mmat)
        MNR = self.solve_unmarg(white_noise_helpers, self.Mmat, right)
            
        cf = jsl.cho_factor(MNM)
        term2 = LNM @ jsl.cho_solve(cf, MNR)

        if not return_logdet:
            return term1 - term2
        else:
            return term1 - term2, self.logdet_unmarg(white_noise_helpers)\
                                 + 2 * jnp.sum(jnp.log(cf[0].diagonal())) + self.Mprior

    @jit_method
    def logdet_unmarg(self, white_noise_helpers):
        """Compute the log-determinant of the white-noise covariance matrix.

        Parameters
        ----------
        white_noise_helpers: tuple
            (nvec, jvec) where jvec = None
        
        Returns
        -------
        float
            The log-determinant of the covariance matrix.
        """
        nvec, jvec = white_noise_helpers
        return jnp.sum(jnp.log(nvec))
    

class SinglePulsarWhiteCov:
    """A complete TOA covariance matrix with EFAC, EQUAD, and ECORR.

    This class implements the full white noise covariance matrix for a pulsar 
    with a fixed pulsar timing model. This model includes contributions from EFAC, 
    EQUAD, and ECORR parameters. 
    The covariance matrix can be written as: 
        N_ij = delta_ij EFAC_i^2 * (sigma_i^2 + EQUAD_i^2) + ECORR_ij^2
        - if i and j are within the same epoch, otherwise N_ij = 0

    This class uses the Sherman-Morrison formula to calculate the linear system
    left^T N^{-1} right efficiently without explicitly inverting the covariance matrix.
    """

    def __init__(self, psr,
                marg = False,
                efac_prior_bounds = (0.01, 10), # (low, high)
                efac_prior_normal = (1., 0.25), # (mean, std)
                log10equad_prior_bounds = (-9, -5), # (low, high)
                log10ecorr_prior_bounds = (-9, -5), # (low, high)
                include_ecorr = True
                ):
        """A constructor for the full white noise covariance matrix.

        Parameters
        ----------
        psr : object
            the pulsar object
        marg: bool
            Do you want to marginalize over linear timing model errors?
        *prior_bounds : tuple, optional
            the lower and upper prior bounds for white noise params
        include_ecorr : bool, optional
            Whether to give each backend an ECORR parameter. Set this to False
            for datasets whose epochs contain a single TOA each -- an epoch of
            size one makes ECORR exactly degenerate with EQUAD, since the
            rank-one per-epoch update collapses onto that TOA's own diagonal
            entry. Sampling it anyway costs a parameter per backend and buys
            an unidentifiable direction. With False, each backend has only
            EFAC and log10_t2equad, and ``jvec`` is identically zero (which
            makes the Sherman-Morrison correction and its log-determinant
            contribution vanish exactly). Defaults to True.
        """
        backends, B, U_pad, U_mask, V = _get_psr_wn_attributes(psr)

        if include_ecorr and len(U_pad) == 0:
            raise ValueError(
                f"{psr.name}: no epoch contains more than one TOA (within "
                f"{EPOCH_THRESHOLD} s), so ECORR has nothing to correlate and is "
                "entirely unidentifiable -- sampling it would explore the prior "
                "with a flat likelihood. Pass include_ecorr=False."
            )

        self.marg = marg
        self.include_ecorr = include_ecorr

        # Only _solve_marg reads these, so only build them when marginalising.
        # PTA_Data already holds its own _timing_model_svd of the same design
        # matrices; with marg=False this second copy is never read, and on the
        # NANOGrav 15-year set it is ~1.1 GB of device memory that does nothing.
        if self.marg:
            self.Mmat = _timing_model_svd(psr.Mmat)
            self.Mprior = self.Mmat.shape[1] * jnp.log(1e40)
        else:
            self.Mmat = None
            self.Mprior = None

        # Static attributes ----------------------------------------------------
        self.psr_name = psr.name
        self.ntoas = len(psr.toas)
        self.nepochs = len(U_pad)
        self.toaerrs = psr.toaerrs

        self.backends = backends
        self.n_backends = len(backends)
        # Free parameters per backend: EFAC, EQUAD and (optionally) ECORR.
        self.n_params_per_backend = 3 if self.include_ecorr else 2
        self.n_params = self.n_params_per_backend * self.n_backends
        self.B = B # To construct nvec
        self.U_pad = U_pad
        self.U_mask = U_mask
        self.V = V # To construct jvec
        # Per-TOA epoch index for segment sums: toa_epoch[i] = the ECORR epoch
        # of TOA i, or nepochs (a dummy segment) for TOAs in no epoch.  Lets
        # the Sherman-Morrison terms be summed without padded gathers.
        toa_epoch = np.full(self.ntoas, self.nepochs, dtype=np.int32)
        for b in range(self.nepochs):
            members = np.asarray(U_pad[b])[np.asarray(U_mask[b]) > 0]
            toa_epoch[members] = b
        self.toa_epoch = jnp.asarray(toa_epoch)

        # Prior Ranges--------------------------------------
        self.lower_efac = efac_prior_bounds[0] 
        self.upper_efac = efac_prior_bounds[1]
        self.center_efac = efac_prior_normal[0]
        self.sigma_efac = efac_prior_normal[1]
        self.lower_log10equad = log10equad_prior_bounds[0]
        self.upper_log10equad = log10equad_prior_bounds[1]
        self.lower_log10ecorr = log10ecorr_prior_bounds[0]
        self.upper_log10ecorr = log10ecorr_prior_bounds[1]

        #jitted function that returns (left.T N right)
        self.solve = self._solve_func_maker(return_logdet = False)
        #unmarg solver to be used within marg solver function later
        self.solve_unmarg = partial(self._solve_unmarg, return_logdet = False) 
        #jitted function that returns (left.T N right, logDetN)
        self.solve_with_logdet = self._solve_func_maker(return_logdet = True)

        #Function used for performing cholesky decomp on N
        self._chol_diag_plus_rank1_batched = jax.vmap(self._chol_diag_plus_rank1, in_axes=(0, 0))

    def params_dict_to_vector(self, params):
        """Extract white noise params from a dict into a flat JAX array.

        Call this ONCE outside the sampler to pay the Python dict-lookup
        and host→device transfer cost.

        Layout: [ef_0, ..., ef_nb, eq_0, ..., eq_nb, ec_0, ..., ec_nb]
        where nb = len(self.backends).

        Parameters
        ----------
        params : dict

        Returns
        -------
        wn_vec : jnp.ndarray [n_params]  (n_params = 3 * nbackends, or 2 * nbackends
            without ECORR)
        """
        names = self.get_param_names()
        v = jnp.array([params[n] for n in names])
        return v
    

    def get_prior_bounds(self):
        """Return lower and upper bounds for all white-noise parameters.

        Returns
        -------
        tuple[list, list]
            Two lists containing the lower and upper bounds,
            respectively, ordered as

                [EFACs | log10_t2equads | log10_ecorrs].
        """
        low  = (
            [self.lower_efac]       * len(self.backends) +
            [self.lower_log10equad] * len(self.backends)
        )
        high = (
            [self.upper_efac]       * len(self.backends) +
            [self.upper_log10equad] * len(self.backends)
        )
        if self.include_ecorr:
            low  += [self.lower_log10ecorr] * len(self.backends)
            high += [self.upper_log10ecorr] * len(self.backends)
        return low, high

    def prior_draw(self, uniform_efac = True):
        """Draw white-noise parameters from the prior distribution.

        Generates one EFAC, log10_t2equad, and log10_ecorr parameter for
        each backend and returns them as a flat vector with layout

            [ef_0, ..., ef_nb | eq_0, ..., eq_nb | ec_0, ..., ec_nb]

        Parameters
        ----------
        uniform_efac : bool, optional
            If True, draw EFAC values from a uniform distribution
            between ``lower_efac`` and ``upper_efac``. Otherwise,
            draw from a normal distribution centered on
            ``center_efac`` with width ``sigma_efac`` and reflected
            about ``lower_efac`` to enforce positivity.

        Returns
        -------
        jnp.ndarray
            White-noise parameter vector of shape ``[n_params]``
            (``3 * n_backends``, or ``2 * n_backends`` without ECORR).
        """
        ef, eq, ec = [], [], []
        for b in self.backends:   # same iteration order as params_dict_to_vector
            if uniform_efac:
                ef.append(np.random.uniform(self.lower_efac, self.upper_efac))
            else:
                ef.append(self.lower_efac + 
                        np.abs(np.random.normal(loc = self.center_efac, scale = self.sigma_efac)))
            eq.append(np.random.uniform(self.lower_log10equad, self.upper_log10equad))
            if self.include_ecorr:
                ec.append(np.random.uniform(self.lower_log10ecorr, self.upper_log10ecorr))
        # Concatenate in the same [ef | eq | ec] order that params_dict_to_vector uses
        blocks = [jnp.stack(ef), jnp.stack(eq)]
        if self.include_ecorr:
            blocks.append(jnp.stack(ec))
        return jnp.concatenate(blocks)

    def prior_draw_dictionary(self, 
                            param_values = np.array([False]), 
                            uniform_efac = False):
        """Return white-noise parameters as a dictionary.

        Creates a parameter dictionary whose keys match
        ``get_param_names()``. Values are taken from
        ``param_values`` if provided; otherwise they are drawn
        from the prior distribution.

        Parameters
        ----------
        param_values : array-like, optional
            Parameter values ordered as

                [EFACs | log10_t2equads | log10_ecorrs].

        uniform_efac : bool, optional
            Passed to ``prior_draw()`` when generating new
            samples.

        Returns
        -------
        dict
            Mapping from parameter names to parameter values.
        """
        names = self.get_param_names()
        if np.any(param_values):
            xs = [float(x) for x in param_values]
        else:
            xs = [float(x) for x in self.prior_draw(uniform_efac = uniform_efac)]
        return dict(zip(names, xs))

    def get_param_names(self):
        """Return the white-noise parameter names for this pulsar.

        Names are returned in the order

            [EFACs | log10_t2equads | log10_ecorrs]

        and correspond directly to the parameter-vector layout
        used by ``params_dict_to_vector()``.

        Returns
        -------
        list[str]
            White-noise parameter names.
        """
        # Params is a dictionary of parameter values for each backends
        ef = [f'{self.psr_name}_{b}_efac'           for b in self.backends]
        eq = [f'{self.psr_name}_{b}_log10_t2equad'   for b in self.backends]
        if not self.include_ecorr:
            return ef + eq
        ec = [f'{self.psr_name}_{b}_log10_ecorr'     for b in self.backends]
        return ef + eq + ec

    @jit(static_argnums = 0)
    def get_nvec_jvec(self, wn_vec):
        """Compute nvec and jvec from a parameter vector.

        This is the sampler-hot-path replacement for dictionary-based
        parameter access. The computation uses only JAX operations and
        is fully JIT compatible.

        Parameters
        ----------
        wn_vec : jnp.ndarray
            White-noise parameter vector of shape ``[n_params]`` with layout

                [EFACs | log10_t2equads | log10_ecorrs (if include_ecorr)].

        Returns
        -------
        tuple
            Tuple ``(nvec, jvec)`` where

            - ``nvec`` contains the diagonal white-noise variances
              for each TOA.
            - ``jvec`` contains the ECORR variance associated with
              each epoch.
        """
        nb = len(self.backends)
        ef = wn_vec[:nb]
        eq = wn_vec[nb:2*nb]

        ef2 = ef ** 2
        eq2 = 10.0 ** (2.0 * eq)

        nvec = ef2[self.B] * (self.toaerrs ** 2 + eq2[self.B])

        if self.include_ecorr:
            ec2 = 10.0 ** (2.0 * wn_vec[2*nb:])
            jvec = ec2[self.V]
        else:
            # jvec == 0 makes the Sherman-Morrison numerator and the
            # log(1 + jvec * epoch_sums) correction vanish identically, so the
            # solvers reduce exactly to the diagonal case with no extra branch.
            jvec = jnp.zeros_like(self.V, dtype=nvec.dtype)

        return nvec, jvec

    def _solve_func_maker(self, return_logdet):
        """Create a solver function for the covariance matrix.

        This method returns a JIT-compiled function that solves the linear system
        left^T N^{-1} right, where N is the covariance matrix. If ``self.marg`` is
        set, N is replaced by the timing-model marginalized covariance D. The
        returned function can optionally also return the log-determinant.

        Parameters
        ----------
        return_logdet : bool
            If True, the returned solver also returns the log-determinant of the
            covariance matrix. If False, only the solution is returned.

        Returns
        -------
        function
            A JIT-compiled function with signature
            ``f(white_noise_helpers, left, right)``.
        """
        if self.marg:
            new_func = partial(self._solve_marg, return_logdet = return_logdet)
            return jit(new_func)
        else:
            new_func = partial(self._solve_unmarg, return_logdet = return_logdet)
            return jit(new_func)

    def _apply_inv_unmarg(self, white_noise_helpers, right):
        """``N^{-1} right`` with N = diag(nvec) + sum_b jvec_b u_b u_b^T.

        Sherman-Morrison per epoch (epochs are disjoint):

            N^{-1} R = A^{-1} R - A^{-1} U diag(j / (1 + j u^T A^{-1} u)) U^T A^{-1} R
        """
        nvec, jvec = white_noise_helpers
        Ainv = 1 / nvec
        AinvR = Ainv[:, None] * right                                       # [n, M]
        nb = self.nepochs
        if nb == 0:
            return AinvR
        seg = self.toa_epoch                                                # [n] in [0, nb]
        # u_b^T A^{-1} R and u_b^T A^{-1} u_b for every epoch, without padding
        UtAinvR = jax.ops.segment_sum(AinvR, seg, num_segments=nb + 1)[:nb]    # [b, M]
        epoch_Ainv = jax.ops.segment_sum(Ainv, seg, num_segments=nb + 1)[:nb]  # [b]
        w = UtAinvR * (jvec / (1.0 + jvec * epoch_Ainv))[:, None]          # [b, M]
        # each TOA takes its epoch's correction (the dummy segment adds 0)
        w = jnp.concatenate([w, jnp.zeros((1, w.shape[1]), w.dtype)], axis=0)
        return AinvR - Ainv[:, None] * w[seg]

    def apply_inv(self, white_noise_helpers, right):
        """``N^{-1} right`` (or ``D^{-1} right`` when the timing model is
        marginalised), as a matrix the size of ``right``.

        For ``left^T N^{-1} right`` with a small ``right`` and a wide ``left``,
        ``left.T @ apply_inv(right)`` touches ``left`` only through one matrix
        product, while ``solve`` also runs its epoch gathers over ``left``.
        """
        x = self._apply_inv_unmarg(white_noise_helpers, right)
        if not self.marg:
            return x
        NiM = self._apply_inv_unmarg(white_noise_helpers, self.Mmat)       # [n, k]
        cf = jsl.cho_factor(self.Mmat.T @ NiM)
        return x - NiM @ jsl.cho_solve(cf, NiM.T @ right)

    def _solve_unmarg(self, white_noise_helpers, left, right, return_logdet):
        """Solve the linear system ``left^T N^{-1} right``.

        N is the diagonal part plus a per-epoch rank-one ECORR term,

            N = diag(nvec) + U diag(jvec) U^T

        where U is the epoch indicator. The inverse is applied with the
        Sherman-Morrison formula, so N is never formed or inverted explicitly.

        NOTE: the left matrix will be transposed for you and should be [N_toa, N]

        Parameters
        ----------
        white_noise_helpers : tuple
            Tuple ``(nvec, jvec)`` containing the diagonal and ECORR
            covariance components.
        left : array-like
            Left-hand matrix of shape ``[N_toa, N]``.
        right : array-like
            Right-hand matrix of shape ``[N_toa, M]``.
        return_logdet : bool
            Whether to also return log|N|.

        Returns
        -------
        array-like or tuple
            ``left^T N^{-1} right`` of shape ``[N, M]``; if ``return_logdet``
            is True, the tuple ``(solve_result, logdet_N)`` instead.
        """
        # N^{-1} (Sherman-Morrison per epoch) is applied to whichever of
        # left / right has fewer columns, then ONE matrix product:
        #     left^T N^{-1} right = (N^{-1} left)^T right = left^T (N^{-1} right)
        # The epoch sums are segment sums (no padded gathers), so the cost is
        # ~ n_toa * min(N, M) for the inverse plus one GEMM, which reaches
        # cuBLAS on GPU.
        if left.shape[1] <= right.shape[1]:
            solve_result = self._apply_inv_unmarg(white_noise_helpers, left).T @ right
        else:
            solve_result = left.T @ self._apply_inv_unmarg(white_noise_helpers, right)

        if return_logdet:
            return solve_result, self.logdet_unmarg(white_noise_helpers)
        else:
            return solve_result
   
    def _solve_marg(self, white_noise_helpers, left, right, return_logdet):
        """Solve a linear equation (left).T @ D^{-1} @ right.

        This method solves the linear equation (left).T @ D^{-1} @ right, where
        D is the timing-model marginalized covariance matrix,

            D = N + M (prior) M^T,

        applied in the limit of an infinite timing-model prior. D^{-1} reduces to
        N^{-1} - N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1}, which is what is
        evaluated below.

        NOTE: the left matrix will be transposed for you and should be [N_toa, N]

        Parameters
        ----------
        white_noise_helpers : tuple
            Tuple ``(nvec, jvec)`` containing the diagonal and ECORR covariance components.
        left : array-like
            The left-hand side of the equation. [Ntoa, n]
        right : array-like
            The right-hand side of the equation. [Ntoa, m]
        return_logdet : bool
            Whether to also return log|D|, which includes the log-determinant of
            M^T N^{-1} M and the ``Mprior`` normalisation.

        Returns
        -------
        array-like or tuple
            The result of the linear equation, shape ``[n, m]``; if
            ``return_logdet`` is True, the tuple ``(result, logdet_D)`` instead.
        """
        # D^{-1} = N^{-1} - N^{-1} M (M^T N^{-1} M)^{-1} M^T N^{-1}, applied
        # to the narrower of left / right, then one matrix product.  N^{-1} M
        # is formed once per call (it was formed three times).
        NiM = self._apply_inv_unmarg(white_noise_helpers, self.Mmat)       # [n, k]
        cf = jsl.cho_factor(self.Mmat.T @ NiM)
        narrow_left = left.shape[1] <= right.shape[1]
        X = left if narrow_left else right
        DiX = (self._apply_inv_unmarg(white_noise_helpers, X)
               - NiM @ jsl.cho_solve(cf, NiM.T @ X))
        result = DiX.T @ right if narrow_left else left.T @ DiX

        if return_logdet:
            logdet = self.logdet_unmarg(white_noise_helpers)\
                 + 2 * jnp.sum(jnp.log(cf[0].diagonal())) + self.Mprior
            return result, logdet
        else:
            return result

    @jit_method
    def logdet_unmarg(self, white_noise_helpers):
        """Compute the log-determinant of the white-noise covariance matrix.

        Parameters
        ----------
        helpers : tuple
            Tuple ``(nvec, jvec)`` containing the diagonal and ECORR
            covariance components.

        Returns
        -------
        float
            The log-determinant of the covariance matrix.
        """
        # Diagonal contribution
        nvec, jvec = white_noise_helpers
        logdet_D = jnp.sum(jnp.log(nvec))

        # A^{-1}
        Ainv = 1.0 / nvec

        # For each epoch: sum_{i in epoch} Ainv_i
        epoch_sums = jnp.einsum(
            'bi,bi->b',
            Ainv[self.U_pad],
            self.U_mask
        )

        # Low-rank correction
        logdet_corr = jnp.sum(jnp.log(1.0 + jvec * epoch_sums))

        return logdet_D + logdet_corr

    def _chol_diag_plus_rank1(self, d, s):
        """Lower-triangular Cholesky factor of A = diag(d) + outer(s, s).

        Rows/cols with s_i = 0 are fully decoupled from the rest (safe to
        use for padded entries — d there just needs to be > 0).

        Parameters
        ----------
        d : (n,) array   diagonal entries, all > 0
        s : (n,) array   rank-1 vector

        Returns
        -------
        L : (n, n) lower-triangular array with L @ L.T == A
        """
        n = d.shape[0]
        L0 = jnp.diag(jnp.sqrt(d))
        idx = jnp.arange(n)

        def step(carry, k):
            L, x = carry
            Lkk = L[k, k]
            xk = x[k]
            r = jnp.sqrt(Lkk**2 + xk**2)
            a = r / Lkk
            b = xk / Lkk

            below = idx > k
            col_k = L[:, k]
            new_col_k = jnp.where(below, (col_k + b * x) / a, col_k)
            new_col_k = new_col_k.at[k].set(r)

            new_x = jnp.where(below, a * x - b * new_col_k, x)

            L = L.at[:, k].set(new_col_k)
            return (L, new_x), None

        (L, _), _ = jax.lax.scan(step, (L0, s), idx)
        return L

    
    def CholN_dot(self, white_noise_helpers, right):
        """Compute ``Cholesky(N) @ right``.

        N is block-diagonal (diag(nvec) plus disjoint per-epoch ECORR
        rank-1 blocks), so its Cholesky factor is too: a plain diagonal
        sqrt everywhere, overridden on ECORR rows by a small dense
        triangular block factor.

        Parameters
        ----------
        white_noise_helpers : tuple
            Tuple ``(nvec, jvec)`` containing the diagonal and ECORR
            covariance components.
        right : array-like
            Matrix of shape ``[N_toa, M]`` to left-multiply by chol(N).

        Returns
        -------
        array
            ``Cholesky(N) @ right``, shape ``[N_toa, M]``.
        """
        nvec, jvec = white_noise_helpers

        # Diagonal-only
        result = jnp.sqrt(nvec)[:, None] * right

        # --- per-block dense Cholesky ---------------------------------
        d_blocks = nvec[self.U_pad]                      # (b, n) always > 0
        s_blocks = jnp.sqrt(jvec)[:, None] * self.U_mask  # (b, n), 0 on padding
        
        L_blocks = self._chol_diag_plus_rank1_batched(d_blocks, s_blocks)  # (b, n, n)

        right_blocks = right[self.U_pad, :]                            # (b, n, M)
        out_blocks = jnp.einsum('bij,bjm->bim', L_blocks, right_blocks)

        # delta = (full block answer) - (diagonal-only baseline already
        # sitting in `result`); zero on padding so duplicate/dummy scatter
        # indices just add zero, avoiding any set()-collision issues.
        diag_only = jnp.sqrt(d_blocks)[..., None] * right_blocks
        delta = (out_blocks - diag_only) * self.U_mask[..., None]

        result = result.at[self.U_pad, :].add(delta)

        return result

class WhiteCov:
    """The Multi-pulsar white noise covariance matrix handler (also works with one pulsar!).

    Exploits the block-diagonal structure of N across pulsars to efficiently
    compute T^T N^{-1} T and T^T N^{-1} r for all pulsars simultaneously.

    Because N is block-diagonal across pulsars, we have:
        T^T N^{-1} T = sum_p  T_p^T N_p^{-1} T_p
        T^T N^{-1} r = sum_p  T_p^T N_p^{-1} r_p
    where the _p subscript selects the TOA rows belonging to pulsar p.

    Within each per-pulsar block the Sherman-Morrison formula from
    :class:`SinglePulsarWhiteCov` is re-used, so ECORR is handled exactly.
    """

    def __init__(self, 
                data,
                stabilize_TNT = False,
                efac_prior_bounds = (0.01, 10), # (low, high)
                efac_prior_normal = (1., 0.25), # (mean, std)
                log10equad_prior_bounds = (-9, -5), # (low, high)
                log10ecorr_prior_bounds = (-9, -5), # (low, high)
                include_ecorr = None
                ):
        """Construct a multi-pulsar white noise covariance handler.

        Builds one per-pulsar covariance object for every pulsar in ``data``. The
        per-pulsar class is chosen by ``data.diag_white_cov``:
        :class:`DiagSinglePulsarWhiteCov` when set (TOA errors only, no free
        parameters), :class:`SinglePulsarWhiteCov` otherwise.

        NOTE: this constructor mutates its argument -- it calls
        ``data.add_white_noise_cov(self)`` on the way out, so ``data.Nmat``
        points at the handler afterwards.

        Parameters
        ----------
        data : object
            ATLAS data object. Supplies ``psrs``, ``npsrs``, ``diag_white_cov``
            and ``marg``.
        stabilize_TNT : bool, optional
            Condition-stabilize TNT (and TDNTD) in the solvers before returning.
            Defaults to False. NOTE: currently honoured only when
            ``data.diag_white_cov`` is False.
        efac_prior_bounds : tuple, optional
            ``(low, high)`` bounds on EFAC. Defaults to ``(0.01, 10)``.
        efac_prior_normal : tuple, optional
            ``(mean, std)`` for the truncated-normal EFAC prior, used when EFAC
            is not sampled uniformly. Defaults to ``(1., 0.25)``.
        log10equad_prior_bounds : tuple, optional
            ``(low, high)`` bounds on log10_t2equad. Defaults to ``(-9, -5)``.
        log10ecorr_prior_bounds : tuple, optional
            ``(low, high)`` bounds on log10_ecorr. Defaults to ``(-9, -5)``.
        include_ecorr : bool or None, optional
            Give every backend an ECORR parameter? Set False when the epochs
            hold one TOA each, which makes ECORR degenerate with EQUAD. See
            :class:`SinglePulsarWhiteCov`. Ignored when ``data.diag_white_cov``
            is set, since that model has no free parameters at all.
            Defaults to None, which takes the answer from ``data.include_ecorr``
            (detected from the TOA epochs by :class:`ATLAS.data.PTA_Data`) and
            falls back to True if the data object does not carry the attribute.
        """
        if include_ecorr is None:
            include_ecorr = getattr(data, 'include_ecorr', None)
            if include_ecorr is None:
                include_ecorr = True

        # Extracting the data analysis settings
        self.data = data
        self.diag_white_cov = self.data.diag_white_cov
        self.marg = self.data.marg
        self.npulsars = self.data.npsrs
        self.stabilize = stabilize_TNT
        self.include_ecorr = include_ecorr

        self.cov_matrices = []
        pbar = tqdm(range(self.npulsars))
        for pidx in pbar:
            psr = self.data.psrs[pidx]
            pbar.set_description(f"Constructing the white noise cov matrix for {psr.name}")
            if not self.diag_white_cov:
                self.cov_matrices.append(SinglePulsarWhiteCov(
                                                            psr, 
                                                            marg = self.marg, 
                                                            efac_prior_bounds = efac_prior_bounds,
                                                            efac_prior_normal = efac_prior_normal,
                                                            log10equad_prior_bounds = log10equad_prior_bounds,
                                                            log10ecorr_prior_bounds = log10ecorr_prior_bounds,
                                                            include_ecorr = include_ecorr
                                                            ))
            else:
                self.cov_matrices.append(DiagSinglePulsarWhiteCov(psr, 
                                                            marg = self.marg
                                                            ))

        self.ntoas_per_psr = tuple(len(psr.toas) for psr in self.data.psrs)
        self.total_ntoas = sum(self.ntoas_per_psr)
        # Precompute static slice boundaries for each pulsar in the global array.
        cumulative = np.cumsum([0] + list(self.ntoas_per_psr))
        self.toa_starts = tuple(int(c) for c in cumulative[:-1])
        self.toa_ends   = tuple(int(c) for c in cumulative[1:])
        self.pulsar_idxs = jnp.arange(self.npulsars)

        self.data.add_white_noise_cov(self)

    # ------------------------------------------------------------------
    # Parameter interface
    # ------------------------------------------------------------------

    def get_prior_bounds(self):
        """Concatenate per-pulsar prior bounds.

        Returns
        -------
        concatenated prior bounds across pulsars and backends.

        """
        return jnp.concat([jnp.array(cov.get_prior_bounds()) for cov in self.cov_matrices], axis = -1)

    def params_dict_to_vector(self, params):
        """Concatenate per-pulsar wn_vecs into one flat array.

        Returns
        -------
        wn_vec : jnp.ndarray [sum_p n_params_p]
            Concatenation of each per-pulsar params_dict_to_vector output.
        """
        return jnp.concatenate([cov.params_dict_to_vector(params)
                                for cov in self.cov_matrices])


    def prior_draw(self, uniform_efac = False):
        """Sample white noise for all pulsars and return a concatenated wn_vec.

        The layout mirrors MultiPsr_WN_Cov.params_dict_to_vector:
            [wn_vec_psr0 | wn_vec_psr1 | ...]
        where each per-pulsar block is [ef | eq | ec] for that pulsar's backends.

        Parameters
        ----------
        multi_cov : MultiPsr_WN_Cov

        Returns
        -------
        wn_vec : jnp.ndarray [sum_p n_params_p]
        """
        return jnp.concatenate([cov.prior_draw(uniform_efac = uniform_efac) for cov in self.cov_matrices])

    def prior_draw_dictionary(self, param_values = None, uniform_efac = False):
        """Sample white noise for all pulsars and return a concatenated wn_vec.

        The layout mirrors MultiPsr_WN_Cov.params_dict_to_vector:
            [wn_vec_psr0 | wn_vec_psr1 | ...]
        where each per-pulsar block is [ef | eq | ec] for that pulsar's backends.

        Parameters
        ----------
        multi_cov : MultiPsr_WN_Cov

        Returns
        -------
        wn_vec : jnp.ndarray [sum_p n_params_p]
        """
        names = self.get_param_names()
        if np.any(param_values):
            xs = [float(x) for x in param_values]
        else:
            xs = [float(x) for x in self.prior_draw(uniform_efac = uniform_efac)]
        return dict(zip(names, xs))

    def get_param_names(self):
        """Sample white noise for all pulsars and return a concatenated wn_vec.

        The layout mirrors MultiPsr_WN_Cov.params_dict_to_vector:
            [wn_vec_psr0 | wn_vec_psr1 | ...]
        where each per-pulsar block is [ef | eq | ec] for that pulsar's backends.

        Parameters
        ----------
        multi_cov : MultiPsr_WN_Cov

        Returns
        -------
        wn_vec : jnp.ndarray [sum_p n_params_p]
        """
        return list(itertools.chain.from_iterable([cov.get_param_names() for cov in self.cov_matrices]))

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def get_red_helpers(self, red_noise_basis, residuals, white_noise_params):
        """Compute both F_p^T N_p^{-1} F_p and F_p^T N_p^{-1} r_p per pulsar.
        NOTE: The F-matrix can be replaced with a T-matrix. F and T are interchangable.

        Appends r_p as an extra column of the right-hand side so only one
        call to solve() is needed per pulsar:
            solve(F_p, [F_p | r_p]) = [F_p^T N_p^{-1} F_p | F_p^T N_p^{-1} r_p]

        Parameters
        ----------
        red_noise_basis : array [total_ntoas, N_basis]
            Global Fourier design matrix (F-matrix).
        residuals : array [total_ntoas]
            Global residuals vector.
        white_noise_params: array
            The Global (concatenated) white noise values
        Returns
        -------
        FNF : array [npulsars, N_basis, N_basis]
        FNr : array [npulsars, N_basis]
        rNr: array [1]
        log_det_N: array [1]
        """
        N_basis = red_noise_basis.shape[1]
        residuals = jnp.reshape(residuals, (-1,))
        FNF = jnp.zeros((self.npulsars, N_basis, N_basis))
        FNr = jnp.zeros((self.npulsars, N_basis))

        log_det_N = 0
        rNr = 0
        wn_params_start_idx = 0

        if self.diag_white_cov:
            for pidx, cov, start, end in zip(self.pulsar_idxs, 
                                            self.cov_matrices, 
                                            self.toa_starts, 
                                            self.toa_ends):

                white_noise_helper = cov.get_nvec_jvec()
                F_p  = red_noise_basis[start:end, :]                    # [n_p, N_basis]
                r_p  = residuals[start:end]                             # [n_p]
                # ONE Gram per pulsar: G = [T_p | r_p]^T N^{-1} [T_p | r_p]
                # holds TNT, TNr and rNr; log|N| comes with it.
                Fr_p = jnp.concatenate([F_p, r_p[:, None]], axis=1)     # [n_p, N_basis+1]
                G, y = cov.solve_with_logdet(white_noise_helper, Fr_p, Fr_p)
                G = 0.5 * (G + G.T)

                FNF = FNF.at[pidx].set(G[:N_basis, :N_basis])
                FNr = FNr.at[pidx].set(G[:N_basis, N_basis])
                rNr += G[N_basis, N_basis]
                log_det_N += y

            if self.stabilize:
                return stabilize_TNT(FNF, FNF.shape[-1]), FNr, rNr, log_det_N
            return FNF, FNr, rNr, log_det_N

        else:
            for pidx, cov, start, end in zip(self.pulsar_idxs, 
                                            self.cov_matrices, 
                                            self.toa_starts, 
                                            self.toa_ends):

                wn_params_end_idx = wn_params_start_idx + cov.n_params
                wn_params = white_noise_params[wn_params_start_idx: wn_params_end_idx]
                wn_params_start_idx = wn_params_end_idx

                white_noise_helper = cov.get_nvec_jvec(wn_params)
                F_p  = red_noise_basis[start:end, :]                    # [n_p, N_basis]
                r_p  = residuals[start:end]                             # [n_p]
                # ONE Gram per pulsar: G = [T_p | r_p]^T N^{-1} [T_p | r_p]
                # holds TNT, TNr and rNr; log|N| comes with it.
                Fr_p = jnp.concatenate([F_p, r_p[:, None]], axis=1)     # [n_p, N_basis+1]
                G, y = cov.solve_with_logdet(white_noise_helper, Fr_p, Fr_p)
                G = 0.5 * (G + G.T)

                FNF = FNF.at[pidx].set(G[:N_basis, :N_basis])
                FNr = FNr.at[pidx].set(G[:N_basis, N_basis])
                rNr += G[N_basis, N_basis]
                log_det_N += y

            if self.stabilize:
                return stabilize_TNT(FNF, FNF.shape[-1]), FNr, rNr, log_det_N
            else:
                return FNF, FNr, rNr, log_det_N

    def _per_pulsar_wn(self, white_noise_params):
        """(cov, start, end, white_noise_helper) per pulsar."""
        out, i0 = [], 0
        for cov, start, end in zip(self.cov_matrices, self.toa_starts, self.toa_ends):
            if self.diag_white_cov:
                wh = cov.get_nvec_jvec()
            else:
                i1 = i0 + cov.n_params
                wh = cov.get_nvec_jvec(white_noise_params[i0:i1])
                i0 = i1
            out.append((cov, start, end, wh))
        return out

    def get_static_helpers(self, red_noise_basis, white_noise_params):
        """The residual-independent parts of the helpers, computed once.

        For runs where T and the white noise are fixed but the residuals change
        every step (a sampled nonlinear timing model).  Returns
        ``(TNT, NiT, logdet_N)``:

        TNT      [npsr, N_basis, N_basis]  T^T N^{-1} T (stabilised if enabled)
        NiT      [total_ntoas, N_basis]    N^{-1} T (D^{-1} T when marginalised)
        logdet_N scalar

        Pair with ``get_red_helpers_from_static`` each step.  Memory is one
        extra T-sized array.
        """
        N_basis = red_noise_basis.shape[1]
        TNT = jnp.zeros((self.npulsars, N_basis, N_basis))
        NiT = []
        logdet = 0.0
        for pidx, (cov, start, end, wh) in zip(self.pulsar_idxs,
                                               self._per_pulsar_wn(white_noise_params)):
            T_p = red_noise_basis[start:end, :]
            NiT_p = cov.apply_inv(wh, T_p)
            G = NiT_p.T @ T_p
            TNT = TNT.at[pidx].set(0.5 * (G + G.T))
            NiT.append(NiT_p)
            logdet = logdet + _logdet_total(cov, wh)
        if self.stabilize:
            TNT = stabilize_TNT(TNT, N_basis)
        return TNT, jnp.concatenate(NiT, axis=0), logdet

    def get_red_helpers_from_static(self, static, residuals, white_noise_params):
        """``get_red_helpers`` output from ``get_static_helpers`` + new residuals.

        Per step only T^T N^{-1} r = (N^{-1} T)^T r and r^T N^{-1} r are formed:
        cost ~ n_toa * N_basis instead of n_toa * N_basis^2.  ``static`` must
        come from the same T and white noise.
        """
        TNT, NiT, logdet = static
        residuals = jnp.reshape(residuals, (-1,))
        TNr = jnp.zeros((self.npulsars, TNT.shape[-1]))
        rNr = 0.0
        for pidx, (cov, start, end, wh) in zip(self.pulsar_idxs,
                                               self._per_pulsar_wn(white_noise_params)):
            r_p = residuals[start:end]
            TNr = TNr.at[pidx].set(NiT[start:end].T @ r_p)
            rNr = rNr + r_p @ cov.apply_inv(wh, r_p[:, None])[:, 0]
        return TNT, TNr, rNr, logdet

    def get_red_helpers_partial(self, red_noise_basis, cols, base_helpers,
                                residuals, white_noise_params):
        """Update ``get_red_helpers`` output where only some columns of T changed.

        With ``C_p = T_p[:, cols]`` (the changed columns) and ``R_p = [T_p | r_p]``,
        one solve per pulsar gives ``C_p^T N_p^{-1} R_p``.  That is exactly the
        rows ``cols`` of ``T^T N^{-1} T`` (and, by symmetry, its columns
        ``cols``) and the entries ``cols`` of ``T^T N^{-1} r``; everything else
        in ``base_helpers`` is unchanged:

            [ S'N^-1 S   S'N^-1 C ]      S'N^-1 S  : kept from base_helpers
            [ C'N^-1 S   C'N^-1 C ]      the rest   : recomputed here

        ``rNr`` and ``log|N|`` do not depend on T and are returned as given.
        Cost per pulsar ~ n_toa * n_cols * (N_basis + 1) instead of
        n_toa * N_basis * (N_basis + 1).

        VALID ONLY IF ``base_helpers`` were computed with the same
        ``white_noise_params`` and ``residuals``; the columns outside ``cols``
        must also be unchanged.  Not supported with ``stabilize_TNT=True``: the
        stabilising shift depends on the whole diagonal, so a partial update
        cannot reproduce it.

        Parameters
        ----------
        red_noise_basis : array [total_ntoas, N_basis]
            The updated T-matrix.
        cols : static 1-D int array or slice
            Columns of T that changed.  Must be concrete (not traced).
        base_helpers : tuple
            ``(TNT, TNr, rNr, logdet_N)`` from ``get_red_helpers`` for the
            previous T, same white noise and residuals.
        residuals : array [total_ntoas]
        white_noise_params : array
            The global (concatenated) white-noise values (ignored for a
            diagonal white-noise covariance).

        Returns
        -------
        (TNT, TNr, rNr, logdet_N) with the same shapes as ``get_red_helpers``.
        """
        if self.stabilize:
            raise NotImplementedError(
                "partial helper updates are not available with stabilize_TNT=True; "
                "use get_red_helpers")
        TNT, TNr, rNr, log_det_N = base_helpers
        N_basis = red_noise_basis.shape[1]
        if isinstance(cols, slice):
            cols = np.arange(*cols.indices(N_basis))
        cols = np.asarray(cols, dtype=int)
        if cols.size == 0:
            return TNT, TNr, rNr, log_det_N
        if cols.min() < 0 or cols.max() >= N_basis:
            raise ValueError(f"cols out of range for a basis with {N_basis} columns")
        if np.unique(cols).size != cols.size:
            raise ValueError("cols must not repeat")

        wn_params_start_idx = 0
        for pidx, cov, start, end in zip(self.pulsar_idxs,
                                         self.cov_matrices,
                                         self.toa_starts,
                                         self.toa_ends):
            if self.diag_white_cov:
                white_noise_helper = cov.get_nvec_jvec()
            else:
                wn_params_end_idx = wn_params_start_idx + cov.n_params
                wn_params = white_noise_params[wn_params_start_idx: wn_params_end_idx]
                wn_params_start_idx = wn_params_end_idx
                white_noise_helper = cov.get_nvec_jvec(wn_params)

            T_p = red_noise_basis[start:end, :]                     # [n_p, N_basis]
            r_p = residuals[start:end]                              # [n_p]
            C_p = T_p[:, cols]                                      # [n_p, n_cols]
            # N^{-1} is applied to the narrow C_p only; T_p enters through one
            # matrix product.  (cov.solve(C_p, [T_p | r_p]) gives the same
            # numbers but runs its per-epoch gathers over every column of T_p.)
            NiC = cov.apply_inv(white_noise_helper, C_p)            # [n_p, n_cols]
            CNT = NiC.T @ T_p                                       # [n_cols, N_basis]
            CNr = NiC.T @ r_p                                       # [n_cols]

            A = TNT[pidx]
            A = A.at[cols, :].set(CNT)
            A = A.at[:, cols].set(CNT.T)
            TNT = TNT.at[pidx].set(A)
            TNr = TNr.at[pidx, cols].set(CNr)

        return TNT, TNr, rNr, log_det_N

    def get_red_det_helpers(self, red_noise_basis, det_signal_basis,
                            residuals, white_noise_params):
        """Compute the red-noise *and* deterministic-signal N^{-1} helpers per pulsar.

        This extends :meth:`get_red_helpers` with a second (independently sized)
        design matrix ``det_signal_basis`` (denoted "TD") for a deterministic
        signal. In addition to the usual red-noise helpers it returns every
        N^{-1}-weighted cross product between the red-noise basis ``T``, the
        deterministic basis ``TD``, and the residuals ``r``.

        Parameters
        ----------
        red_noise_basis : array [total_ntoas, N_basis]
            Global red-noise (Fourier / T-matrix) design matrix.
        det_signal_basis : array [total_ntoas, N_det_basis]
            Global deterministic-signal design matrix. May have a different
            number of columns than ``red_noise_basis``.
        residuals : array [total_ntoas]
            Global residuals vector.
        white_noise_params : array
            The global (concatenated) white noise values.

        Returns
        -------
        TNT : array [npulsars, N_basis, N_basis]
            T^T N^{-1} T for each pulsar.
        TNr : array [npulsars, N_basis]
            T^T N^{-1} r for each pulsar.
        rNr : array [1]
            r^T N^{-1} r summed over pulsars.
        logdet_N : array [1]
            log|N| summed over pulsars.
        TDNTD : array [npulsars, N_det_basis, N_det_basis]
            TD^T N^{-1} TD for each pulsar.
        TDNr : array [npulsars, N_det_basis]
            TD^T N^{-1} r for each pulsar.
        TNTD : array [npulsars, N_basis, N_det_basis]
            T^T N^{-1} TD for each pulsar.
        """
        N_basis = red_noise_basis.shape[1]
        N_det_basis = det_signal_basis.shape[1]
        Ntot = N_basis + N_det_basis

        TNT   = jnp.zeros((self.npulsars, N_basis, N_basis))
        TNr   = jnp.zeros((self.npulsars, N_basis))
        TDNTD = jnp.zeros((self.npulsars, N_det_basis, N_det_basis))
        TDNr  = jnp.zeros((self.npulsars, N_det_basis))
        TNTD  = jnp.zeros((self.npulsars, N_basis, N_det_basis))

        logdet_N = 0
        rNr = 0
        wn_params_start_idx = 0

        for pidx, cov, start, end in zip(self.pulsar_idxs,
                                         self.cov_matrices,
                                         self.toa_starts,
                                         self.toa_ends):

            if self.diag_white_cov:
                white_noise_helper = cov.get_nvec_jvec()
            else:
                wn_params_end_idx = wn_params_start_idx + cov.n_params
                wn_params = white_noise_params[wn_params_start_idx: wn_params_end_idx]
                wn_params_start_idx = wn_params_end_idx
                white_noise_helper = cov.get_nvec_jvec(wn_params)

            T_p  = red_noise_basis[start:end, :]                   # [n_p, N_basis]
            TD_p = det_signal_basis[start:end, :]                  # [n_p, N_det_basis]
            r_p  = residuals[start:end]                            # [n_p]

            # Stack once: M_p = [T_p | TD_p | r_p]  -> [n_p, Ntot + 1]
            M_p = jnp.concatenate([T_p, TD_p, r_p[:, None]], axis=1)

            # One solve per pulsar gives the full symmetric Gram matrix + logdet.
            # G = M_p^T N_p^{-1} M_p  -> [Ntot + 1, Ntot + 1]
            G, y = cov.solve_with_logdet(white_noise_helper, M_p, M_p)

            TNT   = TNT.at[pidx].set(G[:N_basis, :N_basis])
            TNTD  = TNTD.at[pidx].set(G[:N_basis, N_basis:Ntot])
            TDNTD = TDNTD.at[pidx].set(G[N_basis:Ntot, N_basis:Ntot])
            TNr   = TNr.at[pidx].set(G[:N_basis, Ntot])
            TDNr  = TDNr.at[pidx].set(G[N_basis:Ntot, Ntot])

            rNr += G[Ntot, Ntot]
            logdet_N += y
        if self.stabilize:
            # The parentheses are load-bearing: without them the trailing comma
            # ended the return statement after the first element and the next
            # two lines parsed as unreachable expression statements, so this
            # branch returned a 1-tuple and every caller failed to unpack it.
            return (stabilize_TNT(TNT, TNT.shape[-1]),
                    TNr, rNr, logdet_N, stabilize_TDNTD(TDNTD, TDNTD.shape[-1]),
                    TDNr, TNTD)
        else:
            return TNT, TNr, rNr, logdet_N, TDNTD, TDNr, TNTD