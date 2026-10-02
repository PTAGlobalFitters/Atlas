"""The NumPyro model at the top of an ATLAS run.

``model_maker`` is the actual entry point of a fit: it samples the timing model
(optionally), the white noise (optionally), the red-noise spectrum and the
non-centred coefficients, then hands the log-density to NumPyro.

The ``numpyro.factor('lnpost', lprob + 0.5 * sum(z_a**2))`` line at the end is
load-bearing and not obviously so.  Both likelihoods return a density in the
reparameterised coordinate, and NumPyro has already added the N(0, 1) log-prior
for ``z_a`` from the ``sample`` statement above; the ``+0.5 sum(z**2)`` cancels
that double-count.  Removing it silently corrupts the posterior rather than
raising, which is why ``tests/test_identities.py`` pins the identity directly.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

__all__ = ["model_maker"]


def model_maker(raw_residuals,
                super_sig,
                marg_over_non_gwb,
                vary_white=False,
                wn_lower_bound=None,
                wn_upper_bound=None,
                tm_model=None,
                helpers=None,
                save_red_coeff=False,
                fixed_white_noise_params=None,
                red_noise_basis=None,
                tm_direct_sampling_type='klam',
                varied_chrom_index = False
                ):
    """NumPyro model for the ATLAS global fit.

    The function is the NumPyro model: call it via ``MCMC(model_maker, ...)``,
    passing keyword arguments through ``MCMC.run``.  NumPyro traces through it
    to build the joint density; do not call it directly.

    Parameters
    ----------
    raw_residuals : list of array
        Per-pulsar timing residuals in seconds, ``[npsr][n_toa_p]``.  Used as
        the stochastic residuals when no timing model is sampled (``tm_model``
        is None).
    super_sig : SuperSignal
        The assembled signal (T-matrix + phi model + likelihood helpers).
    marg_over_non_gwb : bool
        When True, the non-GWB coefficients (timing, IRN, DM, GTM) are
        marginalised analytically and only the GWB coefficients are
        reparameterised as ``g = g_hat + L^-T z_a``.  When False, all
        coefficients are reparameterised jointly via ``lnposterior_reparam``.
    vary_white : bool, default False
        Sample the white-noise parameters.  Requires ``wn_lower_bound`` and
        ``wn_upper_bound``.
    wn_lower_bound, wn_upper_bound : array, optional
        Prior bounds for the white-noise parameters; required when
        ``vary_white=True``.
    tm_model : MultiPsrTimingModel, optional
        The nonlinear timing model.  When not None the timing parameters are
        sampled and the residuals are recomputed from them; when None the
        ``raw_residuals`` are used as-is.
    helpers : tuple, optional
        Pre-built (TNT, TNr, rNr, logdet_N) from ``super_sig.get_helpers``.
        Used when both ``vary_white=False`` and ``tm_model is None``.  Must
        not be None in that case.
    save_red_coeff : bool, default False
        Register the Fourier coefficients as a ``numpyro.deterministic`` site
        (``'coeff'``) so they are stored in the MCMC samples.
    fixed_white_noise_params : array, optional
        White-noise parameters to pass to ``get_helpers`` when ``tm_model``
        is not None and ``vary_white=False``.  If the white noise is fixed in
        the data object (``super_sig.fixed_wn=True``), this is not needed and
        can be left None.
    red_noise_basis : array, optional
        The T-matrix, ``[n_toas, nmodes]``.  Pass it as a model argument
        through ``MCMC.run`` with ``jit_model_args=True`` whenever the helpers
        are rebuilt inside the model (``vary_white=True`` or ``tm_model``
        is not None); otherwise XLA writes it into the potential-energy
        executable as a compile-time constant, producing executables of several
        GB on large arrays.  Defaults to ``super_sig.get_Fmat_concat``.
    tm_direct_sampling_type : str, default ``'klam'``
        Parameterisation for the timing model when ``tm_model`` is not None.

        ``'klam'``
            Samples a global scale ``timing_lam ~ HalfNormal(10)`` and
            per-parameter unit coordinates
            ``timing_k ~ Normal(0, 50)[nparams_total]``.  The product
            ``timing_k * timing_lam`` is passed as the normalised z-vector to
            ``tm_model.residuals``; the effective prior in physical units is
            Normal(0, sigma_JUG * timing_lam) scaled by up to 50.

        Any other value
            Calls ``tm_model.sample_residuals()``, which samples each
            pulsar's physical timing parameters directly from their bounded
            priors (SINI, ECC from Uniform; affine params from a wide Normal).
    varied_chrom_index : bool, default ``False``
        Varying the chromatic index from 0 to 6
        
    Notes
    -----
    The ``numpyro.factor`` cancellation:
        Both likelihoods return log p(data | z_a, ...) + terms in phi, but
        *not* log p(z_a).  NumPyro adds log p(z_a) = -0.5 sum(z_a**2) from
        the ``sample`` statement.  The ``+0.5 sum(z_a**2)`` in the factor call
        cancels it, leaving only the correct marginalised density.

    Raises
    ------
    ValueError
        If ``vary_white=False``, ``tm_model`` is None, and ``helpers`` is
        None; the model has no way to build the white-noise products.
    """
    if not vary_white and tm_model is None and helpers is None:
        raise ValueError(
            "helpers must be supplied when vary_white=False and tm_model is "
            "None; call super_sig.get_helpers() first and pass the result.")

    # ------------------------------------------------------------------ #
    #  Timing model                                                         #
    # ------------------------------------------------------------------ #
    if tm_model is not None:
        if tm_direct_sampling_type == 'klam':
            lam = numpyro.sample("timing_lam", dist.HalfNormal(10.0))
            k = numpyro.sample(
                "timing_k",
                dist.Normal(0, 10).expand([tm_model.nparams_total]))
            stochastic_res = tm_model.residuals(k * lam)
        else:
            stochastic_res = tm_model.sample_residuals()
    else:
        stochastic_res = raw_residuals

    # ------------------------------------------------------------------ #
    #  Chromatic Noise (beyond DM)                                       #
    # ------------------------------------------------------------------ #
    if varied_chrom_index:
        chrom_index = numpyro.sample('chromatic_index', 
                                    dist.Uniform(0, 6).expand((super_sig.data.npsrs,)))        
        red_noise_basis = super_sig.update_red_basis(chrom_index = chrom_index)

    # ------------------------------------------------------------------ #
    #  White-noise helper products                                          #
    # ------------------------------------------------------------------ #
    if vary_white:
        theta_wn = numpyro.sample(
            'white_noise', dist.Uniform(wn_lower_bound, wn_upper_bound))
        helpers_now = super_sig.get_helpers(
            reff=stochastic_res,
            white_noise_params=theta_wn,
            red_noise_basis=red_noise_basis,
        )
    elif tm_model is not None:
        # The timing model changed the residuals; rebuild helpers even though
        # white noise is fixed.
        helpers_now = super_sig.get_helpers(
            reff=stochastic_res,
            white_noise_params=fixed_white_noise_params,
            red_noise_basis=red_noise_basis,
        )
    else:
        # White noise and residuals are both fixed; use pre-built helpers.
        helpers_now = helpers

    # ------------------------------------------------------------------ #
    #  Red-noise spectrum                                                   #
    # ------------------------------------------------------------------ #
    xs = numpyro.sample(
        'red_noise',
        dist.Uniform(super_sig.model.lower_prior_lim_all,
                     super_sig.model.upper_prior_lim_all))

    # ------------------------------------------------------------------ #
    #  Deterministic signals                                                #
    # ------------------------------------------------------------------ #
    if super_sig.has_det:
        det = super_sig.det_signal
        det_params = numpyro.sample(
            'det_params',
            dist.Uniform(det.det_param_mins, det.det_param_maxs))
        if not det.with_psr_params:
            D_params = (det_params, None, None)
        else:
            data = super_sig.data
            psr_phases = numpyro.sample(
                'psr_phases',
                dist.Uniform(0., 2. * jnp.pi).expand((data.npsrs,)))
            psr_dists_standard = numpyro.sample(
                'psr_dists_standard',
                dist.Normal().expand((data.npsrs,)))
            psr_dists = numpyro.deterministic(
                'psr_dists',
                data.psr_dists_mean + psr_dists_standard * data.psr_dists_std)
            D_params = (det_params, psr_phases, psr_dists)
            numpyro.factor('psr_dist_lnprior', det.psr_dists_lnprior(det_params, psr_phases, psr_dists))
    else:
        D_params = None

    # ------------------------------------------------------------------ #
    #  Log-posterior                                                        #
    # ------------------------------------------------------------------ #
    if marg_over_non_gwb:
        # n_gwb_modes from the model's own attribute, not from data.num_gwb_bins,
        # so the shape matches the GWB slice that partial_marg_lnposterior
        # actually consumes even if data and model were built with different
        # bin counts.
        n_gwb_modes = 2 * super_sig.model.crn_bins
        z_a = numpyro.sample(
            'z_a',
            dist.Normal(0, 1).expand((super_sig.npsrs, n_gwb_modes)))
        lprob, coeff = super_sig.partial_marg_lnposterior(
            helpers=helpers_now, red_params=xs, z=z_a, D_params=D_params)
    else:
        n_reparam = super_sig.nmodes - (
            super_sig.det_signal.num_coeff_det if super_sig.has_det else 0)
        z_a = numpyro.sample(
            'z_a',
            dist.Normal(0, 1).expand((super_sig.npsrs, n_reparam)))
        lprob, coeff = super_sig.lnposterior_reparam(
            helpers=helpers_now, red_params=xs, z=z_a, D_params=D_params)

    # Cancel the N(0,1) log-prior NumPyro adds for z_a from the sample site
    # above; the likelihoods return the density in the reparameterised
    # coordinate, which does not include that term.
    numpyro.factor('lnpost', lprob + 0.5 * jnp.sum(z_a ** 2))

    if save_red_coeff:
        numpyro.deterministic('coeff', coeff)