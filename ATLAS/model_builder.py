from ATLAS.signals import signals_utils as sutils
import jax
from ATLAS.signals.factorized.base import Red, SuperSignal, GaussianTiming
from ATLAS.signals.factorized.chromatic import (
    ChromaticRed, ChromaticQuad, TimeDomainSolarWind, DEFAULT_CHROM_INDEX)
from ATLAS.signals.correlated.base import Correlated
from ATLAS.nMatrix.base import WhiteCov
from ATLAS.signals.deterministic.base import Deterministic


class ModelBuilder:
    """Factory for the three model components consumed by ``model_maker``.

    Separates construction (par/tim loading, PSD/ORF wiring, white-noise
    setup) from sampling, so each component can be built and inspected
    independently before the NumPyro model is assembled.

    Parameters
    ----------
    data : PTA_Data
        The loaded dataset, providing par/tim paths, residuals, bin counts
        and pulsar metadata.
    explicit_timing_model_params_to_sample : list of list of str, optional
        Per-pulsar parameter lists, e.g. ``[['F0','F1','DM'], ['F0','DM']]``.
        When ``None`` the lists are derived from the par files at
        ``make_timing_model`` call time.
    """

    def __init__(self,
                 data):
        self.data = data
        self._sample_init = None

    def make_timing_model(self,
                          explicit_timing_model_params_to_sample=None,
                          load_how_many_in_parallel=None):
        """Build and return the non-linear timing model.

        The JUG import is deliberately function-level: JUG is not installable
        from PyPI, so a module-level import would make every other builder
        method (``make_white_noise``, ``make_red_noise``) unusable in a bare
        environment.

        Parameter resolution order for the sample list
        -----------------------------------------------
        1. ``explicit_timing_model_params_to_sample`` passed here (this call).
        2. The list supplied to ``__init__``.
        3. Derived from the par files: ``psr.fitpars[1:]`` (enterprise) or
           ``psr.fit_param_names`` (JUG-native).

        ``fitpars[1:]`` drops the leading ``Offset`` entry, which JUG always
        marginalises analytically and must never appear in ``sample_list``.

        ``A1DOT`` is renamed to ``XDOT`` because that is the design-label JUG
        uses internally (the par stores ``A1DOT``, but the design matrix and
        the ``uncertainties`` dict use ``XDOT``).

        This method does not cache the derived list.  Call it with the same
        arguments each time, or supply the list explicitly.

        Parameters
        ----------
        enterprise_data : bool, default True
            When True, reads ``psr.fitpars``; otherwise ``psr.fit_param_names``.
        explicit_timing_model_params_to_sample : list of list of str, optional
            Override for this call only; does not update ``self._sample_init``.
        load_how_many_in_parallel : int, optional
            Number of par/tim pairs to load in parallel via joblib.  Defaults
            to ``min(npsrs, 10)``.  Set to 1 to disable parallelism (required
            when the caller's environment cannot pickle JAX closures).

        Returns
        -------
        MultiPsrTimingModel
        """
        from ATLAS.signals.timing.base import build_multi_psr_timing_model

        # Resolve the sample list without mutating instance state.
        sample = (explicit_timing_model_params_to_sample
                  or self._sample_init)

        if sample is None:
            try:
                sample = [
                    list(self.data.psrs[pidx].fitpars[1:])
                    for pidx in range(self.data.npsrs)
                ]
                for sublist in sample:
                    for j, s in enumerate(sublist):
                        sublist[j] = s.replace('A1DOT', 'XDOT')
            except:
                sample = [
                    list(self.data.psrs[pidx].fit_param_names)
                    for pidx in range(self.data.npsrs)
                ]

        njobs = (load_how_many_in_parallel
                 if load_how_many_in_parallel is not None
                 else min(self.data.npsrs, 10))

        return build_multi_psr_timing_model(
            self.data.parfiles,
            self.data.timfiles,
            sample,
            load_how_many_in_parallel=njobs,
            data=self.data,
        )

    def make_white_noise(self, stabilize_TNT = True, include_ecorr = None):
        """Build the white noise covariance.

        ``include_ecorr=None`` takes the answer from ``data.include_ecorr``,
        which PTA_Data detects from the TOA epochs; True/False overrides it.
        """
        wn_model = WhiteCov(data=self.data, stabilize_TNT=stabilize_TNT,
                            include_ecorr=include_ecorr)
        return wn_model

    def make_red_noise(self,
                       red_noise_combination_string,
                       use_pulsar_tspan=False,
                       irn_psd_function=None,
                       gwb_psd_function=None,
                       det_delay_function=None,
                       orf_function=None,
                       dm_psd_function=None,
                       upper_bound_orf=None,
                       lower_bound_orf=None,
                       irn_lower_bound_psd=None,
                       irn_upper_bound_psd=None,
                       dm_lower_bound_psd=None,
                       dm_upper_bound_psd=None,
                       gwb_lower_bound_psd=None,
                       gwb_upper_bound_psd=None,
                       det_parameter_bounds=None,
                       gt_psd_val=None,
                       # ---- chromatic GP (`chrom`) ----
                       chrom_psd_function=None,
                       chrom_lower_bound_psd=None,
                       chrom_upper_bound_psd=None,
                       num_chrom_bins=30,
                       chrom_index=DEFAULT_CHROM_INDEX,
                       # ---- chromatic quadratic (`cquad`) ----
                       cquad_log10_amp=0.0,
                       cquad_project_timing=True,
                       # ---- time-domain solar wind (`sw`) ----
                       sw_geometry=None,
                       sw_kwargs=None):
        """Assemble and return a ``SuperSignal`` from named components.

        Which components are built is determined entirely by the names that
        appear in ``red_noise_combination_string`` (parsed by
        ``sutils.parse_basis_string``).  A component whose name is absent from
        the string is ignored even if its PSD function / bounds are supplied.

        Components
        ----------
        ``'unc'``
            Per-pulsar intrinsic red noise (``Red``).  Requires
            ``irn_psd_function`` and its bound arrays.
        ``'cor'``
            Common correlated process (``Correlated``).  Requires
            ``gwb_psd_function``, ``orf_function`` and their bound arrays.
        ``'dm'``
            DM noise (``Red`` with ``name='dm'``).  Requires
            ``dm_psd_function`` and its bound arrays.
        ``'gtm'``
            Gaussian timing model (``GaussianTiming``).  Uses
            ``data.adaptus_basis`` as the precomputed per-pulsar basis and
            ``gt_psd_val`` as the optional fixed phi array
            ``(adaptus_size, npsrs)``.  When ``gt_psd_val`` is None the GTM
            PSD is held at unit variance (one free parameter per basis column
            would require ``lower/upper_bound_psd`` instead).
        ``'det'``
            Deterministic signal (``Deterministic``).  Requires
            ``det_delay_function`` and ``det_parameter_bounds``.
        ``'chrom'``
            Chromatic (scattering-like) Fourier GP, ``ChromaticRed``, scaled
            by ``(f_ref/nu)**chrom_index``.  Requires ``chrom_psd_function``
            and its bound arrays; ``num_chrom_bins`` frequencies.
        ``'cquad'``
            Chromatic quadratic (enterprise ``chrom_quad``), ``ChromaticQuad``,
            at the same ``chrom_index``; marginalised with a fixed broad prior
            (``cquad_log10_amp``), optionally with the timing model projected
            out (``cquad_project_timing``).  No sampled parameters.
        ``'sw'``
            Time-domain solar-wind GP, ``TimeDomainSolarWind`` (port of PINT
            ``TimeDomainSWNoise``).  Requires ``sw_geometry``; kernel, nodes
            and priors go in ``sw_kwargs`` (see that class, and
            ``read_tdsw_par`` for par-file settings).

        The extra blocks must be separate blocks listed after ``dm``/``gtm``
        in the string, in a fixed order, e.g.
        ``"ltm|unc+cor->unc ; dm, chrom, cquad, sw"``.  Sample the shared
        chromatic index with ``model_maker(..., varied_chrom_gp_index=True)``.

        Parameters
        ----------
        red_noise_combination_string : str
            Combination string passed directly to ``SuperSignal``, e.g.
            ``"[T]:unc+cor->unc | dm"``.
        use_pulsar_tspan : bool, default False
            Use per-pulsar timespans for the ``unc`` and ``dm`` frequency
            grids; otherwise the PTA timespan is used.
        irn_psd_function : callable, optional
            ``psd_func(f, df, *params) -> (num_irn_bins,)``.
        gwb_psd_function : callable, optional
            ``psd_func(f, df, *params) -> (num_gwb_bins,)``.
        orf_function : callable, optional
            ``orf_func(angle, *params) -> (n_pairs,)``.
        dm_psd_function : callable, optional
            ``psd_func(f, df, *params) -> (num_dm_bins,)``.
        det_delay_function : callable, optional
            Deterministic delay model.
        irn_lower_bound_psd, irn_upper_bound_psd : array, optional
            Prior bounds for the IRN PSD's varied parameters, one per
            parameter (not one per bin unless the PSD is a free spectrum).
        dm_lower_bound_psd, dm_upper_bound_psd : array, optional
            As above for DM noise.
        gwb_lower_bound_psd, gwb_upper_bound_psd : array, optional
            As above for the GWB PSD.
        upper_bound_orf, lower_bound_orf : array, optional
            Prior bounds for the ORF's free parameters.
        det_parameter_bounds : array, optional
            Prior bounds for the deterministic signal parameters.
        gt_psd_val : array, optional
            Fixed GTM phi, shape ``(adaptus_size, npsrs)``.  One value per
            basis column per pulsar.  When None the GTM PSD is held at unit
            variance and no GTM parameters are sampled.

        Returns
        -------
        SuperSignal
        """
        parsed = sutils.parse_basis_string(red_noise_combination_string)
        red_signal_names = parsed['shared_names'] + parsed['separate_names']

        signals = []

        if 'unc' in red_signal_names:
            signals.append(Red(
                name='unc',
                psd_function=irn_psd_function,
                nfreqs=self.data.num_irn_bins,
                lower_bound_psd=irn_lower_bound_psd,
                upper_bound_psd=irn_upper_bound_psd,
                data=self.data,
                use_pulsar_tspan=use_pulsar_tspan,
            ))

        if 'gtm' in red_signal_names:
            # `basis` and `gtm_psd` are both supplied here; GaussianTiming
            # uses `basis` for the column vectors and `gtm_psd` for a fixed
            # phi (no sampled GTM parameters when gt_psd_val is not None).
            # The GTM PSD is always mode-resolved: one value per basis column,
            # not one per sin/cos pair.  `adaptus_size` is the column count
            # (nmodes), so bounds passed here must have that many entries.
            signals.append(GaussianTiming(
                name='gtm',
                nmodes=self.data.adaptus_size,
                gtm_psd=gt_psd_val,
                lower_bound_psd=None,
                upper_bound_psd=None,
                data=self.data,
                basis=self.data.adaptus_basis,
            ))

        if 'cor' in red_signal_names:
            signals.append(Correlated(
                name='cor',
                psd_function=gwb_psd_function,
                orf_function=orf_function,
                nfreqs=self.data.num_gwb_bins,
                lower_bound_psd=gwb_lower_bound_psd,
                upper_bound_psd=gwb_upper_bound_psd,
                upper_bound_orf=upper_bound_orf,
                lower_bound_orf=lower_bound_orf,
                data=self.data,
            ))

        if 'dm' in red_signal_names:
            signals.append(Red(
                name='dm',
                psd_function=dm_psd_function,
                nfreqs=self.data.num_dm_bins,
                lower_bound_psd=dm_lower_bound_psd,
                upper_bound_psd=dm_upper_bound_psd,
                data=self.data,
                use_pulsar_tspan=use_pulsar_tspan,
            ))

        if 'chrom' in red_signal_names:
            signals.append(ChromaticRed(
                name='chrom',
                psd_function=chrom_psd_function,
                nfreqs=num_chrom_bins,
                lower_bound_psd=chrom_lower_bound_psd,
                upper_bound_psd=chrom_upper_bound_psd,
                data=self.data,
                use_pulsar_tspan=use_pulsar_tspan,
                chrom_index=chrom_index,
            ))
        if 'cquad' in red_signal_names:
            signals.append(ChromaticQuad(
                name='cquad',
                data=self.data,
                chrom_index=chrom_index,
                log10_amp=cquad_log10_amp,
                project_timing=cquad_project_timing,
            ))
        if 'sw' in red_signal_names:
            if sw_geometry is None:
                raise ValueError("an 'sw' block needs sw_geometry (per-pulsar, in pc).")
            signals.append(TimeDomainSolarWind(
                name='sw',
                data=self.data,
                sw_geometry=sw_geometry,
                **(sw_kwargs or {}),
            ))
        if 'det' in red_signal_names:
            signals.append(Deterministic(
                name='det',
                data=self.data,
                nfreqs_det=self.data.num_det_bins,
                get_delays_func=det_delay_function,
                det_parameter_bounds=det_parameter_bounds,
            ))

        return SuperSignal(
            signal_list=signals,
            signal_combination_string=red_noise_combination_string,
            data=self.data,
        )