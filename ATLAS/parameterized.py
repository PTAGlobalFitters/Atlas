import numpy as np
import warnings, inspect, jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from ATLAS.utils import jit_method

fref = 1 / (1 * 365.25 * 24 * 60 * 60)   # 1/yr, reference frequency


_AMPLITUDE_NAME_MARKERS = ("log10_a", "log10_rho", "log10_amp")


def _renorm_shift(param_names, logrenorm_offset):
    """Per-parameter additive shift of the prior bounds for a unit change.

    ``renorm_const`` rescales the residuals, so only LOG-AMPLITUDE parameters
    move under it.  A spectral index (``gamma``), a bend frequency, a turnover
    exponent or an ORF parameter is dimensionless and must not be shifted --
    but the old code added ``logrenorm_offset`` to every bound of every PSD,
    silently sliding those priors whenever ``renorm_const != 1``.

    Amplitude-ness is decided by name: anything containing ``log10_A``,
    ``log10_rho`` (hence ``halflog10_rho``) or ``log10_amp``, case-insensitively.
    A free spectrum is all-amplitude, so its behaviour is unchanged.
    """
    is_amp = np.array(
        [any(m in str(n).lower() for m in _AMPLITUDE_NAME_MARKERS)
         for n in np.atleast_1d(param_names)],
        dtype=bool,
    )
    return jnp.where(jnp.asarray(is_amp), logrenorm_offset, 0.0)


def _signature_param_names(func):
    """Ordered parameter names of `func`, excluding only ``**kwargs``.

    ``VAR_POSITIONAL`` is KEPT.  A free-spectral PSD is written as
    ``free_spectrum(f, df, *halflog10_rho)``, so the sentinel that marks it as
    a free spectrum lives on the starred parameter; dropping variadics here
    would return an empty name list and every free spectrum would be
    misparsed as a zero-parameter model.

    ``**kwargs`` is excluded by kind rather than by the old
    ``"args" not in name`` substring test, which also dropped any parameter
    whose name merely contained the substring (``nargs``, ``target_args``).
    """
    return np.array([
        p.name for p in inspect.signature(func).parameters.values()
        if p.kind is not inspect.Parameter.VAR_KEYWORD
    ])

# ---------------------------------------------------------------------------
# Name helpers (companions to _parse_psd_func / _parse_orf_func)
# ---------------------------------------------------------------------------
def _psd_signature_names(psd_func, n_bins, free_spec_sentinel="halflog10_rho"):
    """Ordered names of a PSD function's parameters, excluding ``f`` and ``df``.

    Parsed exactly as ``_parse_psd_func`` parses them, so indexing this array
    with the matching ``*_varied_indxs`` lines the names up with the sampled
    parameters (and hence with the prior-bound arrays).

    A free-spectral model carries a single sentinel argument in its signature
    but ``n_bins`` actual parameters, so names are generated as
    ``f"{free_spec_sentinel}_{i}"`` for i in range(n_bins).  ``n_bins`` must be
    whatever was passed to ``_parse_psd_func`` for the same function -- for a
    mode-resolved GTM block that is ``n_gtm_eval``, not ``gtm_bins``.

    Parameters
    ----------
    psd_func : callable            signature ``(f, df, *params)``
    n_bins : int                   parameter count for the free-spectral case
    free_spec_sentinel : str       signature name marking a free spectrum

    Returns
    -------
    np.ndarray of str
    """
    sigs = _signature_param_names(psd_func)[2:]  # skip leading `f`, `df`
    if free_spec_sentinel in sigs:
        return np.array([f"{free_spec_sentinel}_{i}" for i in range(n_bins)])
    return sigs


def _orf_signature_names(orf_func):
    """Ordered names of an ORF function's parameters, excluding ``angle``.

    Parameters
    ----------
    orf_func : callable            signature ``(angle, *params)``

    Returns
    -------
    np.ndarray of str              empty when the ORF takes no free parameters
    """
    return _signature_param_names(orf_func)[1:]  # skip leading `angle`

# ---------------------------------------------------------------------------
# Helper to parse a PSD function's helper-dictionary the same way it is done
# for the GWB PSD.  Returns (param_value_container, varied_param_indxs).
# ---------------------------------------------------------------------------
def _parse_psd_func(psd_func, helper_dict, n_bins, free_spec_sentinel="halflog10_rho"):
    """
    Inspect `psd_func` and build the parameter-container / varied-index arrays
    used to evaluate the PSD inside JAX-JIT code.

    The function signature is assumed to start with (f, df, *params).  The
    sentinel `free_spec_sentinel` marks the free-spectral (bin-by-bin) model
    whose parameter count equals `n_bins` rather than the number of named
    keyword arguments.

    Parameters
    ----------
    psd_func : callable
        The PSD function, e.g. from a GWBFunctions-like class.
    helper_dict : dict
        A dictionary that mirrors the structure of `gwb_helper_dictionary` but
        for this particular PSD.  Expected keys (when applicable):
          * "psd_param_upper_lim"   – upper prior bounds for varied params
          * "psd_param_lower_lim"   – lower prior bounds for varied params
          * "ordered_psd_model_params" – param name array for ordering check
          * "fixed_psd_param_indices"  – indices of fixed params (optional)
          * "fixed_psd_param_values"   – values of those fixed params (optional)
    n_bins : int
        Number of frequency bins (used only for the free-spectral model).
    free_spec_sentinel : str
        Name in the function signature that marks the free-spectral model.

    Returns
    -------
    param_value_container : jnp.ndarray
        Zero-filled container of length ``len(signature params)`` (or
        ``n_bins`` for a free spectrum); fixed slots are pre-filled with their
        values, so evaluating the PSD only needs the varied slots written.
    varied_param_indxs : jnp.ndarray[int]
        Indices of the parameters that are *not* fixed (i.e. sampled).  Its
        length is the per-pulsar parameter count, and the helper dictionary's
        prior-bound arrays must have exactly that length.

    Notes
    -----
    On the free-spectral branch every bin is varied: ``fixed_psd_param_indices``
    is ignored there (a warning is emitted if it was supplied).
    """
    sigs = _signature_param_names(psd_func)[2:]  # skip leading `f` and `df`

    if free_spec_sentinel in sigs:
        # On the free-spectral branch every bin is varied -- "fixed_psd_param_indices"
        # is NOT honoured for a free spectrum.  A caller that expects to hold
        # bins fixed gets the opposite, and its prior-bound arrays will then be
        # shorter than the parameter block this reports.  Say so loudly rather
        # than letting the length mismatch surface much later.
        if "fixed_psd_param_indices" in helper_dict:
            warnings.warn(
                f"{getattr(psd_func, '__name__', psd_func)!r} is treated as a "
                "free spectrum, so 'fixed_psd_param_indices' is IGNORED and all "
                f"{n_bins} bins are sampled.",
                stacklevel=2,
            )
        container = jnp.zeros(n_bins)
        varied = jnp.arange(n_bins, dtype=int)
        return container, varied

    container = jnp.zeros(len(sigs))
    if "fixed_psd_param_indices" in helper_dict:
        fixed_idx = helper_dict["fixed_psd_param_indices"]
        fixed_val = helper_dict["fixed_psd_param_values"]
        container = container.at[fixed_idx].set(fixed_val)
        # `i not in fixed_idx` on a jnp array goes through __contains__ on a
        # traced/complex array type; force a plain Python set of ints.
        fixed_set = {int(i) for i in np.asarray(fixed_idx).reshape(-1)}
        varied = jnp.array(
            [i for i in range(len(sigs)) if i not in fixed_set], dtype=int
        )
    else:
        varied = jnp.arange(len(sigs), dtype=int)

    # Ordering safety-check
    if "ordered_psd_model_params" in helper_dict:
        supplied = helper_dict["ordered_psd_model_params"]
        if not (np.all(supplied[varied] == sigs[varied])):
            raise ValueError((
                f"PSD param ordering mismatch for {psd_func.__name__}. "
                f"Signature expects {sigs[varied]}, got {supplied[varied]}."
            ))

    return container, varied


def _parse_orf_func(orf_func, helper_dict):
    """
    Inspect `orf_func` and determine whether it has free parameters
    beyond the angular separation) or has free parameters to be sampled.

    The function signature is assumed to start with (angle, *params).

    Fixed-ness is decided by the PRESENCE of ``"ordered_orf_model_params"`` in
    the helper dictionary, not by the signature; a parametrised ORF without
    that key is therefore treated as fixed and evaluated at its defaults, which
    this warns about.

    Parameters
    ----------
    orf_func : callable            signature ``(angle, *params)``
    helper_dict : dict             the GWB helper dictionary

    Returns
    -------
    orf_fixed : bool
        True when the ORF is to be called as ``orf_func(xi)`` with no sampled
        parameters.
    orf_signs : np.ndarray of str
        Parameter names after ``angle`` (whether or not they are sampled).
    """
    orf_signs = _signature_param_names(orf_func)[1:]  # skip leading `angle`

    if "ordered_orf_model_params" in helper_dict:
        supplied = helper_dict["ordered_orf_model_params"]
        if not (np.all(supplied == orf_signs)):
            raise ValueError((
                f"ORF param ordering mismatch.  Signature expects {orf_signs}, "
                f"got {supplied}."
            ))
        return False, orf_signs

    # Fixed-ness is decided by the PRESENCE OF A DICT KEY, not by the signature.
    # An ORF that really does take free parameters but whose helper dictionary
    # omits 'ordered_orf_model_params' is silently treated as fixed and then
    # called as orf_func(xi) -- which either raises far downstream or, worse,
    # quietly runs on the parameters' defaults.
    if len(orf_signs) > 0:
        warnings.warn(
            f"ORF {getattr(orf_func, '__name__', orf_func)!r} has free "
            f"parameters {list(orf_signs)} but the helper dictionary has no "
            "'ordered_orf_model_params' key, so it is being treated as FIXED "
            "and will be evaluated at its default parameter values.",
            stacklevel=2,
        )
    return True, orf_signs


def _bin_idx_to_mode_idx(idx_arr):
    """
    Expand an array of frequency-*bin* indices into the corresponding
    frequency-*mode* (quadrature, sin/cos) indices, in a way that matches
    `jnp.repeat(x, 2, axis=0)`: bin index ``i`` maps to modes ``2*i`` and
    ``2*i + 1`` (row ``i`` of a bin-resolution array is duplicated into rows
    ``2*i`` and ``2*i+1`` of the mode-resolution array).
    """
    idx_arr = jnp.asarray(idx_arr)
    return jnp.stack([2 * idx_arr, 2 * idx_arr + 1], axis=1).reshape(-1)

# ---------------------------------------------------------------------------
# Generic extra per-pulsar blocks (chromatic GP, chromatic quadratic,
# time-domain solar wind, ...)
# ---------------------------------------------------------------------------

class _ExtraBlock:
    """Bookkeeping for one extra block; plain attributes, built once."""
    __slots__ = ("name", "psd_func", "bins", "bin_slice", "f", "df",
                 "param_container", "varied_indxs", "n_varied", "p0", "p1",
                 "fidxs", "K", "DIR", "fidxs_modes", "K_modes", "DIR_modes")


class _ExtraBlocksMixin:
    """Per-pulsar, pulsar-diagonal phi blocks beyond IRN / DM / GTM.

    ``extra_components`` is a list of dicts, one per block, in T-matrix column
    order::

        {"name": str,               # key into signal_indices
         "psd_func": callable,      # psd(f, df, *params) -> (bins,)
         "helper_dictionary": dict, # as for dm (make_irn_model output)
         "bins": int,               # = block columns / 2
         "freqs": array}            # (bins,) or (Npulsars, bins)

    Each block is bin-resolved like ``dm``: one phi value per column pair.
    Its sampled parameters are appended to ``xs`` after every other block,
    pulsar-major, starting at ``self.extra_start_idx``.
    """

    def _init_extra_blocks(self, extra_components, signal_indices, tm_offset,
                           Npulsars, param_start):
        self.extra_blocks = []
        self.extra_start_idx = int(param_start)
        idx = int(param_start)
        upper, lower = [], []
        diag_idx = jnp.arange(Npulsars)
        for comp in (extra_components or []):
            blk = _ExtraBlock()
            blk.name = comp["name"]
            sl = signal_indices[blk.name]
            ncol = sl.stop - sl.start
            if (sl.start - tm_offset) % 2 or ncol % 2:
                raise ValueError(
                    f"extra block {blk.name!r} must start on a column pair and "
                    f"have an even width; got columns {sl} (timing offset "
                    f"{tm_offset}).  Pad its basis with a zero column.")
            blk.bin_slice = slice((sl.start - tm_offset) // 2,
                                  (sl.stop - tm_offset) // 2)
            blk.bins = ncol // 2
            if comp.get("bins") is not None and int(comp["bins"]) != blk.bins:
                raise ValueError(
                    f"extra block {blk.name!r}: bins={comp['bins']} but "
                    f"signal_indices spans {blk.bins} column pairs.")
            f = jnp.asarray(comp["freqs"], dtype=float)
            if f.ndim == 1:
                f = jnp.broadcast_to(f, (Npulsars, blk.bins))
            if f.shape != (Npulsars, blk.bins):
                raise ValueError(
                    f"extra block {blk.name!r}: freqs has shape {f.shape}, "
                    f"needs ({blk.bins},) or ({Npulsars}, {blk.bins}).")
            blk.f = f
            blk.df = jnp.diff(jnp.concatenate((jnp.zeros((Npulsars, 1)), f), axis=1))
            blk.psd_func = comp["psd_func"]
            helper = comp["helper_dictionary"]
            blk.param_container, blk.varied_indxs = _parse_psd_func(
                blk.psd_func, helper, blk.bins)
            blk.n_varied = int(len(blk.varied_indxs))
            blk.p0, blk.p1 = idx, idx + blk.n_varied * Npulsars
            idx = blk.p1
            if blk.n_varied:
                shift = _renorm_shift(
                    _psd_signature_names(blk.psd_func, blk.bins)[
                        np.asarray(blk.varied_indxs)],
                    self.logrenorm_offset)
                upper.append(jnp.tile(
                    jnp.asarray(helper["psd_param_upper_lim"]) + shift, Npulsars))
                lower.append(jnp.tile(
                    jnp.asarray(helper["psd_param_lower_lim"]) + shift, Npulsars))
            blk.fidxs = jnp.arange(blk.bin_slice.start, blk.bin_slice.stop)
            blk.K = jnp.repeat(blk.fidxs[:, None], Npulsars, axis=1)
            blk.DIR = jnp.repeat(diag_idx[None, :], blk.bins, axis=0)
            blk.fidxs_modes = _bin_idx_to_mode_idx(blk.fidxs)
            blk.K_modes = jnp.repeat(blk.fidxs_modes[:, None], Npulsars, axis=1)
            blk.DIR_modes = jnp.repeat(diag_idx[None, :], 2 * blk.bins, axis=0)
            self.extra_blocks.append(blk)
        self.extra_end_idx = idx
        self.num_extra_params = idx - int(param_start)
        self.has_extra = len(self.extra_blocks) > 0
        cat = lambda xs: jnp.concatenate(xs) if xs else jnp.array([])
        return cat(upper), cat(lower)

    def _extra_stops(self):
        return [b.bin_slice.stop for b in getattr(self, "extra_blocks", [])]

    def _eval_extra_psd(self, blk, xs):
        """PSD of one extra block for every pulsar -> (bins, Npulsars)."""
        per_psr = xs[blk.p0:blk.p1].reshape(self.Npulsars, blk.n_varied)

        def _single(params, freqs_, df_):
            filled = blk.param_container.at[blk.varied_indxs].set(params)
            return blk.psd_func(freqs_, df_, *filled)

        return jax.vmap(_single)(per_psr, blk.f, blk.df).T

    def _add_extra_bins(self, phi_diag, xs):
        """Add every extra block to a bin-resolution phi diagonal."""
        for blk in self.extra_blocks:
            phi_diag = phi_diag.at[blk.bin_slice].add(self._eval_extra_psd(blk, xs))
        return phi_diag

    def _extra_param_names(self):
        names = []
        for blk in self.extra_blocks:
            pn = _psd_signature_names(blk.psd_func, blk.bins)[
                np.asarray(blk.varied_indxs)]
            for psr in self.pulsar_names:
                names += [f"{psr}_{blk.name}_{q}" for q in pn]
        return names


# ---------------------------------------------------------------------------
# Shared GTM (Gaussian timing model) geometry
# ---------------------------------------------------------------------------

class _GTMModeMixin:
    """GTM sizing and evaluation shared by both covariance classes.

    The GTM basis columns are principal components of the timing-model prior
    predictive, NOT sin/cos quadrature pairs of a frequency.  Pairing
    consecutive columns into a "bin" and handing both the same PSD value --
    which is what the bin-resolution path does, via ``jnp.repeat(x, 2)`` --
    forces two unrelated PCs to share a variance.

    With ``gtm_mode_resolved=True`` (the default) the GTM PSD is instead
    carried at BASIS-COLUMN resolution: one value, and one sampled parameter
    per pulsar, for every column.  ``phi`` is then returned at mode resolution
    (``nmodes = 2 * n_total_bins`` along axis 0) throughout the class, with the
    non-GTM blocks' bin values duplicated across their two quadrature modes.
    Supplying a precomputed ``gtm_psd`` array is the fixed-value special case
    of exactly that layout.

    Set ``gtm_mode_resolved=False`` to recover the old bin-resolution
    behaviour (one PSD value per pair of columns).
    """

    def _init_gtm_geometry(self, gtm_bins_arg, f_gtm,
                           gtm_psd_func, gtm_helper_dictionary):
        """Size the GTM block and validate everything that describes it.

        Must be called after ``self.GTM_slice``, ``self.has_gtm_direct``,
        ``self.gtm_mode_resolved``, ``self.gtm_psd_input`` and
        ``self.Npulsars`` are set, and before the GTM PSD is parsed (which
        needs ``n_gtm_eval``).

        Sets
        ----
        gtm_bins : int          GTM width in frequency bins, taken from
                                ``signal_indices['gtm']`` (the ``gtm_bins``
                                argument is only cross-checked against it).
        n_gtm_modes : int       ``2 * gtm_bins`` -- the number of basis columns.
        GTM_slice_modes : slice the GTM block in column space.
        n_gtm_eval : int        points at which the PSD function is evaluated:
                                ``n_gtm_modes`` when mode-resolved, else
                                ``gtm_bins``.  This is the per-pulsar parameter
                                count for a free spectrum.
        f_gtm, df_gtm : arrays  ``(Npulsars, n_gtm_eval)``; only set when the
                                PSD comes from a function.

        Parameters
        ----------
        gtm_bins_arg : int or None   caller's bin count, cross-checked
        f_gtm : array or None        frequencies, at bin or column length
        gtm_psd_func : callable or None
        gtm_helper_dictionary : dict or None

        Raises
        ------
        ValueError
            On any disagreement between ``gtm_bins``, ``signal_indices``,
            ``f_gtm`` and a directly-supplied ``gtm_psd``.
        """
        n_bins = self.GTM_slice.stop - self.GTM_slice.start
        self.gtm_bins = n_bins
        self.n_gtm_modes = 2 * n_bins
        self.GTM_slice_modes = slice(2 * self.GTM_slice.start,
                                     2 * self.GTM_slice.stop)
        if gtm_bins_arg is not None and int(gtm_bins_arg) != n_bins:
            raise ValueError(
                f"gtm_bins={gtm_bins_arg} disagrees with signal_indices['gtm'], "
                f"which spans {n_bins} frequency bins / {self.n_gtm_modes} "
                "basis columns.")

        # Points at which the PSD function is evaluated: one per basis column
        # when mode-resolved, one per sin/cos pair otherwise.
        self.n_gtm_eval = self.n_gtm_modes if self.gtm_mode_resolved else n_bins

        if self.has_gtm_direct:
            arr = jnp.asarray(self.gtm_psd_input)
            if arr.shape != (self.n_gtm_modes, self.Npulsars):
                raise ValueError(
                    f"gtm_psd must have shape ({self.n_gtm_modes}, "
                    f"{self.Npulsars}) -- one value per basis column per pulsar "
                    f"-- to match the {n_bins}-bin GTM slice in signal_indices; "
                    f"got {arr.shape}.")
            self.gtm_psd_input = arr
            return

        if gtm_psd_func is None or f_gtm is None or gtm_helper_dictionary is None:
            raise ValueError(
                "gtm_psd_func, f_gtm and gtm_helper_dictionary must all be "
                "supplied when a GTM block is present and no explicit gtm_psd "
                "array is given.")

        f = jnp.asarray(f_gtm)
        if f.ndim == 1:
            if f.shape[0] == n_bins and self.n_gtm_eval == self.n_gtm_modes:
                # Bin-resolution placeholder frequencies for a mode-resolved
                # block: the PCs have no real frequency, so duplicate rather
                # than reject.
                f = jnp.repeat(f, 2)
            if f.shape[0] != self.n_gtm_eval:
                raise ValueError(
                    f"f_gtm has {f.shape[0]} entries but the GTM block needs "
                    f"{self.n_gtm_eval} "
                    f"({'one per basis column' if self.gtm_mode_resolved else 'one per frequency bin'}).")
            f = jnp.broadcast_to(f, (self.Npulsars, self.n_gtm_eval))
        elif f.shape != (self.Npulsars, self.n_gtm_eval):
            raise ValueError(
                f"f_gtm must have shape ({self.n_gtm_eval},) or "
                f"({self.Npulsars}, {self.n_gtm_eval}); got {f.shape}.")
        self.f_gtm = f
        self.df_gtm = jnp.diff(
            jnp.concatenate((jnp.zeros((self.Npulsars, 1)), self.f_gtm), axis=1))

    @jit_method
    def _gtm_mode_values(self, gtm_flat):
        """GTM phi at basis-column resolution.

        Parameters
        ----------
        gtm_flat : jnp.ndarray, shape (num_GTM_params,)
            The GTM slice of ``xs``; ignored when the block carries no sampled
            parameters.

        Returns
        -------
        jnp.ndarray, shape (n_gtm_modes, Npulsars)
            The directly-supplied ``gtm_psd`` if there is one; ones when the
            block has no varied parameters; otherwise the evaluated PSD, with
            each value duplicated across its two quadrature modes if the block
            is bin-resolved rather than mode-resolved.
        """
        if self.has_gtm_direct:
            return self.gtm_psd_input
        if self._do_not_vary_gtm:
            return jnp.ones((self.n_gtm_modes, self.Npulsars))
        vals = self._eval_gtm_psd_all(gtm_flat)      # (n_gtm_eval, Npulsars)
        if self.n_gtm_eval != self.n_gtm_modes:      # bin-resolved fallback
            vals = jnp.repeat(vals, 2, axis=0)
        return vals


# ---------------------------------------------------------------------------
# Unified red-noise covariance-matrix class
# ---------------------------------------------------------------------------

class PerPulsarRedNoise(_GTMModeMixin, _ExtraBlocksMixin):
    """Red-noise covariance (phi) for a PTA with NO cross-pulsar correlations.

    Combines up to three per-pulsar spectral components -- intrinsic red noise
    (``irn``), DM noise (``dm``) and the Gaussian timing model (``gtm``) -- each
    described by a callable PSD function plus a helper dictionary carrying its
    prior bounds and optional fixed parameters.  At least one must be supplied.
    For a model that also carries a common, ORF-correlated process, use
    ``CorrelatedPulsarRedNoise``.

    Because the pulsars are independent, phi is diagonal in the pulsar index
    and is carried as that diagonal rather than as a square matrix:
    ``get_phi_mat`` and ``get_phi_mat_full`` are aliases of ``get_phi_diag``
    and all return shape ``(n_rows, Npulsars)``.  ``get_phi_mat_inv`` is the
    one method that expands to ``(n_rows, Npulsars, Npulsars)``, for interface
    parity with the correlated class.

    Parameter vector layout
    -----------------------
    The flat ``xs`` vector consumed by ``get_phi_diag`` is ordered as::

        xs = [ irn_psd_params (num_IR_params),
               dm_psd_params  (num_DM_params),
               gtm_psd_params (num_GTM_params) ]

    Each block is present only when its component is; each is flat with the
    pulsar index varying *slowest* (all parameters for pulsar 0, then pulsar 1,
    ...).  There are no GWB or ORF parameters in this class.  ``get_param_names``
    returns the matching names, and ``lower/upper_prior_lim_all`` the matching
    bounds -- their length is checked against this layout at construction.

    PSD function convention
    -----------------------
    Every PSD function must have the signature::

        psd_func(f, df, *params) -> jnp.ndarray of shape (n_points,)

    where ``params`` is the parameter tuple *for a single pulsar*; the function
    is called once per pulsar inside a ``vmap``.

    Row resolution
    --------------
    Rows are frequency BINS, shape ``(n_total_bins, Npulsars)``, except when the
    GTM block is mode-resolved -- ``gtm_mode_resolved=True`` (the default) or a
    direct ``gtm_psd`` array -- in which case every returned array is at BASIS
    COLUMN resolution, ``(2 * n_total_bins, Npulsars)``: the IRN/DM/GWB bin
    values duplicated across their two quadrature modes, and one independent
    value per GTM column.  See ``_GTMModeMixin`` for why the GTM block is
    treated this way.  ``get_phi_mat_inv`` follows whichever resolution its
    input is in.

    Helper dictionary keys
    ----------------------
    ``irn_helper_dictionary`` / ``dm_helper_dictionary`` /
    ``gtm_helper_dictionary``:
        * ``"psd_param_upper_lim"``      (required) bounds for the VARIED params
        * ``"psd_param_lower_lim"``      (required)
        * ``"ordered_psd_model_params"`` (optional) ordering cross-check
        * ``"fixed_psd_param_indices"``  (optional) ignored for a free spectrum
        * ``"fixed_psd_param_values"``   (optional)

    Bounds are shifted by ``0.5*log10(renorm_const)`` for LOG-AMPLITUDE
    parameters only (see ``_renorm_shift``).

    Authors
    -------
    Nima Laal (original pandora classes, 02/12/2025)
    Unified refactor: Ge (06/2025)
    """

    def __init__(
        self,
        signal_indices,          # from build_basis
        linear_timing_model_size,
        irn_psd_func=None,
        irn_helper_dictionary=None,
        irn_bins=None,
        f_irn=None,
        # ---- DM noise (optional) ----
        dm_psd_func=None,
        dm_helper_dictionary=None,
        dm_bins=None,
        f_dm=None,
        # ---- GTM noise (optional) ----
        gtm_psd_func=None,
        gtm_helper_dictionary=None,
        gtm_bins=None,
        f_gtm=None,
        gtm_psd=None,            # direct, mode-resolution GTM phi: (2*gtm_nfreqs, Npulsars)
        gtm_mode_resolved=True,  # one PSD value per basis COLUMN (see _GTMModeMixin)
        # ---- extra per-pulsar blocks: chrom / cquad / sw (optional) ----
        extra_components=None,
        # ---- shared ----
        renorm_const=1.0,
        Npulsars=1,
        pulsar_names=None
    ):
        """
        Parameters
        ----------
        signal_indices : dict[str, slice]
            Column slices of the T-matrix per signal, as returned by
            ``SuperSignal.build_basis``.  The keys ``'unc'`` / ``'dm'`` /
            ``'gtm'`` are read for whichever components are present, and they
            fix each block's width -- the ``*_bins`` arguments are only
            cross-checked against them.
        linear_timing_model_size : int
            Number of leading timing-model columns in the T-matrix; subtracted
            before converting column indices to frequency bins.
        irn_psd_func, dm_psd_func, gtm_psd_func : callable, optional
            ``psd_func(f, df, *params)`` for one pulsar.  Passing None omits
            that component; at least one component is required.
        irn_helper_dictionary, dm_helper_dictionary, gtm_helper_dictionary : dict, optional
            Prior bounds and optional fixed-parameter info (see class docstring).
            Required whenever the matching PSD function is given.
        irn_bins, dm_bins, gtm_bins : int, optional
            Number of frequency bins for each component.
        f_irn, f_dm, f_gtm : array, optional
            Frequencies, ``(n_bins,)`` or ``(Npulsars, n_bins)``.  For a
            mode-resolved GTM block ``f_gtm`` may also be given at column
            resolution; a bin-length array is duplicated.
        gtm_psd : array, optional
            Precomputed GTM phi, ``(2*gtm_bins, Npulsars)`` -- one value per
            basis column per pulsar.  When given, no GTM parameters are sampled
            and ``gtm_psd_func``/``gtm_helper_dictionary`` are unused.
        gtm_mode_resolved : bool, default True
            One PSD value (and one sampled parameter) per GTM basis column
            rather than one per sin/cos pair.  See ``_GTMModeMixin``.
        renorm_const : float, default 1.0
            Unit rescaling of the residuals; shifts log-amplitude prior bounds
            by ``0.5*log10(renorm_const)``.
        Npulsars : int
        pulsar_names : list of str, optional
            Required only by ``get_param_names``.

        Raises
        ------
        ValueError
            If no component is supplied, if any block's declared size
            disagrees with ``signal_indices``, or if the prior bounds do not
            cover exactly the sampled parameters.
        """
        self.Npulsars = Npulsars
        if pulsar_names is not None:
            if not (len(pulsar_names) == Npulsars):
                raise ValueError((
                    f"len(pulsar_names)={len(pulsar_names)} does not match "
                    f"Npulsars={Npulsars}."
                ))
        self.pulsar_names = pulsar_names

        # ------------------------------------------------------------------ #
        #  Frequency index arrays from signal_indices                          #
        # ------------------------------------------------------------------ #
        tm_offset = linear_timing_model_size

        def _fourier_slice(name):
            """Column slice of `name` in the T-matrix -> its frequency-bin slice."""
            sl = signal_indices[name]
            return slice((sl.start - tm_offset) // 2,
                        (sl.stop  - tm_offset) // 2)

        # ------------------------------------------------------------------ #
        #  Basic bookkeeping                                                   #
        # ------------------------------------------------------------------ #
        self.irn_psd_func = irn_psd_func
        self.dm_psd_func = dm_psd_func
        self.gtm_psd_func = gtm_psd_func
        self.renorm_const = renorm_const
        self.logrenorm_offset = 0.5 * jnp.log10(renorm_const)

        has_irn = irn_psd_func is not None
        self.has_irn = has_irn
        has_dm = dm_psd_func is not None
        self.has_dm = has_dm
        # A GTM component may be specified either via a callable PSD model
        # (gtm_psd_func) or via a precomputed, mode-resolution phi array
        # (gtm_psd). The latter takes precedence if both are supplied.
        has_gtm = (gtm_psd_func is not None) or (gtm_psd is not None)
        self.has_gtm = has_gtm
        self.has_gtm_direct = gtm_psd is not None
        self.gtm_psd_input = gtm_psd
        # A directly-supplied gtm_psd is mode-resolution by definition.  False
        # when there is no GTM block at all, so every `if self.gtm_mode_resolved`
        # below is also a "there is a GTM block" test.
        self.gtm_mode_resolved = bool(
            has_gtm and (gtm_mode_resolved or self.has_gtm_direct))
        # `has_*` are booleans, never None, so the old `is None` test could never
        # fire; with no component at all the failure was a bare
        # `max([])  ->  ValueError` from the n_total_bins line below.
        if not (self.has_irn or self.has_dm or self.has_gtm or extra_components):
            raise ValueError(
                "PerPulsarRedNoise needs at least one of `irn_psd_func`, "
                "`dm_psd_func`, `gtm_psd_func`/`gtm_psd`, `extra_components`.")
        self._extra_components_arg = extra_components

        if has_irn:
            self.IRN_slice = _fourier_slice('unc')

        if has_dm:
            self.DM_slice = _fourier_slice('dm')

        if has_gtm:
            self.GTM_slice = _fourier_slice('gtm')

        # n_total_bins = stop of the last occupied slice
        all_stops = []
        if self.has_irn: all_stops.append(self.IRN_slice.stop)
        if self.has_dm:  all_stops.append(self.DM_slice.stop)
        if self.has_gtm:  all_stops.append(self.GTM_slice.stop)
        all_stops += [((signal_indices[c["name"]].stop - tm_offset) // 2)
                      for c in (extra_components or [])]
        self.n_total_bins = max(all_stops)

        # ------------------------------------------------------------------ #
        #  IRN bookkeeping                                                     #
        # ------------------------------------------------------------------ #
        if self.has_irn:
            if not (irn_bins is not None and f_irn is not None):
                raise ValueError((
                    "irn_bins and f_irn must be supplied when irn_psd_func is given."
                ))
            if not (irn_helper_dictionary is not None):
                raise ValueError((
                    "irn_helper_dictionary must be supplied when irn_psd_func is given."
                ))
            self.irn_bins = irn_bins
            self.f_irn = f_irn if f_irn.ndim == 2 else jnp.broadcast_to(f_irn, (self.Npulsars, self.irn_bins)) 
            self.df_irn = jnp.diff(jnp.concatenate((jnp.zeros((self.Npulsars, 1)), self.f_irn), axis = 1))

        # ------------------------------------------------------------------ #
        #  DM bookkeeping                                                      #
        # ------------------------------------------------------------------ #
        if self.has_dm:
            if not (dm_bins is not None and f_dm is not None):
                raise ValueError((
                    "dm_bins and f_dm must be supplied when dm_psd_func is given."
                ))
            if not (dm_helper_dictionary is not None):
                raise ValueError((
                    "dm_helper_dictionary must be supplied when dm_psd_func is given."
                ))
            self.dm_bins = dm_bins
            self.f_dm = f_dm if f_dm.ndim == 2 else jnp.broadcast_to(f_dm, (self.Npulsars, self.dm_bins))
            self.df_dm = jnp.diff(jnp.concatenate((jnp.zeros((self.Npulsars, 1)), self.f_dm), axis = 1))

        # ------------------------------------------------------------------ #
        #  GTM bookkeeping                                                      #
        # ------------------------------------------------------------------ #
        if self.has_gtm:
            self._init_gtm_geometry(gtm_bins, f_gtm,
                                    gtm_psd_func, gtm_helper_dictionary)

        # ------------------------------------------------------------------ #
        #  Parse IRN PSD                                                       #
        # ------------------------------------------------------------------ #
        if self.has_irn:
            self.irn_param_container, self.irn_varied_indxs = _parse_psd_func(
                irn_psd_func, irn_helper_dictionary, irn_bins
            )
            # Number of varied params per pulsar
            self.n_irn_varied = int(len(self.irn_varied_indxs))
            self.num_IR_params = self.n_irn_varied * self.Npulsars
        else:
            self.num_IR_params = 0

        # ------------------------------------------------------------------ #
        #  Parse DM PSD                                                        #
        # ------------------------------------------------------------------ #
        if self.has_dm:
            self.dm_param_container, self.dm_varied_indxs = _parse_psd_func(
                dm_psd_func, dm_helper_dictionary, dm_bins
            )
            self.n_dm_varied = int(len(self.dm_varied_indxs))
            self.num_DM_params = self.n_dm_varied * self.Npulsars
        else:
            self.num_DM_params = 0

        # ------------------------------------------------------------------ #
        #  Parse GTM PSD                                                        #
        # ------------------------------------------------------------------ #
        if self.has_gtm and not self.has_gtm_direct:
            # n_gtm_eval, not gtm_bins: a mode-resolved free spectrum carries
            # one parameter per basis column.
            self.gtm_param_container, self.gtm_varied_indxs = _parse_psd_func(
                gtm_psd_func, gtm_helper_dictionary, self.n_gtm_eval
            )
            self.n_gtm_varied = int(len(self.gtm_varied_indxs))
            self.num_GTM_params = self.n_gtm_varied * self.Npulsars
        else:
            # No sampled GTM params either because there's no GTM component
            # at all, or because it was supplied directly via `gtm_psd`.
            self.num_GTM_params = 0

        # ------------------------------------------------------------------ #
        #  Parameter-vector slice indices                                      #
        # ------------------------------------------------------------------ #
        # Layout: [ IRN(0..num_IR_params) | DM(..+num_DM_params) | GTM(..+num_GTM_params)
        self.irn_end_idx = self.num_IR_params
        self.dm_end_idx = self.irn_end_idx + self.num_DM_params
        self.gtm_end_idx = self.dm_end_idx + self.num_GTM_params
        if self.gtm_end_idx == self.dm_end_idx:
            self._do_not_vary_gtm = True
        else:
            self._do_not_vary_gtm = False
        # ------------------------------------------------------------------ #
        #  Prior bounds                                                        #
        # ------------------------------------------------------------------ #
        upper, lower = jnp.array([]), jnp.array([])

        if self.has_irn:
            irn_shift = _renorm_shift(
                _psd_signature_names(irn_psd_func, self.irn_bins)[
                    np.asarray(self.irn_varied_indxs)],
                self.logrenorm_offset)
            irn_upper = jnp.tile(
                jnp.asarray(irn_helper_dictionary["psd_param_upper_lim"]) + irn_shift,
                self.Npulsars
            )
            irn_lower = jnp.tile(
                jnp.asarray(irn_helper_dictionary["psd_param_lower_lim"]) + irn_shift,
                self.Npulsars
            )
            upper = jnp.concatenate([upper, irn_upper])
            lower = jnp.concatenate([lower, irn_lower])

        if self.has_dm:
            dm_shift = _renorm_shift(
                _psd_signature_names(dm_psd_func, self.dm_bins)[
                    np.asarray(self.dm_varied_indxs)],
                self.logrenorm_offset)
            dm_upper = jnp.tile(
                jnp.asarray(dm_helper_dictionary["psd_param_upper_lim"]) + dm_shift,
                self.Npulsars
            )
            dm_lower = jnp.tile(
                jnp.asarray(dm_helper_dictionary["psd_param_lower_lim"]) + dm_shift,
                self.Npulsars
            )
            upper = jnp.concatenate([upper, dm_upper])
            lower = jnp.concatenate([lower, dm_lower])

        if self.has_gtm and not self.has_gtm_direct:
            gtm_shift = _renorm_shift(
                _psd_signature_names(gtm_psd_func, self.n_gtm_eval)[
                    np.asarray(self.gtm_varied_indxs)],
                self.logrenorm_offset)
            gtm_upper = jnp.tile(
                jnp.asarray(gtm_helper_dictionary["psd_param_upper_lim"]) + gtm_shift,
                self.Npulsars
            )
            gtm_lower = jnp.tile(
                jnp.asarray(gtm_helper_dictionary["psd_param_lower_lim"]) + gtm_shift,
                self.Npulsars
            )
            upper = jnp.concatenate([upper, gtm_upper])
            lower = jnp.concatenate([lower, gtm_lower])

        x_upper, x_lower = self._init_extra_blocks(
            extra_components, signal_indices, tm_offset, self.Npulsars,
            param_start=self.gtm_end_idx)
        upper = jnp.concatenate([upper, x_upper])
        lower = jnp.concatenate([lower, x_lower])
        self.upper_prior_lim_all = upper
        self.lower_prior_lim_all = lower
        # The parameter block sizes come from the PSD SIGNATURES, the prior
        # bounds come from the HELPER DICTIONARIES, and nothing tied the two
        # together.  A mismatch (e.g. a free-spectrum PSD, whose every bin is
        # varied, paired with an empty bounds array because the caller meant to
        # hold it fixed) silently produced an `xs` layout the prior does not
        # cover -- get_lnprior then compares arrays of different length.
        n_expected = self.extra_end_idx
        if upper.shape[0] != n_expected:
            raise ValueError(
                f"Prior bounds have {upper.shape[0]} entries but the parameter "
                f"layout needs {n_expected} (IRN {self.num_IR_params} + DM "
                f"{self.num_DM_params} + GTM {self.num_GTM_params} + extra "
                f"{self.num_extra_params}).  Check that "
                "each helper dictionary's psd_param_*_lim covers exactly the "
                "varied parameters of its PSD function.")

        if renorm_const != 1:
            warnings.warn(
                "You have chosen to change units.  Make sure your amplitude "
                "priors reflect that!"
            )

    # ---------------------------------------------------------------------- #
    #  Internal PSD helpers                                                   #
    # ---------------------------------------------------------------------- #

    @jit_method
    def _eval_irn_psd_all(self, irn_params_flat):
        """
        Evaluate the IRN PSD for *all* pulsars.

        Parameters
        ----------
        irn_params_flat : jnp.ndarray, shape (num_IR_params,)
            Flat array; reshaped to (Npulsars, n_irn_varied) before vmapping.

        Returns
        -------
        jnp.ndarray, shape (irn_bins, Npulsars)
        """
        per_psr = irn_params_flat.reshape(self.Npulsars, self.n_irn_varied)

        def _single(params, freqs_, df_):
            filled = self.irn_param_container.at[self.irn_varied_indxs].set(params)
            return self.irn_psd_func(freqs_, df_, *filled)

        return jax.vmap(_single)(per_psr, self.f_irn, self.df_irn).T  # → (irn_bins, Npulsars)

    @jit_method
    def _eval_dm_psd_all(self, dm_params_flat):
        """
        Evaluate the DM PSD for *all* pulsars.

        Parameters
        ----------
        dm_params_flat : jnp.ndarray, shape (num_DM_params,)
            Flat array; reshaped to (Npulsars, n_dm_varied) before vmapping.

        Returns
        -------
        jnp.ndarray, shape (dm_bins, Npulsars)
        """
        per_psr = dm_params_flat.reshape(self.Npulsars, self.n_dm_varied)

        def _single(params, freqs_, df_):
            filled = self.dm_param_container.at[self.dm_varied_indxs].set(params)
            return self.dm_psd_func(freqs_, df_, *filled)

        return jax.vmap(_single)(per_psr, self.f_dm, self.df_dm).T  # → (dm_bins, Npulsars)

    @jit_method
    def _eval_gtm_psd_all(self, gtm_params_flat):
        """
        Evaluate the GTM PSD for *all* pulsars (function-based model only;
        not used when `gtm_psd` was supplied directly).

        Parameters
        ----------
        gtm_params_flat : jnp.ndarray, shape (num_GTM_params,)
            Flat array; reshaped to (Npulsars, n_gtm_varied) before vmapping.

        Returns
        -------
        jnp.ndarray, shape (n_gtm_eval, Npulsars)
            One row per basis column when the block is mode-resolved, else one
            row per frequency bin.  Prefer ``_gtm_mode_values``, which always
            returns column resolution.
        """
        per_psr = gtm_params_flat.reshape(self.Npulsars, self.n_gtm_varied)

        def _single(params, freqs_, df_):
            filled = self.gtm_param_container.at[self.gtm_varied_indxs].set(params)
            return self.gtm_psd_func(freqs_, df_, *filled)

        return jax.vmap(_single)(per_psr, self.f_gtm, self.df_gtm).T  # → (gtm_bins, Npulsars)

    # ---------------------------------------------------------------------- #
    #  Parameter unpacking                                                    #
    # ---------------------------------------------------------------------- #

    @jit_method
    def _unpack(self, xs):
        """Split ``xs`` into its (irn_flat, dm_flat, gtm_flat) blocks.

        Blocks belonging to absent components come back empty.
        """
        irn = xs[:self.irn_end_idx]
        dm  = xs[self.irn_end_idx:self.dm_end_idx]
        gtm = xs[self.dm_end_idx:self.gtm_end_idx]
        return irn, dm, gtm

    # ---------------------------------------------------------------------- #
    #  Core phi builders                                                      #
    # ---------------------------------------------------------------------- #

    @jit_method
    def get_phi_diag(self, xs):
        """
        Build the phi diagonal: the prior variance of every basis coefficient.

        Parameters
        ----------
        xs : jnp.ndarray
            Flat parameter vector; see the class docstring for its layout.

        Returns
        -------
        phi_diag : jnp.ndarray
            ``(n_total_bins, Npulsars)`` when the GTM block is bin-resolved (or
            absent).  When it is mode-resolved -- ``gtm_mode_resolved=True`` or
            a direct ``gtm_psd`` -- the whole array comes back at BASIS COLUMN
            resolution instead, ``(2 * n_total_bins, Npulsars)``: each non-GTM
            bin duplicated across its two quadrature modes, and one independent
            value per GTM column.
        """
        irn_flat, dm_flat, gtm_flat = self._unpack(xs)

        phi_diag = jnp.zeros((self.n_total_bins, self.Npulsars))

        if self.has_irn:
            phi_diag = phi_diag.at[self.IRN_slice].add(
                self._eval_irn_psd_all(irn_flat)
            )

        if self.has_dm:
            phi_diag = phi_diag.at[self.DM_slice].add(
                self._eval_dm_psd_all(dm_flat)
            )

        if self.has_gtm and not self.gtm_mode_resolved:
            if not self._do_not_vary_gtm:
                gtm_psd = self._eval_gtm_psd_all(gtm_flat)
            else:
                gtm_psd = 1.
            phi_diag = phi_diag.at[self.GTM_slice].add(gtm_psd)

        phi_diag = self._add_extra_bins(phi_diag, xs)

        if self.gtm_mode_resolved:
            # Expand every bin into its two quadrature modes, then overwrite the
            # GTM columns with one value per column.
            phi_diag = jnp.repeat(phi_diag, 2, axis=0)
            phi_diag = phi_diag.at[self.GTM_slice_modes].set(
                self._gtm_mode_values(gtm_flat))

        return phi_diag

    @jit_method
    def get_phi_mat(self, xs):
        """
        Alias of ``get_phi_diag``.

        This model has NO cross-pulsar correlations, so phi is diagonal in the
        pulsar index and is carried as the (n_total_bins, Npulsars) diagonal
        rather than an (n_total_bins, Npulsars, Npulsars) matrix.  The name is
        kept so callers can use either model class interchangeably; the old
        docstring claimed a square matrix that was never built.
        """
        return self.get_phi_diag(xs)

    @jit_method
    def get_phi_mat_full(self, xs):
        """Alias of ``get_phi_diag`` -- see ``get_phi_mat``."""
        return self.get_phi_diag(xs)

    @jit_method
    def get_phi_mat_CURN(self, xs):
        """
        Return ``(phi_diag, psd_common)`` for interface parity with
        ``CorrelatedPulsarRedNoise.get_phi_mat_CURN``.

        There is no common process in this model, so ``psd_common`` is None.
        This used to return the bare array, which made the documented (and
        actually used) ``phi, psd_common = model.get_phi_mat_CURN(params)``
        unpack the leading axis of phi -- a ValueError, or two rows of phi
        silently mistaken for phi and psd_common when n_total_bins == 2.

        Returns
        -------
        phi_diag : jnp.ndarray  (n_total_bins, Npulsars)
        psd_common : None
        """
        return self.get_phi_diag(xs), None

    # ---------------------------------------------------------------------- #
    #  Inversion                                                              #
    # ---------------------------------------------------------------------- #

    @jit_method
    def get_phi_mat_inv(self, phi):
        """
        Invert the phi diagonal and embed it as a (diagonal) matrix per row.

        The inversion is elementwise -- this model has no cross-pulsar
        correlations -- and the result is widened to ``(..., Npulsars,
        Npulsars)`` only so that callers can treat it like the correlated
        class's output.

        Parameters
        ----------
        phi : jnp.ndarray
            The output of ``get_phi_diag``: ``(n_total_bins, Npulsars)`` when
            bin-resolved, ``(2*n_total_bins, Npulsars)`` when the GTM block is
            mode-resolved.

        Returns
        -------
        phiinv : jnp.ndarray
            Bin-resolved input: ``(2*n_total_bins, Npulsars, Npulsars)`` -- each
            bin repeated across its two quadrature modes.  Mode-resolved input:
            ``(2*n_total_bins, Npulsars, Npulsars)`` with no repetition, since
            the input already has one row per basis column.
        logdet_phi : float
            log|phi| summed over all modes and pulsars (hence the factor 2 in
            the bin-resolved branch).
        """
        if self.gtm_mode_resolved:
            phiinv = 1.0 / phi
            log_det_phi = jnp.sum(jnp.log(phi))
        else:
            phiinv = jnp.repeat(1 / phi, 2, axis=0)
            log_det_phi = 2.0 * jnp.sum(jnp.log(phi))
        return phiinv[..., None] * jnp.eye(phiinv.shape[-1]) , log_det_phi 

    # ---------------------------------------------------------------------- #
    #  Prior                                                                  #
    # ---------------------------------------------------------------------- #

    @jit_method
    def get_lnprior(self, xs):
        """UNNORMALISED uniform log-prior over ``xs``.

        Returns an arbitrary constant (see ``_spit_neg_number``) when every
        parameter is strictly inside its bounds and -inf otherwise; the
        constant is not -log(prior volume), so this is usable for MCMC
        acceptance ratios but not for evidence.
        """
        in_bounds = jnp.logical_and(
            xs > self.lower_prior_lim_all, xs < self.upper_prior_lim_all
        ).all()
        return jax.lax.cond(in_bounds, self._spit_neg_number, self._spit_neg_infinity)

    def get_lnprior_numpy(self, xs):
        """``get_lnprior`` as a host-side numpy scalar."""
        return self.get_lnprior(xs).__array__()

    @jit_method
    def make_initial_guess(self, key):
        """Draw one uniform sample from inside the prior bounds. [n_params]"""
        return jr.uniform(
            key,
            shape=(self.upper_prior_lim_all.shape[0],),
            minval=self.lower_prior_lim_all,
            maxval=self.upper_prior_lim_all,
        )

    # ---------------------------------------------------------------------- #
    #  Utilities                                                              #
    # ---------------------------------------------------------------------- #

    def get_param_names(self):
        """
        Return the flat list of parameter names in the same order as the
        `xs` vector consumed by `get_phi_diag` / `get_phi_mat`.

        Order: [irn_psd_params (pulsar-major, param-minor),
                dm_psd_params  (pulsar-major, param-minor),
                gtm_psd_params (pulsar-major, param-minor)]

        Names are ``f"{pulsar}_{component}_{param}"``.  A directly-supplied
        ``gtm_psd`` contributes no names (it has no sampled parameters), and a
        mode-resolved GTM free spectrum contributes one name per basis column.

        Returns
        -------
        list of str, the same length as ``lower/upper_prior_lim_all``.

        Raises
        ------
        ValueError
            If ``pulsar_names`` was not supplied at construction.
        """
        if not (self.pulsar_names is not None):
            raise ValueError((
                "pulsar_names must be supplied at construction to get param names."
            ))

        names = []

        if self.has_irn:
            irn_names = _psd_signature_names(self.irn_psd_func, self.irn_bins)[
                np.asarray(self.irn_varied_indxs)
            ]
            for psr in self.pulsar_names:
                names += [f"{psr}_irn_{p}" for p in irn_names]

        if self.has_dm:
            dm_names = _psd_signature_names(self.dm_psd_func, self.dm_bins)[
                np.asarray(self.dm_varied_indxs)
            ]
            for psr in self.pulsar_names:
                names += [f"{psr}_dm_{p}" for p in dm_names]

        if self.has_gtm and not self.has_gtm_direct:
            gtm_names = _psd_signature_names(self.gtm_psd_func, self.n_gtm_eval)[
                np.asarray(self.gtm_varied_indxs)
            ]
            for psr in self.pulsar_names:
                names += [f"{psr}_gtm_{p}" for p in gtm_names]
        names += self._extra_param_names()
        return names

    def get_param_names_and_priors(self):
        """``{param_name: (lower_bound, upper_bound)}`` in ``xs`` order."""
        return dict(
                zip(
                self.get_param_names(), 
                zip(self.lower_prior_lim_all,
                    self.upper_prior_lim_all
                    )))
                    
    def jax_to_numpy_CPU(self, jax_CPU_array):
        """Zero-copy view of a CPU-resident JAX array as a numpy array."""
        return np.from_dlpack(jax_CPU_array)

    def _spit_neg_infinity(self):
        """-inf branch of ``get_lnprior`` (lax.cond needs a callable)."""
        return -jnp.inf

    def _spit_neg_number(self):
        """In-bounds branch of ``get_lnprior`` (lax.cond needs a callable).

        An arbitrary constant, NOT -log(prior volume): the prior is
        unnormalised, so this cancels in MCMC ratios but makes any evidence or
        cross-model comparison built on ``get_lnprior`` meaningless.
        """
        return -8.01


class CorrelatedPulsarRedNoise(_GTMModeMixin, _ExtraBlocksMixin):
    """Red-noise covariance (phi) for a PTA WITH a common, ORF-correlated process.

    Same per-pulsar components as ``PerPulsarRedNoise`` (``irn``, ``dm``,
    ``gtm``, all optional) plus a mandatory common process whose PSD is shared
    by every pulsar and whose cross-pulsar correlations are set by an overlap
    reduction function of the pulsar-pair angular separations.  phi is
    therefore a genuine ``(n_rows, Npulsars, Npulsars)`` matrix: diagonal
    everywhere except in the GWB bins, where the off-diagonals are
    ``orf(xi) * psd_common``.

    Parameter vector layout
    -----------------------
    The flat ``xs`` vector is ordered as::

        xs = [ irn_psd_params (num_IR_params),
               dm_psd_params  (num_DM_params),
               gtm_psd_params (num_GTM_params),
               gwb_psd_params (n_gwb_varied),
               orf_params     (n_orf_varied) ]    <- only if the ORF is free

    The three per-pulsar blocks are flat with the pulsar index varying
    *slowest*; the GWB and ORF blocks are shared across pulsars.  Block
    boundaries are ``irn_end_idx``, ``dm_end_idx``, ``gtm_end_idx``,
    ``gwb_psd_end_idx``; the total is checked against the prior bounds at
    construction.

    Row resolution
    --------------
    Rows are frequency BINS, except when the GTM block is mode-resolved --
    ``gtm_mode_resolved=True`` (the default) or a direct ``gtm_psd`` array --
    in which case ``get_phi_diag`` / ``get_phi_mat`` / ``get_phi_mat_full`` /
    ``get_phi_mat_CURN`` / ``get_phi_mat_inv`` all work at BASIS COLUMN
    resolution (``2 * n_total_bins`` rows): IRN/DM/GWB bin values duplicated
    across their two quadrature modes, one independent value per GTM column.
    ``get_phi_mat_from_diag`` and ``_get_phi_diag_bins`` are the exceptions --
    they are bin-resolution only.

    Helper dictionaries
    -------------------
    ``gwb_helper_dictionary``:
        * ``"gwb_psd_param_upper_lim"`` / ``"gwb_psd_param_lower_lim"``
        * ``"ordered_gwb_psd_model_params"``  (optional, ordering cross-check)
        * ``"fixed_gwb_psd_param_indices"`` / ``"fixed_gwb_psd_param_values"``
          (optional)
        * ``"ordered_orf_model_params"``      (optional; its PRESENCE is what
          marks the ORF as free)
        * ``"orf_param_upper_lim"`` / ``"orf_param_lower_lim"`` (required when
          the ORF is free)

    ``irn_helper_dictionary`` / ``dm_helper_dictionary`` /
    ``gtm_helper_dictionary`` use the same keys with the ``"psd_"`` prefix, as
    documented on ``PerPulsarRedNoise``.  Log-amplitude bounds are shifted by
    ``0.5*log10(renorm_const)``; dimensionless ones are not.

    Authors
    -------
    Nima Laal (original pandora classes, 02/12/2025)
    Unified refactor: Ge (06/2025)
    """
    def __init__(
        self,
        psr_pos,
        Npulsars,
        signal_indices,
        linear_timing_model_size,
        # ---- GWB ----
        gwb_psd_func,
        orf_func,
        gwb_helper_dictionary,
        crn_bins,
        f_common,
        # ---- non-GWB intrinsic red noise (optional) ----
        irn_psd_func=None,
        irn_helper_dictionary=None,
        irn_bins=None,
        f_irn=None,
        # ---- DM noise (optional) ----
        dm_psd_func=None,
        dm_helper_dictionary=None,
        dm_bins=None,
        f_dm=None,
        # ---- GTM noise (optional) ----
        gtm_psd_func=None,
        gtm_helper_dictionary=None,
        gtm_bins=None,
        f_gtm=None,
        gtm_psd=None,            # direct, mode-resolution GTM phi: (2*gtm_nfreqs, Npulsars)
        gtm_mode_resolved=True,  # one PSD value per basis COLUMN (see _GTMModeMixin)
        # ---- extra per-pulsar blocks: chrom / cquad / sw (optional) ----
        extra_components=None,
        # ---- shared ----
        renorm_const=1.0,
        pulsar_names=None
    ):
        """
        Parameters
        ----------
        psr_pos : array, shape (Npulsars, 3)
            Unit position vectors; used only for the pair separations ``xi``.
        Npulsars : int
        signal_indices : dict[str, slice]
            T-matrix column slices per signal (``'cor'`` required; ``'unc'`` /
            ``'dm'`` / ``'gtm'`` as present), from ``SuperSignal.build_basis``.
        linear_timing_model_size : int
            Leading timing-model columns, subtracted before converting column
            indices to frequency bins.
        gwb_psd_func : callable
            ``gwb_psd_func(f, df, *params) -> (crn_bins,)``.
        orf_func : callable
            ``orf_func(angle, *params) -> (n_pairs,)``.
        gwb_helper_dictionary : dict
            Bounds and fixed-parameter info for the GWB PSD and the ORF.
        crn_bins : int
            Number of common-process bins; must equal the width of
            ``signal_indices['cor']``.
        f_common : array, shape (crn_bins,)
        irn_psd_func, dm_psd_func, gtm_psd_func, and their
        ``*_helper_dictionary`` / ``*_bins`` / ``f_*`` companions : optional
            Per-pulsar components, exactly as in ``PerPulsarRedNoise``.
        gtm_psd : array, optional
            Precomputed GTM phi, ``(2*gtm_bins, Npulsars)``.
        gtm_mode_resolved : bool, default True
            One PSD value per GTM basis column.  See ``_GTMModeMixin``.
        renorm_const : float, default 1.0
        pulsar_names : list of str, optional

        Raises
        ------
        ValueError
            If ``crn_bins`` or any component's size disagrees with
            ``signal_indices``, or if the prior bounds do not cover exactly the
            sampled parameters.
        """
        self.Npulsars = Npulsars
        if pulsar_names is not None:
            if not (len(pulsar_names) == Npulsars):
                raise ValueError((
                    f"len(pulsar_names)={len(pulsar_names)} does not match "
                    f"Npulsars={Npulsars}."
                ))
        self.pulsar_names = pulsar_names
        
        self.signal_indices = signal_indices
        tm_offset = linear_timing_model_size

        def _fourier_slice(name):
            """Column slice of `name` in the T-matrix -> its frequency-bin slice."""
            sl = self.signal_indices[name]
            start_col = sl.start - tm_offset
            stop_col  = sl.stop  - tm_offset
            return slice(start_col // 2, stop_col // 2)

        self.GWB_slice     = _fourier_slice('cor')
        self.GWB_fidxs     = jnp.arange(self.GWB_slice.start, self.GWB_slice.stop)
        # `_eye` is built with crn_bins rows and is cho_solve'd against
        # phi[GWB_fidxs]; if the two disagree the failure is a batch-shape error
        # deep inside get_phi_mat_inv, so check it here.
        n_gwb_slice = self.GWB_slice.stop - self.GWB_slice.start
        if n_gwb_slice != crn_bins:
            raise ValueError(
                f"crn_bins={crn_bins} but signal_indices['cor'] spans "
                f"{n_gwb_slice} frequency bins.")
        self.crn_bins      = crn_bins

        # ------------------------------------------------------------------ #
        #  Basic bookkeeping                                                   #
        # ------------------------------------------------------------------ #
        self.gwb_helper_dictionary = gwb_helper_dictionary
        self.psr_pos = psr_pos
        self.gwb_psd_func = gwb_psd_func
        self.orf_func = orf_func
        self.irn_psd_func = irn_psd_func
        self.dm_psd_func = dm_psd_func
        self.gtm_psd_func = gtm_psd_func
        self.diag_idx = jnp.arange(Npulsars)
        self.ppair_number = int(Npulsars * (Npulsars - 1) * 0.5)
        self.renorm_const = renorm_const
        self.logrenorm_offset = 0.5 * jnp.log10(renorm_const)

        self.f_common = f_common[..., None]
        self.df_gwb = jnp.diff(jnp.concatenate((jnp.zeros((1)), 
                                                f_common), 
                                                axis = 0))[..., None]

        # ------------------------------------------------------------------ #
        #  IRN bookkeeping                                                     #
        # ------------------------------------------------------------------ #
        has_irn = irn_psd_func is not None
        self.has_irn = has_irn
        if has_irn:
            if not (irn_bins is not None and f_irn is not None):
                raise ValueError((
                    "irn_bins and f_irn must be supplied when irn_psd_func is given."
                ))
            if not (irn_helper_dictionary is not None):
                raise ValueError((
                    "irn_helper_dictionary must be supplied when irn_psd_func is given."
                ))
            self.irn_bins = irn_bins   # (the old `if has_irn else crn_bins` was dead: guarded by `if has_irn`)
            self.f_irn = f_irn if f_irn.ndim == 2 else jnp.broadcast_to(f_irn, (self.Npulsars, self.irn_bins)) 
            self.df_irn = jnp.diff(jnp.concatenate((jnp.zeros((self.Npulsars, 1)), self.f_irn), axis = 1))

        # ------------------------------------------------------------------ #
        #  DM bookkeeping                                                      #
        # ------------------------------------------------------------------ #
        has_dm = dm_psd_func is not None
        self.has_dm = has_dm
        if has_dm:
            if not (dm_bins is not None and f_dm is not None):
                raise ValueError((
                    "dm_bins and f_dm must be supplied when dm_psd_func is given."
                ))
            if not (dm_helper_dictionary is not None):
                raise ValueError((
                    "dm_helper_dictionary must be supplied when dm_psd_func is given."
                ))
            self.dm_bins = dm_bins
            self.f_dm = f_dm if f_dm.ndim == 2 else jnp.broadcast_to(f_dm, (self.Npulsars, self.dm_bins))
            self.df_dm = jnp.diff(jnp.concatenate((jnp.zeros((self.Npulsars, 1)), self.f_dm), axis = 1))

        # ------------------------------------------------------------------ #
        #  GTM bookkeeping                                                      #
        # ------------------------------------------------------------------ #
        # A GTM component may be specified either via a callable PSD model
        # (gtm_psd_func) or via a precomputed, mode-resolution phi array
        # (gtm_psd). The latter takes precedence if both are supplied.
        has_gtm = (gtm_psd_func is not None) or (gtm_psd is not None)
        self.has_gtm = has_gtm
        self.has_gtm_direct = gtm_psd is not None
        self.gtm_psd_input = gtm_psd
        # A directly-supplied gtm_psd is mode-resolution by definition.  False
        # when there is no GTM block at all, so every `if self.gtm_mode_resolved`
        # below is also a "there is a GTM block" test.
        self.gtm_mode_resolved = bool(
            has_gtm and (gtm_mode_resolved or self.has_gtm_direct))
        # Actual sizing happens in the frequency-index section below, once
        # GTM_slice is known (_init_gtm_geometry).

        # ------------------------------------------------------------------ #
        #  Frequency index arrays                                              #
        # ------------------------------------------------------------------ #
        if has_irn:
            self.IRN_slice = _fourier_slice('unc')
            _gwb_set = {int(i) for i in np.asarray(self.GWB_fidxs)}
            self.nonGWB_fidxs = jnp.array(
                [i for i in range(int(self.IRN_slice.start), int(self.IRN_slice.stop))
                 if i not in _gwb_set],
                dtype=int,   # stays integer when empty; jnp.array([]) is float
            )
            # `.any()` asks whether any INDEX IS NON-ZERO, so a single IRN-only
            # bin at index 0 disabled the separate inversion entirely: that bin's
            # phiinv row stayed zero and its logdet term went missing.
            self.separate_inversion_strat = bool(self.nonGWB_fidxs.size > 0)

        if has_dm:
            self.DM_slice  = _fourier_slice('dm')
            self.DM_fidxs  = jnp.arange(self.DM_slice.start, self.DM_slice.stop)

        if has_gtm:
            self.GTM_slice = _fourier_slice('gtm')
            self.GTM_fidxs = jnp.arange(self.GTM_slice.start, self.GTM_slice.stop)
            self._init_gtm_geometry(gtm_bins, f_gtm,
                                    gtm_psd_func, gtm_helper_dictionary)

        # n_total_bins is now just the stop of the last signal slice
        all_stops = [self.GWB_slice.stop]
        if has_irn: all_stops.append(self.IRN_slice.stop)
        if has_dm:  all_stops.append(self.DM_slice.stop)
        if has_gtm: all_stops.append(self.GTM_slice.stop)
        all_stops += [((signal_indices[c["name"]].stop - tm_offset) // 2)
                      for c in (extra_components or [])]
        self.n_total_bins = max(all_stops)

        # ------------------------------------------------------------------ #
        #  Pulsar-pair angular separations & multi-dim index arrays           #
        # ------------------------------------------------------------------ #
        self._eye = jnp.repeat(np.eye(Npulsars)[None], crn_bins, axis=0)

        I_tri, J_tri = np.tril_indices(Npulsars)
        a, b = [], []
        for i, j in zip(I_tri, J_tri):
            if i != j:
                a.append(i)
                b.append(j)
        a = np.array(a, dtype=int)
        b = np.array(b, dtype=int)

        self.xi = jnp.array(
            [np.arccos(np.dot(psr_pos[i], psr_pos[j])) for i, j in zip(a, b)]
        )

        self.I = jnp.repeat(a[None, :], crn_bins, axis=0)
        self.J = jnp.repeat(b[None, :], crn_bins, axis=0)
        self.KGW = jnp.repeat(self.GWB_fidxs[:, None], len(a), axis=1)

        if has_irn and self.separate_inversion_strat:
            self.DIR = jnp.repeat(self.diag_idx[None, :],
                                  len(self.nonGWB_fidxs), axis=0)
            self.KIR = jnp.repeat(self.nonGWB_fidxs[:, None], Npulsars, axis=1)

        if has_dm:
            self.DIRDM = jnp.repeat(self.diag_idx[None, :], dm_bins, axis=0)
            self.KDM = jnp.repeat(self.DM_fidxs[:, None], Npulsars, axis=1)

        if has_gtm:
            # self.gtm_bins, not the raw `gtm_bins` argument: on the direct
            # gtm_psd path the argument may be None or stale, and DIRGTM must
            # match KGTM (which is built from the GTM slice).
            self.DIRGTM = jnp.repeat(self.diag_idx[None, :], self.gtm_bins, axis=0)
            self.KGTM = jnp.repeat(self.GTM_fidxs[:, None], Npulsars, axis=1)

        # ---- mode-resolution companions, needed whenever phi is mode-resolved #
        if self.has_gtm and self.gtm_mode_resolved:
            self.GWB_fidxs_modes = _bin_idx_to_mode_idx(self.GWB_fidxs)
            self._eye_modes = jnp.repeat(self._eye, 2, axis=0)

            self.GTM_fidxs_modes = _bin_idx_to_mode_idx(self.GTM_fidxs)
            self.KGTM_modes = jnp.repeat(self.GTM_fidxs_modes[:, None], Npulsars, axis=1)
            self.DIRGTM_modes = jnp.repeat(self.diag_idx[None, :], len(self.GTM_fidxs_modes), axis=0)

            if has_irn:
                self.nonGWB_fidxs_modes = _bin_idx_to_mode_idx(self.nonGWB_fidxs)
                self.KIR_modes = jnp.repeat(self.nonGWB_fidxs_modes[:, None], Npulsars, axis=1)
                self.DIR_modes = jnp.repeat(self.diag_idx[None, :], len(self.nonGWB_fidxs_modes), axis=0)

            if has_dm:
                self.DM_fidxs_modes = _bin_idx_to_mode_idx(self.DM_fidxs)
                self.KDM_modes = jnp.repeat(self.DM_fidxs_modes[:, None], Npulsars, axis=1)
                self.DIRDM_modes = jnp.repeat(self.diag_idx[None, :], len(self.DM_fidxs_modes), axis=0)

        # ------------------------------------------------------------------ #
        #  Parse GWB PSD + ORF                                                #
        # ------------------------------------------------------------------ #
        _gwb_hd_adapted = {
            k.replace("gwb_psd_", "psd_").replace("gwb_", "psd_"): v
            for k, v in gwb_helper_dictionary.items()
            if k not in ("ordered_orf_model_params",)
        }
        if "ordered_gwb_psd_model_params" in gwb_helper_dictionary:
            _gwb_hd_adapted["ordered_psd_model_params"] = (
                gwb_helper_dictionary["ordered_gwb_psd_model_params"]
            )

        self.gwb_param_container, self.gwb_varied_indxs = _parse_psd_func(
            gwb_psd_func, _gwb_hd_adapted, crn_bins
        )

        self.orf_fixed, _ = _parse_orf_func(orf_func, gwb_helper_dictionary)
        if self.orf_fixed:
            self.orf_val = orf_func(self.xi)

        # ------------------------------------------------------------------ #
        #  Parse IRN PSD                                                       #
        # ------------------------------------------------------------------ #
        if has_irn:
            self.irn_param_container, self.irn_varied_indxs = _parse_psd_func(
                irn_psd_func, irn_helper_dictionary, irn_bins
            )
            self.n_irn_varied = int(len(self.irn_varied_indxs))
            self.num_IR_params = self.n_irn_varied * Npulsars
        else:
            self.num_IR_params = 0

        # ------------------------------------------------------------------ #
        #  Parse DM PSD                                                        #
        # ------------------------------------------------------------------ #
        if has_dm:
            self.dm_param_container, self.dm_varied_indxs = _parse_psd_func(
                dm_psd_func, dm_helper_dictionary, dm_bins
            )
            self.n_dm_varied = int(len(self.dm_varied_indxs))
            self.num_DM_params = self.n_dm_varied * Npulsars
        else:
            self.num_DM_params = 0

        # ------------------------------------------------------------------ #
        #  Parse GTM PSD                                                        #
        # ------------------------------------------------------------------ #
        if has_gtm and not self.has_gtm_direct:
            # n_gtm_eval, not gtm_bins: a mode-resolved free spectrum carries
            # one parameter per basis column.
            self.gtm_param_container, self.gtm_varied_indxs = _parse_psd_func(
                gtm_psd_func, gtm_helper_dictionary, self.n_gtm_eval
            )
            self.n_gtm_varied = int(len(self.gtm_varied_indxs))
            self.num_GTM_params = self.n_gtm_varied * Npulsars
        else:
            # No sampled GTM params either because there's no GTM component
            # at all, or because it was supplied directly via `gtm_psd`.
            self.num_GTM_params = 0

        # ------------------------------------------------------------------ #
        #  Parameter-vector slice indices                                      #
        # ------------------------------------------------------------------ #
        # Layout: [ IRN(0..num_IR_params) | DM(..+num_DM_params) |
        #           GTM(..+num_GTM_params) | GWB_PSD(..+n_gwb) | ORF(..+n_orf) ]
        self.irn_end_idx = self.num_IR_params
        self.dm_end_idx = self.irn_end_idx + self.num_DM_params
        self.gtm_end_idx = self.dm_end_idx + self.num_GTM_params
        if self.gtm_end_idx == self.dm_end_idx:
            self._do_not_vary_gtm = True
        else:
            self._do_not_vary_gtm = False
        self.gwb_psd_end_idx = self.gtm_end_idx + int(len(self.gwb_varied_indxs))

        # ------------------------------------------------------------------ #
        #  Prior bounds                                                        #
        # ------------------------------------------------------------------ #
        upper, lower = jnp.array([]), jnp.array([])

        if has_irn:
            irn_shift = _renorm_shift(
                _psd_signature_names(irn_psd_func, self.irn_bins)[
                    np.asarray(self.irn_varied_indxs)],
                self.logrenorm_offset)
            irn_upper = jnp.tile(
                jnp.asarray(irn_helper_dictionary["psd_param_upper_lim"]) + irn_shift,
                Npulsars
            )
            irn_lower = jnp.tile(
                jnp.asarray(irn_helper_dictionary["psd_param_lower_lim"]) + irn_shift,
                Npulsars
            )
            upper = jnp.concatenate([upper, irn_upper])
            lower = jnp.concatenate([lower, irn_lower])

        if has_dm:
            dm_shift = _renorm_shift(
                _psd_signature_names(dm_psd_func, self.dm_bins)[
                    np.asarray(self.dm_varied_indxs)],
                self.logrenorm_offset)
            dm_upper = jnp.tile(
                jnp.asarray(dm_helper_dictionary["psd_param_upper_lim"]) + dm_shift,
                Npulsars
            )
            dm_lower = jnp.tile(
                jnp.asarray(dm_helper_dictionary["psd_param_lower_lim"]) + dm_shift,
                Npulsars
            )
            upper = jnp.concatenate([upper, dm_upper])
            lower = jnp.concatenate([lower, dm_lower])

        if has_gtm and not self.has_gtm_direct:
            gtm_shift = _renorm_shift(
                _psd_signature_names(gtm_psd_func, self.n_gtm_eval)[
                    np.asarray(self.gtm_varied_indxs)],
                self.logrenorm_offset)
            gtm_upper = jnp.tile(
                jnp.asarray(gtm_helper_dictionary["psd_param_upper_lim"]) + gtm_shift,
                Npulsars
            )
            gtm_lower = jnp.tile(
                jnp.asarray(gtm_helper_dictionary["psd_param_lower_lim"]) + gtm_shift,
                Npulsars
            )
            upper = jnp.concatenate([upper, gtm_upper])
            lower = jnp.concatenate([lower, gtm_lower])

        # The GWB bounds previously got NO renorm shift at all while the IRN/DM
        # bounds got it on every parameter -- two different wrong conventions in
        # one constructor.  Both now use the amplitude-only shift.
        gwb_shift = _renorm_shift(
            _psd_signature_names(gwb_psd_func, crn_bins)[
                np.asarray(self.gwb_varied_indxs)],
            self.logrenorm_offset)
        upper = jnp.concatenate(
            [upper, jnp.asarray(gwb_helper_dictionary["gwb_psd_param_upper_lim"]) + gwb_shift]
        )
        lower = jnp.concatenate(
            [lower, jnp.asarray(gwb_helper_dictionary["gwb_psd_param_lower_lim"]) + gwb_shift]
        )

        if not self.orf_fixed:
            upper = jnp.concatenate(
                [upper, jnp.array(gwb_helper_dictionary["orf_param_upper_lim"])]
            )
            lower = jnp.concatenate(
                [lower, jnp.array(gwb_helper_dictionary["orf_param_lower_lim"])]
            )
            self.n_orf_varied = int(
                len(gwb_helper_dictionary["orf_param_upper_lim"]))
        else:
            self.n_orf_varied = 0
        self.orf_end_idx = self.gwb_psd_end_idx + self.n_orf_varied

        # Extra per-pulsar blocks go LAST, after the ORF parameters, so the
        # established IRN | DM | GTM | GWB | ORF offsets do not move.
        x_upper, x_lower = self._init_extra_blocks(
            extra_components, signal_indices, tm_offset, Npulsars,
            param_start=self.orf_end_idx)
        upper = jnp.concatenate([upper, x_upper])
        lower = jnp.concatenate([lower, x_lower])

        # See PerPulsarRedNoise for why this is checked: block sizes come from
        # the PSD signatures, bounds come from the helper dictionaries, and
        # nothing else ties the two together.
        n_expected = self.extra_end_idx
        if upper.shape[0] != n_expected:
            raise ValueError(
                f"Prior bounds have {upper.shape[0]} entries but the parameter "
                f"layout needs {n_expected} (IRN {self.num_IR_params} + DM "
                f"{self.num_DM_params} + GTM {self.num_GTM_params} + GWB "
                f"{int(len(self.gwb_varied_indxs))} + ORF {self.n_orf_varied} "
                f"+ extra {self.num_extra_params}).")

        self.upper_prior_lim_all = upper
        self.lower_prior_lim_all = lower

        if renorm_const != 1:
            warnings.warn(
                "You have chosen to change units.  Make sure your amplitude "
                "priors reflect that!"
            )

    # ---------------------------------------------------------------------- #
    #  Internal PSD helpers                                                   #
    # ---------------------------------------------------------------------- #

    @jit_method
    def _eval_gwb_psd(self, gwb_psd_params):
        """Evaluate the common-process PSD on ``self.f_common``.

        Returns
        -------
        jnp.ndarray, shape (crn_bins, 1)
            The trailing axis is kept so the result broadcasts against
            ``(crn_bins, Npulsars)`` phi blocks and ``(crn_bins, n_pairs)``
            ORF products.
        """
        filled = self.gwb_param_container.at[self.gwb_varied_indxs].set(
            gwb_psd_params
        )
        return self.gwb_psd_func(self.f_common, self.df_gwb, *filled)

    @jit_method
    def _eval_irn_psd_all(self, irn_params_flat):
        """Evaluate the IRN PSD for all pulsars. -> (irn_bins, Npulsars)"""
        per_psr = irn_params_flat.reshape(self.Npulsars, self.n_irn_varied)

        def _single(params, freqs_, df_):
            filled = self.irn_param_container.at[self.irn_varied_indxs].set(params)
            return self.irn_psd_func(freqs_, df_, *filled)

        return jax.vmap(_single)(per_psr, self.f_irn, self.df_irn).T

    @jit_method
    def _eval_dm_psd_all(self, dm_params_flat):
        """Evaluate the DM PSD for all pulsars. -> (dm_bins, Npulsars)"""
        per_psr = dm_params_flat.reshape(self.Npulsars, self.n_dm_varied)

        def _single(params, freqs_, df_):
            filled = self.dm_param_container.at[self.dm_varied_indxs].set(params)
            return self.dm_psd_func(freqs_, df_, *filled)

        return jax.vmap(_single)(per_psr, self.f_dm, self.df_dm).T

    @jit_method
    def _eval_gtm_psd_all(self, gtm_params_flat):
        """
        Evaluate the GTM PSD for *all* pulsars (function-based model only;
        not used when `gtm_psd` was supplied directly).

        Parameters
        ----------
        gtm_params_flat : jnp.ndarray, shape (num_GTM_params,)
            Flat array; reshaped to (Npulsars, n_gtm_varied) before vmapping.

        Returns
        -------
        jnp.ndarray, shape (n_gtm_eval, Npulsars)
            One row per basis column when the block is mode-resolved, else one
            row per frequency bin.  Prefer ``_gtm_mode_values``, which always
            returns column resolution.
        """
        per_psr = gtm_params_flat.reshape(self.Npulsars, self.n_gtm_varied)

        def _single(params, freqs_, df_):
            filled = self.gtm_param_container.at[self.gtm_varied_indxs].set(params)
            return self.gtm_psd_func(freqs_, df_, *filled)

        return jax.vmap(_single)(per_psr, self.f_gtm, self.df_gtm).T  # → (gtm_bins, Npulsars)

    # ---------------------------------------------------------------------- #
    #  Parameter unpacking                                                    #
    # ---------------------------------------------------------------------- #

    @jit_method
    def _unpack(self, xs):
        """Split ``xs`` into (irn, dm, gtm, gwb_psd, orf) blocks.

        Blocks belonging to absent components -- and ``orf`` when the ORF is
        fixed -- come back empty.
        """
        irn = xs[:self.irn_end_idx]
        dm  = xs[self.irn_end_idx:self.dm_end_idx]
        gtm = xs[self.dm_end_idx:self.gtm_end_idx]
        gwb = xs[self.gtm_end_idx:self.gwb_psd_end_idx]
        orf = xs[self.gwb_psd_end_idx:self.orf_end_idx]
        return irn, dm, gtm, gwb, orf

    # ---------------------------------------------------------------------- #
    #  Core phi builders                                                      #
    # ---------------------------------------------------------------------- #

    @jit_method
    def _get_phi_diag_bins(self, xs):
        """Bin-resolution phi diagonal, always ``(n_total_bins, Npulsars)``.

        Internal: the GTM block contributes here only when it is bin-resolved.
        A mode-resolved GTM block (the default, and always the case for a
        direct ``gtm_psd``) is written in afterwards at column resolution by
        the public builders.

        Returns
        -------
        phi_diag : jnp.ndarray, (n_total_bins, Npulsars)
        psd_common : jnp.ndarray, (crn_bins, 1)
        """
        irn_flat, dm_flat, gtm_flat, gwb_params, _ = self._unpack(xs)
        psd_common = self._eval_gwb_psd(gwb_params)

        phi_diag = jnp.zeros((self.n_total_bins, self.Npulsars))

        if self.has_irn:
            irn_psd = self._eval_irn_psd_all(irn_flat)          # (irn_bins, Npulsars)
            phi_diag = phi_diag.at[self.IRN_slice].add(irn_psd)

        if self.has_dm:
            dm_psd = self._eval_dm_psd_all(dm_flat)             # (dm_bins, Npulsars)
            phi_diag = phi_diag.at[self.DM_slice].add(dm_psd)

        if self.has_gtm and not self.gtm_mode_resolved:
            if not self._do_not_vary_gtm:
                gtm_psd = self._eval_gtm_psd_all(gtm_flat)          # (gtm_bins, Npulsars)
            else:
                gtm_psd = 1.
            phi_diag = phi_diag.at[self.GTM_slice].add(gtm_psd)

        phi_diag = self._add_extra_bins(phi_diag, xs)
        phi_diag = phi_diag.at[self.GWB_slice].add(psd_common)
        return phi_diag, psd_common

    @jit_method
    def get_phi_diag(self, xs):
        """
        Build the phi diagonal, ignoring cross-pulsar correlations.

        Parameters
        ----------
        xs : jnp.ndarray
            Flat parameter vector; see the class docstring for its layout.

        Returns
        -------
        phi_diag : jnp.ndarray
            ``(n_total_bins, Npulsars)`` when the GTM block is bin-resolved or
            absent; ``(2 * n_total_bins, Npulsars)`` when it is mode-resolved
            (non-GTM bins duplicated across their quadrature modes, one
            independent value per GTM column).
        psd_common : jnp.ndarray, (crn_bins, 1)
            The common-process PSD, at bin resolution either way.
        """
        phi_diag, psd_common = self._get_phi_diag_bins(xs)

        if self.gtm_mode_resolved and self.has_gtm:
            *_, gtm_flat, _, _ = self._unpack(xs)
            phi_diag = jnp.repeat(phi_diag, 2, axis=0)
            phi_diag = phi_diag.at[self.GTM_slice_modes].set(
                self._gtm_mode_values(gtm_flat))

        return phi_diag, psd_common

    @jit_method
    def get_phi_mat(self, xs):
        """Full phi with the GWB cross terms in the LOWER triangle only.

        Parameters
        ----------
        xs : jnp.ndarray            flat parameter vector

        Returns
        -------
        phi : jnp.ndarray
            ``(n_rows, Npulsars, Npulsars)`` -- ``n_rows`` is ``n_total_bins``,
            or ``2 * n_total_bins`` when the GTM block is mode-resolved.  Only
            the diagonal and the lower-triangular GWB entries are filled; use
            ``get_phi_mat_full`` builds an explicitly symmetric matrix.
        """
        phi_diag, psd_common = self._get_phi_diag_bins(xs)
        n_total = phi_diag.shape[0]

        phi = jnp.zeros((n_total, self.Npulsars, self.Npulsars))
        phi = phi.at[:, self.diag_idx, self.diag_idx].set(phi_diag)

        *_, orf_params = self._unpack(xs)
        orf_val = self.orf_val if self.orf_fixed else self.orf_func(self.xi, *orf_params)
        phi = phi.at[self.KGW, self.I, self.J].set(orf_val * psd_common)

        if self.gtm_mode_resolved and self.has_gtm:
            *_, gtm_flat, _, _ = self._unpack(xs)
            phi = jnp.repeat(phi, 2, axis=0)
            phi = phi.at[self.KGTM_modes, self.DIRGTM_modes, self.DIRGTM_modes].set(
                self._gtm_mode_values(gtm_flat)
            )

        return phi

    @jit_method
    def get_phi_mat_full(self, xs):
        """As ``get_phi_mat``, but with both triangles filled (symmetric).

        This is the form the reparameterised likelihoods consume.
        """
        phi_diag, psd_common = self._get_phi_diag_bins(xs)
        n_total = phi_diag.shape[0]

        phi = jnp.zeros((n_total, self.Npulsars, self.Npulsars))
        phi = phi.at[:, self.diag_idx, self.diag_idx].set(phi_diag)

        *_, orf_params = self._unpack(xs)
        orf_val = self.orf_val if self.orf_fixed else self.orf_func(self.xi, *orf_params)
        phi = phi.at[self.KGW, self.I, self.J].set(orf_val * psd_common)
        phi = phi.at[self.KGW, self.J, self.I].set(orf_val * psd_common)

        if self.gtm_mode_resolved and self.has_gtm:
            *_, gtm_flat, _, _ = self._unpack(xs)
            phi = jnp.repeat(phi, 2, axis=0)
            phi = phi.at[self.KGTM_modes, self.DIRGTM_modes, self.DIRGTM_modes].set(
                self._gtm_mode_values(gtm_flat)
            )

        return phi

    @jit_method
    def get_phi_mat_CURN(self, xs):
        """CURN view: ``(phi_diag, psd_common)`` with the ORF ignored.

        Common Uncorrelated Red Noise -- the common process keeps its shared
        PSD but contributes no cross-pulsar terms.  Identical to
        ``get_phi_diag``; named separately for call-site clarity.
        """
        return self.get_phi_diag(xs)

    @jit_method
    def get_phi_mat_from_diag(self, phi_diag, psd_common, orf_params=None):
        """Rebuild phi from an already-computed diagonal plus the ORF terms.

        Parameters
        ----------
        phi_diag : jnp.ndarray, (n_total_bins, Npulsars)
            BIN resolution only, as produced by ``_get_phi_diag_bins``.  A
            mode-resolved model must use ``get_phi_mat`` instead -- this method
            has no GTM handling at all.
        psd_common : jnp.ndarray, (crn_bins, 1)
        orf_params : array, optional
            Required when the ORF has free parameters.

        Returns
        -------
        phi : jnp.ndarray, (n_total_bins, Npulsars, Npulsars)
            Lower triangle only, as in ``get_phi_mat``.
        """
        n_total = phi_diag.shape[0]
        phi = jnp.zeros((n_total, self.Npulsars, self.Npulsars))
        phi = phi.at[:, self.diag_idx, self.diag_idx].set(phi_diag)

        if self.orf_fixed:
            orf_val = self.orf_val
        else:
            if not (orf_params is not None):
                raise ValueError((
                    "orf_params must be provided when the ORF has free parameters."
                ))
            orf_val = self.orf_func(self.xi, *orf_params)

        return phi.at[self.KGW, self.I, self.J].set(orf_val * psd_common)

    # ---------------------------------------------------------------------- #
    #  Inversion                                                              #
    # ---------------------------------------------------------------------- #

    @jit_method
    def get_phi_mat_inv(self, phi):
        """
        Invert phi with a mixed Cholesky + diagonal strategy.

        Only the GWB rows are dense (the ORF couples pulsars there), so those
        are Cholesky-inverted as a batch while the IRN-only, DM and GTM rows --
        which are diagonal in the pulsar index -- are inverted elementwise.

        Parameters
        ----------
        phi : jnp.ndarray
            Output of ``get_phi_mat`` / ``get_phi_mat_full``:
            ``(n_total_bins, Npulsars, Npulsars)`` when bin-resolved,
            ``(2*n_total_bins, Npulsars, Npulsars)`` when the GTM block is
            mode-resolved.

        Returns
        -------
        phiinv : jnp.ndarray, (2*n_total_bins, Npulsars, Npulsars)
            Bin-resolved input is repeated across the two quadrature modes;
            mode-resolved input already has one row per basis column and is
            returned as-is.
        logdet_phi : float
            log|phi| summed over all modes (hence the doubling in the
            bin-resolved branch).
        """
        if self.has_gtm and self.gtm_mode_resolved:
            phiinv = jnp.zeros_like(phi)

            # --- GWB modes: Cholesky, done directly at mode resolution ---
            cp = jsp.linalg.cho_factor(phi[self.GWB_fidxs_modes], lower=True)
            phiinv = phiinv.at[self.GWB_fidxs_modes].set(
                jsp.linalg.cho_solve(cp, self._eye_modes)
            )
            logdet_phi = 2.0 * jnp.sum(jnp.log(cp[0].diagonal(axis1=-2, axis2=-1)))

            # --- IRN-only modes: diagonal inversion ---
            if self.has_irn and self.separate_inversion_strat:
                diags_irn = phi[self.nonGWB_fidxs_modes].diagonal(axis1=-2, axis2=-1)
                phiinv = phiinv.at[self.KIR_modes, self.DIR_modes, self.DIR_modes].set(
                    1.0 / diags_irn
                )
                logdet_phi = logdet_phi + jnp.sum(jnp.log(diags_irn))

            # --- DM modes: diagonal inversion ---
            if self.has_dm:
                diags_dm = phi[self.DM_fidxs_modes].diagonal(axis1=-2, axis2=-1)
                phiinv = phiinv.at[self.KDM_modes, self.DIRDM_modes, self.DIRDM_modes].set(
                    1.0 / diags_dm
                )
                logdet_phi = logdet_phi + jnp.sum(jnp.log(diags_dm))

            # --- GTM modes: diagonal inversion, already at true mode resolution ---
            diags_gtm = phi[self.GTM_fidxs_modes].diagonal(axis1=-2, axis2=-1)
            phiinv = phiinv.at[self.KGTM_modes, self.DIRGTM_modes, self.DIRGTM_modes].set(
                1.0 / diags_gtm
            )
            logdet_phi = logdet_phi + jnp.sum(jnp.log(diags_gtm))

            # --- extra blocks (chrom / cquad / sw): diagonal inversion ---
            for blk in self.extra_blocks:
                diags_x = phi[blk.fidxs_modes].diagonal(axis1=-2, axis2=-1)
                phiinv = phiinv.at[blk.K_modes, blk.DIR_modes, blk.DIR_modes].set(
                    1.0 / diags_x)
                logdet_phi = logdet_phi + jnp.sum(jnp.log(diags_x))

            # No outer doubling here: `phi` already has one row per true
            # Fourier mode, so summing logdet contributions over all of its
            # rows already accounts for both quadrature components.
            return phiinv, logdet_phi

        phiinv = jnp.zeros_like(phi)

        # --- GWB bins: Cholesky ---
        cp = jsp.linalg.cho_factor(phi[self.GWB_fidxs], lower=True)
        phiinv = phiinv.at[self.GWB_fidxs].set(
            jsp.linalg.cho_solve(cp, self._eye)
        )
        logdet_phi = 2.0 * jnp.sum(jnp.log(cp[0].diagonal(axis1=-2, axis2=-1)))

        # --- IRN-only bins: diagonal inversion ---
        if self.has_irn and self.separate_inversion_strat:
            diags_irn = phi[self.nonGWB_fidxs].diagonal(axis1=-2, axis2=-1)
            phiinv = phiinv.at[self.KIR, self.DIR, self.DIR].set(1.0 / diags_irn)
            logdet_phi = logdet_phi + jnp.sum(jnp.log(diags_irn))

        # --- DM bins: diagonal inversion ---
        if self.has_dm:
            diags_dm = phi[self.DM_fidxs].diagonal(axis1=-2, axis2=-1)
            phiinv = phiinv.at[self.KDM, self.DIRDM, self.DIRDM].set(1.0 / diags_dm)
            logdet_phi = logdet_phi + jnp.sum(jnp.log(diags_dm))

        # --- GTM bins: diagonal inversion ---
        if self.has_gtm:
            diags_gtm = phi[self.GTM_fidxs].diagonal(axis1=-2, axis2=-1)
            phiinv = phiinv.at[self.KGTM, self.DIRGTM, self.DIRGTM].set(1.0 / diags_gtm)
            logdet_phi = logdet_phi + jnp.sum(jnp.log(diags_gtm))

        # --- extra blocks (chrom / cquad / sw): diagonal inversion ---
        for blk in self.extra_blocks:
            diags_x = phi[blk.fidxs].diagonal(axis1=-2, axis2=-1)
            phiinv = phiinv.at[blk.K, blk.DIR, blk.DIR].set(1.0 / diags_x)
            logdet_phi = logdet_phi + jnp.sum(jnp.log(diags_x))

        return jnp.repeat(phiinv, 2, axis=0), 2.0 * logdet_phi

    @jit_method
    def partial_reparm_helper(self, xs, pad_mask):
        """Split phi into a dense GWB block and a diagonal everything-else block.

        Feeds ``SuperSignal.partial_marg_lnposterior``, which keeps the GWB
        coefficients and marginalises the rest analytically, so the two halves
        are returned separately and both at MODE resolution.

        Parameters
        ----------
        xs : jnp.ndarray
            Flat parameter vector; see the class docstring.
        pad_mask : jnp.ndarray, (Npulsars, linear_timing_model_size)
            1 where a linear-timing column is padding, 0 where it is real.  The
            padded slots get unit prior precision (so HMC sees curvature) and
            the real ones ~0 (an effectively flat timing prior).

        Returns
        -------
        phiinv_gwb : jnp.ndarray, (2*crn_bins, Npulsars, Npulsars)
            Inverse of the dense GWB block, repeated across quadrature modes.
        logdet_phi_gwb : float
        phiinv_non_gwb : jnp.ndarray, (Npulsars, ltm_size + n_non_gwb_modes)
            Diagonal precision of [timing | IRN | DM | GTM], transposed.
        logdet_phi_non_gwb : float
            Excludes the (constant) timing-model block.

        Notes
        -----
        The non-GWB block is laid out IRN | DM | GTM and assumes those basis
        columns are contiguous and in that order; ``_check_non_gwb_block_order``
        enforces it at trace time.
        """
        irn_flat, dm_flat, gtm_flat, gwb_params, orf_params = self._unpack(xs)
        psd_common = self._eval_gwb_psd(gwb_params)
        orf_val = self.orf_val if self.orf_fixed else self.orf_func(self.xi, *orf_params)

        # Build the non-GWB (IRN + DM + GTM) diagonal directly at MODE
        # resolution: IRN/DM (and function-based GTM) are computed at bin
        # resolution and duplicated across their two quadrature modes,
        # while a direct `gtm_psd` is inserted as-is (it is already at mode
        # resolution and may differ between the two modes of a bin).
        n_irn_modes = 2 * (self.IRN_slice.stop - self.IRN_slice.start) if self.has_irn else 0
        n_dm_modes  = 2 * (self.DM_slice.stop  - self.DM_slice.start)  if self.has_dm  else 0
        n_gtm_modes = 2 * (self.GTM_slice.stop - self.GTM_slice.start) if self.has_gtm else 0
        n_extra_modes = sum(2 * b.bins for b in self.extra_blocks)
        n_non_gwb_modes = n_irn_modes + n_dm_modes + n_gtm_modes + n_extra_modes

        phi_diag_non_gwb = jnp.zeros((n_non_gwb_modes, self.Npulsars))

        # These local slices lay the non-GWB block out as IRN | DM | GTM.  That
        # is only the right ordering if the T-matrix columns are in that order
        # and contiguous -- the consumer slices TNT by column index, so a model
        # string that puts, say, 'dm' before 'unc' would pair this phi with the
        # wrong TNT rows.  Checked once, at trace time, where it is cheap.
        self._check_non_gwb_block_order()

        irn_local = slice(0, n_irn_modes) if self.has_irn else None
        dm_local  = slice(irn_local.stop if irn_local else 0,
                        (irn_local.stop if irn_local else 0) + n_dm_modes) if self.has_dm else None
        _dm_or_irn_stop = dm_local.stop if dm_local else (irn_local.stop if irn_local else 0)
        gtm_local = slice(_dm_or_irn_stop, _dm_or_irn_stop + n_gtm_modes) if self.has_gtm else None
        _x0 = _dm_or_irn_stop + (n_gtm_modes if self.has_gtm else 0)

        if self.has_irn:
            phi_diag_non_gwb = phi_diag_non_gwb.at[irn_local].set(
                jnp.repeat(self._eval_irn_psd_all(irn_flat), 2, axis=0)
            )
        if self.has_dm:
            phi_diag_non_gwb = phi_diag_non_gwb.at[dm_local].set(
                jnp.repeat(self._eval_dm_psd_all(dm_flat), 2, axis=0)
            )
        if self.has_gtm:
            # One value per basis column when mode-resolved; the mixin handles
            # the bin-resolved fallback (and the fixed gtm_psd case).
            phi_diag_non_gwb = phi_diag_non_gwb.at[gtm_local].set(
                self._gtm_mode_values(gtm_flat))
        for blk in self.extra_blocks:
            phi_diag_non_gwb = phi_diag_non_gwb.at[_x0:_x0 + 2 * blk.bins].set(
                jnp.repeat(self._eval_extra_psd(blk, xs), 2, axis=0))
            _x0 += 2 * blk.bins

        phi_gwb = jnp.zeros((self.crn_bins, self.Npulsars, self.Npulsars))
        phi_gwb = phi_gwb.at[:, self.diag_idx, self.diag_idx].set(psd_common)
        phi_gwb = phi_gwb.at[self.KGW, self.I, self.J].set(orf_val * psd_common)
        phi_gwb = phi_gwb.at[self.KGW, self.J, self.I].set(orf_val * psd_common)

        # per-pulsar phi (already at mode resolution)
        phiinv_non_gwb = 1 / phi_diag_non_gwb
        logdet_phi_non_gwb = jnp.sum(jnp.log(phi_diag_non_gwb))
        ltm_size = pad_mask.shape[-1]
        concat_phiinv_non_gwb = jnp.zeros((phi_diag_non_gwb.shape[0] + ltm_size, self.Npulsars))
        concat_phiinv_non_gwb = concat_phiinv_non_gwb.at[ltm_size:].set(phiinv_non_gwb)
        concat_phiinv_non_gwb = concat_phiinv_non_gwb.at[:ltm_size].set(pad_mask.mT)
        concat_phiinv_non_gwb += 1e-40

        # gwb phi
        cp = jsp.linalg.cho_factor(phi_gwb, lower=True)
        phiinv_gwb = jsp.linalg.cho_solve(cp, self._eye)
        logdet_phi_gwb = 4.0 * jnp.sum(jnp.log(cp[0].diagonal(axis1=-2, axis2=-1)))     # x4 b/c Cholesky + nfreq size
        
        result = (jnp.repeat(phiinv_gwb, repeats=2, axis=0),
                  logdet_phi_gwb,
                  concat_phiinv_non_gwb.mT,
                  logdet_phi_non_gwb)
        
        return result

    def _check_non_gwb_block_order(self):
        """Raise if the non-GWB basis blocks are not contiguous and ordered IRN, DM, GTM.

        Raises
        ------
        ValueError
            If they are not: ``partial_reparm_helper`` builds its phi block in
            that order, while the consumer slices TNT by column index, so any
            other ordering silently pairs phi with the wrong rows.
        """
        present = [(name, sl) for name, sl in
                   (('unc', getattr(self, 'IRN_slice', None)),
                    ('dm',  getattr(self, 'DM_slice',  None)),
                    ('gtm', getattr(self, 'GTM_slice', None)),
                    *[(b.name, b.bin_slice) for b in self.extra_blocks])
                   if sl is not None]
        for (n0, s0), (n1, s1) in zip(present, present[1:]):
            if s0.stop != s1.start:
                raise ValueError(
                    "partial_reparm_helper lays the non-GWB phi block out as "
                    "IRN | DM | GTM | extra blocks (in model-string order), so "
                    "those basis columns must be contiguous "
                    f"and in that order; got {n0}={s0} then {n1}={s1}.  Reorder "
                    "the signal-combination string.")

    # ---------------------------------------------------------------------- #
    #  Prior                                                                  #
    # ---------------------------------------------------------------------- #

    @jit_method
    def get_lnprior(self, xs):
        """UNNORMALISED uniform log-prior over ``xs``.

        Returns an arbitrary constant (see ``_spit_neg_number``) when every
        parameter is strictly inside its bounds and -inf otherwise; the
        constant is not -log(prior volume), so this is usable for MCMC
        acceptance ratios but not for evidence.
        """
        in_bounds = jnp.logical_and(
            xs > self.lower_prior_lim_all, xs < self.upper_prior_lim_all
        ).all()
        return jax.lax.cond(in_bounds, self._spit_neg_number, self._spit_neg_infinity)

    def get_lnprior_numpy(self, xs):
        """``get_lnprior`` as a host-side numpy scalar."""
        return self.get_lnprior(xs).__array__()

    @jit_method
    def make_initial_guess(self, key):
        """Draw one uniform sample from inside the prior bounds. [n_params]"""
        return jr.uniform(
            key,
            shape=(self.upper_prior_lim_all.shape[0],),
            minval=self.lower_prior_lim_all,
            maxval=self.upper_prior_lim_all,
        )

    # ---------------------------------------------------------------------- #
    #  Utilities                                                              #
    # ---------------------------------------------------------------------- #

    def get_param_names(self):
        """
        Return the flat list of parameter names in the same order as the
        `xs` vector consumed by `get_phi_diag` / `get_phi_mat`.

        Order: [irn_psd_params (pulsar-major, param-minor),
                dm_psd_params  (pulsar-major, param-minor),
                gtm_psd_params (pulsar-major, param-minor),
                gwb_psd_params,
                orf_params (only if the ORF has free parameters)]

        Names are ``f"{pulsar}_{component}_{param}"`` for the per-pulsar
        blocks, ``f"gwb_{param}"`` and ``f"orf_{param}"`` for the shared ones.
        A directly-supplied ``gtm_psd`` contributes no names (it has no sampled
        parameters), and a mode-resolved GTM free spectrum contributes one name
        per basis column.

        Returns
        -------
        list of str, the same length as ``lower/upper_prior_lim_all``.

        Raises
        ------
        ValueError
            If ``pulsar_names`` was not supplied at construction.
        """
        if not (self.pulsar_names is not None):
            raise ValueError((
                "pulsar_names must be supplied at construction to get param names."
            ))

        names = []

        if self.has_irn:
            irn_names = _psd_signature_names(self.irn_psd_func, self.irn_bins)[
                np.asarray(self.irn_varied_indxs)
            ]
            for psr in self.pulsar_names:
                names += [f"{psr}_irn_{p}" for p in irn_names]

        if self.has_dm:
            dm_names = _psd_signature_names(self.dm_psd_func, self.dm_bins)[
                np.asarray(self.dm_varied_indxs)
            ]
            for psr in self.pulsar_names:
                names += [f"{psr}_dm_{p}" for p in dm_names]

        if self.has_gtm and not self.has_gtm_direct:
            gtm_names = _psd_signature_names(self.gtm_psd_func, self.n_gtm_eval)[
                np.asarray(self.gtm_varied_indxs)
            ]
            for psr in self.pulsar_names:
                names += [f"{psr}_gtm_{p}" for p in gtm_names]

        gwb_names = _psd_signature_names(self.gwb_psd_func, self.crn_bins)[
            np.asarray(self.gwb_varied_indxs)
        ]
        names += [f"gwb_{p}" for p in gwb_names]

        if not self.orf_fixed:
            orf_names = _orf_signature_names(self.orf_func)
            names += [f"orf_{p}" for p in orf_names]
        names += self._extra_param_names()
        return names

    def get_param_names_and_priors(self):
        """``{param_name: (lower_bound, upper_bound)}`` in ``xs`` order."""
        return dict(
                zip(
                self.get_param_names(), 
                zip(self.lower_prior_lim_all,
                    self.upper_prior_lim_all
                    )))

    def jax_to_numpy_CPU(self, jax_CPU_array):
        """Zero-copy view of a CPU-resident JAX array as a numpy array."""
        return np.from_dlpack(jax_CPU_array)

    def _spit_neg_infinity(self):
        """-inf branch of ``get_lnprior`` (lax.cond needs a callable)."""
        return -jnp.inf

    def _spit_neg_number(self):
        """In-bounds branch of ``get_lnprior`` (lax.cond needs a callable).

        An arbitrary constant, NOT -log(prior volume): the prior is
        unnormalised, so this cancels in MCMC ratios but makes any evidence
        or cross-model comparison built on ``get_lnprior`` meaningless.
        """
        return -8.01