
from ATLAS.utils import get_pulsar_timespan
from ATLAS.utils import psr_has_multi_toa_epoch
from ATLAS.utils import jit, jit_method
from ATLAS.signals.signals_utils import _timing_model_svd

import jax.numpy as jnp


class PTA_Data:
    """A class to hold static data attributes of the PTA dataset.

    This class is intended to hold all the static data attributes of the PTA
    dataset, such as the pulsar objects, their TOAs, positions, and timespans.
    See the following attributes for details.

    Attributes
    ----------
    psrs : list
        A list of enterprise-like pulsar objects.
    npsrs : int
        The number of pulsars in the dataset.
    psr_names : list
        A list of pulsar names corresponding to the pulsar objects.
    fixed_wn : bool
        A flag indicating whether the white noise matrices are fixed.
    toas : list of arrays
        A list where each element is an array of TOAs for a pulsar. [npsrs, ntoas]
    psr_pos : array
        An array containing the positions of the pulsars in unit-Cartesian 
        coordinates. [npsrs, 3]
    pta_tspan : float
        The total timespan covered by the PTA dataset, calculated as the
        difference between the maximum and minimum TOAs across all pulsars.
    psr_tspans : array
        An array containing the individual timespans for each pulsar, calculated
        as the difference between the maximum and minimum TOAs for each pulsar. [npsrs]
    include_ecorr : bool
        Whether ECORR is identifiable for this dataset, detected from the TOA
        epochs. See :meth:`_detect_ecorr`.
    psr_dists_dict : dict
        Dictionary with pulsar names as keys, and tuple values specifying
        (pulsar distance (kpc), uncertainty (kpc), measurement method). The
        measurement method can be 'DM' or 'PX'. If dictionary does not contain
        a particular pulsar, the pulsar objects will be used to define distances.
    """

    def __init__(self, 
                psrs,
                adaptus_basis = None,
                num_gwb_bins = None,
                num_irn_bins = None,
                num_dm_bins = None,
                num_det_bins = None,
                adaptus_size = None, 
                fixed_white_noise_params = None,
                linear_timing = False,
                marg_timing = False,
                diag_white_cov = False,
                fixed_res = False,
                timfiles = None,
                parfiles = None,
                noise_dict = None,
                psr_dists_dict = {},
                dm_ref_freq = 1400): 
        """The constructor for the PTA_Data class

        This class is intended to hold all the static data attributes of the PTA 
        dataset, such as the pulsar objects, their TOAs, positions, and timespans.

        Parameters
        ----------
        psrs : list
            A list of enterprise-like pulsar objects.
        fixed_wn : bool
            Whether the white noise matrices are fixed, by default False.
        """
        self.adaptus_basis = adaptus_basis
        self.num_gwb_bins = num_gwb_bins
        self.num_det_bins = num_det_bins
        self.num_irn_bins = num_irn_bins
        self.num_dm_bins = num_dm_bins
        self.adaptus_size = adaptus_size

        self.noise_dict = noise_dict
        self.parfiles = parfiles
        self.timfiles = timfiles
        self.psrs = psrs # List of pulsar objects
        self.npsrs = len(psrs) # Number of pulsars
        self.npairs = self.npsrs * (self.npsrs - 1) // 2 # Number of unique pulsar pairs
        self.psr_names = [p.name for p in psrs] # List of pulsar names

        # (jagged) List of each pulsar's TOAs (npsrs, ntoas)
        self.toas = [jnp.array(p.toas) for p in psrs]
        # Array of pulsar positions in unit-Cartesian coordinates (npsrs, 3)
        self.psr_pos = jnp.array([p.pos for p in psrs])

        # Total timespan for the whole PTA (float) (maxtoa - mintoa)
        self.pta_tspan = get_pulsar_timespan(psrs)
        # Array of each pulsar's individual timespans (npsrs) 
        self.psr_tspans = jnp.array([get_pulsar_timespan(p) for p in psrs])

        # Raw residuals - List of each pulsar's residuals (npsrs, npsr_toas)
        self.raw_residuals = [jnp.array(p.residuals) for p in psrs]

        # Linear Timing Design Matrix
        self.Mmat = [_timing_model_svd(psr.Mmat) for psr in psrs]

        # pulsar distances
        self.psr_dists_dict = psr_dists_dict
        self.psr_dists_mean = jnp.array([self.psr_dists_dict[psr.name][0] if psr.name in list(self.psr_dists_dict.keys())
                                         else psr.pdist[0] for psr in self.psrs])
        self.psr_dists_std = jnp.array([self.psr_dists_dict[psr.name][1] if psr.name in list(self.psr_dists_dict.keys())
                                        else psr.pdist[1] for psr in self.psrs])
        
        ######################PTA Data Analysis General Settings######################
        # Whether the white noise matrices are fixed (bool)
        self.fixed_wn = fixed_white_noise_params 
        self.fixed_white_noise_params = fixed_white_noise_params
        # No residual subtraction?
        self.fixed_res = fixed_res
        # Linear (M \epsilon) appraoch to modeling the timing model errors
        self.linear_timing = linear_timing
        self.diag_white_cov = diag_white_cov
        self.marg = marg_timing
        # Consulted by WhiteCov when its own include_ecorr is left at None.
        self.include_ecorr = self._detect_ecorr()

        # Radio frequencies
        radio_freqs = jnp.concat([psr.freqs for psr in psrs])
        self.dm_ref_freq = dm_ref_freq
        self.ref_over_radio_freqs = self.dm_ref_freq / radio_freqs

        # A celever way to broadcast DM index over all concatenated toas
        ct = 0
        self.dm_exploder_idxs = []
        for pidx in range(self.npsrs):
            self.dm_exploder_idxs.append(ct * jnp.ones(len(self.toas[pidx])))
            ct+=1
        self.dm_exploder_idxs = jnp.concat(self.dm_exploder_idxs).astype(int)

    @property
    def dm_bins(self):
        """Deprecated alias for :attr:`num_dm_bins`.

        The attribute used to be spelled ``dm_bins`` here while
        ``make_red_noise`` read ``num_dm_bins``, so any model string containing
        ``dm`` raised ``AttributeError`` and DM noise was unreachable through
        the public API. Kept as an alias so external scripts that read the old
        name still work.
        """
        return self.num_dm_bins

    def _detect_ecorr(self):
        """Is ECORR identifiable anywhere in this PTA?

        False only when no pulsar has an epoch holding more than one TOA, which
        makes ECORR exactly degenerate with EQUAD. NOTE: a mixed dataset keeps
        this True, and its epoch-less pulsars still raise in SinglePulsarWhiteCov
        -- ECORR is one flag for the whole PTA, not per pulsar.
        """
        return any(psr_has_multi_toa_epoch(psr) for psr in self.psrs)

    def add_white_noise_cov(self, white_noise_cov):
        self.Nmat = white_noise_cov

    def add_timing_design_matrix(self, Mmat):
        self.Mmat = Mmat
        