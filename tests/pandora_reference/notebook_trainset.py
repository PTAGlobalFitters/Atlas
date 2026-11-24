"""Pandora's training-set code, copied from its notebooks for identity tests.

Pandora has no importable training-set code; it lives in notebook cells. The
functions below are those cells (pandora commit ebd614f) with the minimum
changes needed to call them from a test:

* ``examples/AstroInferenceUpdated.ipynb`` cells 9-11, 16, 17, 20, 23, 25,
  28 and 31: module-level settings became arguments, the LHS takes a ``seed``,
  ``natsorted`` (not installed) became a numeric sort on the draw index,
  which gives the same order for ``{rr}_{yr}yrs.npy`` files, and cell 28's
  sanity check ``par_data[:, :, 100] == par_data[:, :, 89]`` is dropped
  because the tests use fewer than 101 realizations.
* ``examples/NormalizingFlowTrainDEMO.ipynb`` cells 5, 10, 12, 14 and 16:
  the library path, the number of bins and ``Tspan`` became arguments.

Nothing else is changed, so any difference from ATLAS.experimental.astro is a
difference from Pandora.
"""
import glob
import os

import numpy as np
from scipy.stats import qmc
from scipy.stats.distributions import norm, uniform


# ---- AstroInferenceUpdated.ipynb, cells 9-11 --------------------------------

def lhs_draws(astro_draws, seed):
    sampler = qmc.LatinHypercube(d=6, strength=1, seed=seed).random(n=astro_draws)
    lhd = []
    lhd.append(uniform(loc = 0.1, scale = 11 - 0.1).ppf(sampler[:, 0]))
    lhd.append(norm(loc=-2.56, scale=0.4).ppf(sampler[:, 1]))
    lhd.append(norm(loc=10.9, scale=0.4).ppf(sampler[:, 2]))
    lhd.append(norm(loc=8.6, scale=0.2).ppf(sampler[:, 3]))
    lhd.append(norm(loc=0.32, scale=0.15).ppf(sampler[:, 4]))
    lhd.append(uniform(loc = -1.5, scale = 1.5).ppf(sampler[:, 5]))
    lhd = np.array(lhd).T
    return lhd


# ---- cell 16 ----------------------------------------------------------------

def init_sam(sam_shape, params):
    from holodeck import sams, host_relations
    from holodeck.constants import GYR

    gsmf = sams.GSMF_Schechter(
        phi0=params['gsmf_phi0_log10'],
        phiz=params['gsmf_phiz'],
        mchar0_log10=params['gsmf_mchar0_log10'],
        mcharz=params['gsmf_mcharz'],
        alpha0=params['gsmf_alpha0'],
        alphaz=params['gsmf_alphaz'],
    )
    gpf = sams.GPF_Power_Law(
        frac_norm_allq=params['gpf_frac_norm_allq'],
        malpha=params['gpf_malpha'],
        qgamma=params['gpf_qgamma'],
        zbeta=params['gpf_zbeta'],
        max_frac=params['gpf_max_frac'],
    )
    gmt = sams.GMT_Power_Law(
        time_norm=params['gmt_norm']*GYR,
        malpha=params['gmt_malpha'],
        qgamma=params['gmt_qgamma'],
        zbeta=params['gmt_zbeta'],
    )
    mmbulge = host_relations.MMBulge_KH2013(
        mamp_log10=params['mmb_mamp_log10'],
        mplaw=params['mmb_plaw'],
        scatter_dex=params['mmb_scatter_dex'],
    )

    sam = sams.Semi_Analytic_Model(
        gsmf=gsmf, gpf=gpf, gmt=gmt, mmbulge=mmbulge,
        shape=sam_shape,
    )
    return sam

def init_hard(sam, params):
    from holodeck import hardening
    from holodeck.constants import GYR, PC

    hard = hardening.Fixed_Time_2PL_SAM(
        sam,
        params['hard_time']*GYR,
        sepa_init=params['hard_sepa_init']*PC,
        rchar=params['hard_rchar']*PC,
        gamma_inner=params['hard_gamma_inner'],
        gamma_outer=params['hard_gamma_outer'],
    )
    return hard


params = dict(
    hard_time=3.0,          #This will be varied
    hard_sepa_init=1e4,
    hard_rchar=100.0,
    hard_gamma_inner=-1.0, ##This will be varied
    hard_gamma_outer=+2.5,

    gsmf_phi0_log10=-2.77, ##This will be varied
    gsmf_phiz=-0.6,
    gsmf_mchar0_log10=11.24,##This will be varied
    gsmf_mcharz=0.11,
    gsmf_alpha0=-1.21,
    gsmf_alphaz=-0.03,

    gpf_frac_norm_allq=0.025,
    gpf_malpha=0.0,
    gpf_qgamma=0.0,
    gpf_zbeta=1.0,
    gpf_max_frac=1.0,

    gmt_norm=0.5,
    gmt_malpha=0.0,
    gmt_qgamma=-1.0,
    gmt_zbeta=-0.5,

    mmb_mamp_log10=8.69, ##This will be varied
    mmb_plaw=1.10,
    mmb_scatter_dex=0.3, ##This will be varied
)


# ---- cells 17 and 20 --------------------------------------------------------

def make_doit(theta_master, SAVEDIR, SAM_SHAPE, NUM_REALS, PTA_DUR_yr, NUM_FREQS):
    from holodeck import utils
    from holodeck.constants import YR

    PTA_DUR = PTA_DUR_yr * YR
    freqs = np.arange(1/PTA_DUR, (NUM_FREQS+ .001)/PTA_DUR, 1/PTA_DUR)

    def doit(rr):

        theta = theta_master[rr] #randomly selected draws from the distributions
        params['hard_time'] = theta[0]
        params['gsmf_phi0_log10'] = theta[1]
        params['gsmf_mchar0_log10'] = theta[2]
        params['mmb_mamp_log10'] = theta[3]
        params['mmb_scatter_dex'] = theta[4]
        params['hard_gamma_inner'] = theta[5]

        sam = init_sam(sam_shape = SAM_SHAPE, params = params)
        hard = init_hard(sam, params)
        fobs_gw_cents, fobs_gw_edges = utils.pta_freqs(PTA_DUR, NUM_FREQS)

        hc_ss_ph = sam.gwb_new(fobs_gw_edges, hard=hard, realize=NUM_REALS)

        spectrum = 0.5 * np.log10(hc_ss_ph**2/(12*np.pi**2 * freqs[:, None]**3 * PTA_DUR)) #for detection runs we use spectrum instead of strain!

        if np.isfinite(spectrum).all(): # no silly business
            np.save(SAVEDIR + f'/{rr}_{PTA_DUR_yr}yrs.npy', spectrum) # save the spectrum as a npy file (compressed and portable!)

    return doit


# ---- cells 23 and 25 --------------------------------------------------------

def combine(SAVEDIR, PTA_DUR_yr, NUM_REALS, NUM_FREQS):
    paths = sorted(glob.glob(SAVEDIR + f'/*_{PTA_DUR_yr}yrs.npy'),
                   key=lambda p: int(os.path.basename(p).split('_')[0]))
    one_file = np.lib.format.open_memmap(SAVEDIR + f'/gwb_spectrum_samples_{PTA_DUR_yr}yrs.npy',
                        mode='w+',
                        dtype='float64',
                        shape=(len(paths), NUM_REALS , NUM_FREQS),
                        fortran_order=False)

    for idx in range(len(paths)):
        one_file[idx] = np.load(paths[idx]).T # we wont use the `one_file`. it is just here to dump the spectrum into a single file
    one_file.flush()


# ---- cells 28 and 31 --------------------------------------------------------

def prepare(SAVEDIR, theta_master, PTA_DUR_yr):
    gwb_data = np.load(SAVEDIR + f'/gwb_spectrum_samples_{PTA_DUR_yr}yrs.npy', mmap_mode='r')
    par_data = theta_master[:gwb_data.shape[0]] # make sure the right astro samples are used. remove ':gwb_data.shape[0]' if needed

    n_real = gwb_data.shape[1]
    n_samp = gwb_data.shape[0]
    n_pars = par_data.shape[-1]

    par_data = np.broadcast_to(par_data, (n_real, n_samp, n_pars)).transpose((1, 2, 0)) #this just makes sure astro-params have the same shape as gwb spectrum

    ## The first axis is samples, the second is realization
    par_data = par_data.transpose((0, 2, 1))
    ## The third axis combines `gwb_freq` params with other `params`. This is just a matter of organization!
    chain = np.concatenate((par_data, gwb_data), axis = 2)

    assert chain.all()

    B = 5
    min_x = np.min(chain, axis = (0, 1))
    max_x = np.max(chain, axis = (0, 1))
    mean = (max_x + min_x) / 2
    half_range = (max_x - min_x) / 2
    chain = B * (chain - mean) / half_range

    np.save(SAVEDIR + f'/gwb_spectrum_samples_{PTA_DUR_yr}yrs_normalized.npy', chain[..., n_pars:])
    np.save(SAVEDIR + f'/ast_spectrum_samples_{PTA_DUR_yr}yrs_normalized.npy', chain[:, 0, :n_pars])
    np.savez_compressed(SAVEDIR + f'/gwb_spectrum_samples_{PTA_DUR_yr}yrs_mapping_data.npy',
                    B = B, mean = mean, half_range = half_range)


# ---- NormalizingFlowTrainDEMO.ipynb, cells 5, 10, 12, 14, 16 -----------------

def legacy_library_chain(path_to_lib, gwb_freq_bins, Tspan):
    """Returns ``chain`` (n_samp, n_real, n_pars + gwb_freq_bins) before scaling."""
    import h5py

    with h5py.File(path_to_lib, 'r') as data:
        gwb_data = data['gwb'][()][:, 0:gwb_freq_bins, :]
        low_ind = np.where(gwb_data < 1e-20)
        gwb_data[low_ind] = 1e-20

        param_names = data.attrs['param_names'].astype(str)
        par_data = data['sample_params'][()]
        n_pars = par_data.shape[1]
    n_real = gwb_data.shape[-1]
    n_samp = gwb_data.shape[0]
    par_data = np.broadcast_to(par_data, (n_real, n_samp, n_pars)).transpose((1, 2, 0))

    gwb_data = gwb_data.transpose((0, 2, 1))
    par_data = par_data.transpose((0, 2, 1))

    chain = np.concatenate((par_data, gwb_data), axis = 2)

    freqs = np.arange(1/Tspan, (gwb_freq_bins + .001)/Tspan, 1/Tspan)

    chain[..., -gwb_freq_bins:] = 0.5 * np.log10(chain[..., -gwb_freq_bins:]**2/(12*np.pi**2 * freqs[None, None, :]**3 * Tspan))
    return chain, list(param_names)
