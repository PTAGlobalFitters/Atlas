"""Resolved run config -> PTA_Data, white noise, red noise, timing model."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial

import numpy as np
import jax.numpy as jnp

from ATLAS.runner import data_io
from ATLAS.runner.config import ConfigError, load_priors

log = logging.getLogger("ATLAS.runner")

__all__ = ["Built", "build_model"]


@dataclass
class Built:
    data: object
    wn: object
    rn: object
    tm_model: object
    wn_vec: object            # fixed white-noise vector, or None
    wn_lower: object
    wn_upper: object
    present: tuple
    psr_names: list
    residual_offset: object = None   # subtracted from the residuals (GTM/Adaptus mean)

    @property
    def residuals(self):
        """The residuals the likelihood sees: raw minus ``residual_offset``."""
        r = jnp.concat(self.data.raw_residuals)
        return r if self.residual_offset is None else r - self.residual_offset


def _bins(fb, key, present, sig):
    return fb[key] if sig in present else None


def _wn_overrides(run, wn):
    """Optional white-noise bounds from the priors file (``white_noise:``)."""
    entry = load_priors(run).get("white_noise")
    if not entry:
        return
    attrs = {"efac": ("lower_efac", "upper_efac"),
             "log10_equad": ("lower_log10equad", "upper_log10equad"),
             "log10_ecorr": ("lower_log10ecorr", "upper_log10ecorr")}
    unknown = set(entry) - set(attrs)
    if unknown:
        raise ConfigError(f"priors white_noise: unknown keys {sorted(unknown)}; "
                          f"expected {sorted(attrs)}")
    for key, (lo_attr, hi_attr) in attrs.items():
        if key in entry:
            lo, hi = entry[key]
            for cov in wn.cov_matrices:
                setattr(cov, lo_attr, float(lo)); setattr(cov, hi_attr, float(hi))


def build_model(run, parsed, resolved):
    """Everything ``model_maker`` needs for one run.

    ``parsed`` / ``resolved`` come from ``config.validate_run``.
    """
    from ATLAS.data import PTA_Data
    from ATLAS.model_builder import ModelBuilder

    present = tuple(parsed["shared_names"] + parsed["separate_names"])
    d, fb, tm_cfg, sig = run["data"], run["frequency_bins"], run["timing"], run["model"]["signals"]

    psrs = data_io.load_pulsars(run)
    names = [str(p.name) for p in psrs]
    noise_dict = data_io.load_noise_dict(run)

    adaptus_basis = gtm_psd = gtm_means = None
    if "gtm" in present:
        adaptus_basis, gtm_psd, gtm_means = data_io.load_gtm(run, psrs)

    parfiles = timfiles = None
    if tm_cfg["nonlinear"]["enabled"]:
        files = data_io.pulsar_files(run, names)
        parfiles = [files[n][0] for n in names]
        timfiles = [files[n][1] for n in names]

    data = PTA_Data(
        psrs,
        num_irn_bins=_bins(fb, "num_irn_bins", present, "unc"),
        num_gwb_bins=_bins(fb, "num_gwb_bins", present, "cor"),
        num_dm_bins=_bins(fb, "num_dm_bins", present, "dm"),
        num_det_bins=_bins(fb, "num_det_bins", present, "det"),
        adaptus_basis=adaptus_basis,
        adaptus_size=(adaptus_basis[0].shape[1] if adaptus_basis is not None else None),
        linear_timing=tm_cfg["linear_timing"],
        marg_timing=tm_cfg["marg_timing"],
        noise_dict=noise_dict,
        dm_ref_freq=float(d["dm_ref_freq"]),
        psr_dists_dict=data_io.load_distances(run),
        parfiles=parfiles, timfiles=timfiles,
    )
    mb = ModelBuilder(data=data)

    wn_cfg = run["white_noise"]
    ecorr = {"auto": None, True: True, False: False}[wn_cfg["include_ecorr"]]
    wn = mb.make_white_noise(stabilize_TNT=bool(wn_cfg["stabilize_TNT"]), include_ecorr=ecorr)
    _wn_overrides(run, wn)
    wn_lower, wn_upper = (jnp.asarray(x) for x in wn.get_prior_bounds())
    wn_vec = wn.params_dict_to_vector(noise_dict) if noise_dict is not None else None

    kw = dict(use_pulsar_tspan=bool(fb["use_pulsar_tspan"]))
    for which in ("irn", "gwb", "dm", "chrom"):
        uf = data_io.user_freqs(run, which)
        if uf is not None:
            kw[f"{which}_user_freqs"] = jnp.asarray(uf)

    def psd_of(s):
        r = resolved[s]
        fixed = sig[s]["fixed"] or {}
        fn = partial(r["psd"], **fixed) if fixed else r["psd"]
        return fn, jnp.asarray(r["lower"]), jnp.asarray(r["upper"])

    if "unc" in present:
        kw["irn_psd_function"], kw["irn_lower_bound_psd"], kw["irn_upper_bound_psd"] = psd_of("unc")
    if "cor" in present:
        kw["gwb_psd_function"], kw["gwb_lower_bound_psd"], kw["gwb_upper_bound_psd"] = psd_of("cor")
        kw["orf_function"] = resolved["cor"]["orf"]
        if resolved["cor"]["orf_names"]:
            kw["lower_bound_orf"] = jnp.asarray(resolved["cor"]["orf_lower"])
            kw["upper_bound_orf"] = jnp.asarray(resolved["cor"]["orf_upper"])
    if "dm" in present:
        kw["dm_psd_function"], kw["dm_lower_bound_psd"], kw["dm_upper_bound_psd"] = psd_of("dm")
    if "chrom" in present or "cquad" in present:
        idx = data_io.chrom_index_for(run, psrs)
        kw["chrom_index"] = idx
    if "chrom" in present:
        kw["chrom_psd_function"], kw["chrom_lower_bound_psd"], kw["chrom_upper_bound_psd"] = psd_of("chrom")
        kw["num_chrom_bins"] = int(fb["num_chrom_bins"])
    if "cquad" in present:
        kw["cquad_log10_amp"] = float(sig["cquad"]["log10_amp"])
        kw["cquad_project_timing"] = bool(sig["cquad"]["project_timing"])
    if "sw" in present:
        kw["sw_geometry"] = data_io.sw_geometry(run, psrs, compute=True)
        swk = data_io.sw_kwargs_for(run, psrs)
        if "fixed_log10_sigma" not in swk:
            swk["lower_bound_psd"] = tuple(resolved["sw"]["lower"])
            swk["upper_bound_psd"] = tuple(resolved["sw"]["upper"])
        kw["sw_kwargs"] = swk
    if "det" in present:
        from ATLAS.runner import registry
        kw["det_delay_function"] = registry.waveform(sig["det"]["waveform"])
        kw["det_parameter_bounds"] = np.stack([resolved["det"]["lower"],
                                               resolved["det"]["upper"]], axis=1)
    if "gtm" in present and gtm_psd is not None:
        kw["gt_psd_val"] = jnp.asarray(gtm_psd)

    rn = mb.make_red_noise(run["model"]["basis_string"], **kw)

    tm_model = None
    if tm_cfg["nonlinear"]["enabled"]:
        tm_model = mb.make_timing_model(
            explicit_timing_model_params_to_sample=data_io.timing_params_for(run, psrs))

    log.info("built %s: %d pulsars, %d T-matrix columns, %d red-noise parameters",
             run["run"]["name"], len(psrs), rn.nmodes, rn.model.upper_prior_lim_all.shape[0])
    offset = None
    if gtm_means is not None:
        if tm_model is not None:
            raise ConfigError("a gtm mean is subtracted from the data residuals, but with "
                              "nonlinear timing the residuals come from the timing model; "
                              "drop mean_pattern or the nonlinear timing")
        offset = jnp.asarray(np.concatenate(gtm_means))
    return Built(data, wn, rn, tm_model, wn_vec, wn_lower, wn_upper, present, names, offset)
