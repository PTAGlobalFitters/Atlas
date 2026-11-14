"""Point-and-click builder for atlas-run config files (Jupyter / VS Code).

    from ATLAS.runner.wizard import Wizard
    Wizard()                                   # pick a preset, fill the tabs, Validate, Save

    Wizard(save_path="/data/.../runs/my_run.yaml")   # where Save writes by default

The menus come from ATLAS itself (PSDs and ORFs from ATLAS.psd_functions, the
settings and their defaults from ATLAS.runner.config), and Validate runs the
same checks as ``atlas-run check``.  Save writes the run config and its priors
file next to each other.  Every option of the runner stays available by
editing the saved YAML; the preview shows exactly what will be written.
"""
from __future__ import annotations

import copy
import glob
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import yaml

__all__ = ["Wizard", "PRESETS"]

SEPARATE_ORDER = ("dm", "gtm", "chrom", "cquad", "sw", "det")
SW_KERNELS = ("RIDGE",)                 # other kernels need extra hyperparameters
DEFAULT_BOUNDS = {
    "log10_A": (-18.0, -11.0), "gamma": (0.0, 7.0), "halflog10_rho": (-9.0, -4.0),
    "log10_amp": (-3.0, 2.0), "delta": (0.0, 7.0), "log10_fb": (-10.0, -7.0),
    "kappa": (0.01, 0.5), "lf0": (-9.0, -7.0), "beta": (0.0, 7.0), "lfb": (-10.0, -7.0),
    "lfk": (-10.0, -7.0), "a": (-1.0, 1.0), "b": (-1.0, 1.0), "c": (-1.0, 1.0),
    "tau": (-1.5, 1.5), "log10_A_tt": (-18.0, -11.0), "gamma_tt": (0.0, 7.0),
    "log10_A_st": (-18.0, -11.0), "gamma_st": (0.0, 7.0), "log10_A_vl": (-18.0, -11.0),
    "gamma_vl": (0.0, 7.0),
}
CW_BOUNDS = {"log10_mc": (7.2, 9.0), "log10_fgw": (-8.7, -8.2), "cos_inc": (-1.0, 1.0),
             "psi": (0.0, np.pi), "log10_h": (-18.0, -12.0), "cos_theta": (-1.0, 1.0),
             "phi": (0.0, 2 * np.pi), "phase0": (0.0, 2 * np.pi)}

TIMING_CHOICES = [
    ("Standard: marginalise the timing model linearly", "standard"),
    ("Linear: sample the timing coefficients ('ltm|')", "linear"),
    ("Direct: sample timing parameters nonlinearly (JUG)", "direct"),
    ("Target: Adaptus PCA basis instead of the timing model", "target"),
]
EVIDENCE_CHOICES = [
    ("None (ordinary posterior)", "none"),
    ("Model switch TI: ln BF between two ORFs in one integral", "switch"),
    ("Fixed-beta TI: ln Z of each model", "ti"),
]
RUN_LAYOUTS = [
    ("One run (all selected pulsars together)", "single"),
    ("One run per pulsar", "per_psr"),
    ("One run per realization (simulations)", "per_real"),
    ("One run per pulsar and realization", "per_psr_real"),
]

# --------------------------------------------------------------------------- #
#  presets: widget values (keys are Wizard widget keys)
# --------------------------------------------------------------------------- #

_COMMON = {
    "vars": "base: /data/taylor_group/Nima\n",
    "source": "atlas_npz", "npz_files": "${base}/NG15_CNM/{psr}.npz",
    "par_files": "", "tim_files": "", "file_exclude": "ao, gbt",
    "noise_dict": "${base}/15yr_stochastic_analysis/tutorials/data/15yr_wn_dict.json",
    "cache_dir": "${base}/atlas_cache", "dm_ref_freq": 1400.0,
    "timing": "standard", "params_mode": "all", "marg_only_listed": False,
    "marg_rest": True, "parametrization": "simple",
    "use": {"unc": True, "cor": False, "dm": False, "chrom": False, "cquad": False,
            "sw": False, "det": False},
    "wn_vary": False, "ecorr": "auto", "stabilize": True, "use_pulsar_tspan": False,
    "evidence": "none", "marg_over_non_gwb": False,
    "num_warmup": 500, "num_samples": 5000, "num_chains": 1, "chain_method": "vectorized",
    "layout": "single", "run_name": "run",
    "cpus": 4, "gpus": 1.0, "out_root": "${base}/atlas_runs", "save_coeff": False,
}

PRESETS = {
    "Single-pulsar noise (IRN + DM + chromatic + solar wind)": dict(
        _COMMON, use={"unc": True, "cor": False, "dm": True, "chrom": True, "cquad": False,
                      "sw": True, "det": False},
        wn_vary=True, use_pulsar_tspan=True, layout="per_psr", run_name="noise",
        cpus=20, out_root="${base}/NG15_CNM/SinglePulsar_Noise", sw_geometry="stored",
        sw_from_par=False),
    "PTA GWB search (HD, gamma = 13/3)": dict(
        _COMMON, use={"unc": True, "cor": True, "dm": False, "chrom": False, "cquad": False,
                      "sw": False, "det": False},
        cor_orf="hd_orf", cor_fix_gamma=True, run_name="hd", cpus=20,
        out_root="${base}/GWB_runs"),
    "PTA CURN search": dict(
        _COMMON, use={"unc": True, "cor": True, "dm": False, "chrom": False, "cquad": False,
                      "sw": False, "det": False},
        cor_orf="zero_orf", cor_fix_gamma=True, run_name="curn", cpus=20,
        out_root="${base}/GWB_runs"),
    "Evidence: ln BF(HD/CURN), model-switch TI": dict(
        _COMMON, use={"unc": True, "cor": True, "dm": False, "chrom": False, "cquad": False,
                      "sw": False, "det": False},
        cor_orf="hd_orf", cor_fix_gamma=False, stabilize=False, evidence="switch",
        switch_orf="zero_orf", n_beta=17, beta_kind="linear", marg_over_non_gwb=True,
        num_samples=4000, run_name="hd_vs_curn", gpus=0.25,
        out_root="${base}/Evidence/TI_switch_hd_vs_curn"),
    "Evidence: ln Z per model, fixed-beta TI": dict(
        _COMMON, use={"unc": True, "cor": True, "dm": False, "chrom": False, "cquad": False,
                      "sw": False, "det": False},
        cor_orf="hd_orf", cor_fix_gamma=True, stabilize=False, evidence="ti", n_beta=24,
        beta_kind="beta", marg_over_non_gwb=True, num_samples=2000, run_name="hd",
        gpus=0.25, out_root="${base}/Evidence/TI"),
    "Paper2 direct, single pulsar (simulations)": dict(
        _COMMON, vars="base: /data/taylor_group/Nima\nsim_dir: ${base}/Atlas/datasets/NG15/SIM\n",
        npz_files="${sim_dir}/${realization}_{psr}.npz", par_files="${base}/par/*.par",
        tim_files="${base}/tim/*.tim", timing="direct", params_mode="file",
        params_file="${sim_dir}/info.npz", params_key="SAMPLE_LIST", params_index="0",
        wn_vary=True, layout="per_psr_real", real_start=0, real_stop=1,
        run_name="direct", cpus=20, save_coeff=True,
        out_root="${base}/Paper2_Chains/SinglePulsar_runner",
        unc_user_freqs="${sim_dir}/info.npz:gwb_freqs"),
    "Blank (runner defaults)": dict(_COMMON),
}


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _beta_grid(n, kind):
    u = np.linspace(0.0, 1.0, int(n))
    if kind == "linear":
        b = u
    elif kind == "power":
        b = u ** (1 / 0.3)
    else:
        from scipy.stats import beta as _b
        b = _b.ppf(u, 0.3, 0.3)
    b[0], b[-1] = 0.0, 1.0
    return [float(x) for x in np.unique(np.round(b, 6))]


def _split_list(text):
    return [t for t in re.split(r"[,\s]+", text or "") if t]


def _yaml_or_empty(text, what):
    text = (text or "").strip()
    if not text:
        return {}
    val = yaml.safe_load(text)
    if not isinstance(val, dict):
        raise ValueError(f"{what} must be a YAML mapping (key: value lines)")
    return val


class _Dumper(yaml.SafeDumper):
    pass


def _short_lists(dumper, data):
    flow = len(data) <= 12 and all(isinstance(x, (int, float, str)) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _short_lists)


def _dump(d):
    return yaml.dump(d, Dumper=_Dumper, sort_keys=False, width=100, default_flow_style=False)


# --------------------------------------------------------------------------- #
#  the wizard
# --------------------------------------------------------------------------- #

class Wizard:
    """Build, check and save an atlas-run config from menus."""

    def __init__(self, save_path="run_config.yaml", preset=None):
        import ipywidgets as W
        from ATLAS.runner import registry
        self.W = W
        self.registry = registry
        self.psd_names = sorted(registry.available_psds())
        self.orf_names = sorted(registry.available_orfs())
        self.w = {}
        self._prior_widgets = {}          # (prior_key, param) -> (lo, hi)
        self._prior_cache = {}
        self._building = False
        self._ui = self._build(save_path)
        self._initial = {k: w.value for k, w in self.w.items()
                         if k not in ("pulsars_sel", "pulsars_extra")}
        self.apply_preset(preset or next(iter(PRESETS)))

    # ------------------------------------------------------------------ UI
    def _ipython_display_(self):
        from IPython.display import display
        display(self._ui)

    def _add(self, key, widget):
        self.w[key] = widget
        widget.observe(self._changed, names="value")
        return widget

    def _text(self, key, desc, value="", width="95%", ph=""):
        return self._add(key, self.W.Text(value=value, description=desc, placeholder=ph,
                                          style={"description_width": "170px"},
                                          layout=self.W.Layout(width=width)))

    def _dd(self, key, desc, options, value=None, width="95%"):
        return self._add(key, self.W.Dropdown(options=options, value=value, description=desc,
                                              style={"description_width": "170px"},
                                              layout=self.W.Layout(width=width)))

    def _int(self, key, desc, value):
        return self._add(key, self.W.IntText(value=value, description=desc,
                                             style={"description_width": "170px"}))

    def _float(self, key, desc, value):
        return self._add(key, self.W.FloatText(value=value, description=desc,
                                               style={"description_width": "170px"}))

    def _check(self, key, desc, value=False):
        return self._add(key, self.W.Checkbox(value=value, description=desc, indent=False,
                                              layout=self.W.Layout(width="auto")))

    def _note(self, html):
        return self.W.HTML(f"<div style='color:#555;font-size:90%'>{html}</div>")

    def _build(self, save_path):
        W = self.W
        # --- Start
        self.preset_dd = W.Dropdown(options=list(PRESETS), description="Preset",
                                    style={"description_width": "80px"},
                                    layout=W.Layout(width="70%"))
        apply_btn = W.Button(description="Apply preset", button_style="info")
        apply_btn.on_click(lambda _: self.apply_preset(self.preset_dd.value))
        load_txt = W.Text(placeholder="path/to/existing.yaml", description="Load YAML",
                          style={"description_width": "80px"}, layout=W.Layout(width="70%"))
        load_btn = W.Button(description="Load into preview")
        load_btn.on_click(lambda _: self._load_yaml(load_txt.value))
        start = W.VBox([
            self._note("A preset fills every tab with a typical setup; change anything "
                       "afterwards.  The preview (right) is exactly what Save writes."),
            W.HBox([self.preset_dd, apply_btn]),
            self._note("Loading an existing YAML shows it in the preview and validates it; "
                       "the menus are not filled from it."),
            W.HBox([load_txt, load_btn]),
        ])

        # --- Data
        self.psr_select = self._add("pulsars_sel", W.SelectMultiple(
            options=[], rows=10, description="Pulsars", style={"description_width": "170px"},
            layout=W.Layout(width="95%")))
        scan_btn = W.Button(description="Scan data files for pulsars")
        scan_btn.on_click(lambda _: self._scan())
        sel_all = W.Button(description="Select all")
        sel_all.on_click(lambda _: setattr(self.psr_select, "value",
                                           tuple(self.psr_select.options)))
        data = W.VBox([
            self._add("vars", W.Textarea(description="Variables", rows=3,
                                         style={"description_width": "170px"},
                                         layout=W.Layout(width="95%"))),
            self._note("Variables are used in paths as ${name}, e.g. ${base}/NG15_CNM.  "
                       "{psr} in a file pattern is filled with each pulsar's name."),
            self._dd("source", "Data source", [("ATLAS .npz files", "atlas_npz"),
                                               ("par / tim files", "par_tim"),
                                               ("Pickled enterprise pulsars", "enterprise_pickle")]),
            self._text("npz_files", "npz files", ph="${base}/NG15_CNM/{psr}.npz"),
            self._text("pickle", "Pickle file"),
            self._text("par_files", "par files (glob)", ph="${base}/par/*.par"),
            self._text("tim_files", "tim files (glob)", ph="${base}/tim/*.tim"),
            self._text("file_exclude", "Skip par/tim containing", ph="ao, gbt"),
            W.HBox([scan_btn, sel_all]),
            self.psr_select,
            self._text("pulsars_extra", "or type pulsars", ph="B1855+09, J1909-3744"),
            self._note("No pulsar selected and none typed = every pulsar in the data."),
            self._text("noise_dict", "White-noise dict"),
            self._text("cache_dir", "Cache folder"),
            self._float("dm_ref_freq", "DM/chrom ref. freq [MHz]", 1400.0),
        ])

        # --- Timing
        timing = W.VBox([
            self._dd("timing", "Timing model", TIMING_CHOICES, "standard"),
            self._dd("params_mode", "Parameters (direct)", [
                ("Every fitted parameter", "all"), ("A list, same for every pulsar", "list"),
                ("From a file ({psr: [...]})", "file"),
                ("Stored in each pulsar's .npz (sample_params)", "from_pulsar")], "all"),
            self._text("params_list", "Parameter list", ph="F0, F1, RAJ, DECJ"),
            self._text("params_file", "Parameter file", ph="${sim_dir}/info.npz"),
            W.HBox([self._text("params_key", "key", "SAMPLE_LIST", width="45%"),
                    self._text("params_index", "index (blank = none)", "0", width="45%")]),
            self._dd("parametrization", "Direct parametrization",
                     [("simple: bounded per-parameter priors", "simple"),
                      ("klam: k * lambda", "klam")], "simple"),
            self._check("marg_rest", "Direct: marginalise the parameters not sampled "
                                     "(otherwise held at their par values)", True),
            self._check("marg_only_listed", "Standard: marginalise only the listed "
                                            "parameters (others held at par values)", False),
            self._text("gtm_basis", "Adaptus basis (target)", "${base}/AdaptusBasis/{psr}/U.npy"),
            self._text("gtm_mean", "Adaptus mean (target)", "${base}/AdaptusBasis/{psr}/mean.npy"),
            W.HBox([self._int("gtm_size", "PCA columns", 200), self._text("gtm_index", "index", "0",
                                                                          width="40%")]),
            self._note("Standard and Direct use the timing marginalisation; the parameter "
                       "list is needed for Direct, and for Standard only when marginalising "
                       "the listed parameters only."),
        ])

        # --- Noise model
        acc = []
        self._sig_boxes = {}

        def sig_box(name, title, rows):
            cb = self._check(f"use_{name}", title, False)
            box = W.VBox(rows, layout=W.Layout(margin="0 0 0 25px"))
            self._sig_boxes[name] = box
            return W.VBox([cb, box])

        acc.append(sig_box("unc", "Intrinsic red noise (unc)", [
            self._dd("unc_psd", "PSD", self.psd_names, "powerlaw"),
            self._int("unc_bins", "Frequency bins", 30),
            self._text("unc_user_freqs", "Frequencies [Hz] (optional)", ph="file.npz:key or 1e-9, 2e-9"),
        ]))
        acc.append(sig_box("cor", "Common / GWB process (cor)", [
            self._dd("cor_psd", "PSD", [n for n in self.psd_names if n != "free_spectrum"]
                     + ["free_spectrum"], "powerlaw"),
            self._dd("cor_orf", "ORF", self.orf_names, "hd_orf"),
            W.HBox([self._check("cor_fix_gamma", "Fix gamma at", True),
                    self._add("cor_gamma", W.FloatText(value=13 / 3, layout=W.Layout(width="120px")))]),
            self._int("cor_bins", "Frequency bins", 14),
            self._text("cor_user_freqs", "Frequencies [Hz] (optional)"),
        ]))
        acc.append(sig_box("dm", "DM noise (dm, index 2)", [
            self._dd("dm_psd", "PSD", self.psd_names, "powerlaw"),
            self._int("dm_bins", "Frequency bins", 30),
            self._check("dm_varied", "Sample the index of the dm block", False),
        ]))
        acc.append(sig_box("chrom", "Chromatic noise (chrom)", [
            self._dd("chrom_psd", "PSD", self.psd_names, "powerlaw"),
            self._int("chrom_bins", "Frequency bins", 30),
            self._float("chrom_index", "Chromatic index", 4.0),
            W.HBox([self._check("chrom_varied", "Sample the index, bounds", False),
                    self._add("chrom_lo", W.FloatText(value=2.5, layout=W.Layout(width="90px"))),
                    self._add("chrom_hi", W.FloatText(value=6.5, layout=W.Layout(width="90px")))]),
        ]))
        acc.append(sig_box("cquad", "Chromatic quadratic (cquad)", [
            self._float("cquad_amp", "log10 amplitude (fixed)", 0.0),
            self._check("cquad_project", "Project out the timing model", True),
        ]))
        acc.append(sig_box("sw", "Solar wind (sw, time-domain GP)", [
            self._dd("sw_geometry", "Geometry", [("stored in the .npz", "stored"),
                                                 ("PINT (needs par/tim)", "pint"),
                                                 ("approximate (~1%)", "approx")], "stored"),
            self._dd("sw_kernel", "Kernel", list(SW_KERNELS), "RIDGE"),
            self._float("sw_dt", "Node spacing [days]", 30.0),
            self._check("sw_from_par", "Read TDSW* settings from par files", False),
            W.HBox([self._check("sw_fixed", "Fix log10 sigma at", False),
                    self._add("sw_sigma", W.FloatText(value=0.0, layout=W.Layout(width="90px")))]),
        ]))
        acc.append(sig_box("det", "Deterministic signal (det)", [
            self._text("det_waveform", "Waveform", "cw_delay_evolve_float64"),
            self._text("det_dists", "Pulsar distances file"),
            self._int("det_bins", "Frequency bins", 14),
        ]))
        noise = W.VBox([
            self._check("use_pulsar_tspan", "Frequencies from each pulsar's own time span", False),
            *acc,
            self._note("The basis string is assembled for you (shared block first, then "
                       "dm, gtm, chrom, cquad, sw, det).")])

        # --- White noise
        wn = W.VBox([
            self._check("wn_vary", "Sample the white noise (else fixed from the dict)", False),
            self._dd("ecorr", "ECORR", [("auto-detect", "auto"), ("yes", True), ("no", False)], "auto"),
            self._check("stabilize", "stabilize_TNT (off for evidence runs)", True),
        ])

        # --- Priors
        self.prior_box = W.VBox([])
        priors = W.VBox([
            self._note("Uniform bounds [low, high] for every sampled parameter.  The table "
                       "follows the noise model; white-noise bounds use ATLAS's defaults."),
            self.prior_box,
            self._text("priors_name", "Priors file name", "priors.yaml"),
        ])

        # --- Sampler / evidence
        sampler = W.VBox([
            self._int("num_warmup", "Warmup", 500),
            self._int("num_samples", "Samples", 5000),
            self._int("num_chains", "Chains", 1),
            self._dd("chain_method", "Chain method", ["vectorized", "sequential", "parallel"],
                     "vectorized"),
            self._float("target_accept", "Target acceptance", 0.8),
            self._int("max_tree_depth", "Max tree depth", 10),
            self._int("seed", "Seed", 170817),
            self._check("progress_bar", "Show a progress bar (in each run's log / Ray output)", False),
            self._check("marg_over_non_gwb", "Marginalise every non-GWB coefficient "
                                             "(keep only the GWB ones)", False),
            W.HTML("<b>Evidence</b>"),
            self._dd("evidence", "Evidence", EVIDENCE_CHOICES, "none"),
            self._dd("switch_orf", "Model A ORF (beta = 0)", self.orf_names, "zero_orf"),
            W.HBox([self._int("n_beta", "beta points", 17),
                    self._dd("beta_kind", "spacing", [("Beta(0.3,0.3): dense at both ends", "beta"),
                                                      ("linear", "linear"),
                                                      ("power: dense near 0", "power")], "linear",
                             width="60%")]),
            self._text("ti_extra_orfs", "Fixed-beta TI: also run ORFs", ph="zero_orf"),
            self._note("Evidence needs a GWB block; the wizard switches on the GWB-only "
                       "marginalisation and switches off stabilize_TNT for you."),
        ])

        # --- Runs
        runs = W.VBox([
            self._text("run_name", "Run name", "run"),
            self._dd("layout", "Runs", RUN_LAYOUTS, "single"),
            W.HBox([self._int("real_start", "Realizations from", 0),
                    self._int("real_stop", "to (not incl.)", 1)]),
            self._note("For realizations put ${realization} in the data paths; "
                       "seed = seed + realization."),
            self._add("variants", W.Textarea(description="Extra variants", rows=4,
                                             style={"description_width": "170px"},
                                             layout=W.Layout(width="95%"),
                                             placeholder="curn:\n  model.signals.cor.orf: zero_orf")),
            self._note("Each extra variant is a name with dotted-path overrides; it is run "
                       "for every pulsar / realization / beta above, like the main one."),
        ])

        # --- Ray / output
        out = W.VBox([
            self._int("cpus", "CPUs per run", 4),
            self._float("gpus", "GPUs per run", 1.0),
            self._add("max_conc", W.IntText(value=0, description="Max concurrent (0 = any)",
                                            style={"description_width": "170px"})),
            self._text("ray_address", "Ray address (blank = local)"),
            self._text("out_root", "Output folder", "${base}/atlas_runs"),
            self._dd("out_format", "Format", ["npz", "netcdf"], "npz"),
            self._check("save_coeff", "Save the red-noise coefficients 'coeff'", False),
            self._check("skip_completed", "Skip runs already done", True),
        ])

        tabs = W.Tab(children=[start, data, timing, noise, wn, priors, sampler, runs, out])
        for i, t in enumerate(["Start", "Data", "Timing", "Noise model", "White noise",
                               "Priors", "Sampler", "Runs", "Output"]):
            tabs.set_title(i, t)

        # --- right panel: preview, validate, save
        self.preview = W.Textarea(rows=38, layout=W.Layout(width="100%"), disabled=True)
        self.preview.add_class("atlas-wizard-mono")
        self.status = W.HTML()
        self.save_path = W.Text(value=str(save_path), description="Save as",
                                style={"description_width": "60px"}, layout=W.Layout(width="75%"))
        self.check_files = W.Checkbox(value=True, description="also check files exist",
                                      indent=False)
        val_btn = W.Button(description="Validate", button_style="warning")
        val_btn.on_click(lambda _: self.validate())
        save_btn = W.Button(description="Save", button_style="success")
        save_btn.on_click(lambda _: self.save())
        right = W.VBox([W.HTML("<b>Preview</b>"), self.preview,
                        W.HBox([val_btn, self.check_files]),
                        W.HBox([self.save_path, save_btn]), self.status],
                       layout=W.Layout(width="48%"))
        # the preview is a disabled textarea, which Jupyter greys out; force
        # solid black text on white (Chrome also needs -webkit-text-fill-color)
        css = W.HTML("<style>.atlas-wizard-mono textarea, .atlas-wizard-mono textarea:disabled{"
                     "font-family:monospace;font-size:12px;color:#000 !important;"
                     "-webkit-text-fill-color:#000 !important;opacity:1 !important;"
                     "background:#fff !important}</style>")
        return W.VBox([css, W.HBox([W.VBox([tabs], layout=W.Layout(width="52%")), right])])

    # ----------------------------------------------------------- behaviour
    def apply_preset(self, name):
        p = PRESETS[name]
        self._building = True
        try:
            self.preset_dd.value = name
            for k, v in self._initial.items():          # clean slate, pulsars kept
                self.w[k].value = v
            for k, v in p.items():
                if k == "use":
                    for s, on in v.items():
                        self.w[f"use_{s}"].value = on
                elif k in self.w:
                    self.w[k].value = v
        finally:
            self._building = False
        self._changed()

    def _changed(self, *_):
        if self._building:
            return
        self._visibility()
        self._rebuild_priors()
        try:
            cfg, priors = self.config()
            self.preview.value = _dump(cfg) + "\n# ---- " + self.w["priors_name"].value \
                + " ----\n" + _dump(priors)
            self.status.value = ""
        except Exception as exc:                       # bad YAML in a text box, etc.
            self.status.value = f"<span style='color:#b00'>{type(exc).__name__}: {exc}</span>"

    def _visibility(self):
        show = lambda key, on: setattr(self.w[key].layout, "display", None if on else "none")
        t = self.w["timing"].value
        src = self.w["source"].value
        show("npz_files", src == "atlas_npz")
        show("pickle", src == "enterprise_pickle")
        for k in ("params_mode", "parametrization", "marg_rest"):
            show(k, t == "direct")
        show("marg_only_listed", t == "standard")
        need_list = t == "direct" or (t == "standard" and self.w["marg_only_listed"].value)
        mode = self.w["params_mode"].value if t == "direct" else (
            self.w["params_mode"].value if self.w["params_mode"].value != "all" else "file")
        if t == "standard" and self.w["marg_only_listed"].value:
            show("params_mode", True)
        show("params_list", need_list and mode == "list")
        show("params_file", need_list and mode == "file")
        for k in ("params_key", "params_index"):
            show(k, need_list and mode == "file")
        for k in ("gtm_basis", "gtm_mean", "gtm_size", "gtm_index"):
            show(k, t == "target")
        for s, box in self._sig_boxes.items():
            box.layout.display = None if self.w[f"use_{s}"].value else "none"
        ev = self.w["evidence"].value
        show("switch_orf", ev == "switch")
        show("n_beta", ev != "none")
        show("beta_kind", ev != "none")
        show("ti_extra_orfs", ev == "ti")
        lay = self.w["layout"].value
        show("real_start", "real" in lay)
        show("real_stop", "real" in lay)

    # ---------------------------------------------------------- pulsars
    def _vars(self):
        return _yaml_or_empty(self.w["vars"].value, "Variables")

    def _scan(self):
        from ATLAS.runner.config import substitute_vars
        variables = dict(self._vars(), realization=self.w["real_start"].value)
        variables = substitute_vars(variables, variables)
        src = self.w["source"].value
        names = []
        try:
            if src == "atlas_npz":
                pat = substitute_vars(self.w["npz_files"].value, variables)
                rx = re.compile("^" + re.escape(pat).replace(r"\{psr\}", "(.+?)")
                                .replace(r"\*", ".*") + "$")
                for f in sorted(glob.glob(pat.replace("{psr}", "*"))):
                    m = rx.match(f)
                    nm = m.group(1) if (m and m.groups()) else Path(f).stem
                    if nm != "info":
                        names.append(nm)
            elif src == "par_tim":
                from ATLAS.runner.data_io import _par_name
                pars = sorted(glob.glob(substitute_vars(self.w["par_files"].value, variables)))
                excl = _split_list(self.w["file_exclude"].value)
                names = sorted({_par_name(p) for p in pars
                                if not any(x in Path(p).name for x in excl)})
            else:
                raise ValueError("scanning works for .npz and par/tim sources")
            self.psr_select.options = names
            self.status.value = f"found {len(names)} pulsars"
        except Exception as exc:
            self.status.value = f"<span style='color:#b00'>scan: {exc}</span>"

    def _pulsars(self):
        sel = list(self.psr_select.value) + _split_list(self.w["pulsars_extra"].value)
        seen, out = set(), []
        for p in sel:
            if p not in seen:
                out.append(p); seen.add(p)
        return out

    # ------------------------------------------------------------ priors
    def _present(self):
        return [s for s in ("unc", "cor", "dm", "chrom", "cquad", "sw", "det")
                if self.w[f"use_{s}"].value]

    def _prior_rows(self):
        """[(prior_key, label, param, default (lo, hi))] for the current model."""
        rows = []
        present = self._present()
        for s in ("unc", "cor", "dm", "chrom"):
            if s not in present:
                continue
            psd_name = self.w[f"{s}_psd"].value
            fn = self.registry.psd(psd_name, signal=s)
            fixed = ("gamma",) if (s == "cor" and self.w["cor_fix_gamma"].value) else ()
            try:
                names = self.registry.psd_param_names(fn, fixed=fixed, n_bins=self.w[f"{s}_bins"].value)
            except KeyError:
                names = self.registry.psd_param_names(fn, n_bins=self.w[f"{s}_bins"].value)
            key = f"{s}_{psd_name}"
            if names and all(n.startswith("halflog10_rho_") for n in names):
                names = ["halflog10_rho"]
            for n in names:
                rows.append((key, n, n, DEFAULT_BOUNDS.get(n, (0.0, 1.0))))
            if s == "cor":
                for n in self.registry.orf_param_names(self.registry.orf(self.w["cor_orf"].value)):
                    rows.append((key, f"orf: {n}", ("orf", n), DEFAULT_BOUNDS.get(n, (-1.0, 1.0))))
        if "sw" in present and not self.w["sw_fixed"].value:
            rows.append(("sw_ridge", "log10_amp", "log10_amp", DEFAULT_BOUNDS["log10_amp"]))
        if "det" in present:
            for n, b in CW_BOUNDS.items():
                rows.append(("cw", n, n, b))
        return rows

    def _rebuild_priors(self):
        W = self.W
        rows = self._prior_rows()
        sig = tuple((k, str(p)) for k, _, p, _ in rows)
        if getattr(self, "_prior_sig", None) == sig:
            return
        for (k, p), (lo, hi) in self._prior_widgets.items():
            self._prior_cache[(k, p)] = (lo.value, hi.value)
        self._prior_widgets = {}
        children, last = [], None
        for key, label, param, (lo, hi) in rows:
            if key != last:
                children.append(W.HTML(f"<b>{key}</b>"))
                last = key
            lo, hi = self._prior_cache.get((key, str(param)), (lo, hi))
            wlo = W.FloatText(value=lo, layout=W.Layout(width="110px"))
            whi = W.FloatText(value=hi, layout=W.Layout(width="110px"))
            for x in (wlo, whi):
                x.observe(self._changed, names="value")
            self._prior_widgets[(key, str(param))] = (wlo, whi)
            children.append(W.HBox([W.Label(label, layout=W.Layout(width="170px")), wlo, whi]))
        self.prior_box.children = children or [W.HTML("<i>nothing sampled</i>")]
        self._prior_sig = sig

    def _priors_dict(self):
        out = {}
        for key, label, param, _ in self._prior_rows():
            lo, hi = self._prior_widgets[(key, str(param))]
            pair = [float(lo.value), float(hi.value)]
            entry = out.setdefault(key, {})
            if isinstance(param, tuple):
                entry.setdefault("orf", {})[param[1]] = pair
            else:
                entry[param] = pair
        if "cor" in self._present():
            out.setdefault(f"cor_{self.w['cor_psd'].value}", {}).setdefault("orf", {})
        return out

    # ------------------------------------------------------------ config
    def _params_spec(self):
        mode = self.w["params_mode"].value
        if mode == "all":
            return None
        if mode == "list":
            return _split_list(self.w["params_list"].value)
        if mode == "from_pulsar":
            return "from_pulsar"
        spec = {"file": self.w["params_file"].value, "key": self.w["params_key"].value}
        idx = self.w["params_index"].value.strip()
        if idx:
            spec["index"] = int(idx)
        return spec

    def _user_freqs(self, key):
        v = (self.w[key].value or "").strip()
        if not v:
            return None
        if re.fullmatch(r"[\d.eE+\-,\s]+", v):
            return [float(x) for x in _split_list(v)]
        return v

    def basis_string(self):
        present = self._present()
        t = self.w["timing"].value
        shared = [s for s in ("cor", "unc") if s in present]
        if not shared:
            raise ValueError("the model needs 'unc' and/or 'cor'")
        rep = "unc" if "unc" in shared else "cor"
        seps = [s for s in SEPARATE_ORDER if s in present or (s == "gtm" and t == "target")]
        bs = ("ltm|" if t == "linear" else "") + "+".join(shared) + "->" + rep
        if seps:
            bs += " ; " + ", ".join(seps)
        return bs

    def config(self):
        """(run config dict, priors dict) from the current menus."""
        w = self.w
        variables = self._vars()
        t = w["timing"].value
        ev = w["evidence"].value
        lay = w["layout"].value
        present = self._present()
        psrs = self._pulsars()
        per_psr = lay in ("per_psr", "per_psr_real")
        per_real = lay in ("per_real", "per_psr_real")
        if per_psr:
            if not psrs:
                raise ValueError("'one run per pulsar' needs pulsars: Scan or type them on the Data tab")
            variables.setdefault("psr", psrs[0])
        if per_real:
            variables.setdefault("realization", int(w["real_start"].value))

        data = {"source": w["source"].value}
        if w["source"].value == "atlas_npz":
            data["npz_files"] = w["npz_files"].value
        if w["source"].value == "enterprise_pickle":
            data["pickle"] = w["pickle"].value
        data["pulsars"] = ["${psr}"] if per_psr else (psrs or "all")
        for k in ("par_files", "tim_files"):
            if w[k].value.strip():
                data[k] = w[k].value.strip()
        excl = _split_list(w["file_exclude"].value)
        if excl:
            data["file_exclude"] = excl
        if w["noise_dict"].value.strip():
            data["noise_dict"] = w["noise_dict"].value.strip()
        data["dm_ref_freq"] = float(w["dm_ref_freq"].value)
        data["cache_dir"] = w["cache_dir"].value

        timing = {"linear_timing": t == "linear", "marg_timing": t in ("standard", "direct")}
        cols = "all"
        params = self._params_spec() if (t == "direct" or (t == "standard" and w["marg_only_listed"].value)) else None
        if t == "standard" and w["marg_only_listed"].value:
            if params is None:
                raise ValueError("'marginalise only the listed parameters' needs a parameter list")
            cols = "sampled"
        if t == "direct":
            if w["marg_rest"].value:
                cols = "not_sampled"
            else:
                timing["marg_timing"] = False
        if t in ("direct",) or params is not None:
            timing["nonlinear"] = {"enabled": t == "direct", "params_to_sample": params}
            if t == "direct":
                timing["nonlinear"]["parametrization"] = w["parametrization"].value
        if cols != "all":
            timing["design_matrix"] = {"columns": cols, "keep_offset": True}

        stabilize = bool(w["stabilize"].value) and ev == "none"
        white = {"vary": bool(w["wn_vary"].value), "include_ecorr": w["ecorr"].value,
                 "stabilize_TNT": stabilize}

        fb = {"num_irn_bins": int(w["unc_bins"].value) if "unc" in present else None,
              "num_gwb_bins": int(w["cor_bins"].value) if "cor" in present else None,
              "num_dm_bins": int(w["dm_bins"].value) if "dm" in present else None,
              "num_chrom_bins": int(w["chrom_bins"].value) if "chrom" in present else None,
              "use_pulsar_tspan": bool(w["use_pulsar_tspan"].value)}
        if "det" in present:
            fb["num_det_bins"] = int(w["det_bins"].value)
        for s, key in (("unc", "irn_user_freqs"), ("cor", "gwb_user_freqs")):
            uf = self._user_freqs(f"{s}_user_freqs") if s in present else None
            if uf is not None:
                fb[key] = uf

        signals = {}
        for s in ("unc", "cor", "dm", "chrom"):
            if s in present:
                signals[s] = {"psd": w[f"{s}_psd"].value, "priors": f"{s}_{w[f'{s}_psd'].value}"}
        if "cor" in present:
            signals["cor"]["orf"] = w["cor_orf"].value
            if w["cor_fix_gamma"].value:
                signals["cor"]["fixed"] = {"gamma": float(w["cor_gamma"].value)}
        if "dm" in present and w["dm_varied"].value:
            signals["dm"]["varied_index"] = True
        if "chrom" in present:
            signals["chrom"]["chrom_index"] = float(w["chrom_index"].value)
            if w["chrom_varied"].value:
                signals["chrom"]["varied_index"] = True
                signals["chrom"]["index_bounds"] = [float(w["chrom_lo"].value), float(w["chrom_hi"].value)]
        if "cquad" in present:
            signals["cquad"] = {"log10_amp": float(w["cquad_amp"].value),
                                "project_timing": bool(w["cquad_project"].value)}
        if "sw" in present:
            signals["sw"] = {"geometry": w["sw_geometry"].value,
                             "settings_from_par": bool(w["sw_from_par"].value),
                             "defaults": {"kernel": w["sw_kernel"].value,
                                          "dt_days": float(w["sw_dt"].value),
                                          "interp_kind": "linear"}}
            if w["sw_fixed"].value:
                signals["sw"]["fixed_log10_sigma"] = float(w["sw_sigma"].value)
            else:
                signals["sw"]["priors"] = "sw_ridge"
        if t == "target":
            gi = w["gtm_index"].value.strip()
            signals["gtm"] = {"basis_pattern": w["gtm_basis"].value,
                              "mean_pattern": w["gtm_mean"].value or None,
                              "index": int(gi) if gi else None, "size": int(w["gtm_size"].value),
                              "psd_value": 1.0}
        if "det" in present:
            signals["det"] = {"waveform": w["det_waveform"].value, "priors": "cw"}
            if w["det_dists"].value.strip():
                data["pulsar_distances"] = w["det_dists"].value.strip()

        model = {"basis_string": self.basis_string(), "priors_file": w["priors_name"].value,
                 "signals": signals}

        sampler = {"marg_over_non_gwb": bool(w["marg_over_non_gwb"].value) or ev != "none",
                   "num_warmup": int(w["num_warmup"].value),
                   "num_samples": int(w["num_samples"].value),
                   "num_chains": int(w["num_chains"].value),
                   "chain_method": w["chain_method"].value, "seed": int(w["seed"].value),
                   "progress_bar": bool(w["progress_bar"].value),
                   "nuts": {"target_accept_prob": float(w["target_accept"].value),
                            "max_tree_depth": int(w["max_tree_depth"].value)}}
        if per_real:
            sampler["seed_offset_var"] = "realization"
        if ev == "switch":
            sampler["beta"] = None
            sampler["switch_overrides"] = {"model.signals.cor.orf": w["switch_orf"].value}
        elif ev == "ti":
            sampler["beta"] = None

        ray = {"num_cpus_per_run": int(w["cpus"].value), "num_gpus_per_run": float(w["gpus"].value)}
        if w["max_conc"].value > 0:
            ray["max_concurrent_runs"] = int(w["max_conc"].value)
        if w["ray_address"].value.strip():
            ray["address"] = w["ray_address"].value.strip()
        output = {"root": w["out_root"].value, "format": w["out_format"].value,
                  "save": {"red_coefficients": bool(w["save_coeff"].value)},
                  "skip_completed": bool(w["skip_completed"].value)}

        # runs / sweeps
        grid = {}
        if per_real:
            grid["vars.realization"] = {"range": [int(w["real_start"].value), int(w["real_stop"].value)]}
        if per_psr:
            grid["vars.psr"] = psrs
        if ev != "none":
            grid["sampler.beta"] = _beta_grid(w["n_beta"].value, w["beta_kind"].value)
        variants = [(w["run_name"].value or "run", {})]
        for vname, ov in _yaml_or_empty(w["variants"].value, "Extra variants").items():
            if not isinstance(ov, dict):
                raise ValueError(f"variant {vname!r}: give dotted-path overrides under it")
            variants.append((str(vname), ov))
        if ev == "ti":
            for orf in _split_list(w["ti_extra_orfs"].value):
                variants.append((orf.replace("_orf", ""), {"model.signals.cor.orf": orf}))
        runs, sweeps = [], []
        for vname, ov in variants:
            if grid:
                sw = {"name": vname}
                if ov:
                    sw["base_overrides"] = ov
                sw["grid"] = copy.deepcopy(grid)
                sweeps.append(sw)
            else:
                r = {"name": vname}
                if ov:
                    r["overrides"] = ov
                runs.append(r)

        cfg = {"config_version": 1}
        if variables:
            cfg["vars"] = variables
        cfg.update(ray=ray, output=output, data=data, timing=timing, white_noise=white,
                   frequency_bins=fb, model=model, sampler=sampler, runs=runs)
        if sweeps:
            cfg["sweeps"] = sweeps
        return cfg, self._priors_dict()

    # ------------------------------------------------------ validate / save
    def _resolved_runs(self, cfg, priors, config_dir):
        from ATLAS.runner.config import (DEFAULTS, _deep_merge, _check_keys, expand_runs)
        raw = copy.deepcopy(cfg)
        runs, sweeps = raw.pop("runs", None), raw.pop("sweeps", None)
        tmp = Path(tempfile.mkdtemp(prefix="atlas_wizard_")) / "priors.yaml"
        tmp.write_text(_dump(priors))
        raw["model"]["priors_file"] = str(tmp)
        _check_keys(raw, DEFAULTS)
        merged = _deep_merge(DEFAULTS, raw)
        merged["runs"] = runs or ([] if sweeps else [{"name": "run"}])
        merged["sweeps"] = sweeps or []
        merged["_config_path"] = str(Path(config_dir) / "wizard.yaml")
        merged["_config_dir"] = str(config_dir)
        return expand_runs(merged)

    def validate(self, cfg=None, priors=None):
        """Run the atlas-run checks on the current config; returns the errors."""
        from ATLAS.runner.config import validate_run, ConfigError
        try:
            if cfg is None:
                cfg, priors = self.config()
            config_dir = Path(self.save_path.value).expanduser().resolve().parent
            runs = self._resolved_runs(cfg, priors, config_dir)
        except Exception as exc:
            self.status.value = f"<span style='color:#b00'>{type(exc).__name__}: {exc}</span>"
            return [str(exc)]
        errs = []
        for r in runs:
            try:
                validate_run(r, check_files=self.check_files.value)
            except ConfigError as exc:
                errs.append(str(exc))
        names = [r["run"]["name"] for r in runs]
        head = f"{len(runs)} run(s): {', '.join(names[:6])}{' ...' if len(names) > 6 else ''}"
        if errs:
            uniq = []
            for e in errs:
                body = e.split("\n", 1)[-1]
                if body not in uniq:
                    uniq.append(body)
            self.status.value = (f"<b>{head}</b><br><span style='color:#b00'>{len(errs)} run(s) "
                                 "cannot run:</span><pre>" + "\n".join(uniq[:10]) + "</pre>")
        else:
            self.status.value = f"<b>{head}</b><br><span style='color:#080'>all valid</span>"
        return errs

    def save(self, path=None):
        """Write the config and its priors file (same folder).  Returns the config path."""
        path = Path(path or self.save_path.value).expanduser()
        try:
            cfg, priors = self.config()
        except Exception as exc:
            self.status.value = f"<span style='color:#b00'>{type(exc).__name__}: {exc}</span>"
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        ppath = path.parent / cfg["model"]["priors_file"]
        header = ("# Written by ATLAS.runner.wizard.  Check / run with\n"
                  f"#   atlas-run check {path.name}\n#   atlas-run run   {path.name}\n")
        path.write_text(header + _dump(cfg))
        ppath.write_text("# Prior bounds [low, high] for " + path.name + "\n" + _dump(priors))
        errs = self.validate(cfg, priors)
        self.status.value += f"<br>saved <code>{path}</code> and <code>{ppath.name}</code>" + \
            (" (with the problems above)" if errs else "")
        return path

    def _load_yaml(self, path):
        try:
            text = Path(path).expanduser().read_text()
            raw = yaml.safe_load(text)
            self.preview.value = text
            from ATLAS.runner.config import load_config, expand_runs, validate_run, ConfigError
            runs = expand_runs(load_config(path))
            errs = []
            for r in runs:
                try:
                    validate_run(r, check_files=self.check_files.value)
                except ConfigError as exc:
                    errs.append(str(exc))
            self.status.value = (f"loaded {path}: {len(runs)} run(s); "
                                 + ("<span style='color:#080'>all valid</span>" if not errs else
                                    "<span style='color:#b00'>problems:</span><pre>"
                                    + "\n".join(errs[:5]) + "</pre>"))
        except Exception as exc:
            self.status.value = f"<span style='color:#b00'>{type(exc).__name__}: {exc}</span>"
