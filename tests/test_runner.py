"""The config-driven runner (``ATLAS.runner``, ``atlas-run``).

What is pinned:

* config semantics -- defaults, dotted-path overrides that REPLACE values,
  sweeps, run names, seeds, output directories;
* validation rejects, before anything runs, every combination ATLAS cannot
  run or would run wrongly (unknown keys, the ``ltm|`` / ``marg_timing``
  mismatch that gives NaN, missing bins, priors that do not cover exactly the
  sampled parameters, misordered blocks, ...);
* the model the runner builds IS the model ``ModelBuilder`` builds by hand:
  same log-density at the same point;
* execution: outputs, named columns, resume, and one failing run not taking
  the others down -- in-process, and through Ray when it is installed.

The datasets are the synthetic pulsars of ``harness``; no par/tim, PINT or JUG.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
import yaml
import jax.numpy as jnp

from . import harness as H
from ATLAS.runner import config as C
from ATLAS.runner import data_io as D
from ATLAS.runner.build import build_model
from ATLAS.runner.driver import plan_runs, run as run_all

TIGHT = 1e-12

PRIORS = {
    "irn_powerlaw": {"log10_A": [-20.0, -11.0], "gamma": [0.0, 7.0]},
    "gwb_powerlaw": {"log10_A": [-18.0, -11.0]},
    "gwb_powerlaw_free_gamma": {"log10_A": [-18.0, -11.0], "gamma": [0.0, 7.0]},
    "gwb_free_spectrum": {"halflog10_rho": [-9.0, -4.0]},
    "dm_powerlaw": {"log10_A": [-20.0, -11.0], "gamma": [0.0, 7.0]},
    "chrom_powerlaw": {"log10_A": [-20.0, -11.0], "gamma": [0.0, 7.0]},
    "sw_ridge": {"log10_amp": [-3.0, 2.0]},
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Synthetic pulsars, a matching white-noise dict and a priors file."""
    root = tmp_path_factory.mktemp("runner_data")
    psrs = H.load_psrs("synth", 3)
    with open(root / "psrs.pkl", "wb") as fh:
        pickle.dump(psrs, fh)
    nd = {}
    for p in psrs:
        for b in np.unique(p.backend_flags):
            nd[f"{p.name}_{b}_efac"] = H.EFAC
            nd[f"{p.name}_{b}_log10_t2equad"] = H.LOG10_EQUAD
            nd[f"{p.name}_{b}_log10_ecorr"] = H.LOG10_ECORR
    (root / "wn.json").write_text(json.dumps(nd))
    (root / "priors.yaml").write_text(yaml.safe_dump(PRIORS))
    return root


def write_config(dataset, tmp_path, runs, **sections):
    cfg = {
        "ray": {"num_cpus_per_run": 1, "num_gpus_per_run": 0},
        "output": {"root": str(tmp_path / "out")},
        "data": {"source": "enterprise_pickle", "pickle": str(dataset / "psrs.pkl"),
                 "noise_dict": str(dataset / "wn.json"), "cache_dir": str(tmp_path / "cache")},
        "model": {"priors_file": str(dataset / "priors.yaml"),
                  "signals": {"sw": {"geometry": "approx", "settings_from_par": False}}},
        "sampler": {"num_warmup": 5, "num_samples": 5, "seed": 3},
        "runs": runs,
    }
    for k, v in sections.items():
        cfg[k] = C._deep_merge(cfg.get(k, {}), v)
    path = tmp_path / "run_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def one_run(dataset, tmp_path, **overrides):
    path = write_config(dataset, tmp_path, [{"name": "r", "overrides": overrides}])
    return C.expand_runs(C.load_config(path))[0]


# --------------------------------------------------------------------------- #
#  1. Config semantics
# --------------------------------------------------------------------------- #

def test_defaults_fill_everything_not_given(dataset, tmp_path):
    r = one_run(dataset, tmp_path)
    assert r["frequency_bins"]["num_irn_bins"] == C.DEFAULTS["frequency_bins"]["num_irn_bins"]
    assert r["model"]["signals"]["cor"]["orf"] == "hd_orf"
    assert r["data"]["source"] == "enterprise_pickle"           # given value wins


def test_unknown_keys_are_rejected(dataset, tmp_path):
    path = write_config(dataset, tmp_path, [{"name": "r"}],
                        frequency_bins={"num_irn_bin": 30})
    with pytest.raises(C.ConfigError, match="num_irn_bin"):
        C.load_config(path)


def test_overrides_replace_values_and_reject_unknown_paths(dataset, tmp_path):
    path = write_config(dataset, tmp_path, [
        {"name": "a", "overrides": {"model.signals.cor.fixed": {"gamma": 4.0}}},
        {"name": "b", "overrides": {"model.signals.cor.fixed": {}}},
    ])
    a, b = C.expand_runs(C.load_config(path))
    assert a["model"]["signals"]["cor"]["fixed"] == {"gamma": 4.0}
    assert b["model"]["signals"]["cor"]["fixed"] == {}          # replaced, not merged
    bad = write_config(dataset, tmp_path, [{"name": "c", "overrides": {"model.signal.cor.orf": "x"}}])
    with pytest.raises(C.ConfigError, match="does not exist"):
        C.expand_runs(C.load_config(bad))


def test_runs_get_their_own_seed_and_directory(dataset, tmp_path):
    path = write_config(dataset, tmp_path, [
        {"name": "a"}, {"name": "b"}, {"name": "c", "overrides": {"sampler.seed": 99}}])
    a, b, c = C.expand_runs(C.load_config(path))
    assert (a["sampler"]["seed"], b["sampler"]["seed"], c["sampler"]["seed"]) == (3, 4, 99)
    assert len({r["run"]["output_dir"] for r in (a, b, c)}) == 3


def test_sweeps_expand_to_named_runs(dataset, tmp_path):
    path = write_config(dataset, tmp_path, [{"name": "base"}])
    cfg = C.load_config(path)
    cfg["sweeps"] = [{"name": "scan", "grid": {
        "model.signals.chrom.chrom_index": [3.0, 4.4],
        "sampler.marg_over_non_gwb": [False, True]}}]
    names = [r["run"]["name"] for r in C.expand_runs(cfg)]
    assert names[0] == "base" and len(names) == 5
    assert "scan__chrom_index=4.4__marg_over_non_gwb=True" in names


def test_duplicate_run_names_are_rejected(dataset, tmp_path):
    path = write_config(dataset, tmp_path, [{"name": "a"}, {"name": "a"}])
    with pytest.raises(C.ConfigError, match="duplicate"):
        C.expand_runs(C.load_config(path))


# --------------------------------------------------------------------------- #
#  2. Validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("overrides,message", [
    ({"model.basis_string": "unc+cor->unc ; dm", "timing.marg_timing": False}, "ltm"),
    ({"model.basis_string": "ltm|unc+cor->unc"}, "ltm"),
    ({"model.basis_string": "unc+cor->unc ; dm, xyz"}, "unknown signals"),
    ({"model.basis_string": "unc+cor->unc ; sw, chrom"}, "order"),
    ({"model.basis_string": "unc+cor+chrom->unc ; dm"}, "separate"),
    ({"frequency_bins.num_dm_bins": None}, "num_dm_bins"),
    ({"frequency_bins.num_gwb_bins": 40}, "num_gwb_bins must be <="),
    ({"model.signals.unc.psd": "powerlw"}, "unknown PSD"),
    ({"model.signals.cor.orf": "hd"}, "unknown ORF"),
    ({"model.signals.cor.fixed": {"gama": 4.0}}, "not parameters"),
    ({"model.signals.unc.priors": "nope"}, "not a key"),
    ({"model.signals.cor.priors": "gwb_powerlaw"}, "missing \\['gamma'\\]"),
    ({"model.signals.chrom.varied_index": True}, "needs 'chrom' or 'cquad'"),
    ({"model.signals.dm.varied_index": True}, "white_noise.vary"),
    ({"model.basis_string": "unc ; dm", "sampler.marg_over_non_gwb": True}, "needs a 'cor'"),
    ({"sampler.kind": "block_gibbs"}, "not supported"),
    ({"data.noise_dict": None}, "noise_dict"),
])
def test_validation_rejects(dataset, tmp_path, overrides, message):
    r = one_run(dataset, tmp_path, **overrides)
    with pytest.raises(C.ConfigError, match=message):
        C.validate_run(r)


def test_validation_reports_every_problem_at_once(dataset, tmp_path):
    r = one_run(dataset, tmp_path, **{"model.signals.unc.psd": "x", "model.signals.cor.orf": "y"})
    with pytest.raises(C.ConfigError) as exc:
        C.validate_run(r)
    assert "unknown PSD" in str(exc.value) and "unknown ORF" in str(exc.value)


def test_free_spectrum_bounds_broadcast_over_bins(dataset, tmp_path):
    r = one_run(dataset, tmp_path, **{"model.basis_string": "unc+cor->unc",
                                      "model.signals.cor.psd": "free_spectrum",
                                      "model.signals.cor.priors": "gwb_free_spectrum",
                                      "frequency_bins.num_gwb_bins": 5})
    _, res = C.validate_run(r)
    assert res["cor"]["names"] == [f"halflog10_rho_{i}" for i in range(5)]
    assert res["cor"]["lower"] == [-9.0] * 5 and res["cor"]["upper"] == [-4.0] * 5
    assert res["cor"]["psd"].__name__ == "gwb_free_spectrum"     # column-shaped for cor


def test_a_valid_vanilla_run_passes(dataset, tmp_path):
    r = one_run(dataset, tmp_path, **{"model.basis_string": "unc+cor->unc ; dm, chrom, cquad, sw",
                                      "model.signals.chrom.varied_index": True})
    parsed, res = C.validate_run(r)
    assert parsed["separate_names"] == ["dm", "chrom", "cquad", "sw"]
    assert set(res) == {"unc", "cor", "dm", "chrom", "sw"}


# --------------------------------------------------------------------------- #
#  3. The runner builds the hand-built model
# --------------------------------------------------------------------------- #

def test_runner_model_equals_harness_model(dataset, tmp_path):
    """Same data, string, bins, PSDs and bounds as ``H.build`` -> the same
    log-density at the same point."""
    r = one_run(dataset, tmp_path, **{
        "model.basis_string": "ltm|unc+cor->unc", "timing.marg_timing": False,
        "frequency_bins.num_irn_bins": 6, "frequency_bins.num_gwb_bins": 4,
        "model.signals.cor.priors": "gwb_powerlaw_free_gamma",
        "white_noise.include_ecorr": True, "data.pulsars": ["J0000+0000", "J0001+0001"]})
    parsed, res = C.validate_run(r)
    b = build_model(r, parsed, res)
    m = H.build(model_string="ltm|unc+cor->unc", orf_name="hd", n_irn=6, n_gwb=4)
    assert b.rn.model.get_param_names() == m.rn.model.get_param_names()
    assert np.array_equal(np.asarray(b.rn.model.lower_prior_lim_all),
                          np.asarray(m.rn.model.lower_prior_lim_all))
    hb = b.rn.get_helpers(reff=jnp.concat(b.data.raw_residuals), white_noise_params=b.wn_vec)
    red = m.red_params()
    z = jnp.asarray(np.random.default_rng(0).normal(size=(m.npsr, m.rn.nmodes)))
    got = float(b.rn.lnposterior_reparam(hb, red, z)[0])
    want = float(m.rn.lnposterior_reparam(m.helpers, red, z)[0])
    assert abs(got - want) < TIGHT * abs(want)


def test_runner_builds_the_vanilla_blocks(dataset, tmp_path):
    r = one_run(dataset, tmp_path, **{
        "model.basis_string": "unc+cor->unc ; dm, chrom, cquad, sw",
        "frequency_bins.num_irn_bins": 6, "frequency_bins.num_gwb_bins": 4,
        "frequency_bins.num_dm_bins": 5, "frequency_bins.num_chrom_bins": 4})
    parsed, res = C.validate_run(r)
    b = build_model(r, parsed, res)
    assert b.rn.extra_names == ["chrom", "cquad", "sw"]
    names = b.rn.model.get_param_names()
    assert sum(n.endswith("_sw_log10_amp") for n in names) == 3
    assert len(names) == b.rn.model.upper_prior_lim_all.shape[0]


# --------------------------------------------------------------------------- #
#  4. Inputs
# --------------------------------------------------------------------------- #

def test_match_toas_pairs_across_time_references():
    """Topocentric TDB vs barycentric arrival: same TOAs, offsets of minutes."""
    rng = np.random.default_rng(0)
    dst = np.sort(rng.uniform(0, 1e8, 200))
    src = dst + 400.0 * np.sin(2 * np.pi * dst / 3.15e7)       # Roemer-like shift
    perm = rng.permutation(200)
    vals = np.arange(200.0)
    got = D.match_toas(vals[perm], src[perm], dst)
    assert np.array_equal(got, vals)
    with pytest.raises(C.ConfigError, match="do not match"):
        D.match_toas(vals, src + 1e4, dst)


def test_tdsw_settings_from_par_override_defaults(dataset, tmp_path):
    par = tmp_path / "J0000+0000.par"
    par.write_text("PSRJ J0000+0000\nTDSWKERNEL MATERN\nTDSWLOGELL 1.5\n"
                   "TDSWNODE_0001 55000\nTDSWNODE_0002 56000\n")
    tim = tmp_path / "J0000+0000.tim"
    tim.write_text("FORMAT 1\n")
    r = one_run(dataset, tmp_path, **{"data.par_files": str(par), "data.tim_files": str(tim),
                                      "data.pulsars": ["J0000+0000", "J0001+0001"],
                                      "model.signals.sw.settings_from_par": True})
    kw = D.sw_kwargs_for(r, D.load_pulsars(r))
    assert kw["kernel"] == ["MATERN", "RIDGE"]                # par, then config default
    assert kw["log10_ell"] == [1.5, None]
    assert list(kw["nodes_mjd"]["J0000+0000"]) == [55000.0, 56000.0]
    assert kw["nodes_mjd"]["J0001+0001"] is None


def test_chrom_index_file_with_default(dataset, tmp_path):
    f = tmp_path / "chi.yaml"
    f.write_text("J0000+0000: 4.4\ndefault: 4.0\n")
    r = one_run(dataset, tmp_path, **{"model.signals.chrom.chrom_index": str(f)})
    idx = D.chrom_index_for(r, D.load_pulsars(r))
    assert idx == {"J0000+0000": 4.4, "J0001+0001": 4.0, "J0002+0002": 4.0}


# --------------------------------------------------------------------------- #
#  5. Execution
# --------------------------------------------------------------------------- #

RUNS = [
    {"name": "curn", "overrides": {"model.basis_string": "unc+cor->unc",
                                   "model.signals.cor.orf": "zero_orf",
                                   "frequency_bins.num_irn_bins": 5,
                                   "frequency_bins.num_gwb_bins": 3}},
    {"name": "vanilla_chi", "overrides": {
        "model.basis_string": "unc+cor->unc ; dm, chrom, cquad, sw",
        "model.signals.chrom.varied_index": True,
        "frequency_bins.num_irn_bins": 5, "frequency_bins.num_gwb_bins": 3,
        "frequency_bins.num_dm_bins": 4, "frequency_bins.num_chrom_bins": 3}},
]


def check_outputs(path):
    plan = plan_runs(path)
    for r, _ in plan:
        out = r["run"]["output_dir"]
        st = json.loads(open(f"{out}/status.json").read())
        assert st["state"] == "done", st.get("traceback")
        z = np.load(f"{out}/samples.npz")
        assert z["red_noise"].shape[-1] == len(z["red_noise__names"])
        assert ("chrom_gp_index" in z.files) == r["model"]["signals"]["chrom"]["varied_index"]
        summ = json.loads(open(f"{out}/summary.json").read())
        assert set(z["red_noise__names"]) <= set(summ["parameters"])
    return plan


def test_local_execution_outputs_and_resume(dataset, tmp_path):
    path = write_config(dataset, tmp_path, RUNS)
    res = run_all(path, executor="local")
    assert [r["state"] for r in res] == ["done", "done"]
    plan = check_outputs(path)
    assert all(a == "skip (completed)" for _, a in plan)
    assert [r["state"] for r in run_all(path, executor="local")] == ["skipped", "skipped"]


def test_a_failing_run_does_not_stop_the_others(dataset, tmp_path, monkeypatch):
    import ATLAS.runner.build as B
    real = B.build_model

    def flaky(run, parsed, resolved):
        if run["run"]["name"] == "curn":
            raise RuntimeError("boom")
        return real(run, parsed, resolved)

    monkeypatch.setattr(B, "build_model", flaky)
    path = write_config(dataset, tmp_path, RUNS)
    res = {r["run_name"]: r for r in run_all(path, executor="local")}
    assert res["curn"]["state"] == "failed" and "boom" in res["curn"]["error"]
    assert res["vanilla_chi"]["state"] == "done"
    # the failed run is retried next time, the finished one is not
    plan = dict((r["run"]["name"], a) for r, a in plan_runs(path))
    assert plan == {"curn": "run", "vanilla_chi": "skip (completed)"}


def test_ray_execution(dataset, tmp_path):
    ray = pytest.importorskip("ray")
    path = write_config(dataset, tmp_path, RUNS, ray={"max_concurrent_runs": 2})
    try:
        res = run_all(path, executor="ray")
    finally:
        ray.shutdown()
    assert sorted(r["state"] for r in res) == ["done", "done"]
    check_outputs(path)


def test_runner_imports_without_ray_jug_or_pint():
    """Ray, JUG, PINT and enterprise are imported lazily, only where needed."""
    from .test_imports import _import_with_blocked
    out = _import_with_blocked(["ray", "jug", "libstempo", "pint", "enterprise"],
                               mods=["ATLAS.runner", "ATLAS.runner.cli"])
    assert out.returncode == 0, out.stdout + out.stderr


def test_netcdf_output_is_inference_data(dataset, tmp_path):
    if C.netcdf_engine() is None:
        pytest.skip("no NetCDF engine (netCDF4, or h5netcdf with h5py)")
    xr = pytest.importorskip("xarray")
    path = write_config(dataset, tmp_path, RUNS[1:], output={"format": "netcdf"})
    assert [r["state"] for r in run_all(path, executor="local")] == ["done"]
    out = plan_runs(path)[0][0]["run"]["output_dir"]
    post = xr.open_dataset(f"{out}/posterior.nc", group="posterior")
    stats = xr.open_dataset(f"{out}/posterior.nc", group="sample_stats")
    assert post["red_noise"].dims == ("chain", "draw", "red_noise_param")
    assert "gwb_log10_A" in list(post["red_noise_param"].values)
    assert post["chrom_gp_index"].sizes["chrom_gp_index_param"] == 3
    assert stats["diverging"].dims == ("chain", "draw")
