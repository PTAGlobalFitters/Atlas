"""Training-set generation in ATLAS.experimental.astro, checked against Pandora.

The reference is Pandora's notebook code, copied into
tests/pandora_reference/notebook_trainset.py. Identity tests compare with
``np.array_equal``: the port is meant to be exact, not close.

holodeck's GWB realizations are unseeded in Pandora, so both sides run inside
``seeded_holodeck_rng`` with the same seed. That fixes the random numbers
without changing the code under comparison.

Tests that need holodeck skip without the ``[astro]`` extra; the file-format
tests do not need it.
"""
import json
import os

import numpy as np
import pytest

from ATLAS.experimental import astro
from tests.pandora_reference import notebook_trainset as ref

# Small enough to run in seconds; the code paths are the same as at full size.
SAM_SHAPE = 10
N_REAL = 40
N_FREQS = 5
YRS = 20
N_DRAWS = 3
SEED = 11


def _needs_holodeck():
    return pytest.importorskip("holodeck")


@pytest.fixture(scope="module")
def tspan():
    _needs_holodeck()
    from holodeck.constants import YR
    return YRS * YR


@pytest.fixture(scope="module")
def pspace():
    _needs_holodeck()
    import logging
    import holodeck
    holodeck.log.setLevel(logging.WARNING)
    return astro.pandora_phenom_param_space(nsamples=N_DRAWS, sam_shape=SAM_SHAPE, seed=SEED)


@pytest.fixture(scope="module")
def pandora_dir(tmp_path_factory, pspace):
    """Pandora's notebook pipeline, run on the same draws and seeds."""
    d = str(tmp_path_factory.mktemp("pandora"))
    theta = ref.lhs_draws(N_DRAWS, SEED)
    doit = ref.make_doit(theta, d, SAM_SHAPE, N_REAL, YRS, N_FREQS)
    for rr in range(N_DRAWS):
        with astro.seeded_holodeck_rng(SEED + rr):
            doit(rr)
    ref.combine(d, YRS, N_REAL, N_FREQS)
    ref.prepare(d, theta, YRS)
    return d


@pytest.fixture(scope="module")
def atlas_set(tmp_path_factory, pspace, tspan):
    d = str(tmp_path_factory.mktemp("atlas"))
    ts = astro.generate_training_set(pspace, d, tspan, n_freqs=N_FREQS, n_real=N_REAL,
                                     seed=SEED)
    ts.save_normalized(d, ts.metadata["tag"])
    return d, ts


# ---- identity with Pandora ------------------------------------------------------

@pytest.mark.astro
def test_lhs_draws_match_pandora(pspace):
    assert pspace.param_names == list(astro.PANDORA_PHENOM_PARAM_NAMES)
    big = astro.pandora_phenom_param_space(nsamples=200, sam_shape=SAM_SHAPE, seed=3)
    assert np.array_equal(big.param_samples, ref.lhs_draws(200, 3))


@pytest.mark.astro
def test_phenom_defaults_match_pandora():
    assert set(astro.PANDORA_PHENOM_DEFAULTS) == set(ref.params)
    fixed = set(ref.params) - set(astro.PANDORA_PHENOM_PARAM_NAMES)
    assert all(astro.PANDORA_PHENOM_DEFAULTS[k] == ref.params[k] for k in fixed)


@pytest.mark.astro
@pytest.mark.parametrize("name", [f"{i}_{YRS}yrs.npy" for i in range(N_DRAWS)]
                         + [f"gwb_spectrum_samples_{YRS}yrs.npy",
                            f"gwb_spectrum_samples_{YRS}yrs_normalized.npy",
                            f"ast_spectrum_samples_{YRS}yrs_normalized.npy"])
def test_files_match_pandora(pandora_dir, atlas_set, name):
    d, _ = atlas_set
    a, p = np.load(os.path.join(d, name)), np.load(os.path.join(pandora_dir, name))
    assert a.shape == p.shape
    assert np.array_equal(a, p)


@pytest.mark.astro
def test_mapping_matches_pandora(pandora_dir, atlas_set):
    d, ts = atlas_set
    name = f"gwb_spectrum_samples_{YRS}yrs_mapping_data.npy.npz"
    a, p = np.load(os.path.join(d, name)), np.load(os.path.join(pandora_dir, name))
    assert set(a.files) == set(p.files) == {"B", "mean", "half_range"}
    for k in p.files:
        assert a[k].dtype == p[k].dtype
        assert np.array_equal(a[k], p[k])


@pytest.mark.astro
def test_in_memory_normalize_matches_saved(atlas_set):
    d, ts = atlas_set
    rho_n, ast_n, mapping = ts.normalize()
    assert np.array_equal(rho_n, np.load(os.path.join(d, f"gwb_spectrum_samples_{YRS}yrs_normalized.npy")))
    assert np.array_equal(ast_n, np.load(os.path.join(d, f"ast_spectrum_samples_{YRS}yrs_normalized.npy")))
    # Extremes land on +-B up to rounding (also true in Pandora)
    assert np.isclose(np.abs(rho_n).max(), 5) and np.isclose(np.abs(ast_n).max(), 5)


def _write_library(path, tspan, dtype, n_samp=4, n_f=8, n_real=30, seed=0):
    import h5py
    from holodeck import utils as hutils

    rng = np.random.default_rng(seed)
    gwb = 10 ** rng.uniform(-16, -14, size=(n_samp, n_f, n_real))
    gwb[0, 1, :3] = 0.0           # empty bins are floored to 1e-20
    gwb[2, 0, 5] = 1e-22
    cents, edges = hutils.pta_freqs(tspan, n_f)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("fobs_cents", data=cents)
        h5.create_dataset("fobs_edges", data=edges)
        h5.create_dataset("sample_params", data=rng.normal(size=(n_samp, 3)))
        h5.create_dataset("gwb", data=gwb.astype(dtype))
        h5.attrs["param_names"] = np.array(["a", "b", "c"]).astype("S")


@pytest.mark.astro
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_holodeck_library_matches_pandora_legacy(tmp_path, tspan, dtype):
    pytest.importorskip("h5py")
    path = str(tmp_path / "lib.hdf5")
    _write_library(path, tspan, dtype)
    chain, names = ref.legacy_library_chain(path, N_FREQS, tspan)
    ts = astro.load_holodeck_library(path, N_FREQS, tspan=tspan)
    assert ts.param_names == names
    assert np.array_equal(ts.log10_rho, chain[..., -N_FREQS:])
    assert np.array_equal(ts.params, chain[:, 0, :-N_FREQS])
    # tspan inferred from the library's bin width
    assert np.isclose(astro.load_holodeck_library(path, N_FREQS).tspan, tspan, rtol=1e-12)


# ---- behaviour --------------------------------------------------------------------

@pytest.mark.astro
def test_freqs_options(tspan):
    f, edges = astro.gwb_frequencies(tspan, 14)
    # i/T, which is also the centre of holodeck's bins
    assert np.allclose(f, np.arange(1, 15) / tspan, rtol=1e-15, atol=0)
    assert np.allclose(edges, (np.arange(15) + 0.5) / tspan, rtol=1e-15, atol=0)
    custom = np.linspace(1e-9, 5e-9, 14)
    assert np.array_equal(astro.gwb_frequencies(tspan, 14, custom)[0], custom)
    with pytest.raises(ValueError):
        astro.gwb_frequencies(tspan, 14, custom[:3])
    with pytest.raises(ValueError):
        astro.gwb_frequencies(tspan, 14, "centers")


@pytest.mark.astro
def test_freqs_array_changes_conversion(tmp_path, pspace, tspan):
    f = astro.gwb_frequencies(tspan, N_FREQS)[0] * 2
    a = astro.simulate_log10_rho(pspace, pspace.param_samples[0], tspan, N_FREQS, N_REAL, seed=1)
    b = astro.simulate_log10_rho(pspace, pspace.param_samples[0], tspan, N_FREQS, N_REAL,
                                 freqs=f, seed=1)
    # log10_rho ~ -1.5 log10 f at fixed hc
    assert np.allclose(b - a, -1.5 * np.log10(2), atol=1e-12)


@pytest.mark.astro
def test_seed_reproducible_across_workers(tmp_path, pspace, tspan, atlas_set):
    _, serial = atlas_set
    par = astro.generate_training_set(pspace, str(tmp_path), tspan, n_freqs=N_FREQS,
                                      n_real=N_REAL, seed=SEED, n_jobs=2)
    assert np.array_equal(par.log10_rho, serial.log10_rho)
    a = astro.simulate_log10_rho(pspace, pspace.param_samples[0], tspan, N_FREQS, N_REAL, seed=5)
    b = astro.simulate_log10_rho(pspace, pspace.param_samples[0], tspan, N_FREQS, N_REAL, seed=6)
    assert not np.array_equal(a, b)


@pytest.mark.astro
def test_resume_skips_existing_draws(tmp_path, pspace, tspan):
    d = str(tmp_path)
    kw = dict(n_freqs=N_FREQS, n_real=N_REAL, seed=SEED)
    astro.generate_training_set(pspace, d, tspan, indices=[0, 1], combine=False, **kw)
    first = {i: os.stat(os.path.join(d, f"{i}_{YRS}yrs.npy")).st_mtime_ns for i in (0, 1)}
    ts = astro.generate_training_set(pspace, d, tspan, **kw)
    assert {i: os.stat(os.path.join(d, f"{i}_{YRS}yrs.npy")).st_mtime_ns for i in (0, 1)} == first
    assert list(ts.draw_indices) == [0, 1, 2]
    with pytest.raises(ValueError, match="different draws"):
        astro.generate_training_set(pspace, d, tspan, params=pspace.param_samples + 1, **kw)


@pytest.mark.astro
def test_nonfinite_draw_is_dropped_and_recorded(tmp_path, pspace, tspan, monkeypatch):
    real = astro.simulate_log10_rho

    def fake(pspace, params, *args, **kwargs):
        rho = real(pspace, params, *args, **kwargs)
        if np.array_equal(params, pspace.param_samples[1]):
            rho[0, 0] = -np.inf
        return rho

    monkeypatch.setattr(astro, "simulate_log10_rho", fake)
    d = str(tmp_path)
    ts = astro.generate_training_set(pspace, d, tspan, n_freqs=N_FREQS, n_real=N_REAL, seed=SEED)
    assert list(ts.draw_indices) == [0, 2]
    assert np.array_equal(ts.params, pspace.param_samples[[0, 2]])
    with open(os.path.join(d, f"trainset_{YRS}yrs_metadata.json")) as fh:
        assert json.load(fh)["bad_indices"] == [1]


# ---- file format (no holodeck needed) --------------------------------------------

def _fake_pandora_dir(d, indices, n_draws=5, n_f=3, n_real=7, tag="15yrs"):
    rng = np.random.default_rng(0)
    params = rng.normal(size=(n_draws, 2)) + 3
    np.save(os.path.join(d, astro.PARAMS_FILE), params)
    for i in indices:
        np.save(os.path.join(d, f"{i}_{tag}.npy"), rng.normal(size=(n_f, n_real)) - 8)
    return params


def test_combine_keeps_params_aligned_after_dropped_draws(tmp_path):
    """Pandora pairs spectra with ``theta[:n_files]`` (cell 28), which would put
    draw 2's spectrum on draw 1's parameters here. Atlas pairs by draw index."""
    d = str(tmp_path)
    params = _fake_pandora_dir(d, [0, 2, 3, 4])
    ts = astro.combine_training_set(d, "15yrs")
    assert list(ts.draw_indices) == [0, 2, 3, 4]
    assert np.array_equal(ts.params, params[[0, 2, 3, 4]])
    for row, i in enumerate(ts.draw_indices):
        assert np.array_equal(ts.log10_rho[row], np.load(os.path.join(d, f"{i}_15yrs.npy")).T)


def test_from_pandora_files_without_per_draw_files(tmp_path):
    """A directory holding only Pandora's combined file falls back to rows 0..n-1."""
    d = str(tmp_path)
    params = _fake_pandora_dir(d, [0, 1, 2])
    astro.combine_training_set(d, "15yrs")
    for i in range(3):
        os.remove(os.path.join(d, f"{i}_15yrs.npy"))
    ts = astro.TrainingSet.from_pandora_files(d, "15yrs")
    assert np.array_equal(ts.params, params[:3])
    assert ts.param_names == ["param_0", "param_1"]


def test_save_normalized_streams_in_chunks(tmp_path):
    d = str(tmp_path)
    _fake_pandora_dir(d, range(5))
    ts = astro.combine_training_set(d, "15yrs")
    ts.save_normalized(d, "15yrs", chunk=2)
    rho_n, ast_n, mapping = ts.normalize()
    assert np.array_equal(np.load(os.path.join(d, "gwb_spectrum_samples_15yrs_normalized.npy")), rho_n)
    assert np.array_equal(np.load(os.path.join(d, "ast_spectrum_samples_15yrs_normalized.npy")), ast_n)
    assert np.isclose(np.abs(rho_n).max(), 5) and np.isclose(np.abs(ast_n).max(), 5)


def test_normalization_rejects_exact_zeros(tmp_path):
    ts = astro.TrainingSet(params=np.array([[0.0, 1.0]]), log10_rho=-np.ones((1, 2, 3)),
                           param_names=["a", "b"])
    with pytest.raises(ValueError, match="zeros"):
        ts.normalization()
