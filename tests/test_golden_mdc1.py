"""The MDC1 golden run, as a gate.

Two tests. The fast one asserts that the *recorded* posterior in
``tests/golden/mdc1_36psr.json`` meets the physics and sampler-health criteria
-- it needs no GPU and no sampling, and it is what runs in CI. The slow one
re-runs the fit and compares, and is marked ``golden`` so it only runs when
asked:

    pytest -m golden

Regenerate the record with ``python tools/golden_mdc1.py``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "tests" / "golden" / "mdc1_36psr.json"

pytestmark = pytest.mark.skipif(
    not RECORD.exists(),
    reason="no golden record; run tools/golden_mdc1.py")


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD.read_text())


def test_gwb_amplitude_recovers_the_injection(record):
    g = record["posterior"]["gwb_log10_A"]
    truth = record["injected"]["gwb_log10_A"]
    pull = (g["mean"] - truth) / g["std"]
    assert abs(pull) < 3.0, f"gwb_log10_A pull {pull:+.2f} sigma"


def test_gwb_spectral_index_recovers_the_injection(record):
    g = record["posterior"]["gwb_gamma"]
    truth = record["injected"]["gwb_gamma"]
    pull = (g["mean"] - truth) / g["std"]
    assert abs(pull) < 3.0, f"gwb_gamma pull {pull:+.2f} sigma"


def test_posterior_is_informative(record):
    """An upper limit is not a detection. MDC1's background is strong enough
    that a working fit constrains the amplitude to well under a decade."""
    assert record["posterior"]["gwb_log10_A"]["std"] < 0.1
    assert record["posterior"]["gwb_gamma"]["std"] < 0.5


# --------------------------------------------------------------------------- #
#  Sampler health -- the assertions that actually caught the TCB failure
# --------------------------------------------------------------------------- #
#
# When MDC1 was being read as TDB instead of TCB, the residuals were a uniform
# hash over one pulse period and the fit returned gwb_log10_A ~ -8.5, five
# decades high. The chain reported 0 divergences and acceptance probability
# 0.76 -- both healthy. These three are what showed it.

def test_z_a_std_is_order_one(record):
    """The non-centred transform guarantees z ~ N(0, 1) a posteriori. A frozen
    chain shows ~1e-12 here."""
    assert 0.8 < record["health"]["z_a_std"] < 1.25


def test_step_size_is_not_collapsed(record):
    """A frozen chain adapts down to ~1e-13."""
    assert record["health"]["final_step_size"] > 1e-4


def test_not_the_frozen_chain_pathology(record):
    """Running to the tree-depth cap is *not* on its own a symptom.

    With `vary_white=True` this fit sits at the cap on 100% of iterations,
    because the EFAC/EQUAD ridge is a genuinely flat, prior-truncated direction
    -- that is real work, and `sigma_eff` mixes fine even though EFAC and EQUAD
    individually do not. The frozen-chain pathology is the *conjunction* of
    being at the cap with a collapsed step size and a degenerate `z_a`, so only
    the conjunction is gated. The cap fraction is recorded as a datum.
    """
    h = record["health"]
    at_cap = h["frac_at_tree_depth_cap"] > 0.9
    collapsed = h["final_step_size"] < 1e-6 or h["z_a_std"] < 0.5
    assert not (at_cap and collapsed), (
        f"frozen-chain signature: {h['frac_at_tree_depth_cap']*100:.0f}% at cap, "
        f"step size {h['final_step_size']:.1e}, z_a std {h['z_a_std']:.2e}")
    assert 0.0 <= h["frac_at_tree_depth_cap"] <= 1.0


def test_divergences_are_recorded_but_not_gated(record):
    """Deliberately weak: this quantity looked fine while the data was wrong,
    so it is kept as a datum rather than a criterion."""
    assert "n_divergent" in record["health"]


def test_config_is_pinned(record):
    """The recorded posterior is only meaningful at the configuration it was
    produced at, so changing the configuration must invalidate the record."""
    c = record["config"]
    assert c["model_string"] == "ltm|unc+cor->unc"
    assert (c["n_gwb"], c["n_irn"]) == (8, 15)
    assert c["linear_timing"] and not c["marg_timing"]
    assert not c["include_ecorr"]
    assert c["vary_white"]
    assert record["npsr"] == 36


@pytest.mark.golden
@pytest.mark.slow
@pytest.mark.gpu
def test_golden_run_reproduces():
    """Re-run the fit and compare against the record. Minutes on a GPU; the
    tool falls back to CPU, with a warning, when none is available."""
    env = dict(os.environ)
    if env.pop("ATLAS_TEST_CPU_DEFAULT", None):
        # conftest pins the in-process suite to CPU; the fit should use the GPU.
        env.pop("JAX_PLATFORMS", None)
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "golden_mdc1.py"), "--check"],
        cwd=ROOT, capture_output=True, text=True, env=env)
    for line in r.stderr.splitlines():
        if line.startswith("golden_mdc1: WARNING:"):
            warnings.warn(line.split("WARNING:", 1)[1].strip())
    assert r.returncode == 0, r.stdout + r.stderr
    shifts = [float(line.split("shift")[1].split("MC")[0])
              for line in r.stdout.splitlines() if "shift" in line]
    assert shifts, f"could not parse shifts from:\n{r.stdout}"
    assert max(shifts) < 4.0, f"posterior moved by {max(shifts):.2f} MC sigma"
