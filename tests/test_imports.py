"""The likelihood core must import on a bare environment.

`jug` is not installable from PyPI, `libstempo` needs a tempo2 runtime, and
`pint` is only needed to read par/tim files. None of them is required to build a
model, evaluate a likelihood or take a gradient -- and CI depends on that being
true, since it installs none of them.

This is checked in a subprocess with a `sys.meta_path` blocker rather than
in-process, because by the time this test runs the other tests have already
imported half the package and the import would be a no-op.

Regression test: `ATLAS.model_builder` used to import
`ATLAS.signals.timing.base` at module level, which pulls in JUG, so the whole
construction path needed a package that cannot be pip-installed. Local runs did
not catch it because JUG was installed in the dev environment; CI did.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

CORE = [
    "ATLAS.data",
    "ATLAS.model",
    "ATLAS.model_builder",
    "ATLAS.psd_functions",
    "ATLAS.nMatrix.base",
    "ATLAS.signals.signals_utils",
    "ATLAS.signals.factorized.base",
    "ATLAS.signals.factorized.chromatic",
    "ATLAS.signals.correlated.base",
    "ATLAS.signals.deterministic.base",
]

SCRIPT = """
import sys, importlib.abc
BLOCKED = set({blocked!r}.split(","))

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked by the test: " + name)
        return None

sys.meta_path.insert(0, Blocker())
sys.path.insert(0, {root!r})

failed = []
for mod in {mods!r}:
    try:
        __import__(mod)
    except Exception as exc:
        failed.append("%s -> %s: %s" % (mod, type(exc).__name__, exc))
if failed:
    print("\\n".join(failed))
    raise SystemExit(1)
print("ok")
"""


def _import_with_blocked(blocked, mods=CORE):
    code = textwrap.dedent(SCRIPT).format(
        blocked=",".join(blocked), root=str(ROOT), mods=mods)
    return subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                          capture_output=True, text=True,
                          env={"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin",
                               "HOME": str(Path.home())})


@pytest.mark.parametrize("blocked", [
    pytest.param(("jug",), id="no-jug"),
    pytest.param(("libstempo",), id="no-libstempo"),
    pytest.param(("pint",), id="no-pint"),
    pytest.param(("enterprise",), id="no-enterprise"),
    pytest.param(("jug", "libstempo", "pint", "enterprise"), id="none-of-them"),
])
def test_core_imports_without(blocked):
    r = _import_with_blocked(blocked)
    assert r.returncode == 0, (
        f"blocking {blocked} broke the core:\n{r.stdout}\n{r.stderr[-2000:]}")


def test_pulsar_module_is_the_one_that_needs_them():
    """Stated as a fact rather than assumed: ATLAS.pulsar is where the
    unpip-installable dependencies live, and it is never on the test path."""
    r = _import_with_blocked(("jug",), mods=["ATLAS.pulsar"])
    assert r.returncode != 0
    assert "jug" in r.stdout