#!/usr/bin/env python
"""Report the *achieved* margin on each identity, not just pass/fail.

The test suite asserts tolerances; this prints how much room is left, which is
what the Stage 0 gate and the Stage 1 bit-identity harness need as their
characterised noise floors.  Run it on the machine whose floors you want.

    python tools/measure_margins.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import jax
import jax.numpy as jnp

from bench.harness import host_label
from tests import harness as H
from tests import reference as ref

rows = []


def record(name, value, tol, note=""):
    rows.append(dict(identity=name, achieved=float(value), asserted=float(tol), note=note))
    ok = "ok " if value < tol else "FAIL"
    print(f"  {ok} {name:46s} {value:9.2e}  (asserted < {tol:.0e})  {note}")


print("Identity margins")

# curn vs dense marginal
m = H.build(model_string="unc+cor->unc", linear_timing=False, orf_name="zero")
red = m.red_params(irn_overrides={1: (-14.7, 4.0)})
atlas = float(m.rn.ln_likelihood_curn(m.helpers, red))
T, N, Nfull, r = H.dense_bundle(m)
irn, _, gwb = H.psd_pieces(m, red)
phi = ref.build_phi(m.npsr, 2 * m.n_irn, 0, [0] * m.npsr, irn, gwb, orf=None)
exp = ref.marginal_logL(r, T, phi, Nfull) + ref.atlas_offset(r.size)
record("ln_likelihood_curn vs dense marginal", abs(atlas - exp) / abs(exp), 1e-12)

# reparam difference-in-z, both ORFs
for orf in ("zero", "hd"):
    mm = H.build(orf_name=orf)
    rp = mm.red_params(irn_overrides={1: (-14.7, 4.0)})
    Tb, _, Nf, rr = H.dense_bundle(mm)
    ph = H.global_phi(mm, rp)
    d = []
    for z in np.random.default_rng(7).normal(size=(6, mm.npsr, mm.rn.nmodes)):
        lp, c = mm.rn.lnposterior_reparam(mm.helpers, rp, jnp.asarray(z))
        d.append(float(lp) - ref.conditional_logL(rr, Tb, np.asarray(c).ravel(), ph, Nf))
    d = np.array(d)
    record(f"lnposterior_reparam vs dense joint [{orf}]",
           np.ptp(d) / abs(d.mean()), 1e-12, "difference-in-z")

# reparam absolute constant
mm = H.build(orf_name="zero")
rp = mm.red_params(irn_overrides={1: (-14.7, 4.0)})
Tb, Nb, Nf, rr = H.dense_bundle(mm)
z = np.random.default_rng(3).normal(size=(mm.npsr, mm.rn.nmodes))
lp, c = mm.rn.lnposterior_reparam(mm.helpers, rp, jnp.asarray(z))
half = ref.precision_logdet(Tb, Nb, H.phiinv_diag(mm, rp))
joint = ref.conditional_logL(rr, Tb, np.asarray(c).ravel(), H.global_phi(mm, rp), Nf)
n_a = mm.npsr * mm.rn.nmodes
pred = (0.5 * rr.size * np.log(2 * np.pi) + 0.5 * n_a * np.log(2 * np.pi)
        + 0.5 * sum(mm.tm_widths) * np.log(1e40))
record("reparam dropped-constant prediction", abs((float(lp) - (joint - half)) - pred) / pred,
       1e-9, f"= {pred:.6f}")

# z-invariance, zero ORF
v = np.array([float(mm.rn.lnposterior_reparam(mm.helpers, rp, jnp.asarray(zz))[0])
              + 0.5 * np.sum(zz ** 2)
              for zz in np.random.default_rng(11).normal(size=(12, mm.npsr, mm.rn.nmodes))])
record("z-invariance [zero orf]", np.ptp(v) / abs(v.mean()), 1e-12)

# partial_marg vs reparam
for npsr in (2, 3, 4):
    mp = H.build(npsr=npsr, orf_name="zero")
    rp2 = mp.red_params()
    rng = np.random.default_rng(2)
    zr = rng.normal(size=(npsr, mp.rn.nmodes))
    zp = rng.normal(size=(npsr, 2 * mp.n_gwb))
    vr = float(mp.rn.lnposterior_reparam(mp.helpers, rp2, jnp.asarray(zr))[0]) + 0.5 * np.sum(zr ** 2)
    vp = float(mp.rn.partial_marg_lnposterior(mp.helpers, rp2, jnp.asarray(zp))[0]) + 0.5 * np.sum(zp ** 2)
    record(f"partial_marg vs reparam [npsr={npsr}]", abs(vr - vp) / abs(vr), 1e-9)

# gradient vs finite differences
mg = H.build(orf_name="hd")
rpg = mg.red_params(irn_overrides={1: (-14.7, 4.0)})
for fn_name, nz in (("lnposterior_reparam", mg.rn.nmodes),
                    ("partial_marg_lnposterior", 2 * mg.n_gwb)):
    zz = jnp.asarray(np.random.default_rng(8).normal(size=(mg.npsr, nz)) * 0.1)
    fn = getattr(mg.rn, fn_name)
    f = lambda q: fn(mg.helpers, q, zz)[0]
    g = np.asarray(jax.grad(f)(rpg))
    worst = 0.0
    for i in range(rpg.size):
        st = jnp.zeros_like(rpg).at[i].set(1e-4)
        fd = (float(f(rpg + st)) - float(f(rpg - st))) / 2e-4
        worst = max(worst, abs(fd - g[i]) / max(abs(fd), abs(g[i]), 1.0))
    record(f"jax.grad vs central diff [{fn_name}]", worst, 1e-6)

# Real data: unequal timing widths, real backend structure, real epochs.
for fx, ec in (("mdc1_5", False), ("ng15_3", True)):
    if not H.fixture_available(fx):
        print(f"  -- {fx}: not generated, skipped")
        continue
    mr = H.build(npsr=3, model_string="unc+cor->unc", linear_timing=False,
                 orf_name="zero", fixture=fx, include_ecorr=ec)
    rr2 = mr.red_params()
    a2 = float(mr.rn.ln_likelihood_curn(mr.helpers, rr2))
    Tr, Nr_, Nfr, rr3 = H.dense_bundle(mr)
    ir, _, gw = H.psd_pieces(mr, rr2)
    phr = ref.build_phi(mr.npsr, 2 * mr.n_irn, 0, [0] * mr.npsr, ir, gw, orf=None)
    e2 = ref.marginal_logL(rr3, Tr, phr, Nfr) + ref.atlas_offset(rr3.size)
    record(f"ln_likelihood_curn vs dense [{fx}]", abs(a2 - e2) / abs(e2), 1e-9,
           f"{rr3.size} TOAs")

    mq = H.build(npsr=3, orf_name="zero", fixture=fx, include_ecorr=ec)
    rq = mq.red_params()
    rg = np.random.default_rng(21)
    zr2 = rg.normal(size=(mq.npsr, mq.rn.nmodes))
    zp2 = rg.normal(size=(mq.npsr, 2 * mq.n_gwb))
    vr2 = float(mq.rn.lnposterior_reparam(mq.helpers, rq, jnp.asarray(zr2))[0]) + 0.5 * np.sum(zr2 ** 2)
    vp2 = float(mq.rn.partial_marg_lnposterior(mq.helpers, rq, jnp.asarray(zp2))[0]) + 0.5 * np.sum(zp2 ** 2)
    record(f"partial_marg vs reparam [{fx}]", abs(vr2 - vp2) / abs(vr2), 1e-9,
           f"tm cols {mq.tm_widths}")

# Per model string: the corpus the Stage 1 bit-identity harness will run over.
print("\nModel-string corpus")
for case, kw in H.CORPUS.items():
    mc = H.build(orf_name="zero", **kw)
    rc = mc.red_params(irn_overrides={1: (-14.7, 4.0)})
    Dc = mc.det_params()
    rgc = np.random.default_rng(32)
    zrc = rgc.normal(size=(mc.npsr, mc.ncol))
    zpc = rgc.normal(size=(mc.npsr, 2 * mc.n_gwb))
    vrc = float(mc.rn.lnposterior_reparam(mc.helpers, rc, jnp.asarray(zrc), D_params=Dc)[0]) + 0.5 * np.sum(zrc ** 2)
    vpc = float(mc.rn.partial_marg_lnposterior(mc.helpers, rc, jnp.asarray(zpc), D_params=Dc)[0]) + 0.5 * np.sum(zpc ** 2)
    record(f"partial_marg vs reparam [{case}]", abs(vrc - vpc) / abs(vrc), 1e-9,
           f"{mc.rn.nmodes} cols")

out = Path(__file__).resolve().parent / "noise_floors.json"
out.write_text(json.dumps(dict(
    recorded=time.strftime("%Y-%m-%dT%H:%M:%S"),
    host=host_label(),
    platform=jax.default_backend(),
    jax=jax.__version__,
    numpy=np.__version__,
    rows=rows,
), indent=2))
print(f"\nwritten: {out}")
