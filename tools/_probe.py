#!/usr/bin/env python
"""Evaluate ATLAS's log-densities and gradients at a fixed point. Version-blind.

Run by ``tools/regress.py`` inside a git worktree, with that worktree ahead of
everything else on ``sys.path`` so ``import ATLAS`` resolves to *its* revision.

This file is deliberately self-contained: it imports nothing from ``tests/`` and
nothing from the worktree except the ``ATLAS`` package itself, because the
worktree's own copy of the test harness may be older, newer, or absent. All
data -- pulsar arrays, the Adaptus basis, the parameter vectors, the z draws --
arrives pre-computed in a single ``.npz`` written by the driver, so the two
revisions are guaranteed to be evaluated at an identical point.

The comparison code lives in the driver, not here, for the same reason: if the
harness were versioned, two revisions would be compared by two different
implementations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # The worktree must win over anything else, including the driver's repo.
    sys.path.insert(0, args.worktree)

    import numpy as np
    import jax
    import jax.numpy as jnp

    import ATLAS
    if Path(ATLAS.__file__).resolve().parent.parent != Path(args.worktree).resolve():
        raise RuntimeError(
            f"imported ATLAS from {ATLAS.__file__}, expected it under {args.worktree}"
        )

    from ATLAS.data import PTA_Data
    from ATLAS.model_builder import ModelBuilder
    from ATLAS.nMatrix.base import WhiteCov
    from ATLAS.psd_functions import powerlaw, hd_orf

    spec = json.loads(Path(args.spec).read_text())
    z = np.load(args.inputs, allow_pickle=False)

    class ProbePulsar:
        """The eight attributes ATLAS reads off a pulsar."""
        def __init__(self, i):
            self.name = str(z["names"][i])
            self.toas = z[f"psr{i}_toas"]
            self.residuals = z[f"psr{i}_residuals"]
            self.toaerrs = z[f"psr{i}_toaerrs"]
            self.freqs = z[f"psr{i}_freqs"]
            self.pos = z[f"psr{i}_pos"]
            self.Mmat = z[f"psr{i}_Mmat"]
            self.backend_flags = z[f"psr{i}_backend_flags"]
            self.pdist = tuple(z[f"psr{i}_pdist"].tolist())

    psrs = [ProbePulsar(i) for i in range(len(z["names"]))]
    n_gtm = int(spec["n_gtm"])
    n_dm = int(spec["n_dm"])
    n_det = int(spec.get("n_det", 0))

    adaptus = [z[f"adaptus{i}"] for i in range(len(psrs))] if n_gtm else None
    gtm_psd = jnp.asarray(z["gtm_psd"]) if n_gtm else None

    # A det block needs its bin count, waveform, bounds and D_params.  All of
    # them are passed only when there is one, so the probe still runs on
    # revisions that predate the deterministic API.
    if n_det:
        from ATLAS.signals.deterministic.det_signals import cw_delay_evolve_float64
        data_kw = {"num_det_bins": n_det}
        det_kw = {"det_delay_function": cw_delay_evolve_float64,
                  "det_parameter_bounds": z["det_bounds"]}
        D = tuple(jnp.asarray(z[k]) for k in ("det_params", "psr_phases", "psr_dists"))
        dk = {"D_params": D}
    else:
        data_kw, det_kw, dk = {}, {}, {}

    def zero_orf(angle):
        return jnp.zeros_like(jnp.asarray(angle))

    orf = hd_orf if spec["orf"] == "hd" else zero_orf

    data = PTA_Data(
        psrs,
        num_gwb_bins=int(spec["n_gwb"]), num_irn_bins=int(spec["n_irn"]),
        num_dm_bins=(n_dm or None), **data_kw,
        adaptus_basis=adaptus, adaptus_size=(n_gtm or None),
        fixed_white_noise_params=None,
        linear_timing=bool(spec["linear_timing"]),
        marg_timing=bool(spec["marg_timing"]),
        diag_white_cov=False, fixed_res=False,
        timfiles=None, parfiles=None, noise_dict=None, dm_ref_freq=1400,
    )
    WhiteCov(data=data, stabilize_TNT=bool(spec["stabilize_TNT"]),
             include_ecorr=bool(spec["include_ecorr"]))
    rn = ModelBuilder(data=data).make_red_noise(
        spec["model_string"], use_pulsar_tspan=False,
        irn_psd_function=powerlaw, gwb_psd_function=powerlaw, orf_function=orf,
        dm_psd_function=(powerlaw if n_dm else None),
        **det_kw,
        gt_psd_val=gtm_psd,
        irn_lower_bound_psd=jnp.array([-20.0, 0.0]),
        irn_upper_bound_psd=jnp.array([-11.0, 7.0]),
        gwb_lower_bound_psd=jnp.array([-18.0, 0.0]),
        gwb_upper_bound_psd=jnp.array([-11.0, 7.0]),
        dm_lower_bound_psd=(jnp.array([-20.0, 0.0]) if n_dm else None),
        dm_upper_bound_psd=(jnp.array([-11.0, 7.0]) if n_dm else None),
        upper_bound_orf=None, lower_bound_orf=None,
    )

    wn_vec = jnp.asarray(z["wn_vec"])
    red = jnp.asarray(z["red_params"])
    helpers = rn.get_helpers(reff=jnp.concat(data.raw_residuals),
                             white_noise_params=wn_vec)
    TNT, TNr, rNr, logdet_N = helpers

    out = {
        "TNT": np.asarray(TNT), "TNr": np.asarray(TNr),
        "rNr": np.asarray(rNr), "logdet_N": np.asarray(logdet_N),
        "nmodes": np.asarray(rn.nmodes),
        "param_names": np.array(rn.model.get_param_names(), dtype="<U64"),
    }

    # z whitens the stochastic columns only; a det block's are not reparameterised.
    n_reparam = rn.nmodes - (rn.det_signal.num_coeff_det if n_det else 0)
    z_reparam = jnp.asarray(z["z_unit"][:, :n_reparam])
    z_gwb = jnp.asarray(z["z_unit"][:, :2 * int(spec["n_gwb"])])

    lp, coeff = rn.lnposterior_reparam(helpers, red, z_reparam, **dk)
    out["reparam_logp"] = np.asarray(lp)
    out["reparam_coeff"] = np.asarray(coeff)
    gr, gz = jax.grad(
        lambda q, zz: rn.lnposterior_reparam(helpers, q, zz, **dk)[0], argnums=(0, 1)
    )(red, z_reparam)
    out["reparam_grad_red"] = np.asarray(gr)
    out["reparam_grad_z"] = np.asarray(gz)
    if n_det:
        out["reparam_grad_det"] = np.asarray(jax.grad(
            lambda c: rn.lnposterior_reparam(helpers, red, z_reparam,
                                             D_params=(c, *D[1:]))[0])(D[0]))

    pm, g = rn.partial_marg_lnposterior(helpers, red, z_gwb, **dk)
    out["pmarg_logp"] = np.asarray(pm)
    out["pmarg_g"] = np.asarray(g)
    pgr, pgz = jax.grad(
        lambda q, zz: rn.partial_marg_lnposterior(helpers, q, zz, **dk)[0], argnums=(0, 1)
    )(red, z_gwb)
    out["pmarg_grad_red"] = np.asarray(pgr)
    out["pmarg_grad_z"] = np.asarray(pgz)

    # The gradient through the helper rebuild -- the vary_white=True hot path.
    def joint(q, zz, ww):
        hh = rn.get_helpers(reff=jnp.concat(data.raw_residuals), white_noise_params=ww)
        return rn.lnposterior_reparam(hh, q, zz, **dk)[0]
    jr, jz, jw = jax.grad(joint, argnums=(0, 1, 2))(red, z_reparam, wn_vec)
    out["joint_grad_red"] = np.asarray(jr)
    out["joint_grad_z"] = np.asarray(jz)
    out["joint_grad_wn"] = np.asarray(jw)

    try:
        out["curn_logp"] = np.asarray(rn.ln_likelihood_curn(helpers, red))
    except Exception as exc:                      # not every layout supports it
        out["curn_error"] = np.array(f"{type(exc).__name__}: {exc}"[:400])

    np.savez(args.out, **out)
    print(f"probe ok: {Path(args.out).name}  ({len(out)} arrays)")


if __name__ == "__main__":
    main()
