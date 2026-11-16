"""A trained ``SASFlow`` used as the prior of the GWB spectrum parameters.

``model_maker`` samples every red-noise parameter (``red_noise``) from a
uniform box.  With a flow prior, the GWB block ``x_G`` of that vector gets the
flow's density instead, truncated to the same box and renormalised:

    pi(x_G) = q(x_G) / P_q(box)        inside the box,  0 outside,

where ``q`` is the flow density and ``P_q(box)`` the probability the flow puts
inside the box.  In the model this is one factor,

    ln q(x_G) - ln P_q(box) + sum ln(upper - lower),

the last term cancelling the uniform density the ``red_noise`` sample site
already contributes on those coordinates.  The prior is therefore a properly
normalised density, so evidences (fixed-beta TI as well as model-switch TI)
stay correct.  In a model-switch run both models share this prior and its
constants cancel.

Usage
-----
    from ATLAS.experimental.flow_prior import FlowPrior
    prior = FlowPrior.load("gwb_flow.npz", lower, upper,
                           flow_num_layers=4, hidden_size=128, num_bins=8)
    model_maker(..., gwb_prior_flow=prior)

``lower`` / ``upper`` are the uniform bounds of the GWB parameters, in the
order of ``super_sig.model.get_param_names()`` (for a free spectrum: one
``halflog10_rho`` per GWB bin, lowest frequency first).  The flow must have
been trained on vectors in exactly that order.
"""
from __future__ import annotations

import warnings

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

__all__ = ["FlowPrior", "gwb_param_slice"]


def gwb_param_slice(model):
    """(start, stop) of the GWB spectrum parameters inside ``red_noise``."""
    try:
        return int(model.gtm_end_idx), int(model.gwb_psd_end_idx)
    except AttributeError:
        raise ValueError("a GWB prior flow needs a common-process ('cor') block in the "
                         "model") from None


class FlowPrior:
    """Truncated, normalised flow density over the GWB parameter box."""

    def __init__(self, flow, lower, upper, mass_samples=200_000, chunk=50_000, seed=0):
        self.flow = flow
        self.lower = np.asarray(lower, float).ravel()
        self.upper = np.asarray(upper, float).ravel()
        self.D = self.lower.size
        if int(getattr(flow, "D", self.D)) != self.D:
            raise ValueError(f"the flow models {flow.D} parameters but the GWB block has "
                             f"{self.D}; train the flow on exactly the GWB parameters")
        if np.any(self.upper <= self.lower):
            raise ValueError("every upper bound must exceed its lower bound")
        self.log_volume = float(np.sum(np.log(self.upper - self.lower)))

        # P_q(box): Monte-Carlo fraction of flow draws inside the box
        key, inside, n = jr.PRNGKey(seed), 0, 0
        while n < mass_samples:
            key, sub = jr.split(key)
            m = min(chunk, mass_samples - n)
            x = np.asarray(flow.sample(m, key=sub))
            inside += int(np.sum(np.all((x > self.lower) & (x < self.upper), axis=1)))
            n += m
        if inside == 0:
            raise ValueError("no flow draw falls inside the prior box: the flow and the "
                             "GWB prior bounds do not overlap")
        self.mass = inside / n
        self.mass_err = float(np.sqrt(self.mass * (1 - self.mass) / n))
        self.log_mass = float(np.log(self.mass))
        if self.mass < 0.99:
            warnings.warn(f"only {100 * self.mass:.1f}% of the flow lies inside the GWB prior "
                          "box; the prior is truncated there (and renormalised). Widen the "
                          "bounds if the truncation is not intended.")

    # ------------------------------------------------------------------ #
    def log_density(self, x_gwb):
        """ln pi(x_G): the truncated, normalised prior (valid inside the box)."""
        return self.flow.log_prob(jnp.atleast_2d(x_gwb))[0] - self.log_mass

    def numpyro_factor(self, x_gwb):
        """Term to add to the model: replaces the uniform density on x_G by pi(x_G)."""
        return self.log_density(x_gwb) + self.log_volume

    def __repr__(self):
        return (f"FlowPrior(D={self.D}, mass in box={self.mass:.4f} "
                f"± {self.mass_err:.1e})")

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path, lower, upper, flow_num_layers=4, hidden_size=128,
             mlp_num_layers=2, num_bins=8, B=5.0, mass_samples=200_000, seed=0):
        """Rebuild a saved unconditional ``SASFlow`` and wrap it.

        The architecture arguments must match those the flow was trained
        with (they are not stored in the file); the weights and the SAS data
        normalizer are read from ``path`` (``SASFlow.save_params``).
        """
        from ATLAS.experimental.flows import SASFlow

        D = np.asarray(lower).size
        with np.load(path) as f:
            files = set(f.files)
            saved = [f[k].shape for k in sorted((k for k in files if k.startswith("leaf_")),
                                                key=lambda k: int(k[5:]))]
        if not any(k.startswith("norm_") for k in files):
            raise ValueError(f"{path}: no SAS normalizer stored; save the flow with "
                             "SASFlow.save_params")
        if any(k.startswith("ctx_") for k in files):
            raise ValueError(f"{path}: this is a joint/conditional flow; the GWB prior needs "
                             "an unconditional SASFlow trained on the GWB parameters only")
        dummy = np.random.default_rng(seed).normal(size=(max(64, 4 * D), D))
        flow = SASFlow(dummy, flow_num_layers=flow_num_layers, hidden_size=hidden_size,
                       mlp_num_layers=mlp_num_layers, num_bins=num_bins, B=B, seed=seed)
        built = [np.shape(l) for l in jax.tree_util.tree_leaves(flow.params)]
        if built != saved:
            raise ValueError(
                f"{path}: the stored weights do not match the architecture given "
                f"(flow_num_layers={flow_num_layers}, hidden_size={hidden_size}, "
                f"mlp_num_layers={mlp_num_layers}, num_bins={num_bins}, D={D}); use the "
                "values the flow was trained with")
        flow.load_params(path)
        return cls(flow, lower, upper, mass_samples=mass_samples, seed=seed)
