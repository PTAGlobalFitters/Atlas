"""Thermodynamic integration with fixed-beta chains.

Target at each beta (``model_maker(..., marg_over_non_gwb=True, beta=b)``)::

    q_beta(theta, g) = pi(theta) N(g; 0, phi_G(theta)) L_partial(r | g, theta)^beta

where ``L_partial`` is the likelihood with every non-GWB coefficient
(timing, IRN, DM, ...) integrated out analytically and ``g`` are the GWB
coefficients.  q_0 is normalised (the prior), q_1 is the posterior, so::

    ln Z = int_0^1 E_beta[ ln L_partial ] d beta

Each beta is an independent NUTS run; ``lnL_partial`` is recorded per draw.
The integral is a trapezoid over the beta grid:

    sampling error      sqrt( sum_j w_j^2 sd_j^2 / ESS_j )
    discretisation      |I(all points) - I(every other point)| / 3   (Richardson)

``propose_betas`` places new points with density proportional to the spread
sd(beta) of ln L_partial (Neyman allocation; arXiv:2610.24354 Sec. 3), which
is also where E_beta[ln L] changes fastest (d/dbeta E = Var).

ln Z is missing the data constant -(n_toa/2) ln 2 pi, which cancels in a
Bayes factor between models fitted to the same residuals.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

__all__ = ["TIPoint", "TIResult", "ti_estimate", "propose_betas", "initial_betas",
           "bayes_factor", "run_fixed_beta", "point_from_samples"]


@dataclass
class TIPoint:
    """One fixed-beta chain, reduced."""
    beta: float
    mean: float          # E_beta[ln L_partial]
    std: float           # sd of ln L_partial under q_beta
    n_eff: float         # effective sample size of the chain(s)
    n: int = 0
    source: str = ""


@dataclass
class TIResult:
    ln_z: float
    err_sampling: float
    err_discretisation: float
    betas: list = field(default_factory=list)
    means: list = field(default_factory=list)
    stds: list = field(default_factory=list)
    n_effs: list = field(default_factory=list)
    weights: list = field(default_factory=list)

    @property
    def err(self):
        return float(np.hypot(self.err_sampling, self.err_discretisation))

    def as_dict(self):
        d = asdict(self)
        d["err"] = self.err
        return d

    def __str__(self):
        return (f"ln Z = {self.ln_z:.4f} +/- {self.err:.4f} "
                f"(sampling {self.err_sampling:.4f}, discretisation "
                f"{self.err_discretisation:.4f}; {len(self.betas)} beta)")


# --------------------------------------------------------------------------- #
#  reduction of one chain
# --------------------------------------------------------------------------- #

def point_from_samples(beta, lnL, source=""):
    """``TIPoint`` from the ``lnL_partial`` draws of one run, shape
    (draws,) or (chains, draws)."""
    x = np.asarray(lnL, float)
    if x.ndim == 1:
        x = x[None]
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{source or 'chain'} at beta={beta}: non-finite lnL_partial")
    try:
        from numpyro.diagnostics import effective_sample_size
        n_eff = float(effective_sample_size(x)) if x.shape[1] > 3 else float(x.size)
    except Exception:
        n_eff = float(x.size)
    n_eff = float(np.clip(n_eff, 1.0, x.size))
    return TIPoint(float(beta), float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else 0.0,
                   n_eff, int(x.size), source)


# --------------------------------------------------------------------------- #
#  integral and errors
# --------------------------------------------------------------------------- #

def _merge(points):
    """Sort by beta; chains at the same beta are pooled."""
    by = {}
    for p in points:
        by.setdefault(round(float(p.beta), 12), []).append(p)
    out = []
    for b, ps in sorted(by.items()):
        if len(ps) == 1:
            out.append(ps[0]); continue
        n = np.array([p.n for p in ps], float)
        n = np.where(n > 0, n, 1.0)
        m = np.sum(n * [p.mean for p in ps]) / n.sum()
        var = np.sum(n * ([p.std ** 2 for p in ps] + (np.array([p.mean for p in ps]) - m) ** 2)) / n.sum()
        out.append(TIPoint(b, float(m), float(np.sqrt(var)), float(sum(p.n_eff for p in ps)),
                           int(n.sum()), "+".join(p.source for p in ps)))
    return out


def _trapezoid_weights(b):
    b = np.asarray(b, float)
    w = np.zeros_like(b)
    d = np.diff(b)
    w[:-1] += d / 2
    w[1:] += d / 2
    return w


def ti_estimate(points, require_endpoints=True):
    """ln Z from fixed-beta chains (``TIPoint`` list, any order)."""
    pts = _merge(points)
    b = np.array([p.beta for p in pts])
    m = np.array([p.mean for p in pts])
    s = np.array([p.std for p in pts])
    ne = np.array([p.n_eff for p in pts])
    if len(pts) < 2:
        raise ValueError("thermodynamic integration needs at least two beta values")
    if require_endpoints and (b[0] > 1e-12 or b[-1] < 1 - 1e-12):
        raise ValueError(f"the beta grid must include 0 and 1 (has {b[0]:g} .. {b[-1]:g}); "
                         "an unsampled end is not extrapolated")
    w = _trapezoid_weights(b)
    ln_z = float(np.sum(w * m))
    err_s = float(np.sqrt(np.sum(w ** 2 * s ** 2 / ne)))
    if len(b) >= 5:
        keep = np.r_[0:len(b):2]
        if keep[-1] != len(b) - 1:
            keep = np.r_[keep, len(b) - 1]
        coarse = float(np.sum(_trapezoid_weights(b[keep]) * m[keep]))
        err_d = abs(ln_z - coarse) / 3.0
    else:
        err_d = float("nan")
    return TIResult(ln_z, err_s, err_d, b.tolist(), m.tolist(), s.tolist(), ne.tolist(), w.tolist())


def bayes_factor(res_b, res_a):
    """ln BF_{B,A} = ln Z_B - ln Z_A with errors added in quadrature."""
    ln_bf = res_b.ln_z - res_a.ln_z
    return ln_bf, float(np.hypot(res_b.err, res_a.err))


# --------------------------------------------------------------------------- #
#  schedules
# --------------------------------------------------------------------------- #

def initial_betas(n, kind="beta", alpha=0.3):
    """A first schedule with 0 and 1 included.

    kind="beta": quantiles of Beta(alpha, alpha) -- dense at both ends, where
                 the integrand is steepest for a prior-to-posterior path.
    kind="power": (i/(n-1))**(1/alpha) -- dense near 0 only (Xie et al. 2011).
    kind="linear".
    """
    u = np.linspace(0.0, 1.0, int(n))
    if kind == "linear":
        b = u
    elif kind == "power":
        b = u ** (1.0 / alpha)
    elif kind == "beta":
        from scipy.stats import beta as _beta
        b = _beta.ppf(u, alpha, alpha)
    else:
        raise ValueError("kind must be beta | power | linear")
    b[0], b[-1] = 0.0, 1.0
    return np.unique(np.round(b, 12))


def propose_betas(points, n_new, floor=0.02, exclude_existing=True, min_gap=1e-6):
    """New beta values with density proportional to sd(beta) of ln L_partial.

    sd is linearly interpolated between the sampled betas; ``floor`` (a
    fraction of the mean sd) keeps some density everywhere.  Returns the new
    betas only (existing ones are left out unless ``exclude_existing=False``).
    """
    pts = _merge(points)
    b = np.array([p.beta for p in pts])
    s = np.array([p.std for p in pts])
    s = np.maximum(s, floor * max(np.mean(s), 1e-300))
    # cumulative of the piecewise-linear density
    seg = 0.5 * (s[1:] + s[:-1]) * np.diff(b)
    cdf = np.r_[0.0, np.cumsum(seg)]
    cdf /= cdf[-1]
    k = int(n_new) + (len(b) if exclude_existing else 0)
    q = (np.arange(k) + 0.5) / k
    new = np.interp(q, cdf, b)
    new = np.unique(np.round(new, 12))
    if exclude_existing:
        d = np.min(np.abs(new[:, None] - b[None, :]), axis=1)
        new = new[d > min_gap]
        if len(new) > n_new:
            new = new[np.round(np.linspace(0, len(new) - 1, int(n_new))).astype(int)]
    return new


# --------------------------------------------------------------------------- #
#  in-process driver (without the runner)
# --------------------------------------------------------------------------- #

def run_fixed_beta(model_kwargs, betas, num_warmup=500, num_samples=2000, num_chains=1,
                   chain_method="vectorized", seed=0, nuts_kwargs=None, progress_bar=False):
    """Run one NUTS chain set per beta in this process; returns [TIPoint].

    ``model_kwargs`` are the ``model_maker`` arguments (super_sig, helpers,
    raw_residuals, marg_over_non_gwb=True, ...), without ``beta``.  Arrays
    are passed through ``jit_model_args``, so every beta reuses one compiled
    model.
    """
    import jax.numpy as jnp
    import jax.random as jr
    from functools import partial
    from numpyro.infer import MCMC, NUTS
    from ATLAS.model import model_maker

    import jax

    kw = dict(model_kwargs)
    if not kw.get("marg_over_non_gwb", False):
        raise ValueError("thermodynamic integration here needs marg_over_non_gwb=True")
    kw.pop("beta", None)

    def _is_arrays(v):
        leaves = jax.tree_util.tree_leaves(v)
        return bool(leaves) and all(hasattr(x, "shape") for x in leaves)

    # arrays (and tuples of arrays, e.g. helpers) are traced; everything else
    # is bound, so one compiled model serves every beta
    static = {k: kw.pop(k) for k in list(kw)
              if k in ("super_sig", "tm_model") or not _is_arrays(kw[k])}
    model = partial(model_maker, **static)
    mcmc = MCMC(NUTS(model, **(nuts_kwargs or {})), num_warmup=num_warmup,
                num_samples=num_samples, num_chains=num_chains, chain_method=chain_method,
                progress_bar=progress_bar, jit_model_args=True)
    out = []
    for i, b in enumerate(np.asarray(betas, float)):
        # a fresh warmup at every beta (run() restarts from the initial point)
        mcmc.run(jr.key(seed + i), extra_fields=("~z.z_a",), beta=jnp.asarray(b), **kw)
        lnl = mcmc.get_samples(group_by_chain=True)["lnL_partial"]
        out.append(point_from_samples(b, lnl, f"beta={b:g}"))
    return out
