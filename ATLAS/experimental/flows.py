"""
flows.py — normalizing flows and flow matching in JAX.

Contents
--------
Shared helpers
    _params_jit                 jit for methods that read ``self.params``
    power-affine normalizer     _fit_normalizer, _from/_to_coefficients_*
    _iter_batches               memmap-safe shuffled batches (data, context)
    SAS normalizer              fit_sas_normalizer, sas_forward, sas_inverse
    context transforms          fit_context_transform, context_forward
    zuko building blocks        masked MLP masks, rational-quadratic splines
    MADE modules (haiku)        MaskedLinear, MADEBlock, MAFLayer

Flows with power-affine preprocessing
    Flow                        unconditional RQS coupling flow
    ConditionalMADEFlow         conditional MAF with MADE conditioners
    FlowMatching                unconditional flow matching
    ConditionalFlowMatching     conditional flow matching

Flows with sinh-arcsinh (SAS) preprocessing
    SASFlow                     unconditional RQS coupling flow; also the joint
                                p(theta, y) flow of a (n_draws, n_real, n_features) library
    ConditionalFlow             conditional RQS coupling flow (fast both ways)
    SASNSF                      JAX port of zuko's conditional neural spline flow
    (both conditional flows share _ConditionalSASFlow: data handling,
     training, log_prob / sample / forward_pass / latent, save / load)

Notes
-----
* Methods that read ``self.params`` are compiled with ``_params_jit``, which
  passes the parameters as a traced argument.  ``jax.jit(static_argnums=0)``
  would bake them in at the first call, so results would silently keep using
  the old weights after ``fit`` or ``load_params``.
* All arrays are float64 (``jax_enable_x64`` is switched on at import).
"""

import copy
import functools
import inspect
import math
import os
import pickle
import random
from functools import partial

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax.scipy.special import ndtri
from tqdm.auto import trange

jax.config.update("jax_enable_x64", True)

LOG2PI = math.log(2.0 * math.pi)


# ═════════════════════════════════════════════════════════════════════════════
# jit helper
# ═════════════════════════════════════════════════════════════════════════════

def _params_jit(method=None, *, static=()):
    """
    jit a method whose result depends on ``self.params``.

    ``jax.jit(method, static_argnums=0)`` treats ``self`` as a constant, so
    ``self.params`` is baked in at the first trace and later calls keep using
    those weights after training or loading.  This decorator passes
    ``self.params`` as a traced argument instead; other attributes (normalizer
    constants, architecture settings) are still read as constants.

    static : names of arguments that must be static (e.g. ``num_steps``).
    """
    if method is None:
        return partial(_params_jit, static=static)
    sig = inspect.signature(method)
    names = list(sig.parameters)[1:]                  # drop self
    static_argnums = tuple(1 + names.index(s) for s in static)
    key = method.__qualname__

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        root = self.__dict__.get("_params_jit_root", self)
        cache = root.__dict__.setdefault("_params_jit_cache", {})
        fn = cache.get(key)
        if fn is None:
            def pure(params, *a):
                shadow = copy.copy(root)
                shadow._params_jit_root = root
                shadow.params = params
                return method(shadow, *a)
            fn = cache[key] = jax.jit(pure, static_argnums=static_argnums)
        return fn(self.params, *bound.args[1:])

    return wrapper

# ═════════════════════════════════════════════════════════════════════════════
# Power-affine normalizer (Flow, ConditionalMADEFlow, flow matching)
# ═════════════════════════════════════════════════════════════════════════════

def _fit_normalizer(min_x, max_x, p, B):
    """
    Fit the power-affine normalizer from user-supplied range statistics.

    The forward direction (physical → normalized) is:
        dummy = sign(x) * |x|^exp(alpha)          # power stretch
        norm  = (dummy - mean) / half_range * B   # affine to [-B, B]

    The inverse (normalized → physical) is:
        dummy = half_range/B * norm + mean
        x     = sign(dummy) * |dummy|^(1/exp(alpha))

    Parameters
    ----------
    min_x, max_x : 1-D array-like, shape (D,)
        Per-feature physical-space range boundaries.
    p : float
        Power parameter.  alpha = log(p).
    B : float
        Target half-width of the normalized space.

    Returns
    -------
    dict with keys: alpha, mean, half_range, B
    """
    min_x = np.asarray(min_x, dtype=np.float64)
    max_x = np.asarray(max_x, dtype=np.float64)
    alpha = np.log(float(p))
    gamma = np.exp(alpha)
    dummy_min = np.sign(min_x) * np.abs(min_x) ** gamma
    dummy_max = np.sign(max_x) * np.abs(max_x) ** gamma
    mean      = (dummy_max + dummy_min) / 2.0
    half_range = (dummy_max - dummy_min) / 2.0
    return dict(alpha=alpha, mean=mean, half_range=half_range, B=float(B))


def _from_coefficients_np(arr, norm):
    """Physical → normalized  (numpy, used during batched training reads)."""
    alpha, mean, half_range, B = norm['alpha'], norm['mean'], norm['half_range'], norm['B']
    dummy = np.sign(arr) * np.abs(arr) ** np.exp(alpha)
    return (dummy - mean) / half_range * B


def _to_coefficients_np(arr, norm):
    """Normalized → physical  (numpy, used at inference)."""
    alpha, mean, half_range, B = norm['alpha'], norm['mean'], norm['half_range'], norm['B']
    dummy = half_range / B * arr + mean
    return np.sign(dummy) * np.abs(dummy) ** (1.0 / np.exp(alpha))


def _from_coefficients_jnp(arr, alpha, mean, half_range, B):
    """Physical → normalized  (jax, used inside jit-compiled methods)."""
    dummy = jnp.sign(arr) * jnp.abs(arr) ** jnp.exp(alpha)
    return (dummy - mean) / half_range * B


def _to_coefficients_jnp(arr, alpha, mean, half_range, B):
    """Normalized → physical  (jax, used inside jit-compiled methods)."""
    dummy = half_range / B * arr + mean
    return jnp.sign(dummy) * jnp.abs(dummy) ** (1.0 / jnp.exp(alpha))


def _logdet_from_coefficients_jnp(arr, alpha, mean, half_range, B):
    """
    Log |det J| of the physical → normalized map.
    J is diagonal so log|det J| = sum_i log|df_i/dx_i|.

    df/dx = exp(alpha) * |x|^(exp(alpha)-1) * (B / half_range)
    """
    gamma = jnp.exp(alpha)
    return jnp.sum(
        jnp.log(gamma)
        + (gamma - 1.0) * jnp.log(jnp.abs(arr))
        + jnp.log(B / half_range),
        axis=-1,
    )


def _logdet_to_coefficients_jnp(arr, alpha, mean, half_range, B):
    """
    Log |det J| of the normalized → physical map.
    """
    beta = 1.0 / jnp.exp(alpha)
    dummy = half_range / B * arr + mean
    return jnp.sum(
        jnp.log(half_range / B)
        + jnp.log(beta)
        + (beta - 1.0) * jnp.log(jnp.abs(dummy)),
        axis=-1,
    )

# ═════════════════════════════════════════════════════════════════════════════
# Memmap-safe batch iterator
# ═════════════════════════════════════════════════════════════════════════════

def _iter_batches(data, context, x_norm, c_norm, batch_size, key):
    """
    Yield (x_batch, c_batch) pairs in random order without loading the
    full dataset into memory.

    Parameters
    ----------
    data, context : array-like with shape[0] == N
        Raw memmapped (or in-memory) arrays.  Only small index-selected
        slices are materialised at a time.
    x_norm, c_norm : dict
        Normalizer dicts returned by _fit_normalizer for data and context.
    batch_size : int
    key : jax PRNGKey

    Notes
    -----
    *Shuffling* is done by drawing a random permutation of row indices
    (cheap — just integers).  Each batch then reads only `batch_size`
    rows from the mmap file, normalises them on the fly, and converts
    to jax arrays.  Peak memory is O(batch_size * (D + C)), regardless
    of N.
    """
    N = data.shape[0]
    perm = np.array(jr.permutation(key, N))          # numpy for fancy indexing
    for start in range(0, N, batch_size):
        idx = perm[start: start + batch_size]
        if len(idx) < batch_size:
            continue
        # Fancy-index into mmap → materialises only this slice
        x_raw = np.asarray(data[idx], dtype=np.float64)
        c_raw = np.asarray(context[idx], dtype=np.float64)
        x_batch = jnp.asarray(_from_coefficients_np(x_raw, x_norm))
        c_batch = jnp.asarray(_from_coefficients_np(c_raw, c_norm))
        yield x_batch, c_batch

# ═════════════════════════════════════════════════════════════════════════════
# Sinh-arcsinh (SAS) normalizer
# ═════════════════════════════════════════════════════════════════════════════

def _logcosh(a):
    a = jnp.abs(a)
    return a + jnp.log1p(jnp.exp(-2.0 * a)) - jnp.log(2.0)


def _sas_nll_1d(p, xs):
    loc, log_s, log_d, eps = p
    u = (xs - loc) / jnp.exp(log_s)
    a = jnp.exp(log_d) * jnp.arcsinh(u) - eps
    y = jnp.sinh(a)
    logjac = log_d - log_s + _logcosh(a) - jnp.log(jnp.hypot(1.0, u))
    return jnp.mean(0.5 * y**2 + 0.5 * LOG2PI - logjac)


_nll_v = jax.vmap(_sas_nll_1d, in_axes=(0, 1))
_grad_v = jax.vmap(jax.grad(_sas_nll_1d), in_axes=(0, 1))
_hess_v = jax.vmap(jax.hessian(_sas_nll_1d), in_axes=(0, 1))


@partial(jax.jit, static_argnames=("num_iters",))
def _fit_sas_chunk(xs, p0, num_iters=100):
    """Levenberg-Marquardt on every column of xs at once (exact 4x4 Hessians)."""
    eye = jnp.eye(4, dtype=xs.dtype)
    safe = lambda f: jnp.where(jnp.isfinite(f), f, jnp.inf)

    def step(_, state):
        p, f, lam = state
        g, H = _grad_v(p, xs), _hess_v(p, xs)
        d = jnp.linalg.solve(H + lam[:, None, None] * eye, g[..., None])[..., 0]
        p_new = p - d
        f_new = safe(_nll_v(p_new, xs))
        ok = (f_new < f) & jnp.all(jnp.isfinite(p_new), axis=-1)
        return (jnp.where(ok[:, None], p_new, p), jnp.where(ok, f_new, f),
                jnp.clip(jnp.where(ok, lam * 0.3, lam * 5.0), 1e-10, 1e10))

    f0 = safe(_nll_v(p0, xs))
    p, f, _ = jax.lax.fori_loop(0, num_iters, step,
                                (p0, f0, jnp.full(p0.shape[0], 1e-3, xs.dtype)))
    return p, f


def _fit_sas_columns(x, feature_chunk=None, num_iters=100):
    """SAS maximum-likelihood fit of every column of x (n, D), in chunks of columns."""
    x = np.asarray(x, dtype=np.float64)
    n, D = x.shape
    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite values passed to the SAS fit.")
    if feature_chunk is None:
        feature_chunk = int(max(1, min(D, 2e6 // n)))
    out = {k: np.empty(D) for k in ("loc", "s", "delta", "eps", "mu", "sd")}
    starts = np.array([[0.0, np.log(0.1), 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

    for j0 in range(0, D, feature_chunk):
        j1 = min(j0 + feature_chunk, D)
        blk = x[:, j0:j1]
        if j1 - j0 < feature_chunk:                         # pad: no recompilation
            blk = np.concatenate([blk, np.repeat(blk[:, -1:], feature_chunk - (j1 - j0), 1)], 1)
        med = np.median(blk, 0)
        scale = 1.4826 * np.median(np.abs(blk - med), 0)
        scale = np.where(scale > 0, scale, blk.std(0))
        if np.any(scale <= 0):
            raise ValueError(f"A feature in columns {j0}:{j1} has zero spread.")
        xs = jnp.asarray((blk - med) / scale)

        best_p = best_f = None
        for s0 in starts:
            p, f = _fit_sas_chunk(xs, jnp.tile(jnp.asarray(s0), (feature_chunk, 1)),
                                  num_iters=num_iters)
            if best_p is None:
                best_p, best_f = p, f
            else:
                better = f < best_f
                best_p = jnp.where(better[:, None], p, best_p)
                best_f = jnp.where(better, f, best_f)

        p = np.asarray(best_p)
        loc, s = med + scale * p[:, 0], scale * np.exp(p[:, 1])
        delta, eps = np.exp(p[:, 2]), p[:, 3]
        y = np.sinh(delta * np.arcsinh((blk - loc) / s) - eps)
        k = j1 - j0
        for name, v in (("loc", loc), ("s", s), ("delta", delta), ("eps", eps),
                        ("mu", y.mean(0)), ("sd", y.std(0))):
            out[name][j0:j1] = v[:k]
    return out


def fit_sas_normalizer(data, max_samples=200_000, seed=0, feature_chunk=None, num_iters=100):
    """
    Fit a per-feature sinh-arcsinh normalizer by maximum likelihood.

    Each feature is fitted independently, all at once: a damped-Newton
    (Levenberg-Marquardt) iteration with exact 4x4 Hessians, vmapped over
    chunks of features.

    Parameters
    ----------
    data : array-like (N, D), may be a numpy memmap.  Must be finite.
    max_samples : int
        Fit on at most this many randomly chosen rows.
    seed : int
        Seed for the row subsample.
    feature_chunk : int or None
        Features fitted per vmapped batch; None keeps each chunk near 2e6
        elements.  Lower it if you run out of (GPU) memory.
    num_iters : int
        Levenberg-Marquardt iterations.

    Returns
    -------
    dict of float64 arrays of shape (D,): loc, s, delta, eps, mu, sd
    """
    N = data.shape[0]
    if N > max_samples:
        idx = np.sort(np.random.default_rng(seed).choice(N, size=max_samples, replace=False))
        x = np.asarray(data[idx], dtype=np.float64)
    else:
        x = np.asarray(data, dtype=np.float64)
    return _fit_sas_columns(x, feature_chunk=feature_chunk, num_iters=num_iters)


def _sas_forward_np(x, n):
    u = (x - n["loc"]) / n["s"]
    return (np.sinh(n["delta"] * np.arcsinh(u) - n["eps"]) - n["mu"]) / n["sd"]


def sas_forward(x, n):
    """Physical -> normalized; returns (y, log|dy/dx| summed over features)."""
    u = (x - n["loc"]) / n["s"]
    a = n["delta"] * jnp.arcsinh(u) - n["eps"]
    y = (jnp.sinh(a) - n["mu"]) / n["sd"]
    ld = (jnp.log(n["delta"]) - jnp.log(n["s"]) - jnp.log(n["sd"])
          + _logcosh(a) - jnp.log(jnp.hypot(1.0, u)))
    return y, jnp.sum(ld, axis=-1)


def sas_inverse(y, n):
    """Normalized -> physical; returns (x, log|dx/dy| summed over features)."""
    a = jnp.arcsinh(n["sd"] * y + n["mu"])
    x = n["loc"] + n["s"] * jnp.sinh((a + n["eps"]) / n["delta"])
    return x, -sas_forward(x, n)[1]

# ═════════════════════════════════════════════════════════════════════════════
# Context transforms (SASNSF)
# ═════════════════════════════════════════════════════════════════════════════

def fit_context_transform(c, kind="standardize", bounds=None, pad_frac=1e-3):
    """
    kind : "standardize" | "probit" (uniform design -> N(0,1); needs or infers
           bounds) | "sas" | "none".
    """
    c = np.asarray(c, dtype=np.float64)
    if kind == "none":
        return dict(kind="none")
    if kind == "standardize":
        sd = c.std(0)
        return dict(kind=kind, mean=c.mean(0), std=np.where(sd > 0, sd, 1.0))
    if kind == "probit":
        if bounds is None:
            lo, hi = c.min(0), c.max(0)
            pad = pad_frac * (hi - lo)
            bounds = (lo - pad, hi + pad)
        return dict(kind=kind, lo=np.asarray(bounds[0], float), hi=np.asarray(bounds[1], float))
    if kind == "sas":
        return dict(kind=kind, **fit_sas_normalizer(c))
    raise ValueError("context kind must be 'standardize', 'probit', 'sas' or 'none'")


def context_forward(c, t):
    k = t["kind"]
    if k == "none":
        return c
    if k == "standardize":
        return (c - t["mean"]) / t["std"]
    if k == "probit":
        return ndtri(jnp.clip((c - t["lo"]) / (t["hi"] - t["lo"]), 1e-12, 1 - 1e-12))
    return sas_forward(c, t)[0]

# ═════════════════════════════════════════════════════════════════════════════
# zuko building blocks (SASNSF)
# ═════════════════════════════════════════════════════════════════════════════

def maf_adjacency(features, context, order, passes, total):
    """zuko.flows.autoregressive.MaskedAutoregressiveTransform adjacency."""
    passes = features if passes is None else passes
    passes = min(max(passes, 1), features)
    order = np.asarray(order) // math.ceil(features / passes)
    adj = order[:, None] > order[None, :]
    if context > 0:
        adj = np.concatenate([adj, np.ones((features, context), bool)], axis=1)
    return np.repeat(adj, total, axis=0), passes


def masked_mlp_masks(adjacency, hidden_features):
    """
    zuko.nn.MaskedMLP mask construction.  adjacency (out, in) bool.
    Returns a list of masks, one per layer, each shaped (in_l, out_l).
    """
    adjacency = np.asarray(adjacency, bool)
    adj_u, inverse = np.unique(adjacency, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    a = adj_u.astype(np.float64)
    precedence = (a @ a.T) == adj_u.sum(-1)        # P_ij: deps(j) subset of deps(i)

    masks, indices = [], None
    for i, feats in enumerate((*hidden_features, adjacency.shape[0])):
        mask = adj_u if i == 0 else precedence[:, indices]
        if (~mask).all():
            raise ValueError("The adjacency matrix leads to a null Jacobian.")
        if i < len(hidden_features):
            reachable = np.nonzero(mask.sum(-1))[0]
            indices = reachable[np.arange(feats) % len(reachable)]
            mask = mask[indices]
        else:
            mask = mask[inverse]
        masks.append(mask.T.copy())
    return masks


def init_masked_mlp(key, masks, zero_init_last, dtype):
    """torch.nn.Linear default init: W, b ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    params = []
    for l, m in enumerate(masks):
        fan_in, fan_out = m.shape
        key, kw, kb = jr.split(key, 3)
        bound = 1.0 / math.sqrt(fan_in)
        if zero_init_last and l == len(masks) - 1:
            W, b = jnp.zeros((fan_in, fan_out), dtype), jnp.zeros(fan_out, dtype)
        else:
            W = jr.uniform(kw, (fan_in, fan_out), dtype, -bound, bound)
            b = jr.uniform(kb, (fan_out,), dtype, -bound, bound)
        params.append(dict(W=W, b=b))
    return params


_ACTIVATIONS = dict(relu=jax.nn.relu, elu=jax.nn.elu, silu=jax.nn.silu,
                    gelu=jax.nn.gelu, tanh=jnp.tanh)


def masked_mlp_apply(params, masks, x, activation):
    h = x
    for l, (p, m) in enumerate(zip(params, masks)):
        h = h @ (p["W"] * m) + p["b"]
        if l < len(params) - 1:
            h = activation(h)
    return h


def _rqs_knots(w, h, d, bound, slope):
    """zuko.transforms.MonotonicRQSTransform.__init__"""
    ls = math.log(slope)
    w = w / (1 + jnp.abs(2 * w / ls))
    h = h / (1 + jnp.abs(2 * h / ls))
    d = d / (1 + jnp.abs(d / ls))
    pad1 = [(0, 0)] * (w.ndim - 1)
    w = jnp.pad(jax.nn.softmax(w, axis=-1), pad1 + [(1, 0)])
    h = jnp.pad(jax.nn.softmax(h, axis=-1), pad1 + [(1, 0)])
    d = jnp.pad(d, pad1 + [(1, 1)])
    xs = bound * (2 * jnp.cumsum(w, axis=-1) - 1)
    ys = bound * (2 * jnp.cumsum(h, axis=-1) - 1)
    return xs, ys, jnp.exp(d)


def _rqs_bin(k, xs, ys, ds, bins):
    mask = (k >= 0) & (k < bins)
    k = k % bins
    g = lambda a, i: jnp.take_along_axis(a, i[..., None], axis=-1)[..., 0]
    x0, x1 = g(xs, k), g(xs, k + 1)
    y0, y1 = g(ys, k), g(ys, k + 1)
    d0, d1 = g(ds, k), g(ds, k + 1)
    return mask, x0, x1, y0, y1, d0, d1, (y1 - y0) / (x1 - x0)


def rqs_forward(x, xs, ys, ds):
    """Spline value and log-derivative; identity outside [-B, B]."""
    bins = xs.shape[-1] - 1
    k = jnp.sum(xs < x[..., None], axis=-1) - 1
    mask, x0, x1, y0, y1, d0, d1, s = _rqs_bin(k, xs, ys, ds, bins)
    z = mask * (x - x0) / (x1 - x0)
    den = s + (d0 + d1 - 2 * s) * z * (1 - z)
    y = y0 + (y1 - y0) * (s * z**2 + d0 * z * (1 - z)) / den
    jac = s**2 * (2 * s * z * (1 - z) + d0 * (1 - z) ** 2 + d1 * z**2) / den**2
    return jnp.where(mask, y, x), mask * jnp.log(jac)


def rqs_inverse(y, xs, ys, ds):
    bins = xs.shape[-1] - 1
    k = jnp.sum(ys < y[..., None], axis=-1) - 1
    mask, x0, x1, y0, y1, d0, d1, s = _rqs_bin(k, xs, ys, ds, bins)
    y_ = mask * (y - y0)
    a = (y1 - y0) * (s - d0) + y_ * (d0 + d1 - 2 * s)
    b = (y1 - y0) * d0 - y_ * (d0 + d1 - 2 * s)
    c = -s * y_
    z = 2 * c / (-b - jnp.sqrt(jnp.maximum(b**2 - 4 * a * c, 0.0)))
    return jnp.where(mask, x0 + z * (x1 - x0), y)

# ═════════════════════════════════════════════════════════════════════════════
# MADE modules (ConditionalMADEFlow)
# ═════════════════════════════════════════════════════════════════════════════

class MaskedLinear(hk.Module):
    """
    A linear layer whose weight matrix is elementwise-multiplied by a
    fixed binary mask, enforcing the autoregressive ordering.

    mask : (out_features, in_features) bool array
    """

    def __init__(self, out_features, mask, with_bias=True, name=None):
        super().__init__(name=name)
        self.out_features = out_features
        self.mask         = mask          # (out, in) — fixed at construction
        self.with_bias    = with_bias

    def __call__(self, x):
        # x : (..., in_features)
        w = hk.get_parameter(
            'w',
            shape=(self.mask.shape[1], self.out_features),
            dtype=x.dtype,
            init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        )
        # Apply mask: zero out forbidden connections
        w_masked = w * self.mask.T          # (in, out)
        out = x @ w_masked
        if self.with_bias:
            b = hk.get_parameter(
                'b',
                shape=(self.out_features,),
                dtype=x.dtype,
                init=jnp.zeros,
            )
            out = out + b
        return out


class MADEBlock(hk.Module):
    """
    One MADE block: masked autoencoder for distribution estimation.

    Produces, for each input dimension i, the parameters of a
    RationalQuadraticSpline conditioned on x_{<i} and context c.

    Architecture
    ------------
    Input layer:  MaskedLinear  (D → hidden)   mask enforces autoregression
    Hidden layers: MaskedLinear (hidden → hidden) + activation
    Output layer: MaskedLinear  (hidden → D * num_params)  strict mask

    Context injection
    -----------------
    A linear projection of c is *added* to every hidden layer.  Adding
    keeps each hidden unit's autoregressive degree, so output_i still
    depends only on x_{<i}.  With context, hidden units of degree -1 see
    only c, so the first feature is conditioned on c as well.

    Parameters
    ----------
    D             : int   input / output dimension
    C             : int   context dimension (0 for unconditional)
    hidden_size   : int   width of hidden layers
    num_hidden    : int   number of hidden layers
    num_params    : int   spline params per dimension  (3*num_bins + 1)
    """

    def __init__(self, D, C, hidden_size, num_hidden, num_params, name=None):
        super().__init__(name=name)
        self.D          = D
        self.C          = C
        self.hidden_size = hidden_size
        self.num_hidden  = num_hidden
        self.num_params  = num_params

        # ── Degree assignment (Germain et al. 2015) ───────────────────────
        # Input degrees:  0, 1, ..., D-1
        # Hidden degrees: cycle through 0..D-2 (never D-1, so a valid path
        #                 reaches every output); with context, also -1:
        #                 context-only units that feed the first output.
        # Output degrees: 0, 1, ..., D-1  (same as input)
        self.m_input = np.arange(D)                                  # (D,)
        if C > 0:
            self.m_hidden = np.arange(hidden_size) % D - 1           # (H,)
        else:
            self.m_hidden = np.arange(hidden_size) % max(1, D - 1)   # (H,)
        self.m_output = np.arange(D)                                 # (D,)

        # ── Masks ─────────────────────────────────────────────────────────
        # input → first hidden:  h_k receives x_j  iff  m_h[k] >= m_in[j]
        self.mask_in = jnp.asarray(
            self.m_hidden[:, None] >= self.m_input[None, :],  # (H, D)
            dtype=jnp.float32,
        )

        # hidden → hidden:  m_h[k] >= m_h[j]
        self.mask_hh = jnp.asarray(
            self.m_hidden[:, None] >= self.m_hidden[None, :],  # (H, H)
            dtype=jnp.float32,
        )

        # last hidden → output:  out_i depends on x_{<i}
        # strict inequality: output_i must NOT see x_i
        # We tile to cover all num_params outputs per dimension
        mask_out_base = jnp.asarray(
            self.m_output[:, None] > self.m_hidden[None, :],  # (D, H)
            dtype=jnp.float32,
        )
        # Repeat for num_params outputs per dimension: (D*P, H)
        self.mask_out = jnp.repeat(mask_out_base, num_params, axis=0)

    def __call__(self, x, c=None):
        """
        x : (batch, D)
        c : (batch, C) or None

        Returns
        -------
        params : (batch, D, num_params)
        """
        # ── Input projection (masked) ─────────────────────────────────────
        h = MaskedLinear(self.hidden_size, self.mask_in, name='lin_in')(x)

        # ── Hidden layers: inject context after each masked projection ────
        for i in range(self.num_hidden):
            h = jax.nn.tanh(h)
            # Add a projection of the context to every hidden unit.  Mixing
            # [h, c] through a dense layer would let every unit see every x
            # and break the autoregressive structure (and the log-det).
            if c is not None:
                h = h + hk.Linear(
                    self.hidden_size,
                    name=f'ctx_proj_{i}',
                    w_init=hk.initializers.VarianceScaling(
                        1.0, 'fan_avg', 'uniform'),
                    b_init=jnp.zeros,
                )(c)
            if i < self.num_hidden - 1:
                h = MaskedLinear(
                    self.hidden_size, self.mask_hh, name=f'lin_h{i}'
                )(h)

        h = jax.nn.tanh(h)

        # ── Output projection (strictly masked) ───────────────────────────
        out = MaskedLinear(
            self.D * self.num_params,
            self.mask_out,
            name='lin_out',
        )(h)
        # Reshape: (batch, D*P) → (batch, D, P)
        return out.reshape(x.shape[0], self.D, self.num_params)


# ─────────────────────────────────────────────────────────────────────────────
# MAF layer: one autoregressive RQS transformation
# ─────────────────────────────────────────────────────────────────────────────

class MAFLayer(hk.Module):
    """
    One MAF layer wrapping a MADEBlock with RQS bijectors.

    Inverse pass (x → z, used for log_prob): parallel O(1)
        For each i: z_i = RQS_i^{-1}(x_i | params_i(x_{<i}, c))
        All i computed simultaneously because params depend on x (known).

    Forward pass (z → x, used for sample): sequential O(D)
        For each i: x_i = RQS_i(z_i | params_i(x_{<i}, c))
        x_i needed to compute params_{i+1} — inherently sequential.
    """

    def __init__(self, D, C, hidden_size, num_hidden, num_bins, B, name=None):
        super().__init__(name=name)
        self.D          = D
        self.num_bins   = num_bins
        self.B          = B
        self.made       = MADEBlock(
            D, C, hidden_size, num_hidden,
            num_params=3 * num_bins + 1,
            name=f'{name}_made' if name else 'made',
        )

    def inverse_and_log_det(self, x, c=None):
        """
        x → z  (fast: all params computable in parallel)

        Returns z, log|det J_{x→z}|
        """
        # MADE forward pass: (batch, D, num_params)
        params = self.made(x, c)

        # Apply RQS inverse to each dimension independently
        def invert_dim(x_i, params_i):
            spline = distrax.RationalQuadraticSpline(
                params_i[None, :],          # (1, num_params)
                range_min=-(self.B + 1),
                range_max=  self.B + 1,
            )
            z_i, ld_i = spline.inverse_and_log_det(x_i[None])
            return z_i[0], ld_i[0]

        # vmap over the D dimensions, then over the batch
        def invert_sample(x_row, params_row):
            # x_row: (D,), params_row: (D, P)
            z_row, ld_row = jax.vmap(invert_dim)(x_row, params_row)
            return z_row, ld_row.sum()

        z, log_det = jax.vmap(invert_sample)(x, params)
        return z, log_det

    def forward_single(self, z_row, c_row=None):
        """
        z → x for a SINGLE sample — sequential loop over dimensions.
        Used inside jax.lax.scan during sampling.

        z_row   : (D,)
        c_row   : (C,) or None

        Returns x_row : (D,)
        """
        c_2d = c_row[None, :] if c_row is not None else None

        num_params = 3 * self.num_bins + 1

        def fori_fn(i, x_so_far):
            # All MADE params from current x_so_far — (1, D, P)
            all_params = self.made(x_so_far[None, :], c_2d)

            # Extract params for dimension i via dynamic_slice (JAX-safe)
            # all_params[0] shape: (D, P) → slice row i → (1, P)
            params_i = jax.lax.dynamic_slice(
                all_params[0], (i, 0), (1, num_params)
            )  # (1, P)

            z_i = jax.lax.dynamic_slice(z_row, (i,), (1,))  # (1,)

            spline = distrax.RationalQuadraticSpline(
                params_i,               # (1, P)
                range_min=-(self.B + 1),
                range_max=  self.B + 1,
            )
            x_i, _ = spline.forward_and_log_det(z_i)        # (1,)
            return x_so_far.at[i].set(x_i[0])

        x_row = jax.lax.fori_loop(0, self.D, fori_fn, jnp.zeros_like(z_row))
        return x_row

# ═════════════════════════════════════════════════════════════════════════════
# Flow
# ═════════════════════════════════════════════════════════════════════════════

class Flow:
    """
    Rational Quadratic Spline normalizing flow with
    custom power-law preprocessing.

    This class wraps a full training + inference pipeline including:
    - preprocessing
    - model definition
    - optimization
    - sampling and evaluation
    """

    def __init__(
        self,
        data,
        data_min=None,
        data_max=None,
        tset_paths=None,
        flow_num_layers=4,
        hidden_size=128,
        mlp_num_layers=2,
        num_bins=8,
        learning_rate=1e-4,
        B=6.0,
        p=0.3,
        seed=0,
    ):

        """
        Flow: Rational Quadratic Spline Normalizing Flow
        ================================================

        This module defines the `Flow` class, a full implementation of a
        normalizing flow using JAX, Haiku, Distrax, and Optax.

        The implementation is adapted from:
        https://cocalc.com/github/probml/pyprobml/blob/master/deprecated/notebooks/flow_spline_mnist_jax.ipynb

        Original authors:
        Murphy, K., Soliman, M., Duran-Martin, G., Kara, A., Liang Ang, M., Reddy, S., & Patel, D. (2021)
        PyProbML library for Probabilistic Machine Learning

        -----------------------------------------------------------------------
        OVERVIEW
        -----------------------------------------------------------------------

        This model implements a density estimator using a sequence of invertible
        transformations (a normalizing flow). The transformations map data between:

            physical space (original data)
                ↔
            normalized bounded space
                ↔
            latent Gaussian space

        The pipeline consists of:

        1. Power-law preprocessing
        2. Affine normalization to a bounded domain [-B, B]
        3. A stack of rational quadratic spline coupling layers
        4. A standard Gaussian base distribution

        -----------------------------------------------------------------------
        PREPROCESSING TRANSFORM
        -----------------------------------------------------------------------

        Before training, each input x is transformed as follows:

        Step 1: Power transform
            x' = sign(x) * |x|^gamma

            where gamma = exp(alpha)

        Step 2: Compute statistics from transformed data
            mean = (max(x') + min(x')) / 2
            half_range = (max(x') - min(x')) / 2

        Step 3: Normalize to bounded space
            x_norm = (x' - mean) / half_range * B

        Full forward transform:
            x_norm = (sign(x) * |x|^gamma - mean) / half_range * B

        Inverse transform:
            z = (half_range / B) * x_norm + mean
            x = sign(z) * |z|^(1/gamma)

        Jacobian corrections are applied during likelihood evaluation.

        -----------------------------------------------------------------------
        FEATURES
        -----------------------------------------------------------------------

        - Exact log-likelihood computation
        - Invertible transformations
        - Efficient sampling
        - JIT-compiled operations (via JAX)
        - Automatic differentiation

        -----------------------------------------------------------------------

        Initialize the flow model.

        Parameters
        ----------
        data : array (N, D)
            Training dataset
        data_min, data_max : array (D,) or None
            Physical-space feature bounds; computed from `data` if None.
        tset_paths : list of str or None
            .npy training files, used by fit(mode='from_disk').
        flow_num_layers : int
            Number of coupling layers
        hidden_size : int
            Width of MLP hidden layers
        mlp_num_layers : int
            Number of hidden layers in conditioner networks
        num_bins : int
            Number of bins in spline transform
        learning_rate : float
            Adam optimizer learning rate
        B : float
            Bound for normalized space
        p : float
            Power transform parameter
        seed : int
            Random seed
        """
        self.tset_paths = tset_paths

        self.N, self.D = data.shape
        self.B = float(B) #avoiding boundry problems
        self.alpha = jnp.log(p)
        self.gamma = jnp.exp(self.alpha)

        # ------------------------------------------------------------
        # Power transform parameters
        # ------------------------------------------------------------
        if (data_min is not None and data_max is not None
                and np.any(data_min) and np.any(data_max)):
            min_x = data_min
            max_x = data_max
            dummy_min = np.sign(min_x) * np.abs(min_x) ** self.gamma
            dummy_max = np.sign(max_x) * np.abs(max_x) ** self.gamma
            self.mean_x      = (dummy_max + dummy_min) / 2.0
            self.half_range  = (dummy_max - dummy_min) / 2.0
        else:
            dummy = jnp.sign(data) * jnp.abs(data) ** self.gamma
            min_x = dummy.min(axis =0)
            max_x = dummy.max(axis = 0)
            self.mean_x = (max_x + min_x) / 2.0
            self.half_range = (max_x - min_x) / 2.0

        # ------------------------------------------------------------
        # Normalize data
        # ------------------------------------------------------------

        self.x_norm = self.to_unit_interval(data)
        # assert self.x_norm.min() > -B
        # assert self.x_norm.max() < B
        # ------------------------------------------------------------
        # Flow config
        # ------------------------------------------------------------

        self.event_shape = (self.D,)
        self.flow_num_layers = flow_num_layers
        self.hidden_sizes = [hidden_size] * mlp_num_layers
        self.num_bins = num_bins
        self.learning_rate = learning_rate

        # ------------------------------------------------------------
        # RNG
        # ------------------------------------------------------------

        self.key = jr.PRNGKey(seed)

        # ------------------------------------------------------------
        # Build/init
        # ------------------------------------------------------------

        self._build_model()
        self._init_params()

    # ================================================================
    # Data transforms
    # ================================================================
    @partial(jax.jit, static_argnums=0)
    def from_unit_interval(self, arr):
        """
        Transform normalized values back to physical space.

        Implements the inverse of the preprocessing transform.
        """
        dummy = (self.half_range[None, :] / self.B * arr + self.mean_x[None, :])
        return jnp.sign(dummy) * jnp.abs(dummy) ** (1.0 / jnp.exp(self.alpha))

    @partial(jax.jit, static_argnums=0)
    def to_unit_interval(self, arr):
        """
        Transform physical data into normalized bounded space [-B, B].
        """
        dummy = jnp.sign(arr) * jnp.abs(arr) ** (jnp.exp(self.alpha))
        return (dummy - self.mean_x[None, :]) / self.half_range[None, :] * self.B


    # ================================================================
    # Jacobian: physical → normalized
    # ================================================================
    @partial(jax.jit, static_argnums=0)
    def logdet_to_unit_interval(self, arr):
        """
        Log absolute determinant of the Jacobian of the transformation
        from physical space to normalized space.
        """

        gamma = jnp.exp(self.alpha)

        logdet = jnp.sum(
            jnp.log(gamma)
            + (gamma - 1.0) * jnp.log(jnp.abs(arr))
            + jnp.log(self.B / self.half_range),
            axis=-1,
        )

        return logdet


    # ================================================================
    # Jacobian: normalized → physical
    # ================================================================
    @partial(jax.jit, static_argnums=0)
    def logdet_from_unit_interval(self, arr):
        """
        Log absolute determinant of the Jacobian of the transformation
        from normalized space back to physical space.
        """

        beta = 1.0 / jnp.exp(self.alpha)

        dummy = (
            self.half_range[None, :] / self.B * arr
            + self.mean_x[None, :]
        )

        logdet = jnp.sum(
            jnp.log(self.half_range / self.B)
            + jnp.log(beta)
            + (beta - 1.0) * jnp.log(jnp.abs(dummy)),
            axis=-1,
        )

        return logdet


    # ================================================================
    # Batching
    # ================================================================

    @staticmethod
    def get_batches(data, batch_size, key):
        """
        Generate shuffled mini-batches of data.
        """

        N = data.shape[0]

        idx = jax.random.permutation(key, N)

        for i in range(0, N, batch_size):
            yield data[idx[i:i + batch_size]]

    # ================================================================
    # Conditioner
    # ================================================================

    @staticmethod
    def make_conditioner(
        event_shape,
        hidden_sizes,
        num_bijector_params,
    ):
        """
        Construct the neural network used to parameterize spline bijectors.
        """

        return hk.Sequential([
            hk.Flatten(
                preserve_dims=-len(event_shape)
            ),

            hk.nets.MLP(
                hidden_sizes,
                activate_final=True,
            ),

            hk.Linear(
                np.prod(event_shape)
                * num_bijector_params,

                w_init=jnp.zeros,
                b_init=jnp.zeros,
            ),

            hk.Reshape(
                event_shape
                + (num_bijector_params,),
                preserve_dims=-1,
            ),
        ])

    # ================================================================
    # Flow definition
    # ================================================================

    @staticmethod
    def make_flow_model(
        event_shape,
        num_layers,
        hidden_sizes,
        num_bins,
        B,
    ):
        """
        Build the complete normalizing flow model.

        The flow consists of multiple masked coupling layers using
        rational quadratic spline transformations.
        """

        mask = (
            jnp.arange(np.prod(event_shape))
            % 2
        )

        mask = mask.reshape(event_shape).astype(bool)

        def bijector_fn(params):
            return distrax.RationalQuadraticSpline(
                params,
                range_min=-B - 1,
                range_max=B + 1,
            )

        num_bijector_params = 3 * num_bins + 1

        layers = []

        for _ in range(num_layers):

            layers.append(
                distrax.MaskedCoupling(
                    mask=mask,

                    bijector=bijector_fn,

                    conditioner=Flow.make_conditioner(
                        event_shape,
                        hidden_sizes,
                        num_bijector_params,
                    ),
                )
            )

            mask = jnp.logical_not(mask)

        flow = distrax.Inverse(
            distrax.Chain(layers)
        )

        base_dist = distrax.Independent(
            distrax.Normal(
                loc=jnp.zeros(event_shape),
                scale=jnp.ones(event_shape),
            ),
            reinterpreted_batch_ndims=len(event_shape),
        )

        return distrax.Transformed(
            base_dist,
            flow,
        )

    # ================================================================
    # Haiku transforms
    # ================================================================

    def _build_model(self):
        """
        Define Haiku-transformed functions for log-probability,
        sampling, and forward transformations.
        """

        @hk.without_apply_rng
        @hk.transform
        def log_prob_fn(data):

            model = Flow.make_flow_model(
                self.event_shape,
                self.flow_num_layers,
                self.hidden_sizes,
                self.num_bins,
                self.B,
            )

            return model.log_prob(data)

        @hk.without_apply_rng
        @hk.transform
        def sample_fn(key, num_samples):

            model = Flow.make_flow_model(
                self.event_shape,
                self.flow_num_layers,
                self.hidden_sizes,
                self.num_bins,
                self.B,
            )

            return model.sample(
                seed=key,
                sample_shape=[num_samples],
            )

        @hk.without_apply_rng
        @hk.transform
        def transform_fn(z):

            model = Flow.make_flow_model(
                self.event_shape,
                self.flow_num_layers,
                self.hidden_sizes,
                self.num_bins,
                self.B,
            )

            return model.bijector.forward(z)

        @hk.without_apply_rng
        @hk.transform
        def inverse_transform_fn(z_inv):

            model = Flow.make_flow_model(
                self.event_shape,
                self.flow_num_layers,
                self.hidden_sizes,
                self.num_bins,
                self.B,
            )

            return model.bijector.inverse(z_inv)

        @hk.without_apply_rng
        @hk.transform
        def transform_and_jac_fn(z):

            model = Flow.make_flow_model(
                self.event_shape,
                self.flow_num_layers,
                self.hidden_sizes,
                self.num_bins,
                self.B,
            )

            return model.bijector.forward_and_log_det(z)

        self.log_prob_fn = log_prob_fn
        self.sample_fn = sample_fn
        self.transform_fn = transform_fn
        self.inv_transform_fn = inverse_transform_fn
        self.transform_and_jac_fn = transform_and_jac_fn

    # ================================================================
    # Initialization
    # ================================================================

    def _init_params(self):
        """
        Initialize model parameters and optimizer state.
        """

        self.key, init_key = jr.split(self.key)

        self.params = self.log_prob_fn.init(
            init_key,
            self.x_norm[:1],
        )

        self.optimizer = optax.adam(
            self.learning_rate
        )

        self.opt_state = self.optimizer.init(
            self.params
        )

    # ================================================================
    # Training
    # ================================================================

    def loss_fn(self, params, batch):
        """
        Negative log-likelihood loss function.
        The same thing as KL-divergence!
        """

        logp = self.log_prob_fn.apply(
            params,
            batch,
        )

        return -jnp.mean(logp)

    @partial(jax.jit, static_argnums=0)
    def update(
        self,
        params,
        opt_state,
        batch,
    ):

        """
        Perform one gradient update step using Adam.
        """

        grads = jax.grad(
            self.loss_fn
        )(params, batch)

        updates, opt_state = self.optimizer.update(
            grads,
            opt_state,
        )

        params = optax.apply_updates(
            params,
            updates,
        )

        return params, opt_state

    def fit(
        self,
        mode = 'pre-loaded',
        num_epochs=100,
        batch_size=256,
        checkpoint=None,
    ):
        """
        Train the model using mini-batch gradient descent.

        mode : str
        num_epochs : int
        batch_size : int
        checkpoint : path or None
            Write training state here after every epoch if not None,
            and resume from last epoch recorded here if it already exists.
        """


        if mode == 'pre-loaded':
            # load from last checkpointed epoch
            start_epoch = 0
            if checkpoint is not None and os.path.exists(checkpoint):
                start_epoch = self._load_checkpoint(checkpoint, batch_size)
                print(f"[flow] resuming from {checkpoint} at epoch "
                        f"{start_epoch}/{num_epochs}", flush=True)
            self.resumed_from = start_epoch # needed for interpreting wall-clock timing on a resumed run

            for ep in trange(start_epoch, num_epochs,
                            initial=start_epoch, total=num_epochs,
                            colour = 'blue', desc = "Training the flow from pre-loaded data"):

                self.key, subkey = jr.split(self.key)

                for batch in self.get_batches(
                    self.x_norm,
                    batch_size,
                    subkey,
                ):

                    self.params, self.opt_state = (
                        self.update(
                            self.params,
                            self.opt_state,
                            batch,
                        )
                    )

                if checkpoint is not None:
                    self._save_checkpoint(checkpoint, ep + 1, batch_size)

        if mode == 'from_disk':
            if checkpoint is not None:
                err = "Checkpointing is only supported for mode='pre-loaded'."
                raise ValueError(err)
            for _ in trange(num_epochs, colour = 'blue', desc = "Training the flow from disk"):
                rand_idx = random.randint(0, len(self.tset_paths) - 1)
                training_set = jnp.load(self.tset_paths[rand_idx])
                idxs = jnp.array(random.sample(range(training_set.shape[0]),
                                k = min(batch_size, training_set.shape[0])))
                batch =  self.to_unit_interval(training_set[idxs])
                self.params, self.opt_state = self.update(self.params,
                                                            self.opt_state,
                                                            batch)

    # ================================================================
    # Public API
    # ================================================================

    @_params_jit
    def log_prob(self, x):
        """
        Compute log-probability in physical space.
        """

        x_norm = self.to_unit_interval(x)

        logp_norm = self.log_prob_fn.apply(
            self.params,
            x_norm,
        )

        logdet = self.logdet_to_unit_interval(x)

        return logp_norm + logdet

    def sample(self, num_samples):
        """
        Draw samples in physical space.
        """

        self.key, subkey = jr.split(self.key)

        samples_norm = self.sample_fn.apply(
            self.params,
            subkey,
            num_samples,
        )

        return self.from_unit_interval(
            samples_norm
        )

    @_params_jit
    def forward_pass(self, z):
        """
        Map latent samples to physical space.
        """

        physics_norm = self.transform_fn.apply(
            self.params,
            z,
        )

        return self.from_unit_interval(
            physics_norm
        )

    @_params_jit
    def backward_pass(self, target_samp):
        """
        Map latent samples to physical space.
        """
        x = self.to_unit_interval(target_samp)

        physics_norm = self.inv_transform_fn.apply(
            self.params,
            x,
        )

        return physics_norm


    @_params_jit
    def forward_pass_with_logdet(self, z):
        """
        Forward transform with full Jacobian correction.
        """

        physics_norm, logdet_flow = self.transform_and_jac_fn.apply(
            self.params,
            z,
        )

        physics = self.from_unit_interval(physics_norm)

        logdet_phys = self.logdet_from_unit_interval(physics_norm)

        total_logdet = logdet_flow + logdet_phys

        return physics, total_logdet

    def save_params(self, flow_save_path):
        leaves, treedef = jax.tree_util.tree_flatten(self.params)
        save_dict = {f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)}
        save_dict["treedef"] = np.array(str(treedef))
        np.savez(flow_save_path, **save_dict)

    def load_params(self, flow_load_path):
        data = np.load(flow_load_path, allow_pickle=True)
        _, treedef = jax.tree_util.tree_flatten(self.params)
        leaves = [jnp.asarray(data[f"leaf_{i}"]) for i in range(treedef.num_leaves)]
        self.params = jax.tree_util.tree_unflatten(treedef, leaves)


    # ------------- Checkpointing -----------
    # To resume flow training if interrupted partway through

    def _checkpoint_signature(self, batch_size):
        # Configuration must match for a checkpoint to belong to this run.
        # Prevents resuming into a different state
        return dict(
            N=int(self.N), D=int(self.D),
            flow_num_layers=int(self.flow_num_layers),
            hidden_sizes=[int(h) for h in self.hidden_sizes],
            num_bins=int(self.num_bins),
            learning_rate=float(self.learning_rate),
            B=float(self.B), gamma=float(self.gamma),
            batch_size=int(batch_size),
        )

    def _save_checkpoint(self, path, epoch, batch_size):
        """Params, optimizer state, RNG key and epoch counter."""
        # `opt_state` is saved, not just `params`.  `save_params` is deliberately
        # not reused here: it writes the parameter leaves through `np.savez` plus
        # a stringified treedef, which cannot carry an optax NamedTuple tree or a
        # PRNG key.  Resuming without Adam's first and second moments would
        # restart them at zero and put a visible transient in the loss.
        blob = dict(
            params=jax.tree_util.tree_map(np.asarray, self.params),
            opt_state=jax.tree_util.tree_map(np.asarray, self.opt_state),
            key=np.asarray(self.key),
            epoch=int(epoch),
            signature=self._checkpoint_signature(batch_size),
        )
        # Write to a temporary file and rename.  The atomic rename is the whole
        # reason this is safe to interrupt: a kill during the write leaves the
        # PREVIOUS checkpoint intact rather than a half-written file that cannot
        # be unpickled -- which would turn one lost epoch into all of them.
        tmp = f'{path}.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def _load_checkpoint(self, path, batch_size):
        """Restore training state; return the number of epochs already done."""
        with open(path, 'rb') as f:
            blob = pickle.load(f)

        want, got = self._checkpoint_signature(batch_size), blob['signature']
        if got != want:
            differ = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
            raise ValueError(
                f"{path} was written by a different configuration: "
                f"(checkpoint, this run) differ on {differ}.  Train into a "
                f"different directory, or delete the checkpoint to start over.")

        self.params = jax.tree_util.tree_map(jnp.asarray, blob['params'])
        self.opt_state = jax.tree_util.tree_map(jnp.asarray, blob['opt_state'])
        # Restoring the key is what makes a resume CONTINUE the RNG stream.
        # Without it the resumed epochs would replay the batch order of the
        # epochs already trained on.
        self.key = jnp.asarray(blob['key'])
        return int(blob['epoch'])

# ═════════════════════════════════════════════════════════════════════════════
# ConditionalMADEFlow
# ═════════════════════════════════════════════════════════════════════════════

class ConditionalMADEFlow:
    """
    Conditional Masked Autoregressive Flow (MAF) with RQS bijectors.

    Masked Autoregressive Flow (MAF) with MADE conditioners.

    Architecture
    ------------
    Instead of RQS masked-coupling layers (as in ConditionalFlow), this uses
    Masked Autoregressive Flow (MAF) layers, each built from a MADE block.

    MADE vs MLP conditioner
    -----------------------
    MaskedCoupling (ConditionalFlow, SASFlow):
        - Splits x into two halves by index parity (mask).
        - The *unchanged* half is fed through an unconstrained MLP to produce
          spline params for the *transformed* half.
        - Parallelisable: all transformed dims updated simultaneously.
        - Requires num_layers coupling layers to mix all dimensions.

    MADE (what this class uses):
        - A single network with masked weights enforces the autoregressive
          property: output_i depends only on inputs x_{<i}.
        - Every dimension gets its own dedicated conditioner output in one
          forward pass — no split needed.
        - More expressive per layer: D autoregressive conditionals vs D/2.
        - Forward (z→x) is sequential O(D) — slow for sampling.
        - Inverse (x→z) is parallel O(1) — fast for density evaluation.
        - This is MAF: fast log_prob, slow sample.
        - Contrast with IAF (fast sample, slow log_prob).

    For PTA / GW inference you usually call log_prob far more often than
    sample, so MAF is the natural choice.

    Mask construction (Germain et al. 2015)
    ----------------------------------------
    Assign each unit an integer "degree" m:
        Input  units: m(x_i) = i          (1-indexed, 0..D-1 here)
        Hidden units: m(h_k) ~ Uniform{min_degree, D-1}
        Output units: m(out_i) = i

    Weight mask W_{kj} = 1  iff  m(k) >= m(j)   (hidden→hidden, hidden→input)
               W_{kj} = 1  iff  m(k) >  m(j)   (output→hidden)

    This guarantees output_i depends only on x_{0..i-1}.

    Context conditioning
    --------------------
    A linear projection of c is added to every MADE hidden layer, after the
    masked input projection.  c conditions every output (including the
    first, through context-only hidden units) without breaking the
    autoregressive masks.

    Stacked MAF layers, each backed by a MADE block, with the power-affine
    normalisation of data and context.

    Key properties vs a coupling flow
    ----------------------------------
    log_prob  : O(num_layers) parallel MADE passes  — same speed
    sample    : O(num_layers × D) sequential steps  — SLOWER
    forward_pass : slow (sequential per dimension)
    backward_pass / inverse_pass : fast (parallel)

    Use this class when:
    - You evaluate log_prob much more often than you sample.
    - You want strictly autoregressive conditionals (no coupling approximation).
    - D is moderate (≤ ~50); for large D the sequential sampling is painful.

    Normalisation, save/load, fit modes and public API match
    ConditionalFlowMatching (power-affine interface).  For SAS preprocessing
    and the newer interface, use SASNSF.
    """

    def __init__(
        self,
        data,
        context,
        data_min=None,
        data_max=None,
        context_min=None,
        context_max=None,
        normalizer_dict=None,
        tset_paths=None,
        last_feature_index=None,
        flow_save_path=None,
        flow_load_path=None,
        p=0.1,
        B=4.0,
        flow_num_layers=4,
        hidden_size=128,
        mlp_num_layers=2,       # number of MADE hidden layers per MAF layer
        num_bins=8,
        learning_rate=1e-4,
        seed=0,
    ):
        self.N, self.D      = data.shape[0], data.shape[1]
        self.C              = context.shape[1]
        self.tset_paths     = tset_paths
        self.last_feature_idx = last_feature_index
        self.flow_save_path = flow_save_path
        self.flow_load_path = flow_load_path

        self._data    = data
        self._context = context

        # ── Normalizers ───────────────────────────────────────────────────
        if normalizer_dict is not None and any(normalizer_dict):
            self.x_norm = normalizer_dict['data']
            self.c_norm = normalizer_dict['context']
        else:
            self.x_norm = _fit_normalizer(data_min, data_max, p, B)
            self.c_norm = _fit_normalizer(context_min, context_max, p, B)

        self._x_alpha      = jnp.float64(self.x_norm['alpha'])
        self._x_mean       = jnp.asarray(self.x_norm['mean'])
        self._x_half_range = jnp.asarray(self.x_norm['half_range'])
        self._x_B          = jnp.float64(self.x_norm['B'])

        self._c_alpha      = jnp.float64(self.c_norm['alpha'])
        self._c_mean       = jnp.asarray(self.c_norm['mean'])
        self._c_half_range = jnp.asarray(self.c_norm['half_range'])
        self._c_B          = jnp.float64(self.c_norm['B'])

        # ── Flow config ───────────────────────────────────────────────────
        self.flow_num_layers = flow_num_layers
        self.hidden_size     = hidden_size
        self.num_hidden      = mlp_num_layers   # MADE hidden layers
        self.num_bins        = num_bins
        self.learning_rate   = learning_rate
        self.B_float         = float(self.x_norm['B'])

        self.key = jr.PRNGKey(seed)
        self._build_model()
        self._init_params()

    # ====================================================================
    # Normalisation  (power-affine family)
    # ====================================================================

    @partial(jax.jit, static_argnums=0)
    def normalize_data(self, x):
        return _from_coefficients_jnp(
            x, self._x_alpha, self._x_mean, self._x_half_range, self._x_B)

    @partial(jax.jit, static_argnums=0)
    def denormalize_data(self, x_norm):
        return _to_coefficients_jnp(
            x_norm, self._x_alpha, self._x_mean, self._x_half_range, self._x_B)

    @partial(jax.jit, static_argnums=0)
    def normalize_context(self, c):
        return _from_coefficients_jnp(
            c, self._c_alpha, self._c_mean, self._c_half_range, self._c_B)

    @partial(jax.jit, static_argnums=0)
    def denormalize_context(self, c_norm):
        return _to_coefficients_jnp(
            c_norm, self._c_alpha, self._c_mean, self._c_half_range, self._c_B)

    @partial(jax.jit, static_argnums=0)
    def _logdet_normalize_data(self, x):
        return _logdet_from_coefficients_jnp(
            x, self._x_alpha, self._x_mean, self._x_half_range, self._x_B)

    @partial(jax.jit, static_argnums=0)
    def _logdet_denormalize_data(self, x_norm):
        return _logdet_to_coefficients_jnp(
            x_norm, self._x_alpha, self._x_mean, self._x_half_range, self._x_B)

    # ====================================================================
    # Model construction
    # ====================================================================

    def _make_maf_stack(self, x, c, inverse=True):
        """
        Build and apply all MAF layers.

        inverse=True  : x → z  (fast, for log_prob)
        inverse=False : z → x  (slow, for sample)
        """
        D, C         = self.D, self.C
        hidden_size  = self.hidden_size
        num_hidden   = self.num_hidden
        num_bins     = self.num_bins
        B            = self.B_float
        num_layers   = self.flow_num_layers

        # Between MAF layers we permute dimensions to improve mixing.
        # Use a fixed reversal permutation (simple and effective).
        perm    = jnp.arange(D - 1, -1, -1)   # reverse order
        inv_perm = jnp.argsort(perm)

        total_log_det = jnp.zeros(x.shape[0])

        if inverse:
            # x → z: apply layers in forward order, permute between them
            h = x
            for i in range(num_layers):
                layer = MAFLayer(
                    D, C, hidden_size, num_hidden, num_bins, B,
                    name=f'maf_{i}',
                )
                h, ld = layer.inverse_and_log_det(h, c)
                total_log_det = total_log_det + ld
                if i < num_layers - 1:
                    h = h[:, perm]          # permute between layers
            return h, total_log_det

        else:
            # z → x: apply layers in reverse order, un-permute between them
            h = x
            for i in range(num_layers - 1, -1, -1):
                # Un-permute before applying inverse layer
                if i < num_layers - 1:
                    h = h[:, inv_perm]

                layer = MAFLayer(
                    D, C, hidden_size, num_hidden, num_bins, B,
                    name=f'maf_{i}',
                )
                # forward_single must be vmapped over the batch
                if c is not None:
                    h = jax.vmap(layer.forward_single)(h, c)
                else:
                    h = jax.vmap(lambda z_row: layer.forward_single(z_row))(h)

            return h

    def _build_model(self):
        @hk.without_apply_rng
        @hk.transform
        def log_prob_fn(x_norm, c_norm):
            """x_norm → log p(x_norm | c_norm) in normalised space."""
            z, log_det = self._make_maf_stack(x_norm, c_norm, inverse=True)
            # Base distribution: N(0, I)
            log_p_z = jnp.sum(
                -0.5 * z ** 2 - 0.5 * jnp.log(2 * jnp.pi),
                axis=-1,
            )
            # log p(x) = log p(z) + log|det J_{x→z}|
            return log_p_z + log_det

        @hk.without_apply_rng
        @hk.transform
        def inverse_fn(x_norm, c_norm):
            """x_norm → z  (the fast direction)."""
            z, _ = self._make_maf_stack(x_norm, c_norm, inverse=True)
            return z

        @hk.without_apply_rng
        @hk.transform
        def inverse_and_logdet_fn(x_norm, c_norm):
            return self._make_maf_stack(x_norm, c_norm, inverse=True)

        @hk.without_apply_rng
        @hk.transform
        def forward_fn(z, c_norm):
            """z → x_norm  (the slow, sequential direction)."""
            return self._make_maf_stack(z, c_norm, inverse=False)

        self.log_prob_fn          = log_prob_fn
        self.inverse_fn           = inverse_fn
        self.inverse_and_logdet_fn = inverse_and_logdet_fn
        self.forward_fn           = forward_fn

    # ====================================================================
    # Initialisation
    # ====================================================================

    def _init_params(self):
        self.key, init_key = jr.split(self.key)

        x0 = jnp.asarray(
            _from_coefficients_np(
                np.asarray(self._data[:2], dtype=np.float64), self.x_norm))
        c0 = jnp.asarray(
            _from_coefficients_np(
                np.asarray(self._context[:2], dtype=np.float64), self.c_norm))

        self.params    = self.log_prob_fn.init(init_key, x0, c0)
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    # ====================================================================
    # Training  (power-affine family)
    # ====================================================================

    def loss_fn(self, params, x_batch, c_batch):
        return -jnp.mean(self.log_prob_fn.apply(params, x_batch, c_batch))

    @partial(jax.jit, static_argnums=0, donate_argnums=(1, 2))
    def _update(self, params, opt_state, x_batch, c_batch):
        grads = jax.grad(self.loss_fn)(params, x_batch, c_batch)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def fit(self, mode='pre-loaded', num_epochs=100, batch_size=256):
        if mode == 'pre-loaded':
            for _ in trange(num_epochs, colour='green',
                            desc='Training ConditionalMADEFlow (pre-loaded)'):
                self.key, subkey = jr.split(self.key)
                for x_batch, c_batch in _iter_batches(
                    self._data, self._context,
                    self.x_norm, self.c_norm,
                    batch_size, subkey,
                ):
                    self.params, self.opt_state = self._update(
                        self.params, self.opt_state, x_batch, c_batch)
        else:
            for _ in trange(num_epochs, colour='blue',
                            desc='Training ConditionalMADEFlow (on-the-fly)'):
                rand_idx     = random.randint(0, len(self.tset_paths) - 1)
                training_set = jnp.load(self.tset_paths[rand_idx])
                idxs         = jnp.array(random.sample(
                    range(training_set.shape[0]),
                    k=min(batch_size, training_set.shape[0]),
                ))
                training_set = training_set[idxs]
                x_batch = self.normalize_data(
                    training_set[:, :self.last_feature_idx])
                c_batch = self.normalize_context(
                    training_set[:, self.last_feature_idx:])
                self.params, self.opt_state = self._update(
                    self.params, self.opt_state, x_batch, c_batch)

    # ====================================================================
    # Public API  (power-affine family)
    # ====================================================================

    @_params_jit
    def log_prob(self, x, c):
        """
        log p(x | c) in physical space.

        Fast: O(num_layers) parallel MADE evaluations.
        """
        x_n = self.normalize_data(x)
        c_n = self.normalize_context(c)
        logp_norm = self.log_prob_fn.apply(self.params, x_n, c_n)
        logdet    = self._logdet_normalize_data(x)
        return logp_norm + logdet

    def sample(self, c, num_samples, chunk_size=None):
        """
        Draw samples from p(x | c).

        SLOW: O(num_layers × D) sequential MADE calls per sample.
        Use chunk_size to bound memory for large num_samples.

        Parameters
        ----------
        c           : (C,) | (1,C) | (num_samples, C)  physical context
        num_samples : int
        chunk_size  : int or None
        """
        c = jnp.atleast_2d(jnp.asarray(c))
        if c.shape[0] == 1:
            c = jnp.tile(c, (num_samples, 1))
        c_n = self.normalize_context(c)

        if chunk_size is None:
            self.key, subkey = jr.split(self.key)
            z        = jr.normal(subkey, (num_samples, self.D))
            x_norm   = self.forward_fn.apply(self.params, z, c_n)
        else:
            chunks = []
            for start in range(0, num_samples, chunk_size):
                end = min(start + chunk_size, num_samples)
                self.key, subkey = jr.split(self.key)
                z       = jr.normal(subkey, (end - start, self.D))
                x_chunk = self.forward_fn.apply(
                    self.params, z, c_n[start:end])
                chunks.append(x_chunk)
            x_norm = jnp.concatenate(chunks, axis=0)

        return self.denormalize_data(x_norm)

    @_params_jit
    def forward_pass(self, z, c):
        """
        Latent z → physical x  (slow, sequential).

        Parameters
        ----------
        z : (batch, D)
        c : (batch, C)  physical-space context
        """
        c_n    = self.normalize_context(c)
        x_norm = self.forward_fn.apply(self.params, z, c_n)
        return self.denormalize_data(x_norm)

    @_params_jit
    def backward_pass(self, x, c):
        """
        Physical x → latent z  (fast, parallel).

        Parameters
        ----------
        x : (batch, D)  physical-space points
        c : (batch, C)  physical-space context
        """
        x_n = self.normalize_data(x)
        c_n = self.normalize_context(c)
        return self.inverse_fn.apply(self.params, x_n, c_n)

    @_params_jit
    def backward_pass_with_logdet(self, x, c):
        """
        Physical x → (z, log|det J|)  (fast, parallel).

        log|det J| here is log|det J_{x→z}| including the normalisation
        Jacobian, so:
            log p(x|c) = log p_z(z) + log|det J_{x→z}|
        """
        x_n = self.normalize_data(x)
        c_n = self.normalize_context(c)
        z, logdet_flow = self.inverse_and_logdet_fn.apply(
            self.params, x_n, c_n)
        logdet_norm = self._logdet_normalize_data(x)
        return z, logdet_flow + logdet_norm

    # ====================================================================
    # Save / load  (power-affine family)
    # ====================================================================

    def save_params(self):
        leaves, _ = jax.tree_util.tree_flatten(self.params)
        save_dict = {f"leaf_{i}": np.asarray(l)
                     for i, l in enumerate(leaves)}
        np.savez(self.flow_save_path, **save_dict)

    def load_params(self):
        data       = np.load(self.flow_load_path, allow_pickle=True)
        _, treedef = jax.tree_util.tree_flatten(self.params)
        leaves     = [jnp.asarray(data[f"leaf_{i}"])
                      for i in range(treedef.num_leaves)]
        self.params = jax.tree_util.tree_unflatten(treedef, leaves)

# ═════════════════════════════════════════════════════════════════════════════
# FlowMatching
# ═════════════════════════════════════════════════════════════════════════════

class FlowMatching:
    """
    Unconditional Flow Matching  (optimal-transport straight paths).

    Replaces the RQS bijector stack with a single vector-field MLP
    v_theta(x_t, t) trained to push N(0,I) → p(x) along straight paths.

    Training objective (CFM / OT-CFM):
        x_t    = (1 - t) * x0 + t * x1          straight path
        target = x1 - (1 - sigma_min) * x0       constant vector field
        loss   = E[ || v(x_t, t) - target ||^2 ]

    Sampling:
        Integrate dx/dt = v_theta(x_t, t) from t=0 to t=1
        using Euler (fast) or RK4 (accurate).

    Pre/post-processing (power-affine normalisation) is identical to
    the Flow class so the two are interchangeable.

    Parameters
    ----------
    data : array (N, D)
        Training dataset.
    hidden_size : int
        Width of vector-field MLP hidden layers.
    num_layers : int
        Number of hidden layers in the vector-field MLP.
    learning_rate : float
        Adam learning rate.
    B : float
        Bound for normalized space (same role as in Flow).
    p : float
        Power transform parameter.
    sigma_min : float
        Minimum path noise; keeps paths slightly stochastic.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        data,
        hidden_size=256,
        num_layers=4,
        learning_rate=1e-4,
        B=6.0,
        p=0.3,
        sigma_min=1e-4,
        seed=0,
    ):
        self.N, self.D = data.shape
        self.B         = float(B) - 1
        self.sigma_min = sigma_min

        # ── Power transform (same as Flow) ───────────────────────────────
        self.alpha      = jnp.log(p)
        dummy           = jnp.sign(data) * jnp.abs(data) ** jnp.exp(self.alpha)
        min_x           = dummy.min(axis=0)
        max_x           = dummy.max(axis=0)
        self.mean_x     = (max_x + min_x) / 2
        self.half_range = (max_x - min_x) / 2

        self.x_norm_data = self.to_unit_interval(data)
        assert self.x_norm_data.min() > -B
        assert self.x_norm_data.max() <  B

        # ── Architecture ─────────────────────────────────────────────────
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.learning_rate = learning_rate
        self.key           = jr.PRNGKey(seed)

        self._build_model()
        self._init_params()

    # ====================================================================
    # Data transforms  (identical to Flow)
    # ====================================================================

    @partial(jax.jit, static_argnums=0)
    def to_unit_interval(self, arr):
        dummy = jnp.sign(arr) * jnp.abs(arr) ** jnp.exp(self.alpha)
        return (dummy - self.mean_x[None, :]) / self.half_range[None, :] * self.B

    @partial(jax.jit, static_argnums=0)
    def from_unit_interval(self, arr):
        dummy = self.half_range[None, :] / self.B * arr + self.mean_x[None, :]
        return jnp.sign(dummy) * jnp.abs(dummy) ** (1.0 / jnp.exp(self.alpha))

    @partial(jax.jit, static_argnums=0)
    def logdet_to_unit_interval(self, arr):
        gamma = jnp.exp(self.alpha)
        return jnp.sum(
            jnp.log(gamma)
            + (gamma - 1.0) * jnp.log(jnp.abs(arr))
            + jnp.log(self.B / self.half_range),
            axis=-1,
        )

    @partial(jax.jit, static_argnums=0)
    def logdet_from_unit_interval(self, arr):
        beta  = 1.0 / jnp.exp(self.alpha)
        dummy = self.half_range[None, :] / self.B * arr + self.mean_x[None, :]
        return jnp.sum(
            jnp.log(self.half_range / self.B)
            + jnp.log(beta)
            + (beta - 1.0) * jnp.log(jnp.abs(dummy)),
            axis=-1,
        )

    # ====================================================================
    # Batching  (identical to Flow)
    # ====================================================================

    @staticmethod
    def get_batches(data, batch_size, key):
        N   = data.shape[0]
        idx = jax.random.permutation(key, N)
        for i in range(0, N, batch_size):
            yield data[idx[i: i + batch_size]]

    # ====================================================================
    # Vector field MLP
    # ====================================================================

    @staticmethod
    def make_vector_field(D, hidden_size, num_layers):

        @hk.without_apply_rng
        @hk.transform
        def vector_field_fn(x_t, t):
            # Sinusoidal time embedding  (replaces raw scalar t)
            half = hidden_size // 2
            freqs = jnp.exp(
                -jnp.log(10000.0) * jnp.arange(half) / (half - 1)
            )                                               # (half,)
            t_feat = jnp.atleast_1d(t)[:, None] * freqs[None, :]  # (batch, half)
            t_emb  = jnp.concatenate(
                [jnp.sin(t_feat), jnp.cos(t_feat)], axis=-1
            )                                               # (batch, hidden_size)

            inp = jnp.concatenate([x_t, t_emb], axis=-1)   # (batch, D + hidden_size)

            layers = []
            for _ in range(num_layers):
                layers += [hk.Linear(hidden_size), jax.nn.silu]
            layers.append(
                hk.Linear(D, w_init=jnp.zeros, b_init=jnp.zeros)
            )
            return hk.Sequential(layers)(inp)

        return vector_field_fn

    # ====================================================================
    # Build / init
    # ====================================================================

    def _build_model(self):
        self.vf_fn = self.make_vector_field(
            self.D, self.hidden_size, self.num_layers
        )

    def _init_params(self):
        self.key, k = jr.split(self.key)
        x0 = jnp.zeros((2, self.D))
        t0 = jnp.zeros((2,))
        self.params    = self.vf_fn.init(k, x0, t0)
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    # ====================================================================
    # Training
    # ====================================================================

    def loss_fn(self, params, x1, key):
        """
        CFM loss  E[ || v(x_t, t) - (x1 - (1-sigma_min)*x0) ||^2 ]
        """
        k1, k2  = jr.split(key)
        x0      = jr.normal(k1, x1.shape)
        t       = jr.uniform(k2, (x1.shape[0],))
        t_b     = t[:, None]
        x_t     = (1.0 - (1.0 - self.sigma_min) * t_b) * x0 + t_b * x1
        target  = x1 - (1.0 - self.sigma_min) * x0
        v_pred  = self.vf_fn.apply(params, x_t, t)
        return jnp.mean((v_pred - target) ** 2)

    @partial(jax.jit, static_argnums=0)
    def update(self, params, opt_state, batch, key):
        loss, grads = jax.value_and_grad(self.loss_fn)(params, batch, key)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def fit(self, num_epochs=100, batch_size=256):
        """
        Train via mini-batch gradient descent.
        """
        for _ in trange(num_epochs, colour='green', desc='Training FlowMatching'):
            self.key, data_key, step_key = jr.split(self.key, 3)
            for batch in self.get_batches(self.x_norm_data, batch_size, data_key):
                self.key, step_key = jr.split(self.key)
                self.params, self.opt_state, _ = self.update(
                    self.params, self.opt_state, batch, step_key
                )

    # ====================================================================
    # ODE integrators  (jax.lax.scan → single XLA program per call)
    # ====================================================================

    @_params_jit(static=("num_steps",))
    def _euler(self, x0, num_steps):
        """
        More steps near t=1 where the tail vector field is largest.
        Uses a cosine schedule instead of uniform spacing.
        """
        # cosine spacing: dense near t=1
        i    = jnp.arange(num_steps)
        ts   = 1.0 - jnp.cos(i / num_steps * jnp.pi / 2)   # (num_steps,)
        dts  = jnp.diff(ts, append=jnp.array([1.0]))         # step sizes

        def step(x, t_and_dt):
            t_scalar, dt = t_and_dt
            t = jnp.full((x.shape[0],), t_scalar)
            return x + dt * self.vf_fn.apply(self.params, x, t), None

        x1, _ = jax.lax.scan(step, x0, (ts, dts))
        return x1

    @_params_jit(static=("num_steps",))
    def _rk4(self, x0, num_steps):
        """RK4 integrator.  4 MLP calls per step, much better quality."""
        dt = 1.0 / num_steps

        def step(x, t_scalar):
            t  = jnp.full((x.shape[0],), t_scalar)
            k1 = self.vf_fn.apply(self.params, x,              t         )
            k2 = self.vf_fn.apply(self.params, x + dt/2 * k1, t + dt/2  )
            k3 = self.vf_fn.apply(self.params, x + dt/2 * k2, t + dt/2  )
            k4 = self.vf_fn.apply(self.params, x + dt   * k3, t + dt    )
            return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4), None

        ts      = jnp.linspace(0.0, 1.0 - dt, num_steps)
        x1, _   = jax.lax.scan(step, x0, ts)
        return x1

    # ====================================================================
    # Public API  (mirrors Flow exactly)
    # ====================================================================

    def sample(self, num_samples, num_steps=10, method='euler'):
        """
        Draw samples in physical space.

        Parameters
        ----------
        num_samples : int
        num_steps   : int    ODE steps.  1–4 for speed, 10–20 for quality.
        method      : 'euler' | 'rk4'
        """
        self.key, subkey = jr.split(self.key)
        x0       = jr.normal(subkey, (num_samples, self.D))
        integrate = self._euler if method == 'euler' else self._rk4
        x1_norm  = integrate(x0, num_steps)
        return self.from_unit_interval(x1_norm)

    @_params_jit(static=("num_steps", "method"))
    def forward_pass(self, z, num_steps=10, method='euler'):
        """
        Map latent samples z ~ N(0,I) to physical space.

        Parameters
        ----------
        z         : (batch, D)
        num_steps : int
        method    : 'euler' | 'rk4'
        """
        integrate = self._euler if method == 'euler' else self._rk4
        x1_norm   = integrate(z, num_steps)
        return self.from_unit_interval(x1_norm)

    def backward_pass(self, x, num_steps=10, method='euler'):
        """
        Map physical-space samples back to latent space  (reverse ODE).

        Runs the ODE backwards from t=1 to t=0.

        Parameters
        ----------
        x         : (batch, D)  physical-space points
        num_steps : int
        method    : 'euler' | 'rk4'
        """
        x_norm = self.to_unit_interval(x)

        dt = 1.0 / num_steps

        def euler_step_back(z, t_scalar):
            t = jnp.full((z.shape[0],), t_scalar)
            return z - dt * self.vf_fn.apply(self.params, z, t), None

        def rk4_step_back(z, t_scalar):
            t  = jnp.full((z.shape[0],), t_scalar)
            k1 = self.vf_fn.apply(self.params, z,              t         )
            k2 = self.vf_fn.apply(self.params, z - dt/2 * k1, t - dt/2  )
            k3 = self.vf_fn.apply(self.params, z - dt/2 * k2, t - dt/2  )
            k4 = self.vf_fn.apply(self.params, z - dt   * k3, t - dt    )
            return z - dt / 6 * (k1 + 2*k2 + 2*k3 + k4), None

        step_fn = euler_step_back if method == 'euler' else rk4_step_back
        ts      = jnp.linspace(1.0, dt, num_steps)          # t=1 → t=0
        z0, _   = jax.lax.scan(step_fn, x_norm, ts)
        return z0

    def save_params(self, path):
        leaves, _  = jax.tree_util.tree_flatten(self.params)
        save_dict  = {f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)}
        np.savez(path, **save_dict)

    def load_params(self, path):
        data     = np.load(path, allow_pickle=True)
        _, treedef = jax.tree_util.tree_flatten(self.params)
        leaves   = [jnp.asarray(data[f"leaf_{i}"]) for i in range(treedef.num_leaves)]
        self.params = jax.tree_util.tree_unflatten(treedef, leaves)

# ═════════════════════════════════════════════════════════════════════════════
# ConditionalFlowMatching
# ═════════════════════════════════════════════════════════════════════════════

class ConditionalFlowMatching:
    """
    Conditional Flow Matching  —  models p(x | c).

    Interface mirrors ConditionalMADEFlow (power-affine family):
    - Same power-affine normalisation for both data and context.
    - Same memmapped-safe batch iterator.
    - Same fit() modes ('pre-loaded' and tset_paths).
    - Same save/load interface.

    The RQS bijector stack is replaced by a single vector-field MLP:
        v_theta(x_t, t, c_norm) → (D,)
    which receives [x_t ‖ t ‖ c_norm] as input.

    Sampling uses Euler or RK4 ODE integration.

    Parameters
    ----------
    data, context : array-like (N, D) / (N, C)
        May be numpy memmaps.
    data_min / max, context_min / max : (D,) / (C,)
        Per-feature physical-space bounds.
    normalizer_dict : dict or None
        Pre-fitted normalizer dict with keys 'data' and 'context'.
        If supplied, data_min/max and context_min/max are ignored.
    tset_paths : list of str, optional
        Paths to .npy training-set files (used in fit mode 'on-the-fly').
    last_feature_index : int, optional
        Column split: data = arr[:, :last_feature_index],
                      context = arr[:, last_feature_index:]
    flow_save_path / flow_load_path : str, optional
        Paths for save_params / load_params.
    p, B : float
        Normalisation hyperparameters.
    hidden_size : int
        Width of vector-field MLP hidden layers.
    num_layers : int
        Number of hidden layers in the vector-field MLP.
    learning_rate : float
    sigma_min : float
        Minimum path noise.
    seed : int
    """

    def __init__(
        self,
        data,
        context,
        data_min=None,
        data_max=None,
        context_min=None,
        context_max=None,
        normalizer_dict=None,
        tset_paths=None,
        last_feature_index=None,
        last_feature_index_for_context=None,
        flow_save_path=None,
        flow_load_path=None,
        p=0.1,
        B=4.0,
        hidden_size=256,
        num_layers=4,
        learning_rate=1e-4,
        sigma_min=1e-4,
        seed=0,
    ):
        self.N, self.D      = data.shape[0], data.shape[1]
        self.C              = context.shape[1]
        self.sigma_min      = sigma_min
        self.tset_paths     = tset_paths
        self.last_feature_idx = last_feature_index
        self.last_feature_index_for_context = last_feature_index_for_context
        self.flow_save_path = flow_save_path
        self.flow_load_path = flow_load_path

        self._data    = data
        self._context = context

        # ── Normalizers ───────────────────────────────────────────────────
        if normalizer_dict is not None and any(normalizer_dict):
            self.x_norm = normalizer_dict['data']
            self.c_norm = normalizer_dict['context']
        else:
            self.x_norm = _fit_normalizer(data_min,    data_max,    p, B)
            self.c_norm = _fit_normalizer(context_min, context_max, p, B)

        self._x_alpha      = jnp.float64(self.x_norm['alpha'])
        self._x_mean       = jnp.asarray(self.x_norm['mean'])
        self._x_half_range = jnp.asarray(self.x_norm['half_range'])
        self._x_B          = jnp.float64(self.x_norm['B'])

        self._c_alpha      = jnp.float64(self.c_norm['alpha'])
        self._c_mean       = jnp.asarray(self.c_norm['mean'])
        self._c_half_range = jnp.asarray(self.c_norm['half_range'])
        self._c_B          = jnp.float64(self.c_norm['B'])

        # ── Architecture ─────────────────────────────────────────────────
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.learning_rate = learning_rate
        self.key           = jr.PRNGKey(seed)

        self._build_model()
        self._init_params()

    # ====================================================================
    # Normalisation  (power-affine family)
    # ====================================================================

    @partial(jax.jit, static_argnums=0)
    def normalize_data(self, x):
        return _from_coefficients_jnp(
            x, self._x_alpha, self._x_mean, self._x_half_range, self._x_B
        )

    @partial(jax.jit, static_argnums=0)
    def denormalize_data(self, x_norm):
        return _to_coefficients_jnp(
            x_norm, self._x_alpha, self._x_mean, self._x_half_range, self._x_B
        )

    @partial(jax.jit, static_argnums=0)
    def normalize_context(self, c):
        return _from_coefficients_jnp(
            c, self._c_alpha, self._c_mean, self._c_half_range, self._c_B
        )

    @partial(jax.jit, static_argnums=0)
    def denormalize_context(self, c_norm):
        return _to_coefficients_jnp(
            c_norm, self._c_alpha, self._c_mean, self._c_half_range, self._c_B
        )

    @partial(jax.jit, static_argnums=0)
    def _logdet_normalize_data(self, x):
        return _logdet_from_coefficients_jnp(
            x, self._x_alpha, self._x_mean, self._x_half_range, self._x_B
        )

    @partial(jax.jit, static_argnums=0)
    def _logdet_denormalize_data(self, x_norm):
        return _logdet_to_coefficients_jnp(
            x_norm, self._x_alpha, self._x_mean, self._x_half_range, self._x_B
        )

    # ====================================================================
    # Vector field MLP
    # ====================================================================

    @staticmethod
    def make_vector_field(D, C, hidden_size, num_layers):

        @hk.without_apply_rng
        @hk.transform
        def vector_field_fn(x_t, t, c_norm):
            # ── Sinusoidal time embedding ───────────────────────────────
            # Gives the network hidden_size features for time instead of 1.
            # Critical for high-dimensional data where the field varies
            # rapidly with t.
            half  = hidden_size // 2
            freqs = jnp.exp(
                -jnp.log(10000.0) * jnp.arange(half) / (half - 1)
            )                                               # (half,)
            t_feat = jnp.atleast_1d(t)[:, None] * freqs[None, :]
            t_emb  = jnp.concatenate(
                [jnp.sin(t_feat), jnp.cos(t_feat)], axis=-1
            )                                               # (batch, hidden_size)

            # ── Input projection ────────────────────────────────────────
            # Project x_t and c separately before combining — avoids the
            # 450-dim input immediately bottlenecking through a 1024-wide
            # layer that also has to process t and c simultaneously.
            x_proj = hk.Linear(hidden_size, name='x_proj')(x_t)
            c_proj = hk.Linear(hidden_size, name='c_proj')(c_norm)

            h = x_proj + t_emb + c_proj    # additive fusion, shape (batch, H)

            # ── Residual MLP ─────────────────────────────────────────────
            # Residual connections help gradients flow in deep networks
            # and are essential for D=450 where the output must closely
            # track the input (the field is often small).
            for _ in range(num_layers):
                h_new = hk.Linear(hidden_size)(h)
                h_new = jax.nn.silu(h_new)
                h_new = hk.Linear(hidden_size)(h_new)
                h     = h + h_new           # residual skip connection

            return hk.Linear(D, w_init=jnp.zeros, b_init=jnp.zeros)(h)

        return vector_field_fn

    # ====================================================================
    # Build / init
    # ====================================================================

    def _build_model(self):
        self.vf_fn = self.make_vector_field(
            self.D, self.C, self.hidden_size, self.num_layers
        )

    def _init_params(self):
        self.key, k = jr.split(self.key)
        x0 = jnp.zeros((2, self.D))
        t0 = jnp.zeros((2,))
        c0 = jnp.zeros((2, self.C))
        self.params    = self.vf_fn.init(k, x0, t0, c0)
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    # ====================================================================
    # Training
    # ====================================================================

    def loss_fn(self, params, x1, c_norm, key):
        """
        Conditional CFM loss  E[ || v(x_t, t, c) - target ||^2 ]
        """
        k1, k2 = jr.split(key)
        x0     = jr.normal(k1, x1.shape)
        t      = jr.uniform(k2, (x1.shape[0],))
        t_b    = t[:, None]
        x_t    = (1.0 - (1.0 - self.sigma_min) * t_b) * x0 + t_b * x1
        target = x1 - (1.0 - self.sigma_min) * x0
        v_pred = self.vf_fn.apply(params, x_t, t, c_norm)
        return jnp.mean((v_pred - target) ** 2)

    @partial(jax.jit, static_argnums=0, donate_argnums=(1, 2))
    def _update(self, params, opt_state, x_batch, c_batch, key):
        loss, grads = jax.value_and_grad(self.loss_fn)(
            params, x_batch, c_batch, key
        )
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def fit(self, mode='pre-loaded', num_epochs=100, batch_size=256):
        """
        Train the model.

        mode='pre-loaded'  : uses self._data / self._context via _iter_batches
                             (memmapped-safe, lazy reads).
        mode='on-the-fly'  : randomly picks a file from self.tset_paths each
                             epoch.
        """
        if mode == 'pre-loaded':
            for _ in trange(num_epochs, colour='green',
                            desc='Training ConditionalFlowMatching (pre-loaded)'):
                self.key, data_key, step_key = jr.split(self.key, 3)
                for x_batch, c_batch in _iter_batches(
                    self._data, self._context,
                    self.x_norm, self.c_norm,
                    batch_size, data_key,
                ):
                    self.key, step_key = jr.split(self.key)
                    self.params, self.opt_state, _ = self._update(
                        self.params, self.opt_state, x_batch, c_batch, step_key
                    )
        else:
            for _ in trange(num_epochs, colour='blue',
                            desc='Training ConditionalFlowMatching (on-the-fly)'):
                rand_idx     = random.randint(0, len(self.tset_paths) - 1)
                training_set = jnp.load(self.tset_paths[rand_idx])
                idxs         = jnp.array(random.sample(
                    range(training_set.shape[0]),
                    k=min(batch_size, training_set.shape[0]),
                ))
                training_set = training_set[idxs]
                x_batch = self.normalize_data(training_set[:, :self.last_feature_idx])
                c_batch = self.normalize_context(training_set[:, self.last_feature_idx:self.last_feature_index_for_context])
                self.key, step_key = jr.split(self.key)
                self.params, self.opt_state, _ = self._update(
                    self.params, self.opt_state, x_batch, c_batch, step_key
                )

    # ====================================================================
    # ODE integrators
    # ====================================================================

    @_params_jit(static=("num_steps",))
    def _euler(self, x0, c_norm, num_steps):
        dt = 1.0 / num_steps

        def step(x, t_scalar):
            t = jnp.full((x.shape[0],), t_scalar)
            return x + dt * self.vf_fn.apply(self.params, x, t, c_norm), None

        ts    = jnp.linspace(0.0, 1.0 - dt, num_steps)
        x1, _ = jax.lax.scan(step, x0, ts)
        return x1

    @_params_jit(static=("num_steps",))
    def _rk4(self, x0, c_norm, num_steps):
        dt = 1.0 / num_steps

        def step(x, t_scalar):
            t  = jnp.full((x.shape[0],), t_scalar)
            k1 = self.vf_fn.apply(self.params, x,              t,        c_norm)
            k2 = self.vf_fn.apply(self.params, x + dt/2 * k1, t + dt/2, c_norm)
            k3 = self.vf_fn.apply(self.params, x + dt/2 * k2, t + dt/2, c_norm)
            k4 = self.vf_fn.apply(self.params, x + dt   * k3, t + dt,   c_norm)
            return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4), None

        ts    = jnp.linspace(0.0, 1.0 - dt, num_steps)
        x1, _ = jax.lax.scan(step, x0, ts)
        return x1

    # ====================================================================
    # Public API  (power-affine family)
    # ====================================================================

    def sample(self, c, num_samples, num_steps=10, method='euler'):
        """
        Draw samples from p(x | c).

        Parameters
        ----------
        c           : (C,) | (1,C) | (num_samples, C)  physical-space context
        num_samples : int
        num_steps   : int    ODE steps
        method      : 'euler' | 'rk4'
        """
        c = jnp.atleast_2d(jnp.asarray(c, dtype=jnp.float64))
        if c.shape[0] == 1:
            c = jnp.tile(c, (num_samples, 1))

        c_n      = self.normalize_context(c)
        integrate = self._euler if method == 'euler' else self._rk4

        self.key, subkey = jr.split(self.key)
        x0      = jr.normal(subkey, (num_samples, self.D))
        x1_norm = integrate(x0, c_n, num_steps)
        return self.denormalize_data(x1_norm)

    @_params_jit(static=("num_steps", "method"))
    def forward_pass(self, z, c, num_steps=10, method='euler'):
        """
        Map latent z ~ N(0,I) to physical space given context c.

        Parameters
        ----------
        z         : (batch, D)
        c         : (batch, C)  physical-space context
        num_steps : int
        method    : 'euler' | 'rk4'
        """
        c_n      = self.normalize_context(c)
        integrate = self._euler if method == 'euler' else self._rk4
        x1_norm  = integrate(z, c_n, num_steps)
        return self.denormalize_data(x1_norm)

    def backward_pass(self, x, c, num_steps=10, method='euler'):
        """
        Map physical-space samples back to latent space (reverse ODE).

        Parameters
        ----------
        x         : (batch, D)  physical-space points
        c         : (batch, C)  physical-space context
        num_steps : int
        method    : 'euler' | 'rk4'
        """
        x_norm = self.normalize_data(x)
        c_n    = self.normalize_context(c)
        dt     = 1.0 / num_steps

        def euler_back(z, t_scalar):
            t = jnp.full((z.shape[0],), t_scalar)
            return z - dt * self.vf_fn.apply(self.params, z, t, c_n), None

        def rk4_back(z, t_scalar):
            t  = jnp.full((z.shape[0],), t_scalar)
            k1 = self.vf_fn.apply(self.params, z,              t,        c_n)
            k2 = self.vf_fn.apply(self.params, z - dt/2 * k1, t - dt/2, c_n)
            k3 = self.vf_fn.apply(self.params, z - dt/2 * k2, t - dt/2, c_n)
            k4 = self.vf_fn.apply(self.params, z - dt   * k3, t - dt,   c_n)
            return z - dt / 6 * (k1 + 2*k2 + 2*k3 + k4), None

        step_fn  = euler_back if method == 'euler' else rk4_back
        ts       = jnp.linspace(1.0, dt, num_steps)
        z0, _    = jax.lax.scan(step_fn, x_norm, ts)
        return z0

    # ====================================================================
    # Save / load  (power-affine family)
    # ====================================================================

    def save_params(self):
        leaves, _  = jax.tree_util.tree_flatten(self.params)
        save_dict  = {f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)}
        np.savez(self.flow_save_path, **save_dict)

    def load_params(self):
        data       = np.load(self.flow_load_path, allow_pickle=True)
        _, treedef = jax.tree_util.tree_flatten(self.params)
        leaves     = [jnp.asarray(data[f"leaf_{i}"]) for i in range(treedef.num_leaves)]
        self.params = jax.tree_util.tree_unflatten(treedef, leaves)

# ═════════════════════════════════════════════════════════════════════════════
# SASFlow
# ═════════════════════════════════════════════════════════════════════════════

def _joint_ctx_forward(c, t, kind):
    """Context columns of a joint row -> normalized, with log|det| (summed)."""
    if kind == "probit":
        u = ndtri((c - t["lo"]) / (t["hi"] - t["lo"]))
        ld = -jnp.log(t["hi"] - t["lo"]) + 0.5 * u**2 + 0.5 * LOG2PI
        return u, jnp.sum(ld, axis=-1)
    if kind == "standardize":
        u = (c - t["mean"]) / t["std"]
        return u, jnp.sum(-jnp.log(t["std"]) * jnp.ones_like(u), axis=-1)
    return sas_forward(c, t)                                  # "sas"


def _joint_ctx_inverse(u, t, kind):
    if kind == "probit":
        return t["lo"] + (t["hi"] - t["lo"]) * jax.scipy.special.ndtr(u)
    if kind == "standardize":
        return t["mean"] + t["std"] * u
    return sas_inverse(u, t)[0]


class SASFlow:
    """
    Unconditional RQS normalizing flow with sinh-arcsinh preprocessing.

    It can also learn a JOINT density p(theta, y) from a simulation library:
    pass data of shape (n_draws, n_real, n_features) and the per-draw
    parameters as `context` (n_draws, P).  Every (draw, realization) pair
    becomes one row [theta_i, y_ij] of dimension P + n_features, without
    ever materializing the broadcast array.  This is the joint counterpart
    of ConditionalFlow(y, theta): same arguments, but theta is modelled
    rather than conditioned on, so forward_pass(z) returns [theta, y].

    Pipeline
    --------
        physical row x = [theta | y]
            |  theta: context transform ("probit" | "standardize" | "sas")
            |  y    : per-feature SAS map, fitted so each marginal is ~ N(0, 1)
            v
        normalized row
            |  stack of RQS masked-coupling layers (learned)
            v
        latent z ~ N(0, I)

    Per-feature SAS normalization (feature j):
        u = (x - loc_j) / s_j
        a = delta_j * arcsinh(u) - eps_j
        y = (sinh(a) - mu_j) / sd_j

        dy/dx = delta_j * cosh(a) / (s_j * sd_j * sqrt(1 + u^2))

    loc, s, delta and eps are fitted by maximum likelihood (so that sinh(a) is
    as close to N(0, 1) as possible); mu and sd are a final exact affine
    correction.  Values that land outside the spline range [-B, B] pass
    through the splines as the identity, so the tails are those of the
    Gaussian base pulled back through the SAS map.

    For theta, "probit" maps a uniform (Latin-hypercube) design exactly to
    N(0, 1), and its inverse keeps every sampled theta inside the bounds.
    All theta transforms include their Jacobian, so log_prob is the density
    of the physical row.

    Design notes
    ------------
    * Every jitted function takes params / normalizers explicitly, so results
      always use the current weights.
    * Batches are read lazily and normalized on the fly, so numpy memmaps
      work.  In the library layout an epoch is one pass over all
      (draw, realization) pairs in random order, so every draw counts equally.
    * For D == 1 every coupling layer transforms the single feature.

    Parameters
    ----------
    data : array-like (N, n_features) or (n_draws, n_real, n_features).
        A 1-D array is treated as (N, 1).  May be a memmap.
    context : array-like (n_draws, P) or None
        Per-draw parameters prepended to every row (library layout), or
        per-row columns prepended in the flat layout.  Modelled jointly.
    context_kind : "probit" | "standardize" | "sas"
        Transform for the context columns (ignored without context).
    context_bounds : (lo, hi) arrays of shape (P,) for "probit".  Default:
        the context min/max widened by 0.1% of the range.  Use your prior
        ranges so new theta values stay inside.
    context_jitter : float
        Std of Gaussian noise added to the normalized context columns of
        every training batch (fresh noise each batch).  A library samples
        theta at a finite set of draws, and a flexible joint flow can put
        spikes of density at exactly those draws instead of interpolating
        between them; jitter smooths p(theta) over the gaps.  0 disables it.
        Choose it by the held-out score (val_frac > 0), starting around
        0.1-0.3 (normalized units are ~N(0, 1)).
    normalizer : dict or None
        Pre-fitted SAS dict for the data features (output of
        `fit_sas_normalizer`).  Fitted from `data` if None.
    normalizer_max_samples : int
        Rows used when fitting the SAS normalizer.
    val_frac : float
        Fraction of draws (library layout) or rows (flat layout) held out.
        fit() then reports the mean held-out log p each epoch.  Held-out
        draws are never seen in training, so this measures interpolation in
        theta.  Default 0 (no validation).
    flow_num_layers, hidden_size, mlp_num_layers, num_bins : int
        Flow architecture.
    B : float
        Spline range is [-B, B].
    learning_rate : float
    grad_clip : float or None
        Global-norm gradient clipping.  None disables it.
    seed : int
    """

    def __init__(
        self,
        data,
        context=None,
        context_kind="probit",
        context_bounds=None,
        context_jitter=0.0,
        normalizer=None,
        normalizer_max_samples=200_000,
        val_frac=0.0,
        flow_num_layers=4,
        hidden_size=128,
        mlp_num_layers=2,
        num_bins=8,
        B=5.0,
        learning_rate=1e-4,
        grad_clip=1.0,
        seed=0,
    ):
        # ---------- layout
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.ndim == 3:
            self.layout = "library"
            self.n_groups, self.R, self.K = data.shape
        elif data.ndim == 2:
            self.layout = "flat"
            (self.n_groups, self.K), self.R = data.shape, 1
        else:
            raise ValueError("data must be (N, n_features) or (n_draws, n_real, n_features); "
                             f"got {data.shape}")
        self._data = data
        self.N = self.n_groups * self.R            # number of training rows (pairs)

        if context is not None:
            context = np.asarray(context, dtype=np.float64)
            if context.ndim == 1:
                context = context[:, None]
            if context.shape[0] != self.n_groups:
                raise ValueError(f"context must have {self.n_groups} rows; got {context.shape}")
        self.context_raw = context
        self.P = 0 if context is None else context.shape[1]
        self.D = self.P + self.K
        self.event_shape = (self.D,)

        self.flow_num_layers = flow_num_layers
        self.hidden_sizes = [hidden_size] * mlp_num_layers
        self.num_bins = num_bins
        self.B = float(B)
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.key = jr.PRNGKey(seed)
        self.rng = np.random.default_rng(seed)
        self.loss_history = []
        self.val_history = []
        self.context_jitter = float(context_jitter)

        # ---------- train / validation split (by draw in the library layout)
        perm = self.rng.permutation(self.n_groups)
        n_val = int(round(val_frac * self.n_groups))
        self.val_groups, self.train_groups = np.sort(perm[:n_val]), np.sort(perm[n_val:])
        rows_of = lambda g: (g[:, None] * self.R + np.arange(self.R)).ravel()
        self.train_rows, self.val_rows = rows_of(self.train_groups), rows_of(self.val_groups)

        # ---------- context transform
        self.context_kind = context_kind if self.P else "none"
        if self.P:
            if context_kind == "probit":
                if context_bounds is None:
                    lo, hi = context.min(0), context.max(0)
                    pad = 1e-3 * (hi - lo)
                    context_bounds = (lo - pad, hi + pad)
                lo, hi = (np.asarray(b, np.float64).reshape(self.P) for b in context_bounds)
                if np.any(context <= lo) or np.any(context >= hi):
                    raise ValueError("context values must lie strictly inside context_bounds")
                ctx = dict(lo=lo, hi=hi)
            elif context_kind == "standardize":
                sd = context.std(0)
                ctx = dict(mean=context.mean(0), std=np.where(sd > 0, sd, 1.0))
            elif context_kind == "sas":
                ctx = fit_sas_normalizer(context, max_samples=normalizer_max_samples, seed=seed)
            else:
                raise ValueError("context_kind must be 'probit', 'standardize' or 'sas'")
            self._set_ctx(ctx)
        else:
            self._set_ctx({})

        # ---------- SAS normalizer for the data features
        if normalizer is None:
            if self.layout == "flat" and n_val == 0:
                normalizer = fit_sas_normalizer(data, max_samples=normalizer_max_samples, seed=seed)
            else:
                n = min(len(self.train_rows), normalizer_max_samples)
                idx = np.sort(self.rng.choice(self.train_rows, size=n, replace=False))
                normalizer = fit_sas_normalizer(self._read_y(idx))
        self.set_normalizer(normalizer)

        self._build_model()
        self._init_params()

    # ================================================================
    # Normalizers
    # ================================================================

    def _set_ctx(self, ctx):
        self.ctx_np = {k: np.asarray(v, np.float64) for k, v in ctx.items()}
        self.ctx = {k: jnp.asarray(v) for k, v in self.ctx_np.items()}
        self.u_all = (np.asarray(_joint_ctx_forward(jnp.asarray(self.context_raw), self.ctx,
                                                    self.context_kind)[0])
                      if self.P else None)

    def set_normalizer(self, normalizer):
        """Set the SAS normalizer of the data features (K = n_features entries)."""
        keys = ("loc", "s", "delta", "eps", "mu", "sd")
        self.normalizer_np = {k: np.asarray(normalizer[k], dtype=np.float64).reshape(self.K)
                              for k in keys}
        self.normalizer = {k: jnp.asarray(v) for k, v in self.normalizer_np.items()}

    def _norms(self):
        return dict(y=self.normalizer, c=self.ctx)

    def _row_forward(self, norms, x):
        """Physical rows [theta | y] -> normalized rows, log|det| (summed)."""
        if not self.P:
            return sas_forward(x, norms["y"])
        u, ld_c = _joint_ctx_forward(x[:, :self.P], norms["c"], self.context_kind)
        yh, ld_y = sas_forward(x[:, self.P:], norms["y"])
        return jnp.concatenate([u, yh], axis=-1), ld_c + ld_y

    def _row_inverse(self, norms, xn):
        """Normalized rows -> physical rows, log|det dx/dxn| (summed)."""
        x = sas_inverse(xn, norms["y"])[0] if not self.P else jnp.concatenate(
            [_joint_ctx_inverse(xn[:, :self.P], norms["c"], self.context_kind),
             sas_inverse(xn[:, self.P:], norms["y"])[0]], axis=-1)
        return x, -self._row_forward(norms, x)[1]

    def normalize(self, x):
        """Physical rows -> normalized rows (no Jacobian)."""
        return self._row_forward(self._norms(), jnp.atleast_2d(jnp.asarray(x, jnp.float64)))[0]

    def denormalize(self, y):
        """Normalized rows -> physical rows (no Jacobian)."""
        return self._row_inverse(self._norms(), jnp.atleast_2d(jnp.asarray(y, jnp.float64)))[0]

    def save_normalizer(self, path):
        """Data-feature SAS normalizer only (the context transform is saved by save_params)."""
        np.savez(path, **self.normalizer_np)

    @staticmethod
    def load_normalizer(path):
        with np.load(path) as f:
            return {k: f[k] for k in f.files}

    # ================================================================
    # Data access
    # ================================================================

    def _read_y(self, idx):
        """Data features of rows idx (sorted pair indices), physical units."""
        if self.layout == "library":
            rows = self._data[idx // self.R, idx % self.R]
        else:
            rows = self._data[idx]
        return np.asarray(rows, dtype=np.float64).reshape(len(idx), self.K)

    def rows(self, idx):
        """Physical joint rows [theta | y] for pair indices idx."""
        idx = np.asarray(idx)
        y = self._read_y(idx)
        if not self.P:
            return y
        return np.concatenate([self.context_raw[idx // self.R], y], axis=1)

    def _batch(self, idx):
        yn = _sas_forward_np(self._read_y(idx), self.normalizer_np)
        if self.P:
            yn = np.concatenate([self.u_all[idx // self.R], yn], axis=1)
        return jnp.asarray(yn)

    # ================================================================
    # Model
    # ================================================================

    @staticmethod
    def make_conditioner(event_shape, hidden_sizes, num_bijector_params):
        return hk.Sequential([
            hk.Flatten(preserve_dims=-len(event_shape)),
            hk.nets.MLP(hidden_sizes, activate_final=True),
            hk.Linear(np.prod(event_shape) * num_bijector_params,
                      w_init=jnp.zeros, b_init=jnp.zeros),
            hk.Reshape(event_shape + (num_bijector_params,), preserve_dims=-1),
        ])

    @staticmethod
    def make_flow_model(event_shape, num_layers, hidden_sizes, num_bins, B):
        D = int(np.prod(event_shape))
        if D == 1:
            mask = jnp.zeros(event_shape, dtype=bool)      # always transform the feature
        else:
            mask = (jnp.arange(D) % 2).reshape(event_shape).astype(bool)

        def bijector_fn(params):
            return distrax.RationalQuadraticSpline(params, range_min=-B, range_max=B)

        layers = []
        for _ in range(num_layers):
            layers.append(distrax.MaskedCoupling(
                mask=mask,
                bijector=bijector_fn,
                conditioner=SASFlow.make_conditioner(event_shape, hidden_sizes, 3 * num_bins + 1),
            ))
            if D > 1:
                mask = jnp.logical_not(mask)

        base = distrax.Independent(
            distrax.Normal(jnp.zeros(event_shape), jnp.ones(event_shape)),
            reinterpreted_batch_ndims=len(event_shape),
        )
        return distrax.Transformed(base, distrax.Inverse(distrax.Chain(layers)))

    def _build_model(self):
        args = (self.event_shape, self.flow_num_layers, self.hidden_sizes, self.num_bins, self.B)

        @hk.without_apply_rng
        @hk.transform
        def log_prob_fn(y):
            return SASFlow.make_flow_model(*args).log_prob(y)

        @hk.without_apply_rng
        @hk.transform
        def forward_fn(z):          # latent -> normalized
            return SASFlow.make_flow_model(*args).bijector.forward_and_log_det(z)

        @hk.without_apply_rng
        @hk.transform
        def inverse_fn(y):          # normalized -> latent
            return SASFlow.make_flow_model(*args).bijector.inverse_and_log_det(y)

        self.log_prob_fn = log_prob_fn
        self.forward_fn = forward_fn
        self.inverse_fn = inverse_fn

        # Pure jitted functions: params and normalizers are arguments, never captured.
        def _log_prob(params, norms, x):
            y, logdet = self._row_forward(norms, x)
            return log_prob_fn.apply(params, y) + logdet

        def _forward(params, norms, z):
            y, ld_flow = forward_fn.apply(params, z)
            x, ld_norm = self._row_inverse(norms, y)
            return x, ld_flow + ld_norm

        def _backward(params, norms, x):
            y, ld_norm = self._row_forward(norms, x)
            z, ld_flow = inverse_fn.apply(params, y)
            return z, ld_norm + ld_flow

        self._log_prob_jit = jax.jit(_log_prob)
        self._forward_jit = jax.jit(_forward)
        self._backward_jit = jax.jit(_backward)

    def _init_params(self):
        self.key, init_key = jr.split(self.key)
        self.params = self.log_prob_fn.init(init_key, self._batch(np.arange(min(2, self.N))))

        tx = [optax.adam(self.learning_rate)]
        if self.grad_clip is not None:
            tx.insert(0, optax.clip_by_global_norm(self.grad_clip))
        self.optimizer = optax.chain(*tx)
        self.opt_state = self.optimizer.init(self.params)

        def loss_fn(params, y):
            return -jnp.mean(self.log_prob_fn.apply(params, y))

        @jax.jit
        def update(params, opt_state, y):
            loss, grads = jax.value_and_grad(loss_fn)(params, y)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        self._update = update

    # ================================================================
    # Training
    # ================================================================

    def _iter_batches(self, batch_size, key):
        """Shuffled, lazily-read, normalized batches over the training (draw, realization) pairs."""
        rows = self.train_rows
        perm = rows[np.asarray(jr.permutation(key, len(rows)))]
        drop_last = len(rows) >= batch_size     # avoid recompiling for a ragged last batch
        for start in range(0, len(rows), batch_size):
            idx = np.sort(perm[start:start + batch_size])   # sorted reads are faster on memmaps
            if drop_last and len(idx) < batch_size:
                break
            x = self._batch(idx)
            if self.context_jitter > 0 and self.P:
                noise = np.zeros(x.shape)
                noise[:, :self.P] = self.context_jitter * self.rng.standard_normal((len(idx), self.P))
                x = x + jnp.asarray(noise)
            yield x

    def fit(self, num_epochs=100, batch_size=256, checkpoint=None, verbose=True):
        """
        Train by maximum likelihood in normalized space.

        checkpoint : path or None
            Save training state after every epoch, and resume from it if it
            already exists.

        Returns
        -------
        list of per-epoch mean training losses (normalized-space NLL).
        With val_frac > 0, `val_history` holds the per-epoch mean held-out
        log p (physical units).
        """
        start_epoch = 0
        if checkpoint is not None and os.path.exists(checkpoint):
            start_epoch = self._load_checkpoint(checkpoint, batch_size)
            print(f"[SASFlow] resuming from {checkpoint} at epoch {start_epoch}/{num_epochs}",
                  flush=True)
        if len(self.val_rows):
            vrng = np.random.default_rng(0)
            vidx = np.sort(vrng.choice(self.val_rows, size=min(20_000, len(self.val_rows)),
                                       replace=False))
            val_x = jnp.asarray(self.rows(vidx))

        bar = trange(start_epoch, num_epochs, initial=start_epoch, total=num_epochs,
                     colour="blue", desc="Training SASFlow", disable=not verbose)
        for ep in bar:
            self.key, subkey = jr.split(self.key)
            losses = []
            for y in self._iter_batches(batch_size, subkey):
                self.params, self.opt_state, loss = self._update(self.params, self.opt_state, y)
                losses.append(loss)
            epoch_loss = float(jnp.mean(jnp.stack(losses)))
            self.loss_history.append(epoch_loss)
            msg = dict(loss=f"{epoch_loss:.4f}")
            if len(self.val_rows):
                self.val_history.append(float(jnp.mean(self.log_prob(val_x))))
                msg["val_logp"] = f"{self.val_history[-1]:.4f}"
            bar.set_postfix(msg)
            if checkpoint is not None:
                self._save_checkpoint(checkpoint, ep + 1, batch_size)
        return self.loss_history

    # ================================================================
    # Public API (inputs / outputs in physical space, rows = [theta | y])
    # ================================================================

    def log_prob(self, x):
        """log p(x) in physical space.  x : (n, D) or (D,), rows [theta | y]."""
        return self._log_prob_jit(self.params, self._norms(), jnp.atleast_2d(x))

    def sample(self, num_samples, key=None):
        """Draw (num_samples, D) rows [theta | y] in physical space."""
        if key is None:
            self.key, key = jr.split(self.key)
        z = jr.normal(key, (num_samples, self.D))
        return self.forward_pass(z)

    def forward_pass(self, z):
        """Latent z -> physical rows [theta | y].  Differentiable (numpyro-ready)."""
        return self._forward_jit(self.params, self._norms(), jnp.atleast_2d(z))[0]

    def forward_pass_with_logdet(self, z):
        """Latent z -> (physical rows, log|det dx/dz|)."""
        return self._forward_jit(self.params, self._norms(), jnp.atleast_2d(z))

    def backward_pass(self, x):
        """Physical rows -> latent z."""
        return self._backward_jit(self.params, self._norms(), jnp.atleast_2d(x))[0]

    def backward_pass_with_logdet(self, x):
        """Physical rows -> (latent z, log|det dz/dx|).  log p(x) = log N(z) + logdet."""
        return self._backward_jit(self.params, self._norms(), jnp.atleast_2d(x))

    # ================================================================
    # Save / load
    # ================================================================

    def save_params(self, path):
        """Flow weights, data normalizer and context transform in one .npz file."""
        leaves, _ = jax.tree_util.tree_flatten(self.params)
        d = {f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)}
        d.update({f"norm_{k}": v for k, v in self.normalizer_np.items()})
        d.update({f"ctx_{k}": v for k, v in self.ctx_np.items()})
        d["context_kind"] = np.array(self.context_kind)
        np.savez(path, **d)

    def load_params(self, path):
        with np.load(path) as f:
            _, treedef = jax.tree_util.tree_flatten(self.params)
            leaves = [jnp.asarray(f[f"leaf_{i}"]) for i in range(treedef.num_leaves)]
            self.params = jax.tree_util.tree_unflatten(treedef, leaves)
            if "norm_loc" in f.files:
                self.set_normalizer({k[5:]: f[k] for k in f.files if k.startswith("norm_")})
            if "context_kind" in f.files:
                kind = str(f["context_kind"])
                if kind != self.context_kind:
                    raise ValueError(f"saved context_kind {kind!r} differs from {self.context_kind!r}")
                self._set_ctx({k[4:]: f[k] for k in f.files if k.startswith("ctx_")})

    # ------------- Checkpointing -------------

    def _checkpoint_signature(self, batch_size):
        return dict(
            N=int(self.N), D=int(self.D), P=int(self.P), R=int(self.R),
            context_kind=self.context_kind, context_jitter=float(self.context_jitter),
            n_train=int(len(self.train_rows)),
            flow_num_layers=int(self.flow_num_layers),
            hidden_sizes=[int(h) for h in self.hidden_sizes],
            num_bins=int(self.num_bins), B=float(self.B),
            learning_rate=float(self.learning_rate),
            grad_clip=None if self.grad_clip is None else float(self.grad_clip),
            batch_size=int(batch_size),
        )

    def _save_checkpoint(self, path, epoch, batch_size):
        blob = dict(
            params=jax.tree_util.tree_map(np.asarray, self.params),
            opt_state=jax.tree_util.tree_map(np.asarray, self.opt_state),
            key=np.asarray(self.key),
            epoch=int(epoch),
            loss_history=list(self.loss_history),
            val_history=list(self.val_history),
            normalizer=self.normalizer_np,
            ctx=self.ctx_np,
            signature=self._checkpoint_signature(batch_size),
        )
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)       # atomic: an interrupted write keeps the old checkpoint

    def _load_checkpoint(self, path, batch_size):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        want, got = self._checkpoint_signature(batch_size), blob["signature"]
        if got != want:
            differ = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
            raise ValueError(f"{path} was written by a different configuration; "
                             f"(checkpoint, this run) differ on {differ}.")
        self.set_normalizer(blob["normalizer"])   # the weights were trained against this map
        if "ctx" in blob:
            self._set_ctx(blob["ctx"])
        self.params = jax.tree_util.tree_map(jnp.asarray, blob["params"])
        self.opt_state = jax.tree_util.tree_map(jnp.asarray, blob["opt_state"])
        self.key = jnp.asarray(blob["key"])
        self.loss_history = list(blob.get("loss_history", []))
        self.val_history = list(blob.get("val_history", []))
        return int(blob["epoch"])


# ═════════════════════════════════════════════════════════════════════════════
# Conditional SAS flows: shared data handling, training and API
# ═════════════════════════════════════════════════════════════════════════════

_E_LOG_CHI2_1 = -1.2703628454614782        # E[log z^2] for z ~ N(0, 1)


def _scale_apply(params, c_hat):
    """Scale network: normalized context (n, C) -> log sigma (n, D)."""
    h = c_hat
    for i, p in enumerate(params):
        h = h @ p["W"] + p["b"]
        if i < len(params) - 1:
            h = jnp.tanh(h)
    return h


class _ConditionalSASFlow:
    """
    Shared machinery of ConditionalFlow and SASNSF: SAS preprocessing of the
    data, a context transform, memmap-friendly batching, training with
    validation by held-out groups, the public API and save/load.

    Subclasses define the transforms through four hooks:
        _setup_architecture()         static structure (masks, layer lists)
        _init_model_params(key)       parameter pytree
        _flow_forward(params, y, c)   normalized data -> (latent, log|det|), one pass
        _flow_inverse(params, z, c)   latent -> normalized data
    and may extend _arch_cfg() / _save_extra() / _check_extra(blob).

    Context-dependent scale (scale_by_context=True)
    -----------------------------------------------
    When the context mostly changes the SIZE of the data (e.g. Fourier
    coefficients, whose amplitude varies by orders of magnitude between
    draws), a flow on the raw values must learn that scale for every
    feature and tends to fall back on the pooled distribution.  With this
    option a small network first learns log sigma_k(c) for every feature,
    by regressing log y_k^2 on the context (robust to heavy tails:
    E[log y^2] = 2 log sigma + E[log chi^2_1] for Gaussian y).  The data are
    divided by sigma(c) before the SAS map and the flow, and the Jacobian is
    included:

        log p(y | c) = log p_flow(SAS(y / sigma(c)) | c) + log|dSAS| - sum_k log sigma_k(c)

    forward_pass multiplies by sigma(c) at the end, so it stays a single
    differentiable pass.  The scale network is fitted once, before the flow,
    and then frozen.

    Data layouts
    ------------
    * library : data (n_theta, R, D), context (n_theta, C).  Pairs
                (theta_i, realization j) are drawn uniformly; validation holds
                out whole theta points.
    * flat    : data (N, D), context (N, C) or None.  Validation holds out rows.
    Both accept numpy memmaps.  context=None gives an unconditional flow.
    """

    _name = "flow"

    def __init__(self, data, context=None, context_kind="standardize", context_bounds=None,
                 val_frac=0.1, normalizer=None, normalizer_samples=50_000,
                 learning_rate=1e-3, grad_clip=1.0, seed=0, dtype=jnp.float64,
                 scale_by_context=False, scale_hidden=(128, 128), scale_steps=3000):
        # ---------- layout
        self.data, self.context_raw = data, None
        if data.ndim == 3:
            if context is None:
                raise ValueError("library layout (n_theta, R, D) needs context (n_theta, C)")
            self.layout = "library"
            self.n_groups, self.R, self.D = data.shape
        elif data.ndim == 2:
            self.layout = "flat"
            self.n_groups, self.D = data.shape
            self.R = 1
        elif data.ndim == 1 and context is None:
            data = data.reshape(-1, 1)
            self.data, self.layout = data, "flat"
            self.n_groups, self.D, self.R = data.shape[0], 1, 1
        else:
            raise ValueError(f"data must be (N, D) or (n_theta, R, D); got {data.shape}")
        if context is not None:
            context = np.asarray(context, dtype=np.float64)
            if context.ndim == 1:
                context = context[:, None]
            if context.shape[0] != self.n_groups:
                raise ValueError("context rows must match data rows / theta points")
            self.context_raw = context
        self.C = 0 if context is None else context.shape[1]
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)
        self.key = jr.PRNGKey(seed)
        self.seed = seed
        self.history = dict(train=[], val=[])
        self.learning_rate, self.grad_clip = learning_rate, grad_clip

        # ---------- split (by theta point in the library layout)
        perm = self.rng.permutation(self.n_groups)
        n_val = int(round(val_frac * self.n_groups))
        self.val_idx, self.train_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])

        # ---------- context transform (fitted on training groups)
        if self.C:
            self.ctx_np = fit_context_transform(self.context_raw[self.train_idx],
                                                context_kind, context_bounds)
        else:
            self.ctx_np = dict(kind="none")
        self._set_ctx()

        # ---------- optional context-dependent scale (fitted before the SAS map)
        self.scale_by_context = bool(scale_by_context)
        self.scale_hidden = tuple(int(h) for h in scale_hidden)
        self.scale_params = None
        if self.scale_by_context:
            if not self.C:
                raise ValueError("scale_by_context needs a context")
            self.scale_params = self._fit_scale(scale_steps)
        self._set_logsig_all()

        # ---------- SAS normalizer for the (scaled) data (training groups only)
        if normalizer is None:
            g, r = self._random_pairs(self.train_idx, normalizer_samples)
            normalizer = fit_sas_normalizer(self._read_scaled(g, r))
        self.set_normalizer(normalizer)

        # ---------- model
        self._setup_architecture()
        self._init_params()
        self._build()

    # ------------------------------------------------------------ hooks
    def _setup_architecture(self):
        raise NotImplementedError

    def _init_model_params(self, key):
        raise NotImplementedError

    def _flow_forward(self, params, y, c):
        raise NotImplementedError

    def _flow_inverse(self, params, z, c):
        raise NotImplementedError

    def _arch_cfg(self):
        return dict(D=self.D, C=self.C)

    def _save_extra(self):
        return {}

    def _check_extra(self, blob):
        pass

    # ------------------------------------------------------------ data access
    def _set_ctx(self):
        self.ctx = {k: (v if k == "kind" else jnp.asarray(v)) for k, v in self.ctx_np.items()}
        self.c_all = (np.asarray(context_forward(jnp.asarray(self.context_raw), self.ctx))
                      if self.C else None)

    def set_normalizer(self, n):
        """Set the SAS normalizer of the (scaled) data."""
        keys = ("loc", "s", "delta", "eps", "mu", "sd")
        self.norm_np = {k: np.asarray(n[k], np.float64).reshape(self.D) for k in keys}
        self.norm = {k: jnp.asarray(v) for k, v in self.norm_np.items()}
        self.norm["scale"] = self.scale_params          # None unless scale_by_context

    # ------------------------------------------------------------ context-dependent scale
    def _fit_scale(self, steps, batch_size=4096, learning_rate=1e-3, verbose=True):
        """Fit log sigma_k(c) by regressing log y_k^2 on the normalized context."""
        g, r = self._random_pairs(self.train_idx, 20_000)
        y0 = self._read(g, r)
        self._scale_floor = float(np.median(y0**2) * 1e-12)      # guards log(0)
        target0 = 0.5 * (np.log(y0**2 + self._scale_floor).mean(0) - _E_LOG_CHI2_1)

        key = jr.PRNGKey(self.seed + 7)
        sizes = (self.C, *self.scale_hidden, self.D)
        params = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            key, k = jr.split(key)
            last = i == len(sizes) - 2
            params.append(dict(W=jnp.zeros((a, b)) if last else jr.normal(k, (a, b)) / np.sqrt(a),
                               b=jnp.asarray(target0) if last else jnp.zeros(b)))

        floor = self._scale_floor

        def loss(p, c, y):
            ls = _scale_apply(p, c)
            return jnp.mean((jnp.log(y**2 + floor) - 2.0 * ls - _E_LOG_CHI2_1) ** 2)

        opt = optax.adam(learning_rate)
        state = opt.init(params)

        @jax.jit
        def step(p, st, c, y):
            l, gr = jax.value_and_grad(loss)(p, c, y)
            upd, st = opt.update(gr, st)
            return optax.apply_updates(p, upd), st, l

        for _ in trange(steps, desc="Fitting context scale", disable=not verbose or steps == 0):
            g, r = self._random_pairs(self.train_idx, batch_size)
            params, state, _ = step(params, state, jnp.asarray(self.c_all[g]), jnp.asarray(self._read(g, r)))

        # held-out check: fraction of the variance of log y^2 explained by the scale
        groups = self.val_idx if len(self.val_idx) else self.train_idx
        g, r = self._random_pairs(groups, 20_000)
        y = self._read(g, r)
        t = np.log(y**2 + floor) - _E_LOG_CHI2_1
        pred = 2.0 * np.asarray(_scale_apply(params, jnp.asarray(self.c_all[g])))
        self.scale_r2 = float(1.0 - np.mean((t - pred) ** 2) / np.mean((t - t.mean(0)) ** 2))
        return params

    def _set_logsig_all(self):
        self.logsig_all = (np.asarray(_scale_apply(self.scale_params, jnp.asarray(self.c_all)))
                           if self.scale_params is not None else None)

    def context_scale(self, c):
        """sigma_k(c) learned by the scale network (physical context), shape (n, D)."""
        if self.scale_params is None:
            raise RuntimeError("scale_by_context is off")
        c_hat = context_forward(jnp.atleast_2d(jnp.asarray(c, jnp.float64)), self.ctx)
        return jnp.exp(_scale_apply(self.scale_params, c_hat))

    def normalize(self, y):
        """Physical data -> SAS-normalized data (no Jacobian)."""
        return sas_forward(jnp.atleast_2d(jnp.asarray(y, jnp.float64)), self.norm)[0]

    def denormalize(self, y_hat):
        """SAS-normalized data -> physical data (no Jacobian)."""
        return sas_inverse(jnp.atleast_2d(jnp.asarray(y_hat, jnp.float64)), self.norm)[0]

    def _random_pairs(self, groups, n):
        g = self.rng.choice(groups, size=n, replace=True)
        r = self.rng.integers(0, self.R, size=n)
        return g, r

    def _read(self, g, r):
        o = np.lexsort((r, g))                    # sorted reads are faster on memmaps
        out = np.empty((len(g), self.D))
        rows = self.data[g[o], r[o]] if self.layout == "library" else self.data[g[o]]
        out[o] = np.asarray(rows, dtype=np.float64).reshape(len(g), self.D)
        if not np.all(np.isfinite(out)):
            raise ValueError("Non-finite data values; remove or handle them before training.")
        return out

    def _read_scaled(self, g, r):
        y = self._read(g, r)
        return y if self.logsig_all is None else y * np.exp(-self.logsig_all[g])

    def _batch(self, g, r):
        y = jnp.asarray(_sas_forward_np(self._read_scaled(g, r), self.norm_np), self.dtype)
        c = jnp.asarray(self.c_all[g], self.dtype) if self.C else None
        return y, c

    # ------------------------------------------------------------ optimisation
    def _init_params(self):
        self.key, k = jr.split(self.key)
        self.params = self._init_model_params(k)
        tx = [optax.adam(self.learning_rate)]
        if self.grad_clip is not None:
            tx.insert(0, optax.clip_by_global_norm(self.grad_clip))
        self.opt = optax.chain(*tx)
        self.opt_state = self.opt.init(self.params)
        self.best_params, self.best_val = self.params, -np.inf

    def _build(self):
        D = self.D

        def norm_lp(params, y, c):                # normalized-space log density
            z, ladj = self._flow_forward(params, y, c)
            return jnp.sum(-0.5 * z**2, -1) - 0.5 * D * LOG2PI + ladj

        def prep_c(c, ctx):
            return None if c is None else context_forward(c, ctx).astype(self.dtype)

        def log_sigma(norm, ch):
            return None if norm["scale"] is None else _scale_apply(norm["scale"], ch)

        def phys_lp(params, norm, ctx, y, c):
            ch = prep_c(c, ctx)
            ls = log_sigma(norm, ch)
            ys, ld_s = (y, 0.0) if ls is None else (y * jnp.exp(-ls), -jnp.sum(ls, -1))
            yh, ld = sas_forward(ys, norm)
            return norm_lp(params, yh.astype(self.dtype), ch) + ld + ld_s

        def phys_sample(params, norm, ctx, z, c):
            ch = prep_c(c, ctx)
            ys = sas_inverse(self._flow_inverse(params, z, ch), norm)[0]
            ls = log_sigma(norm, ch)
            return ys if ls is None else ys * jnp.exp(ls)

        def latent(params, norm, ctx, y, c):
            ch = prep_c(c, ctx)
            ls = log_sigma(norm, ch)
            ys = y if ls is None else y * jnp.exp(-ls)
            yh = sas_forward(ys, norm)[0].astype(self.dtype)
            return self._flow_forward(params, yh, ch)[0]

        def loss(params, y, c):
            return -jnp.mean(norm_lp(params, y, c))

        @jax.jit
        def update(params, opt_state, y, c):
            l, g = jax.value_and_grad(loss)(params, y, c)
            upd, opt_state = self.opt.update(g, opt_state, params)
            return optax.apply_updates(params, upd), opt_state, l

        self._update = update
        # ctx holds a python string, so it is closed over (rebuilt on load)
        self._phys_lp = jax.jit(lambda p, n, y, c: phys_lp(p, n, self.ctx, y, c))
        self._phys_sample = jax.jit(lambda p, n, z, c: phys_sample(p, n, self.ctx, z, c))
        self._latent = jax.jit(lambda p, n, y, c: latent(p, n, self.ctx, y, c))

    # ------------------------------------------------------------ training
    def fit(self, num_epochs=100, batch_size=1024, steps_per_epoch=None, n_val=50_000,
            patience=10, save_path=None, verbose=True):
        """
        Maximum likelihood on uniformly drawn (group, realization) pairs.

        steps_per_epoch : None -> about one pass over the training pairs.
        Validation : mean log p(y | c) in physical units on held-out groups
        (theta points in the library layout).  The best parameters are kept,
        saved to `save_path` on every improvement, and restored at the end;
        training stops after `patience` epochs without improvement.
        """
        n_pairs = len(self.train_idx) * self.R
        if steps_per_epoch is None:
            steps_per_epoch = max(1, n_pairs // batch_size)
        has_val = len(self.val_idx) > 0
        if has_val:
            vg, vr = self._random_pairs(self.val_idx, min(n_val, len(self.val_idx) * self.R))
            vy = jnp.asarray(self._read(vg, vr))
            vc = jnp.asarray(self.context_raw[vg]) if self.C else None
        bad = 0
        bar = trange(num_epochs, desc=f"Training {type(self).__name__}", disable=not verbose)
        for _ in bar:
            losses = []
            for _ in range(steps_per_epoch):
                y, c = self._batch(*self._random_pairs(self.train_idx, batch_size))
                self.params, self.opt_state, l = self._update(self.params, self.opt_state, y, c)
                losses.append(l)
            tr = float(jnp.mean(jnp.stack(losses)))
            self.history["train"].append(tr)
            msg = dict(train_nll=f"{tr:.3f}")
            if has_val:
                v = float(jnp.mean(self.log_prob(vy, vc)))
                self.history["val"].append(v)
                msg["val_logp"] = f"{v:.3f}"
                if v > self.best_val:
                    self.best_val, self.best_params, bad = v, self.params, 0
                    if save_path is not None:
                        self.save(save_path)
                else:
                    bad += 1
            bar.set_postfix(msg)
            if has_val and bad >= patience:
                break
        if has_val:
            self.params = self.best_params
        return self.history

    # ------------------------------------------------------------ public API (physical units)
    def _prep(self, y, c):
        y = jnp.atleast_2d(jnp.asarray(y, jnp.float64))
        if self.C:
            if c is None:
                raise ValueError("this flow is conditional; pass the context")
            c = jnp.atleast_2d(jnp.asarray(c, jnp.float64))
            c = jnp.broadcast_to(c, (y.shape[0], self.C))
        else:
            c = None
        return y, c

    def log_prob(self, y, c=None, chunk=50_000):
        """log p(y | c), physical units.  y (n, D); c (n, C) or (C,)."""
        y, c = self._prep(y, c)
        parts = [self._phys_lp(self.params, self.norm, y[a:a + chunk],
                               None if c is None else c[a:a + chunk])
                 for a in range(0, y.shape[0], chunk)]
        return jnp.concatenate(parts)

    def sample(self, num_samples, c=None, key=None):
        """
        Draw y ~ p(y | c).  c : (C,) for num_samples draws at one context, or
        (num_samples, C) for one draw per context row.
        """
        if key is None:
            self.key, key = jr.split(self.key)
        z = jr.normal(key, (num_samples, self.D), self.dtype)
        return self.forward_pass(z, c)

    def forward_pass(self, z, c=None):
        """
        Latent z -> physical y given context c: y ~ p(y | c) when z ~ N(0, I).

        Pure and differentiable in both z and c, so it can be used inside a
        numpyro model with the base variables z sampled by numpyro.
        z : (n, D) or (D,);  c : (n, C) or (C,) physical-space context.
        """
        z = jnp.atleast_2d(jnp.asarray(z)).astype(self.dtype)
        if self.C:
            if c is None:
                raise ValueError("this flow is conditional; pass the context")
            c = jnp.broadcast_to(jnp.atleast_2d(jnp.asarray(c, jnp.float64)), (z.shape[0], self.C))
        else:
            c = None
        return self._phys_sample(self.params, self.norm, z, c)

    def latent(self, y, c=None):
        """Physical y -> latent z (should be ~N(0, I) for held-out data)."""
        y, c = self._prep(y, c)
        return self._latent(self.params, self.norm, y, c)

    # ------------------------------------------------------------ diagnostics
    def context_ablation_gap(self, n_pairs=20_000, seed=1):
        """mean[log p(y|c_true) - log p(y|c_shuffled)] on held-out groups; ~0 = context ignored."""
        if not self.C:
            raise RuntimeError("unconditional flow")
        rng = np.random.default_rng(seed)
        groups = self.val_idx if len(self.val_idx) else self.train_idx
        g = rng.choice(groups, n_pairs)
        r = rng.integers(0, self.R, n_pairs)
        y = self._read(g, r)
        c, cs = self.context_raw[g], self.context_raw[rng.permutation(g)]
        return float(jnp.mean(self.log_prob(y, c) - self.log_prob(y, cs)))

    def compare_conditional(self, group, num_samples=None, quantiles=(0.05, 0.5, 0.95)):
        """Library vs flow quantiles at one theta point (library layout)."""
        if self.layout != "library":
            raise RuntimeError("needs the library layout")
        lib = np.asarray(self.data[group], dtype=np.float64)
        gen = np.asarray(self.sample(num_samples or max(10 * self.R, 10_000),
                                     self.context_raw[group]))
        return np.quantile(lib, quantiles, axis=0), np.quantile(gen, quantiles, axis=0)

    # ------------------------------------------------------------ save / load
    def save(self, path):
        """Weights, normalizers, split and history in one pickle (atomic write)."""
        blob = dict(params=jax.tree_util.tree_map(np.asarray, self.params),
                    norm=self.norm_np, ctx=self.ctx_np, cfg=self._arch_cfg(),
                    val_idx=self.val_idx, train_idx=self.train_idx, history=self.history,
                    scale_params=(None if self.scale_params is None else
                                  jax.tree_util.tree_map(np.asarray, self.scale_params)),
                    **self._save_extra())
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def load(self, path):
        """
        Load into a flow constructed with the same architecture arguments
        (with scale_by_context=True, pass scale_steps=0 to skip refitting).
        """
        with open(path, "rb") as f:
            blob = pickle.load(f)
        want = self._arch_cfg()
        for k, v in want.items():
            if blob["cfg"].get(k) != v:
                raise ValueError(f"saved model differs on {k}: {blob['cfg'].get(k)} vs {v}")
        self._check_extra(blob)
        if (blob.get("scale_params") is not None) != self.scale_by_context:
            raise ValueError("saved model and this flow differ on scale_by_context")
        self.params = self.best_params = jax.tree_util.tree_map(
            lambda a: jnp.asarray(a, self.dtype), blob["params"])
        self.opt_state = self.opt.init(self.params)
        self.ctx_np = blob["ctx"]
        if self.C:
            self._set_ctx()
        if self.scale_by_context:
            self.scale_params = jax.tree_util.tree_map(jnp.asarray, blob["scale_params"])
            self._set_logsig_all()
        self.set_normalizer(blob["norm"])
        self._build()
        self.val_idx, self.train_idx = blob["val_idx"], blob["train_idx"]
        self.history = blob.get("history", self.history)


# ═════════════════════════════════════════════════════════════════════════════
# ConditionalFlow
# ═════════════════════════════════════════════════════════════════════════════

class ConditionalFlow(_ConditionalSASFlow):
    """
    Conditional RQS coupling flow with SAS preprocessing: the conditional
    form of SASFlow, with the data handling and API of SASNSF.

    Pipeline
    --------
        physical y (D,)  --SAS-->  y_hat  --coupling layers(c_hat)-->  z ~ N(0, I)
        context  c (C,)  --context transform-->  c_hat

        log p(y | c) = log N(z; 0, I) + sum_l log|det dT_l| + log|det dSAS|

    Each layer is a masked coupling: the features kept by the mask, together
    with c_hat, go through an MLP that outputs rational-quadratic spline
    parameters for the other features.  Masks alternate by parity between
    layers (for D == 1 every layer transforms the single feature from c_hat
    alone).  The last MLP layer starts at zero, so the untrained flow is the
    identity.  Splines act on [-B, B] and are the identity outside, so the
    tails come from the SAS map.

    Both directions take one pass per layer, so sampling, forward_pass and
    log_prob are equally fast; this is the flow to use when numpyro samples
    the base variables.

    Parameters
    ----------
    data : (n_theta, R, D) "library" layout, or (N, D) "flat" layout; memmap ok.
    context : (n_theta, C) for library, (N, C) for flat, or None (unconditional).
    context_kind : "standardize" | "probit" | "sas" | "none".
    context_bounds : (lo, hi) for "probit" (inferred from the data if None).
    flow_num_layers, hidden_size, mlp_num_layers, num_bins, B : architecture,
        as in SASFlow.
    val_frac : fraction of theta points (library) or rows (flat) held out.
    normalizer : pre-fitted SAS dict for the data; fitted if None.
    normalizer_samples : rows used for the SAS fit.
    learning_rate, grad_clip, seed : optimisation.
    dtype : jnp.float64 (default) or jnp.float32 (much faster on GPUs).
    scale_by_context : learn a context-dependent scale sigma_k(c) first and
        model y / sigma(c) (see _ConditionalSASFlow).  Use it when the
        context mostly changes the size of the data.
    scale_hidden, scale_steps : hidden layers and training steps of the
        scale network (scale_steps=0 when you will `load` a saved model).
    """

    def __init__(self, data, context=None, context_kind="standardize", context_bounds=None,
                 flow_num_layers=8, hidden_size=256, mlp_num_layers=2, num_bins=8, B=5.0,
                 val_frac=0.1, normalizer=None, normalizer_samples=50_000,
                 learning_rate=1e-3, grad_clip=1.0, seed=0, dtype=jnp.float64,
                 scale_by_context=False, scale_hidden=(128, 128), scale_steps=3000):
        self.flow_num_layers = flow_num_layers
        self.hidden_sizes = [int(hidden_size)] * mlp_num_layers
        self.num_bins = num_bins
        self.B = float(B)
        super().__init__(data, context, context_kind=context_kind, context_bounds=context_bounds,
                         val_frac=val_frac, normalizer=normalizer,
                         normalizer_samples=normalizer_samples, learning_rate=learning_rate,
                         grad_clip=grad_clip, seed=seed, dtype=dtype,
                         scale_by_context=scale_by_context, scale_hidden=scale_hidden,
                         scale_steps=scale_steps)

    def _arch_cfg(self):
        return dict(D=self.D, C=self.C, flow_num_layers=self.flow_num_layers,
                    hidden_sizes=list(self.hidden_sizes), num_bins=self.num_bins, B=self.B)

    def _setup_architecture(self):
        D = self.D
        if D == 1:
            mask = np.zeros(1, bool)
            self.masks = [mask] * self.flow_num_layers
        else:
            base = np.arange(D) % 2 == 1
            self.masks = [base if l % 2 == 0 else ~base for l in range(self.flow_num_layers)]

        D, nparams, B = self.D, 3 * self.num_bins + 1, self.B
        hidden, masks = self.hidden_sizes, self.masks

        def chain(c):
            def conditioner():
                net = hk.Sequential([
                    hk.nets.MLP(hidden, activate_final=True),
                    hk.Linear(D * nparams, w_init=jnp.zeros, b_init=jnp.zeros),
                    hk.Reshape((D, nparams), preserve_dims=-1),
                ])
                if c is None:
                    return net
                return lambda x: net(jnp.concatenate([x, jnp.broadcast_to(
                    c, x.shape[:-1] + c.shape[-1:])], axis=-1))

            layers = [distrax.MaskedCoupling(
                mask=jnp.asarray(m),
                bijector=lambda p: distrax.RationalQuadraticSpline(p, range_min=-B, range_max=B),
                conditioner=conditioner()) for m in masks]
            return distrax.Chain(layers)

        # normalized data -> latent (one pass), and latent -> normalized data (one pass)
        self._hk_fwd = hk.without_apply_rng(hk.transform(
            lambda y, c: chain(c).forward_and_log_det(y)))
        self._hk_inv = hk.without_apply_rng(hk.transform(
            lambda z, c: chain(c).inverse(z)))

    def _init_model_params(self, key):
        y0 = jnp.zeros((2, self.D), self.dtype)
        c0 = jnp.zeros((2, self.C), self.dtype) if self.C else None
        return self._hk_fwd.init(key, y0, c0)

    def _flow_forward(self, params, y, c):
        return self._hk_fwd.apply(params, y, c)

    def _flow_inverse(self, params, z, c):
        return self._hk_inv.apply(params, z, c)


# ═════════════════════════════════════════════════════════════════════════════
# SASNSF
# ═════════════════════════════════════════════════════════════════════════════

class SASNSF(_ConditionalSASFlow):
    """
    Conditional neural spline flow (zuko NSF architecture) with SAS
    preprocessing, in JAX.

    Pipeline (density direction, one parallel pass)
    -----------------------------------------------
        physical y (D,)  --SAS-->  y_hat  --T_1 .. T_n-->  z ~ N(0, I)
        context  c (C,)  --context transform-->  c_hat   (fed to every T_i)

        log p(y | c) = log N(z; 0, I) + sum_i log|det dT_i| + log|det dSAS|

    What is ported from zuko (v1.x), and kept identical
    --------------------------------------------------
    * MaskedAutoregressiveTransform: the hyper-network input is [x, c]; feature
      orders alternate ascending / descending between transforms (or random
      permutations with randperm=True); `passes` groups features as in zuko
      (passes=None -> fully autoregressive, passes=2 -> coupling); context is
      connected to every output.
    * MaskedMLP: the same mask construction (unique dependency sets, precedence
      matrix, cyclic assignment of hidden units), ReLU by default, and torch's
      default Linear initialisation U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
    * MonotonicRQSTransform: bins K, bound B = 5, minimum slope 1e-3, the same
      soft clamping of widths/heights/derivatives, boundary derivatives fixed to
      1, identity outside [-B, B].
    * Sampling inverts each transform with `passes` fixed-point sweeps, as zuko
      does.

    Checked against zuko: with zuko's weights copied in, log p matches to
    machine precision.

    Cost: log_prob takes one pass per transform; sampling / forward_pass take
    `passes` sequential sweeps per transform (D when fully autoregressive).

    Differences from zuko (optional)
    --------------------------------
    * zero_init_last=True starts every transform at the identity (zuko: False).

    Parameters
    ----------
    data, context, context_kind, context_bounds, val_frac, normalizer,
    normalizer_samples, learning_rate, grad_clip, seed, dtype,
    scale_by_context, scale_hidden, scale_steps : as in ConditionalFlow.
    transforms, hidden_features, bins, passes, randperm, activation, slope,
    bound : zuko NSF arguments (defaults as in zuko except hidden_features).
    zero_init_last : start every transform at the identity.
    """

    def __init__(self, data, context=None, context_kind="standardize", context_bounds=None,
                 transforms=5, hidden_features=(256, 256), bins=8, passes=None,
                 randperm=False, activation="relu", slope=1e-3, bound=5.0,
                 zero_init_last=True, val_frac=0.1, normalizer=None,
                 normalizer_samples=50_000, learning_rate=1e-3, grad_clip=1.0,
                 seed=0, dtype=jnp.float64,
                 scale_by_context=False, scale_hidden=(128, 128), scale_steps=3000):
        self.cfg = dict(transforms=transforms, hidden_features=tuple(int(h) for h in hidden_features),
                        bins=bins, passes=passes, randperm=randperm, activation=activation,
                        slope=slope, bound=float(bound), zero_init_last=zero_init_last)
        super().__init__(data, context, context_kind=context_kind, context_bounds=context_bounds,
                         val_frac=val_frac, normalizer=normalizer,
                         normalizer_samples=normalizer_samples, learning_rate=learning_rate,
                         grad_clip=grad_clip, seed=seed, dtype=dtype,
                         scale_by_context=scale_by_context, scale_hidden=scale_hidden,
                         scale_steps=scale_steps)

    def _arch_cfg(self):
        keys = ("transforms", "hidden_features", "bins", "passes", "activation")
        return dict(D=self.D, C=self.C, **{k: self.cfg[k] for k in keys})

    def _save_extra(self):
        return dict(orders=[L["order"] for L in self.layers])

    def _check_extra(self, blob):
        if not all(np.array_equal(a, L["order"]) for a, L in zip(blob["orders"], self.layers)):
            raise ValueError("feature orders differ (randperm with a different seed?)")

    def _setup_architecture(self):
        c = self.cfg
        self.total = 3 * c["bins"] - 1             # widths K, heights K, derivatives K-1
        self.bins = c["bins"]
        orders = [np.arange(self.D), np.arange(self.D)[::-1]]
        prng = np.random.default_rng(self.seed + 12345)
        self.layers = []
        for i in range(c["transforms"]):
            order = prng.permutation(self.D) if c["randperm"] else orders[i % 2]
            adj, p = maf_adjacency(self.D, self.C, order, c["passes"], self.total)
            if adj.any():
                masks = [jnp.asarray(m, self.dtype) for m in masked_mlp_masks(adj, c["hidden_features"])]
            else:                                  # D == 1 and no context: free spline
                masks = None
            self.layers.append(dict(order=order, passes=p, masks=masks))
        self._activation = _ACTIVATIONS[c["activation"]]

    def _init_model_params(self, key):
        params = []
        for L in self.layers:
            key, k = jr.split(key)
            if L["masks"] is None:
                params.append([dict(W=jnp.zeros((0, self.D * self.total), self.dtype),
                                    b=jnp.zeros(self.D * self.total, self.dtype))])
            else:
                params.append(init_masked_mlp(k, [np.asarray(m) for m in L["masks"]],
                                              self.cfg["zero_init_last"], self.dtype))
        return params

    def _spline_params(self, p, L, x, c):
        """Hyper-network output for one transform -> knots, each (..., D, .)."""
        if L["masks"] is None:
            phi = jnp.broadcast_to(p[0]["b"], x.shape[:-1] + p[0]["b"].shape)
        else:
            inp = x if c is None else jnp.concatenate([x, c], axis=-1)
            phi = masked_mlp_apply(p, L["masks"], inp, self._activation)
        phi = phi.reshape(x.shape[:-1] + (self.D, self.total))
        K = self.bins
        w, h, d = phi[..., :K], phi[..., K:2 * K], phi[..., 2 * K:]
        return _rqs_knots(w, h, d, self.cfg["bound"], self.cfg["slope"])

    def _flow_forward(self, params, y, c):
        """normalized data -> latent, summed log|det|  (one parallel pass per transform)"""
        x, ladj = y, 0.0
        for p, L in zip(params, self.layers):
            xs, ys, ds = self._spline_params(p, L, x, c)
            x, l = rqs_forward(x, xs, ys, ds)
            ladj = ladj + jnp.sum(l, axis=-1)
        return x, ladj

    def _flow_inverse(self, params, z, c):
        """latent -> normalized data (zuko: `passes` fixed-point sweeps per transform)"""
        y = z
        for p, L in reversed(list(zip(params, self.layers))):
            target = y

            def sweep(_, x, p=p, L=L, target=target):
                xs, ys, ds = self._spline_params(p, L, x, c)
                return rqs_inverse(target, xs, ys, ds)

            y = jax.lax.fori_loop(0, L["passes"], sweep, jnp.zeros_like(target))
        return y