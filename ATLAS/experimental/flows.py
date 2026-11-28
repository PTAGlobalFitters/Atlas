"""
flows.py: normalizing flows in JAX.

Every flow maps physical samples to a standard normal latent space:

    physical x  --(fixed preprocessing)-->  normalized x  --(learned flow)-->  z ~ N(0, I)

Flows with a power-affine preprocessing
    Flow                   unconditional RQS coupling flow
    ConditionalMADEFlow    conditional masked autoregressive flow (MAF)

Flows with a sinh-arcsinh (SAS) preprocessing
    SASFlow                unconditional RQS coupling flow; also the joint p(theta, y)
                           of a simulation library (n_draws, n_real, n_features)
    ConditionalFlow        conditional RQS coupling flow (fast in both directions)
    SASNSF                 JAX port of zuko's conditional neural spline flow

Flow matching (at the end of the file, unchanged)
    FlowMatching, ConditionalFlowMatching

Early stopping (all flows except flow matching)
    A fraction `val_frac` of the data is held out.  After every epoch:
        1. the mean log p of the held-out data is computed, and
        2. the held-out data are mapped to the latent space (backward pass)
           and tested for being independent standard normal draws.
    By default training stops when either test says so; fit(stop_rule=...)
    can ask for both, or for the log p alone (see `EarlyStopping`).  The
    normality test only counts with >= 500 held-out samples.  The weights
    with the best held-out log p are kept.

Saving and loading (all flows except flow matching)
    flow.save('flow.pkl')           # everything except the training data
    flow = load_flow('flow.pkl')    # ready to use; no data, no constructor call
    flow.load('flow.pkl')           # into an existing flow, e.g. to keep training

All arrays are float64 (`jax_enable_x64` is switched on at import).
"""

import copy
import functools
import inspect
import math
import os
import pickle
import random
import warnings
from functools import partial

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax.scipy.special import ndtri
from scipy import stats
from tqdm.auto import trange

jax.config.update("jax_enable_x64", True)

LOG2PI = math.log(2.0 * math.pi)


##############################################################################################
#                           Power-affine normalizer
##############################################################################################
#
#   physical -> normalized:   x' = sign(x) |x|^p,   x_norm = B (x' - mean) / half_range
#   normalized -> physical:   x' = half_range x_norm / B + mean,   x = sign(x') |x'|^(1/p)
#
#   `alpha` = log(p) is what is stored.


def _fit_normalizer(min_x, max_x, p, B):
    """
    Fit the power-affine normalizer from the physical range of the data.

    :param min_x: per-feature lower bound, shape (D,)
    :param max_x: per-feature upper bound, shape (D,)
    :param p: power of the power transform
    :param B: half-width of the normalized space

    :return: dict with keys alpha, mean, half_range, B
    """
    min_x = np.asarray(min_x, dtype=np.float64)
    max_x = np.asarray(max_x, dtype=np.float64)
    alpha = np.log(float(p))
    gamma = np.exp(alpha)
    dummy_min = np.sign(min_x) * np.abs(min_x) ** gamma
    dummy_max = np.sign(max_x) * np.abs(max_x) ** gamma
    mean = (dummy_max + dummy_min) / 2.0
    half_range = (dummy_max - dummy_min) / 2.0
    return dict(alpha=alpha, mean=mean, half_range=half_range, B=float(B))


def _from_coefficients_np(arr, norm):
    """Physical -> normalized (numpy)."""
    alpha, mean, half_range, B = norm["alpha"], norm["mean"], norm["half_range"], norm["B"]
    dummy = np.sign(arr) * np.abs(arr) ** np.exp(alpha)
    return (dummy - mean) / half_range * B


def _to_coefficients_np(arr, norm):
    """Normalized -> physical (numpy)."""
    alpha, mean, half_range, B = norm["alpha"], norm["mean"], norm["half_range"], norm["B"]
    dummy = half_range / B * arr + mean
    return np.sign(dummy) * np.abs(dummy) ** (1.0 / np.exp(alpha))


def _from_coefficients_jnp(arr, alpha, mean, half_range, B):
    """Physical -> normalized (jax)."""
    dummy = jnp.sign(arr) * jnp.abs(arr) ** jnp.exp(alpha)
    return (dummy - mean) / half_range * B


def _to_coefficients_jnp(arr, alpha, mean, half_range, B):
    """Normalized -> physical (jax)."""
    dummy = half_range / B * arr + mean
    return jnp.sign(dummy) * jnp.abs(dummy) ** (1.0 / jnp.exp(alpha))


def _logdet_from_coefficients_jnp(arr, alpha, mean, half_range, B):
    """log|det J| of physical -> normalized, summed over features."""
    gamma = jnp.exp(alpha)
    return jnp.sum(
        jnp.log(gamma)
        + (gamma - 1.0) * jnp.log(jnp.abs(arr))
        + jnp.log(B / half_range),
        axis=-1,
    )


def _logdet_to_coefficients_jnp(arr, alpha, mean, half_range, B):
    """log|det J| of normalized -> physical, summed over features."""
    beta = 1.0 / jnp.exp(alpha)
    dummy = half_range / B * arr + mean
    return jnp.sum(
        jnp.log(half_range / B)
        + jnp.log(beta)
        + (beta - 1.0) * jnp.log(jnp.abs(dummy)),
        axis=-1,
    )


##############################################################################################
#                    Whitening (linear decorrelation after the normalizer)
##############################################################################################

WHITEN_METHODS = ("zca", "cholesky", "pca")
_WHITE_KEYS = ("w_mean", "W", "Winv", "w_logdet")


def fit_whitener(x, method="zca", shrinkage=0.0, max_samples=200_000, seed=0):
    """
    Fit a linear map y = W (x - m) with Cov(y) = I on normalized data.

    The per-feature normalizers never change how features depend on each
    other.  This map removes the linear part of that dependence, so the flow
    only has to learn the rest.

    :param x:
        normalized data, shape (n, D)
    :param method:
        "zca"      : W = S^(-1/2). Whitened feature j stays close to feature j
                     (good for coupling layers that split features by index).
        "cholesky" : W = L^(-1) with S = L L^T. Whitened feature j only
                     depends on features <= j (good for autoregressive flows).
        "pca"      : W = Lambda^(-1/2) U^T, components by decreasing variance.
    :param shrinkage:
        shrink the covariance towards its diagonal, in [0, 1].
        Use > 0 when features are nearly collinear.
    :param max_samples:
        fit on at most this many random rows
    :param seed:
        seed of the row subsample

    :return: dict of numpy arrays: w_mean (D,), W (D, D), Winv (D, D),
        w_logdet = log|det W| (the constant log-Jacobian of the map)
    """
    if method not in WHITEN_METHODS:
        raise ValueError(f"whiten must be one of {WHITEN_METHODS} or None; got {method!r}")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("fit_whitener expects (n, D) data")
    if len(x) > max_samples:
        x = x[np.random.default_rng(seed).choice(len(x), max_samples, replace=False)]
    if not np.all(np.isfinite(x)):
        raise ValueError("Non-finite values passed to fit_whitener.")

    m = x.mean(0)
    S = np.atleast_2d(np.cov(x, rowvar=False))
    if shrinkage:
        S = (1.0 - shrinkage) * S + shrinkage * np.diag(np.diag(S))
    lam, U = np.linalg.eigh(S)
    if lam.min() <= 1e-12 * lam.max():
        raise ValueError(
            "Covariance is (nearly) singular; set whiten_shrinkage > 0 "
            "or drop redundant features."
        )

    if method == "zca":
        W = (U / np.sqrt(lam)) @ U.T
        Winv = (U * np.sqrt(lam)) @ U.T
    elif method == "pca":
        order = np.argsort(lam)[::-1]
        lam, U = lam[order], U[:, order]
        W = (U / np.sqrt(lam)).T
        Winv = U * np.sqrt(lam)
    else:
        L = np.linalg.cholesky(S)
        W = np.linalg.solve(L, np.eye(len(L)))
        Winv = L
    return dict(w_mean=m, W=W, Winv=Winv, w_logdet=np.float64(np.linalg.slogdet(W)[1]))


def whiten_forward(x, w):
    """Normalized -> whitened. Returns (y, log|det|); the log-det is a constant."""
    return (x - w["w_mean"]) @ w["W"].T, w["w_logdet"]


def whiten_inverse(y, w):
    """Whitened -> normalized. Returns (x, log|det|)."""
    return y @ w["Winv"].T + w["w_mean"], -w["w_logdet"]


def _split_white(d):
    """Split a normalizer dict into (normalizer, whitener or None)."""
    if d is None or "W" not in d:
        return d, None
    rest = {k: v for k, v in d.items() if k not in _WHITE_KEYS}
    return rest, {k: np.asarray(d[k], np.float64) for k in _WHITE_KEYS}


##############################################################################################
#                               Learning-rate schedules
##############################################################################################

LR_SCHEDULES = ("constant", "cosine", "linear", "exponential")


def make_lr_scale(lr_schedule, learning_rate, total_steps, lr_min=None, warmup_steps=0):
    """
    Factor that multiplies every Adam update at each optimizer step.
    (Adam with its updates multiplied by f is Adam with learning rate
    learning_rate * f.)

    :param lr_schedule:
        None or "constant" : constant learning rate
        "cosine"           : cosine decay from learning_rate to lr_min
        "linear"           : linear decay from learning_rate to lr_min
        "exponential"      : geometric decay from learning_rate to lr_min
        callable           : step -> learning rate (e.g. an optax schedule);
                             lr_min and warmup_steps are then ignored
    :param learning_rate:
        the base learning rate
    :param total_steps:
        planned number of steps; after it the rate stays at lr_min
    :param lr_min:
        final learning rate; default learning_rate / 100
    :param warmup_steps:
        linear ramp from ~0 to learning_rate before the decay starts

    :return: function step -> factor. It returns None for a constant rate
        without warm-up, which leaves the updates untouched.
    """
    if lr_schedule is None or lr_schedule == "constant":
        if warmup_steps:
            return lambda t: min(1.0, (t + 1) / warmup_steps)
        return lambda t: None
    if callable(lr_schedule):
        return lambda t: float(lr_schedule(t)) / learning_rate
    if lr_schedule not in LR_SCHEDULES:
        raise ValueError(
            f"lr_schedule must be one of {LR_SCHEDULES}, a callable or None; got {lr_schedule!r}"
        )
    r = (learning_rate / 100.0 if lr_min is None else lr_min) / learning_rate
    if not 0 < r <= 1:
        raise ValueError("need 0 < lr_min <= learning_rate")
    total, w = max(1, int(total_steps)), max(0, int(warmup_steps))

    def factor(t):
        warm = min(1.0, (t + 1) / w) if w else 1.0
        x = min(1.0, max(0, t - w) / max(1, total - w - 1))  # decay progress in [0, 1]
        if lr_schedule == "cosine":
            return warm * (r + (1.0 - r) * 0.5 * (1.0 + np.cos(np.pi * x)))
        if lr_schedule == "linear":
            return warm * (1.0 - (1.0 - r) * x)
        return warm * r**x

    return factor


def _scale_updates(updates, lr_scale):
    if lr_scale is None:
        return updates
    return jax.tree_util.tree_map(lambda u: u * jnp.asarray(lr_scale, u.dtype), updates)


def _make_optimizer(learning_rate, grad_clip=None):
    """Adam, with optional global-norm gradient clipping."""
    tx = [optax.adam(learning_rate)]
    if grad_clip is not None:
        tx.insert(0, optax.clip_by_global_norm(grad_clip))
    return optax.chain(*tx)


##############################################################################################
#                               Rows with NaN / inf
##############################################################################################


def finite_row_mask(data, context=None, chunk_elements=2**22):
    """
    Boolean mask of the rows whose features are all finite.

    :param data:
        (N, D) or (n_groups, R, D); may be a memmap (read in chunks).
        The mask has shape data.shape[:-1].
    :param context:
        (N or n_groups, C) or None. A non-finite context entry marks every
        row of that row / group as invalid.
    """
    n = data.shape[0]
    per_first = int(np.prod(data.shape[1:])) or 1
    step = max(1, chunk_elements // per_first)
    mask = np.empty(data.shape[:-1], dtype=bool)
    for s in range(0, n, step):
        mask[s : s + step] = np.isfinite(np.asarray(data[s : s + step])).all(axis=-1)
    if context is not None:
        c = np.asarray(context, dtype=np.float64).reshape(n, -1)
        mask &= np.isfinite(c).all(axis=1).reshape((n,) + (1,) * (mask.ndim - 1))
    return mask


def _report_nonfinite(name, mask, unit="rows"):
    """Warn about ignored rows and return their number; raise if nothing is left."""
    n_bad = int(mask.size - np.count_nonzero(mask))
    if n_bad == mask.size:
        raise ValueError(f"{name}: every row contains NaN or inf.")
    if n_bad:
        warnings.warn(
            f"{name}: ignoring {n_bad} of {mask.size} {unit} that contain "
            f"NaN or inf ({100 * n_bad / mask.size:.2f}%).",
            stacklevel=3,
        )
    return n_bad


def _split_rows(rows, val_frac, rng):
    """Randomly split row indices into (train, validation), both sorted."""
    rows = np.asarray(rows)
    perm = rng.permutation(len(rows))
    n_val = int(round(val_frac * len(rows)))
    return np.sort(rows[perm[n_val:]]), np.sort(rows[perm[:n_val]])


##############################################################################################
#                         Memmap-safe batches (power-affine flows)
##############################################################################################


def _iter_batches(data, context, x_norm, c_norm, batch_size, key, x_white=None, rows=None):
    """
    Yield normalized (x_batch, c_batch) in random order, reading only one
    batch of rows at a time (so `data` and `context` can be memmaps).
    The last incomplete batch is dropped.

    :param data, context: arrays with the same first dimension N
    :param x_norm, c_norm: power-affine normalizer dicts (`_fit_normalizer`)
    :param batch_size: rows per batch
    :param key: jax PRNG key for the shuffling
    :param x_white: whitener applied to the data after x_norm, or None
    :param rows: row indices to draw from; None uses all N rows
    """
    rows = np.arange(data.shape[0]) if rows is None else np.asarray(rows)
    N = len(rows)
    perm = rows[np.array(jr.permutation(key, N))]
    for start in range(0, N, batch_size):
        idx = perm[start : start + batch_size]
        if len(idx) < batch_size:
            continue
        x_raw = np.asarray(data[idx], dtype=np.float64)
        c_raw = np.asarray(context[idx], dtype=np.float64)
        x_n = _from_coefficients_np(x_raw, x_norm)
        if x_white is not None:
            x_n = whiten_forward(x_n, x_white)[0]
        yield jnp.asarray(x_n), jnp.asarray(_from_coefficients_np(c_raw, c_norm))


##############################################################################################
#                           Sinh-arcsinh (SAS) normalizer
##############################################################################################
#
#   Per feature:   u = (x - loc) / s
#                  a = delta * arcsinh(u) - eps
#                  y = (sinh(a) - mu) / sd
#
#   loc, s, delta, eps are fitted by maximum likelihood so that sinh(a) is
#   close to N(0, 1); mu and sd are a final exact affine correction.


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
        return (
            jnp.where(ok[:, None], p_new, p),
            jnp.where(ok, f_new, f),
            jnp.clip(jnp.where(ok, lam * 0.3, lam * 5.0), 1e-10, 1e10),
        )

    f0 = safe(_nll_v(p0, xs))
    p, f, _ = jax.lax.fori_loop(
        0, num_iters, step, (p0, f0, jnp.full(p0.shape[0], 1e-3, xs.dtype))
    )
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
        ## Pad the last chunk so the fit is not compiled again
        if j1 - j0 < feature_chunk:
            blk = np.concatenate([blk, np.repeat(blk[:, -1:], feature_chunk - (j1 - j0), 1)], 1)

        ## Robust standardization as a starting point
        med = np.median(blk, 0)
        scale = 1.4826 * np.median(np.abs(blk - med), 0)
        scale = np.where(scale > 0, scale, blk.std(0))
        if np.any(scale <= 0):
            raise ValueError(f"A feature in columns {j0}:{j1} has zero spread.")
        xs = jnp.asarray((blk - med) / scale)

        ## Fit from two starting points and keep the better one
        best_p = best_f = None
        for s0 in starts:
            p, f = _fit_sas_chunk(
                xs, jnp.tile(jnp.asarray(s0), (feature_chunk, 1)), num_iters=num_iters
            )
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
        for name, v in (
            ("loc", loc), ("s", s), ("delta", delta), ("eps", eps),
            ("mu", y.mean(0)), ("sd", y.std(0)),
        ):
            out[name][j0:j1] = v[:k]
    return out


def fit_sas_normalizer(data, max_samples=200_000, seed=0, feature_chunk=None, num_iters=100):
    """
    Fit a per-feature sinh-arcsinh normalizer by maximum likelihood.

    :param data: array (N, D); may be a memmap. Must be finite.
    :param max_samples: fit on at most this many random rows
    :param seed: seed of the row subsample
    :param feature_chunk: features fitted at once; lower it if you run out of memory
    :param num_iters: Levenberg-Marquardt iterations

    :return: dict of arrays of shape (D,): loc, s, delta, eps, mu, sd
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
    """Physical -> normalized. Returns (y, log|dy/dx| summed over features)."""
    u = (x - n["loc"]) / n["s"]
    a = n["delta"] * jnp.arcsinh(u) - n["eps"]
    y = (jnp.sinh(a) - n["mu"]) / n["sd"]
    ld = (
        jnp.log(n["delta"]) - jnp.log(n["s"]) - jnp.log(n["sd"])
        + _logcosh(a) - jnp.log(jnp.hypot(1.0, u))
    )
    return y, jnp.sum(ld, axis=-1)


def sas_inverse(y, n):
    """Normalized -> physical. Returns (x, log|dx/dy| summed over features)."""
    a = jnp.arcsinh(n["sd"] * y + n["mu"])
    x = n["loc"] + n["s"] * jnp.sinh((a + n["eps"]) / n["delta"])
    return x, -sas_forward(x, n)[1]


##############################################################################################
#                       Context transforms (SAS family)
##############################################################################################


def fit_context_transform(c, kind="standardize", bounds=None, pad_frac=1e-3):
    """
    :param c: context samples, shape (n, C)
    :param kind:
        "standardize" : (c - mean) / std
        "probit"      : uniform design on [lo, hi] -> N(0, 1); bounds are
                        inferred (padded by pad_frac of the range) if None
        "sas"         : per-column SAS normalizer
        "none"        : identity
    :param bounds: (lo, hi) for "probit"
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
    """Physical context -> normalized context (no Jacobian)."""
    k = t["kind"]
    if k == "none":
        return c
    if k == "standardize":
        return (c - t["mean"]) / t["std"]
    if k == "probit":
        return ndtri(jnp.clip((c - t["lo"]) / (t["hi"] - t["lo"]), 1e-12, 1 - 1e-12))
    return sas_forward(c, t)[0]


##############################################################################################
#                       zuko building blocks (SASNSF)
##############################################################################################


def maf_adjacency(features, context, order, passes, total):
    """Adjacency of zuko.flows.autoregressive.MaskedAutoregressiveTransform."""
    passes = features if passes is None else passes
    passes = min(max(passes, 1), features)
    order = np.asarray(order) // math.ceil(features / passes)
    adj = order[:, None] > order[None, :]
    if context > 0:
        adj = np.concatenate([adj, np.ones((features, context), bool)], axis=1)
    return np.repeat(adj, total, axis=0), passes


def masked_mlp_masks(adjacency, hidden_features):
    """
    Masks of zuko.nn.MaskedMLP for a (out, in) boolean adjacency.
    Returns one mask per layer, each shaped (in_l, out_l).
    """
    adjacency = np.asarray(adjacency, bool)
    adj_u, inverse = np.unique(adjacency, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    a = adj_u.astype(np.float64)
    precedence = (a @ a.T) == adj_u.sum(-1)  # P_ij: deps(j) is a subset of deps(i)

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


_ACTIVATIONS = dict(
    relu=jax.nn.relu, elu=jax.nn.elu, silu=jax.nn.silu, gelu=jax.nn.gelu, tanh=jnp.tanh
)


def masked_mlp_apply(params, masks, x, activation):
    h = x
    for l, (p, m) in enumerate(zip(params, masks)):
        h = h @ (p["W"] * m) + p["b"]
        if l < len(params) - 1:
            h = activation(h)
    return h


def _rqs_knots(w, h, d, bound, slope):
    """Knots of zuko.transforms.MonotonicRQSTransform."""
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


##############################################################################################
#                       RQS coupling flows (Flow, SASFlow)
##############################################################################################


def _make_conditioner(event_shape, hidden_sizes, num_bijector_params):
    """MLP that gives the spline parameters. The last layer starts at zero (identity flow)."""
    return hk.Sequential(
        [
            hk.Flatten(preserve_dims=-len(event_shape)),
            hk.nets.MLP(hidden_sizes, activate_final=True),
            hk.Linear(
                np.prod(event_shape) * num_bijector_params, w_init=jnp.zeros, b_init=jnp.zeros
            ),
            hk.Reshape(event_shape + (num_bijector_params,), preserve_dims=-1),
        ]
    )


def make_coupling_flow(event_shape, num_layers, hidden_sizes, num_bins, range_min, range_max):
    """
    Distrax RQS masked-coupling flow with a standard normal base.
    Masks alternate by feature parity; for one feature every layer transforms it.
    """
    D = int(np.prod(event_shape))
    if D == 1:
        mask = jnp.zeros(event_shape, dtype=bool)
    else:
        mask = (jnp.arange(D) % 2).reshape(event_shape).astype(bool)

    def bijector_fn(params):
        return distrax.RationalQuadraticSpline(params, range_min=range_min, range_max=range_max)

    layers = []
    for _ in range(num_layers):
        layers.append(
            distrax.MaskedCoupling(
                mask=mask,
                bijector=bijector_fn,
                conditioner=_make_conditioner(event_shape, hidden_sizes, 3 * num_bins + 1),
            )
        )
        if D > 1:
            mask = jnp.logical_not(mask)

    base = distrax.Independent(
        distrax.Normal(jnp.zeros(event_shape), jnp.ones(event_shape)),
        reinterpreted_batch_ndims=len(event_shape),
    )
    return distrax.Transformed(base, distrax.Inverse(distrax.Chain(layers)))


##############################################################################################
#                       MADE modules (ConditionalMADEFlow)
##############################################################################################


class MaskedLinear(hk.Module):
    """
    Linear layer whose weights are multiplied by a fixed binary mask.

    :param out_features: number of outputs
    :param mask: (out_features, in_features) array
    :param with_bias: add a bias
    """

    def __init__(self, out_features, mask, with_bias=True, name=None):
        super().__init__(name=name)
        self.out_features = out_features
        self.mask = mask
        self.with_bias = with_bias

    def __call__(self, x):
        w = hk.get_parameter(
            "w",
            shape=(self.mask.shape[1], self.out_features),
            dtype=x.dtype,
            init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        )
        out = x @ (w * self.mask.T)
        if self.with_bias:
            b = hk.get_parameter("b", shape=(self.out_features,), dtype=x.dtype, init=jnp.zeros)
            out = out + b
        return out


class MADEBlock(hk.Module):
    """
    Masked autoencoder (Germain et al. 2015) that gives, for every feature i,
    the spline parameters of x_i from x_{<i} and the context c.

    Every unit has a degree m. A weight from unit j to unit k is kept when
    m(k) >= m(j) (input and hidden layers) or m(k) > m(j) (output layer),
    so output i only sees x_{<i}.  A linear projection of c is added to every
    hidden layer; with context, hidden units of degree -1 see only c, so the
    first feature is conditioned on c as well.

    :param D: number of features
    :param C: number of context features (0 for none)
    :param hidden_size: width of the hidden layers
    :param num_hidden: number of hidden layers
    :param num_params: spline parameters per feature (3 * num_bins + 1)
    """

    def __init__(self, D, C, hidden_size, num_hidden, num_params, name=None):
        super().__init__(name=name)
        self.D = D
        self.C = C
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden
        self.num_params = num_params

        ## Degrees of the input, hidden and output units
        self.m_input = np.arange(D)
        if C > 0:
            self.m_hidden = np.arange(hidden_size) % D - 1
        else:
            self.m_hidden = np.arange(hidden_size) % max(1, D - 1)
        self.m_output = np.arange(D)

        ## Masks from the degrees
        self.mask_in = jnp.asarray(self.m_hidden[:, None] >= self.m_input[None, :], dtype=jnp.float32)
        self.mask_hh = jnp.asarray(self.m_hidden[:, None] >= self.m_hidden[None, :], dtype=jnp.float32)
        mask_out = jnp.asarray(self.m_output[:, None] > self.m_hidden[None, :], dtype=jnp.float32)
        self.mask_out = jnp.repeat(mask_out, num_params, axis=0)

    def __call__(self, x, c=None):
        """
        :param x: (batch, D)
        :param c: (batch, C) or None
        :return: spline parameters, (batch, D, num_params)
        """
        h = MaskedLinear(self.hidden_size, self.mask_in, name="lin_in")(x)
        for i in range(self.num_hidden):
            h = jax.nn.tanh(h)
            ## Adding (not concatenating) the context keeps the autoregressive masks valid
            if c is not None:
                h = h + hk.Linear(
                    self.hidden_size,
                    name=f"ctx_proj_{i}",
                    w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                    b_init=jnp.zeros,
                )(c)
            if i < self.num_hidden - 1:
                h = MaskedLinear(self.hidden_size, self.mask_hh, name=f"lin_h{i}")(h)
        h = jax.nn.tanh(h)
        out = MaskedLinear(self.D * self.num_params, self.mask_out, name="lin_out")(h)
        return out.reshape(x.shape[0], self.D, self.num_params)


class MAFLayer(hk.Module):
    """
    One masked autoregressive layer: a MADEBlock with RQS bijectors.

    x -> z (density direction) is one parallel pass.
    z -> x (sampling direction) is a loop over the D features.
    """

    def __init__(self, D, C, hidden_size, num_hidden, num_bins, B, name=None):
        super().__init__(name=name)
        self.D = D
        self.num_bins = num_bins
        self.B = B
        self.made = MADEBlock(
            D, C, hidden_size, num_hidden,
            num_params=3 * num_bins + 1,
            name=f"{name}_made" if name else "made",
        )

    def _spline(self, params):
        return distrax.RationalQuadraticSpline(
            params, range_min=-(self.B + 1), range_max=self.B + 1
        )

    def inverse_and_log_det(self, x, c=None):
        """x -> z. Returns (z, log|det dz/dx|)."""
        params = self.made(x, c)

        def invert_dim(x_i, params_i):
            z_i, ld_i = self._spline(params_i[None, :]).inverse_and_log_det(x_i[None])
            return z_i[0], ld_i[0]

        def invert_row(x_row, params_row):
            z_row, ld_row = jax.vmap(invert_dim)(x_row, params_row)
            return z_row, ld_row.sum()

        return jax.vmap(invert_row)(x, params)

    def forward_single(self, z_row, c_row=None):
        """z -> x for ONE sample (D,), one feature at a time."""
        c_2d = c_row[None, :] if c_row is not None else None
        num_params = 3 * self.num_bins + 1

        def fill_feature(i, x_so_far):
            all_params = self.made(x_so_far[None, :], c_2d)
            params_i = jax.lax.dynamic_slice(all_params[0], (i, 0), (1, num_params))
            z_i = jax.lax.dynamic_slice(z_row, (i,), (1,))
            x_i, _ = self._spline(params_i).forward_and_log_det(z_i)
            return x_so_far.at[i].set(x_i[0])

        return jax.lax.fori_loop(0, self.D, fill_feature, jnp.zeros_like(z_row))


##############################################################################################
#                   Early stopping: validation log p + normality test
##############################################################################################


def _correlation(a, b):
    """Pearson correlation of every column of a (n, D) with every column of b (n, C)."""
    a = (a - a.mean(0)) / a.std(0)
    b = (b - b.mean(0)) / b.std(0)
    return a.T @ b / len(a)


def _ranks(a):
    """Ranks of every column; correlations of ranks are Spearman correlations."""
    return np.argsort(np.argsort(a, axis=0), axis=0).astype(np.float64)


def _correlation_p(a, b):
    """Two-sided p-values of zero correlation between the columns of a and of b."""
    n = len(a)
    r = _correlation(a, b).ravel()
    t = r * np.sqrt((n - 2) / np.maximum(1.0 - r**2, 1e-300))
    return 2.0 * stats.t.sf(np.abs(t), n - 2)


def _bartlett_p(x):
    """p-value of Bartlett's test that the correlation matrix of x (n, D) is I."""
    n, D = x.shape
    R = np.corrcoef(x, rowvar=False)
    chi2 = -(n - 1 - (2 * D + 5) / 6) * np.linalg.slogdet(R)[1]
    return stats.chi2.sf(chi2, D * (D - 1) / 2)


def normality_test(z, context=None):
    """
    Test whether the rows of `z` are independent draws from N(0, I).

    For a well-trained flow, the latents z = backward_pass(x, c) of held-out
    data are standard normal and independent, at every context.  Checks:
        1. every feature is N(0, 1): mean 0 (z-test), variance 1 (chi^2 test),
           skewness 0 and kurtosis 3 (D'Agostino tests), and shape
           (Kolmogorov-Smirnov)
        2. the features are independent: no correlation between the z's, nor
           between the |z|'s (Bartlett's test; ranks are used for |z|)
        3. z does not depend on the context: no correlation of z, nor of |z|
           (ranks), with any context column
    The combined p-value is the smallest p-value of all checks times their
    number (Bonferroni), so a perfect flow fails the test at level alpha
    with probability at most about alpha.

    :param z:
        latent samples, shape (n, D)
    :param context:
        the (normalized) context of every sample, shape (n, C), or None

    :return:
        dict with the combined `p_value` and the smallest p-value of each
        check: `p_marginal`, `p_dependence` and `p_context`
    """
    z = np.asarray(z, dtype=np.float64)
    n, D = z.shape
    if not np.all(np.isfinite(z)):
        return dict(p_value=0.0, p_marginal=0.0, p_dependence=0.0, p_context=0.0)

    ## First, every feature must be N(0, 1)
    s2 = np.sum(z**2, axis=0)
    p_marginal = np.concatenate([
        2.0 * stats.norm.sf(np.abs(z.mean(0)) * np.sqrt(n)),
        2.0 * np.minimum(stats.chi2.cdf(s2, n), stats.chi2.sf(s2, n)),
        np.atleast_1d(stats.skewtest(z, axis=0).pvalue),
        np.atleast_1d(stats.kurtosistest(z, axis=0).pvalue),
        np.atleast_1d(stats.kstest(z, "norm", axis=0).pvalue),
    ])

    ## Second, the features must be independent
    p_dependence = np.array([])
    if D > 1 and n > D + 10:
        p_dependence = np.array([_bartlett_p(z), _bartlett_p(_ranks(np.abs(z)))])

    ## Third, z must not depend on the context
    p_context = np.array([])
    if context is not None:
        c = np.asarray(context, dtype=np.float64).reshape(n, -1)
        c = c[:, c.std(0) > 0]
        if c.shape[1]:
            p_context = np.concatenate([
                _correlation_p(z, c),
                _correlation_p(_ranks(np.abs(z)), _ranks(c)),
            ])

    ## Finally, combine with a Bonferroni correction
    all_p = np.concatenate([p_marginal, p_dependence, p_context])
    smallest = lambda p: float(p.min()) if len(p) else 1.0
    return dict(
        p_value=min(1.0, len(all_p) * smallest(all_p)),
        p_marginal=smallest(p_marginal),
        p_dependence=smallest(p_dependence),
        p_context=smallest(p_context),
    )


class EarlyStopping(object):
    """
    Decides after every epoch whether training should stop, from held-out data.

    Two tests:
        1. validation log p : the mean log p of the held-out data has not
           improved for `patience` epochs.
        2. normality        : the held-out data, mapped to the latent space by
           the backward pass, pass `normality_test` (p-value > alpha) for
           `normality_patience` epochs in a row.

    :param patience:
        epochs without a better validation log p before test 1 says stop
    :param alpha:
        significance level of the normality test
    :param normality_patience:
        consecutive passed normality tests before test 2 says stop
    :param rule:
        "either" : stop when either test says so
        "both"   : stop only when both tests say so (training can then run
                   to num_epochs if the latents never look normal)
        "logp"   : use the validation log p only (no normality test)
    :param min_samples:
        the normality test only counts when at least this many held-out
        samples are available; with fewer it has no power and would stop
        training too early, so only the validation log p is used

    Author:
    Nima Laal (11/26/2026)
    """

    def __init__(self, patience=10, alpha=0.01, normality_patience=3, rule="either", min_samples=500):
        if rule not in ("either", "both", "logp"):
            raise ValueError("rule must be 'either', 'both' or 'logp'")
        self.patience = patience
        self.alpha = alpha
        self.normality_patience = normality_patience
        self.rule = rule
        self.min_samples = min_samples

        self.best_logp = -np.inf
        self.bad_epochs = 0
        self.normal_epochs = 0
        self.last_p = np.nan
        self.stopped = False

    def update(self, val_logp, z=None, context=None):
        """
        Record one epoch.

        :param val_logp: mean log p of the held-out data
        :param z: latents of held-out data, (n, D)
        :param context: their normalized context, (n, C), or None

        :return: (improved, stop)
        """
        n = 0 if z is None else len(z)

        ## First, the validation log p
        improved = bool(np.isfinite(val_logp) and val_logp > self.best_logp)
        if improved:
            self.best_logp, self.bad_epochs = float(val_logp), 0
        else:
            self.bad_epochs += 1
        logp_says_stop = self.bad_epochs >= self.patience

        ## Second, the normality of the latents
        normal_says_stop = False
        if self.uses_normality(n):
            self.last_p = normality_test(z, context)["p_value"]
            self.normal_epochs = self.normal_epochs + 1 if self.last_p > self.alpha else 0
            normal_says_stop = self.normal_epochs >= self.normality_patience

        ## Third, combine them
        if not self.uses_normality(n):
            stop = logp_says_stop
        elif self.rule == "either":
            stop = logp_says_stop or normal_says_stop
        else:
            stop = logp_says_stop and normal_says_stop
        self.stopped = stop
        return improved, stop

    def uses_normality(self, n):
        """Whether the normality test is used with n held-out samples."""
        return self.rule != "logp" and n >= self.min_samples


##############################################################################################
#                        Shared training loop, saving and loading
##############################################################################################


def load_flow(path):
    """
    Load any flow saved with `flow.save(path)`.  The flow is ready to use
    (log_prob, sample, forward_pass, backward_pass) without its training data
    and without calling the class constructor.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_numpy(a):
    return np.asarray(a) if isinstance(a, jax.Array) else a


class _FlowBase(object):
    """
    What all flows (except flow matching) share:

        * pickling: `save`, `load`, `load_flow`.  The pickle holds every
          attribute except the training data (`_DATA`) and the functions
          made by `_build` (`_BUILT`), which are rebuilt when it is loaded.
        * the training loop with validation, early stopping and checkpoints.

    A subclass defines
        _DATA, _BUILT     attribute names (the first name in _DATA is the
                          training data itself)
        _build()          haiku functions, optimizer, jitted functions
        _arch_cfg()       settings that must match for `load`
        _n_held_out()     number of held-out samples
        _held_out(n)      up to n held-out (x, c) in physical units, without repeats
        _validate(x, c, n_normality) -> (mean log p, latents, normalized context)
        _refresh_data()   recompute arrays derived from the data after `load`
    """

    _DATA = ()
    _BUILT = ()

    ##################################################################
    #                         pickling

    def __getstate__(self):
        drop = set(self._DATA) | set(self._BUILT)
        state = {k: v for k, v in self.__dict__.items() if k not in drop}
        return jax.tree_util.tree_map(_to_numpy, state)

    def __setstate__(self, state):
        self.__dict__.update(state)
        for k in self._DATA:
            setattr(self, k, None)
        self._build()

    def save(self, path):
        """Pickle the flow (everything except the training data). The write is atomic."""
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    @classmethod
    def from_pickle(cls, path):
        """Same as `load_flow`, with a check of the class."""
        flow = load_flow(path)
        if not isinstance(flow, cls):
            raise TypeError(f"{path} holds a {type(flow).__name__}, not a {cls.__name__}")
        return flow

    def load(self, path):
        """
        Load a saved flow into this one, keeping this flow's training data
        (e.g. to keep training).  The architecture must match.
        """
        return self._load_from(load_flow(path), path)

    def _load_from(self, other, path):
        if type(other) is not type(self):
            raise TypeError(f"{path} holds a {type(other).__name__}, not a {type(self).__name__}")
        want, got = self._arch_cfg(), other._arch_cfg()
        differ = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
        if differ:
            raise ValueError(f"{path} has a different setup; (saved, this flow) differ on {differ}")
        self.__dict__.update(other.__getstate__())
        if self._has_data():
            self._refresh_data()
        self._build()
        return self

    ##################################################################
    #                    hooks with defaults

    def _arch_cfg(self):
        return {}

    def _new_history(self):
        self.history = dict(train=[], val=[], normality_p=[], lr=[])
        self.epochs_done = 0
        self.stopper = None

    def _refresh_data(self):
        pass

    def _has_data(self):
        return all(getattr(self, k, None) is not None for k in self._DATA[:1])

    def _require_data(self):
        if not self._has_data():
            raise RuntimeError(
                "This flow was loaded without its training data. To keep training, "
                "construct it with the data and call flow.load(path)."
            )

    ##################################################################
    #                    shared training loop

    def _train(
        self,
        run_epoch,
        num_epochs,
        steps_per_epoch,
        lr_scale,
        checkpoint,
        save_path,
        stopper,
        n_val,
        n_normality,
        desc,
        colour="blue",
        verbose=True,
        val_every=1,
    ):
        """
        Run `run_epoch(step) -> mean training loss` for every epoch, with
        validation, early stopping, checkpoints and saving of the best flow.
        """
        ## First, resume from the checkpoint if there is one
        start = 0
        if checkpoint is not None and os.path.exists(checkpoint):
            self.load(checkpoint)
            start = self.epochs_done
            stopper = self.stopper if self.stopper is not None else stopper
            print(f"[{type(self).__name__}] resuming from {checkpoint} at epoch "
                  f"{start}/{num_epochs}", flush=True)
        else:
            self.epochs_done = 0

        ## Second, fix the held-out samples used after every epoch
        validate = stopper is not None and self._n_held_out() > 0
        self.stopper = stopper if validate else None
        if validate:
            if stopper.stopped:
                self.params = self.best_params
                return self.history
            val_x, val_c = self._held_out(n_val)
            n_test = min(n_normality, len(val_x))
            if stopper.rule != "logp" and not stopper.uses_normality(n_test):
                warnings.warn(f"only {n_test} held-out samples for the normality test "
                              f"(< {stopper.min_samples}); stopping on the validation log p only.")

        ## Third, train
        bar = trange(start, num_epochs, initial=start, total=num_epochs,
                     colour=colour, desc=desc, disable=not verbose)
        for epoch in bar:
            step = epoch * steps_per_epoch
            self.history["lr"].append(self.learning_rate * (lr_scale(step) or 1.0))
            loss = run_epoch(step)
            self.history["train"].append(loss)
            self.epochs_done = epoch + 1
            msg = dict(train_nll=f"{loss:.3f}")

            ## Fourth, validate and decide whether to stop
            stop = False
            if validate and (epoch + 1) % val_every == 0:
                logp, z, c_hat = self._validate(val_x, val_c, n_normality)
                improved, stop = stopper.update(logp, z, c_hat)
                self.history["val"].append(logp)
                self.history["normality_p"].append(stopper.last_p)
                msg.update(val_logp=f"{logp:.3f}", normality_p=f"{stopper.last_p:.2g}")
                if improved:
                    self.best_params = self.params
                    if save_path is not None:
                        self.save(save_path)
            bar.set_postfix(msg)

            if checkpoint is not None:
                self.save(checkpoint)
            if stop:
                break

        ## Finally, keep the weights with the best validation log p
        if validate:
            self.params = self.best_params
        return self.history



##############################################################################################
#                                           Flow
##############################################################################################


class Flow(_FlowBase):
    """
    Unconditional rational-quadratic spline (RQS) coupling flow with a
    power-affine preprocessing.

        physical x --power-affine--> x_norm --(whitening)--> coupling layers --> z ~ N(0, I)

    The power-affine map of every feature is
        x'     = sign(x) |x|^p
        x_norm = B (x' - mean) / half_range
    with mean and half_range from the range of x', so x_norm lies in [-B, B].
    The splines act on [-(B + 1), B + 1].

    Adapted from the pyprobml spline-flow notebook (Murphy et al. 2021):
    https://cocalc.com/github/probml/pyprobml/blob/master/deprecated/notebooks/flow_spline_mnist_jax.ipynb

    :param data:
        training samples, shape (N, D)
    :param data_min:
        per-feature lower bound of the physical data, shape (D,).
        Taken from `data` if None.
    :param data_max:
        per-feature upper bound of the physical data, shape (D,).
        Taken from `data` if None.
    :param tset_paths:
        list of .npy training files, used by fit(mode='from_disk')
    :param flow_num_layers:
        number of coupling layers
    :param hidden_size:
        width of the hidden layers of the conditioner networks
    :param mlp_num_layers:
        number of hidden layers of the conditioner networks
    :param num_bins:
        number of spline bins
    :param learning_rate:
        learning rate of Adam
    :param B:
        half-width of the normalized space
    :param p:
        power of the power transform
    :param val_frac:
        fraction of the rows of `data` held out for early stopping (0 turns it off)
    :param seed:
        random seed
    :param whiten:
        None, "zca", "cholesky" or "pca": decorrelate the normalized data with a
        fixed linear map before the coupling layers (see `fit_whitener`)
    :param whiten_shrinkage:
        covariance shrinkage of the whitener, in [0, 1]
    :param drop_nonfinite:
        ignore the rows of `data` (and of the files read in 'from_disk' mode)
        that contain NaN or inf

    Author:
    Nima Laal (11/26/2026)
    """

    _DATA = ("_data", "x_norm", "train_rows", "val_rows")
    _BUILT = (
        "log_prob_fn", "forward_fn", "inverse_fn", "optimizer",
        "_update", "_log_prob_jit", "_forward_jit", "_backward_jit",
    )

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
        val_frac=0.1,
        seed=0,
        whiten=None,
        whiten_shrinkage=0.0,
        drop_nonfinite=True,
    ):
        self.tset_paths = tset_paths
        self.flow_num_layers = flow_num_layers
        self.hidden_sizes = [hidden_size] * mlp_num_layers
        self.num_bins = num_bins
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.key = jr.PRNGKey(seed)

        ## Drop the rows with NaN / inf
        self.drop_nonfinite = bool(drop_nonfinite)
        self.n_dropped = 0
        if self.drop_nonfinite:
            mask = finite_row_mask(data)
            self.n_dropped = _report_nonfinite("Flow", mask)
            if self.n_dropped:
                data = np.asarray(data)[mask]
        self._data = data
        self.N, self.D = data.shape
        self.event_shape = (self.D,)
        self.train_rows, self.val_rows = _split_rows(np.arange(self.N), val_frac, self.rng)

        ## Power-affine normalizer, from the given bounds or from the data
        self.B = float(B)
        self.alpha = jnp.log(p)
        self.gamma = jnp.exp(self.alpha)
        if (data_min is not None and data_max is not None
                and np.any(data_min) and np.any(data_max)):
            dummy_min = np.sign(data_min) * np.abs(data_min) ** self.gamma
            dummy_max = np.sign(data_max) * np.abs(data_max) ** self.gamma
        else:
            dummy = jnp.sign(data) * jnp.abs(data) ** self.gamma
            dummy_min, dummy_max = dummy.min(axis=0), dummy.max(axis=0)
        self.mean_x = (dummy_max + dummy_min) / 2.0
        self.half_range = (dummy_max - dummy_min) / 2.0

        ## Optional whitening, fitted on the normalized training rows
        self.whiten = whiten
        self.white = self.white_np = None
        if whiten is not None:
            train = np.asarray(data)[self.train_rows]
            dummy = np.sign(train) * np.abs(train) ** float(self.gamma)
            affine = (dummy - np.asarray(self.mean_x)) / np.asarray(self.half_range) * self.B
            self.white_np = fit_whitener(affine, whiten, whiten_shrinkage, seed=seed)
            self.white = {k: jnp.asarray(v) for k, v in self.white_np.items()}

        ## Normalized training data
        self._refresh_data()

        self._new_history()
        self._build()
        self._init_params()

    ##################################################################
    #                    physical <-> normalized

    def to_unit_interval(self, arr):
        """Physical -> normalized space [-B, B] (then whitened, if whiten is set)."""
        dummy = jnp.sign(arr) * jnp.abs(arr) ** jnp.exp(self.alpha)
        out = (dummy - self.mean_x[None, :]) / self.half_range[None, :] * self.B
        if self.white is not None:
            out = whiten_forward(out, self.white)[0]
        return out

    def from_unit_interval(self, arr):
        """Normalized -> physical space."""
        if self.white is not None:
            arr = whiten_inverse(arr, self.white)[0]
        dummy = self.half_range[None, :] / self.B * arr + self.mean_x[None, :]
        return jnp.sign(dummy) * jnp.abs(dummy) ** (1.0 / jnp.exp(self.alpha))

    def logdet_to_unit_interval(self, arr):
        """log|det J| of physical -> normalized, at physical points `arr`."""
        gamma = jnp.exp(self.alpha)
        logdet = jnp.sum(
            jnp.log(gamma)
            + (gamma - 1.0) * jnp.log(jnp.abs(arr))
            + jnp.log(self.B / self.half_range),
            axis=-1,
        )
        if self.white is not None:
            logdet = logdet + self.white["w_logdet"]
        return logdet

    def logdet_from_unit_interval(self, arr):
        """log|det J| of normalized -> physical, at normalized points `arr`."""
        beta = 1.0 / jnp.exp(self.alpha)
        if self.white is not None:
            arr = whiten_inverse(arr, self.white)[0]
        dummy = self.half_range[None, :] / self.B * arr + self.mean_x[None, :]
        logdet = jnp.sum(
            jnp.log(self.half_range / self.B)
            + jnp.log(beta)
            + (beta - 1.0) * jnp.log(jnp.abs(dummy)),
            axis=-1,
        )
        if self.white is not None:
            logdet = logdet - self.white["w_logdet"]
        return logdet

    ##################################################################
    #                    model

    def _build(self):
        """Haiku functions, the optimizer and the jitted functions."""
        args = (self.event_shape, self.flow_num_layers, self.hidden_sizes,
                self.num_bins, -self.B - 1, self.B + 1)
        model = lambda: make_coupling_flow(*args)

        ## Normalized space: log p, latent -> data, data -> latent
        self.log_prob_fn = hk.without_apply_rng(hk.transform(lambda x: model().log_prob(x)))
        self.forward_fn = hk.without_apply_rng(
            hk.transform(lambda z: model().bijector.forward_and_log_det(z)))
        self.inverse_fn = hk.without_apply_rng(
            hk.transform(lambda x: model().bijector.inverse_and_log_det(x)))
        self.optimizer = optax.adam(self.learning_rate)

        ## Physical space; the weights are always an argument
        def log_prob(params, x):
            return self.log_prob_fn.apply(params, self.to_unit_interval(x)) + self.logdet_to_unit_interval(x)

        def forward(params, z):
            x_norm, ld_flow = self.forward_fn.apply(params, z)
            return self.from_unit_interval(x_norm), ld_flow + self.logdet_from_unit_interval(x_norm)

        def backward(params, x):
            return self.inverse_fn.apply(params, self.to_unit_interval(x))[0]

        def loss_fn(params, batch):
            return -jnp.mean(self.log_prob_fn.apply(params, batch))

        @jax.jit
        def update(params, opt_state, batch, lr_scale=None):
            loss, grads = jax.value_and_grad(loss_fn)(params, batch)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            updates = _scale_updates(updates, lr_scale)
            return optax.apply_updates(params, updates), opt_state, loss

        self._update = update
        self._log_prob_jit = jax.jit(log_prob)
        self._forward_jit = jax.jit(forward)
        self._backward_jit = jax.jit(backward)

    def _init_params(self):
        self.key, init_key = jr.split(self.key)
        self.params = self.log_prob_fn.init(init_key, self.x_norm[:1])
        self.opt_state = self.optimizer.init(self.params)
        self.best_params = self.params

    def _arch_cfg(self):
        return dict(
            D=int(self.D), flow_num_layers=int(self.flow_num_layers),
            hidden_sizes=[int(h) for h in self.hidden_sizes], num_bins=int(self.num_bins),
            B=float(self.B), gamma=float(self.gamma), whiten=self.whiten,
        )

    def _refresh_data(self):
        self.x_norm = self.to_unit_interval(np.asarray(self._data[self.train_rows], dtype=np.float64))

    ##################################################################
    #                    training

    @staticmethod
    def get_batches(data, batch_size, key):
        """Shuffled mini-batches of `data`."""
        idx = jax.random.permutation(key, data.shape[0])
        for i in range(0, data.shape[0], batch_size):
            yield data[idx[i : i + batch_size]]

    def fit(
        self,
        mode="pre-loaded",
        num_epochs=100,
        batch_size=256,
        checkpoint=None,
        save_path=None,
        lr_schedule=None,
        lr_min=None,
        warmup_steps=0,
        patience=10,
        normality_alpha=0.01,
        normality_patience=3,
        stop_rule="either",
        stopper=None,
        n_val=50_000,
        n_normality=5000,
        val_every=1,
        verbose=True,
    ):
        """
        Train by maximum likelihood.

        :param mode:
            'pre-loaded' : epochs over the training rows of `data`
            'from_disk'  : every "epoch" is one batch from a random file of tset_paths
        :param num_epochs: number of epochs
        :param batch_size: rows per batch
        :param checkpoint: path; the flow is saved there after every epoch,
            and training resumes from it if it exists
        :param save_path: path; the flow is saved there whenever the validation log p improves
        :param lr_schedule: None, "cosine", "linear", "exponential" or a callable (see `make_lr_scale`)
        :param lr_min: final learning rate of the decay; default learning_rate / 100
        :param warmup_steps: steps of linear warm-up before the decay
        :param patience, normality_alpha, normality_patience, stop_rule:
            early stopping (see `EarlyStopping`); needs val_frac > 0
        :param stopper: an `EarlyStopping` object; if given, it replaces the four above
        :param n_val: held-out rows used for the validation log p
        :param n_normality: held-out rows (out of n_val) used for the normality test
        :param val_every: validate every this many epochs (useful in 'from_disk' mode)
        :param verbose: show a progress bar

        :return: self.history (per-epoch lists: train, val, normality_p, lr)
        """
        self._require_data()
        if mode not in ("pre-loaded", "from_disk"):
            raise ValueError("mode must be 'pre-loaded' or 'from_disk'")
        steps_per_epoch = -(-len(self.train_rows) // batch_size) if mode == "pre-loaded" else 1
        lr_scale = make_lr_scale(lr_schedule, self.learning_rate, num_epochs * steps_per_epoch,
                                 lr_min, warmup_steps)

        def preloaded_epoch(step):
            self.key, subkey = jr.split(self.key)
            losses = []
            for batch in self.get_batches(self.x_norm, batch_size, subkey):
                self.params, self.opt_state, loss = self._update(
                    self.params, self.opt_state, batch, lr_scale(step))
                losses.append(loss)
                step += 1
            return float(jnp.mean(jnp.stack(losses)))

        def from_disk_step(step):
            training_set = np.load(self.tset_paths[random.randint(0, len(self.tset_paths) - 1)])
            if self.drop_nonfinite:
                training_set = training_set[np.all(np.isfinite(training_set), axis=1)]
            idxs = random.sample(range(training_set.shape[0]), k=min(batch_size, training_set.shape[0]))
            batch = self.to_unit_interval(training_set[np.array(idxs)])
            self.params, self.opt_state, loss = self._update(
                self.params, self.opt_state, batch, lr_scale(step))
            return float(loss)

        stopper = stopper or EarlyStopping(patience, normality_alpha, normality_patience, stop_rule)
        return self._train(
            preloaded_epoch if mode == "pre-loaded" else from_disk_step,
            num_epochs, steps_per_epoch, lr_scale, checkpoint, save_path, stopper,
            n_val, n_normality, desc=f"Training Flow ({mode})", colour="blue",
            verbose=verbose, val_every=val_every,
        )

    def _n_held_out(self):
        return len(self.val_rows)

    def _held_out(self, n):
        rng = np.random.default_rng(0)  # same held-out samples in every fit call
        rows = np.sort(rng.choice(self.val_rows, size=min(n, len(self.val_rows)), replace=False))
        x = np.asarray(self._data[rows], dtype=np.float64)
        return x[rng.permutation(len(x))], None

    def _validate(self, x, c, n_normality):
        logp = float(jnp.mean(self.log_prob(x)))
        return logp, self.backward_pass(x[:n_normality]), None

    ##################################################################
    #                    public API (physical space)

    def log_prob(self, x):
        """log p(x), x of shape (n, D)."""
        return self._log_prob_jit(self.params, jnp.atleast_2d(x))

    def sample(self, num_samples):
        """Draw (num_samples, D) samples."""
        self.key, subkey = jr.split(self.key)
        return self.forward_pass(jr.normal(subkey, (num_samples, self.D)))

    def forward_pass(self, z):
        """Latent z (n, D) -> physical x."""
        return self._forward_jit(self.params, jnp.atleast_2d(z))[0]

    def forward_pass_with_logdet(self, z):
        """Latent z -> (physical x, log|det dx/dz|)."""
        return self._forward_jit(self.params, jnp.atleast_2d(z))

    def backward_pass(self, target_samp):
        """Physical x (n, D) -> latent z."""
        return self._backward_jit(self.params, jnp.atleast_2d(target_samp))

    ##################################################################
    #                    weights only (older .npz format)

    def save_params(self, flow_save_path):
        """Save the weights (and the whitener) only. Prefer `save`."""
        leaves, treedef = jax.tree_util.tree_flatten(self.params)
        save_dict = {f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)}
        save_dict["treedef"] = np.array(str(treedef))
        if self.white is not None:
            save_dict.update({f"white_{k}": v for k, v in self.white_np.items()})
        np.savez(flow_save_path, **save_dict)

    def load_params(self, flow_load_path):
        """Load weights saved by `save_params` into a flow built with the same data."""
        data = np.load(flow_load_path, allow_pickle=True)
        saved_white = "white_W" in data.files
        if saved_white != (self.white is not None):
            raise ValueError("saved flow and this flow differ on `whiten`")
        if saved_white and not np.allclose(data["white_W"], self.white_np["W"]):
            raise ValueError("saved whitener differs from the one fitted here; "
                             "construct the flow with the training data")
        _, treedef = jax.tree_util.tree_flatten(self.params)
        leaves = [jnp.asarray(data[f"leaf_{i}"]) for i in range(treedef.num_leaves)]
        self.params = self.best_params = jax.tree_util.tree_unflatten(treedef, leaves)


##############################################################################################
#                                   ConditionalMADEFlow
##############################################################################################


class ConditionalMADEFlow(_FlowBase):
    """
    Conditional masked autoregressive flow (MAF): stacked MAF layers, each a
    MADE network with RQS splines, with the power-affine preprocessing of
    the data and the context.

        physical x --power-affine--> x_norm --(whitening)--> MAF layers(c_norm) --> z ~ N(0, I)
        physical c --power-affine--> c_norm

    Every feature i is transformed by a spline whose parameters depend on
    x_{<i} and c.  The feature order is reversed between layers.

        log_prob, backward_pass : one parallel pass per layer (fast)
        sample, forward_pass    : D sequential steps per layer (slow for large D)

    Use it when you evaluate log p much more often than you sample.

    :param data:
        training samples, shape (N, D); may be a memmap
    :param context:
        context of every sample, shape (N, C); may be a memmap
    :param data_min, data_max:
        per-feature physical range of the data, shape (D,)
    :param context_min, context_max:
        per-feature physical range of the context, shape (C,)
    :param normalizer_dict:
        dict(data=..., context=...) of pre-fitted normalizers (`_fit_normalizer`);
        the ranges above are then ignored
    :param tset_paths:
        list of .npy files used by fit(mode='on-the-fly')
    :param last_feature_index:
        in those files, data = arr[:, :last_feature_index], context = arr[:, last_feature_index:]
    :param flow_save_path, flow_load_path:
        paths used by `save_params` / `load_params`
    :param p:
        power of the power transform
    :param B:
        half-width of the normalized space
    :param flow_num_layers:
        number of MAF layers
    :param hidden_size:
        width of the MADE hidden layers
    :param mlp_num_layers:
        number of MADE hidden layers
    :param num_bins:
        number of spline bins
    :param learning_rate:
        learning rate of Adam
    :param val_frac:
        fraction of the rows held out for early stopping (0 turns it off)
    :param seed:
        random seed
    :param whiten:
        None, "zca", "cholesky" or "pca" (see `fit_whitener`). "cholesky" matches
        the autoregressive order of the first layer. If normalizer_dict['data']
        already holds a whitener, that one is used.
    :param whiten_shrinkage:
        covariance shrinkage of the whitener
    :param drop_nonfinite:
        never train on rows whose data or context contain NaN or inf (nothing is copied)

    Author:
    Nima Laal (11/26/2026)
    """

    _DATA = ("_data", "_context", "train_rows", "val_rows")
    _BUILT = (
        "log_prob_fn", "inverse_fn", "forward_fn", "optimizer",
        "_update", "_log_prob_jit", "_forward_jit", "_backward_jit",
    )

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
        mlp_num_layers=2,
        num_bins=8,
        learning_rate=1e-4,
        val_frac=0.1,
        seed=0,
        whiten=None,
        whiten_shrinkage=0.0,
        drop_nonfinite=True,
    ):
        self.N, self.D = data.shape[0], data.shape[1]
        self.C = context.shape[1]
        self.tset_paths = tset_paths
        self.last_feature_idx = last_feature_index
        self.flow_save_path = flow_save_path
        self.flow_load_path = flow_load_path
        self.flow_num_layers = flow_num_layers
        self.hidden_size = hidden_size
        self.num_hidden = mlp_num_layers
        self.num_bins = num_bins
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.key = jr.PRNGKey(seed)
        self._data = data
        self._context = context

        ## Rows with NaN / inf are never read; split the rest
        rows = np.arange(self.N)
        self.drop_nonfinite = bool(drop_nonfinite)
        self.n_dropped = 0
        if self.drop_nonfinite:
            mask = finite_row_mask(data, context)
            self.n_dropped = _report_nonfinite("ConditionalMADEFlow", mask)
            rows = np.flatnonzero(mask)
        self.train_rows, self.val_rows = _split_rows(rows, val_frac, self.rng)

        ## Power-affine normalizers
        if normalizer_dict is not None and any(normalizer_dict):
            self.x_norm = normalizer_dict["data"]
            self.c_norm = normalizer_dict["context"]
        else:
            self.x_norm = _fit_normalizer(data_min, data_max, p, B)
            self.c_norm = _fit_normalizer(context_min, context_max, p, B)

        ## Optional whitening of the normalized data (stored in self.x_norm)
        self.whiten = whiten
        x_aff_norm, white = _split_white(self.x_norm)
        if white is None and whiten is not None:
            n_fit = min(len(self.train_rows), 200_000)
            idx = np.sort(np.random.default_rng(seed).choice(self.train_rows, n_fit, replace=False))
            x_fit = _from_coefficients_np(np.asarray(data[idx], dtype=np.float64), x_aff_norm)
            white = fit_whitener(x_fit, whiten, whiten_shrinkage, seed=seed)
            self.x_norm = dict(self.x_norm, **white)
        self._x_aff_norm = x_aff_norm
        self._x_white_np = white
        self._x_white = None if white is None else {k: jnp.asarray(v) for k, v in white.items()}

        self._x_alpha = jnp.float64(self.x_norm["alpha"])
        self._x_mean = jnp.asarray(self.x_norm["mean"])
        self._x_half_range = jnp.asarray(self.x_norm["half_range"])
        self._x_B = jnp.float64(self.x_norm["B"])
        self._c_alpha = jnp.float64(self.c_norm["alpha"])
        self._c_mean = jnp.asarray(self.c_norm["mean"])
        self._c_half_range = jnp.asarray(self.c_norm["half_range"])
        self._c_B = jnp.float64(self.c_norm["B"])
        self.B_float = float(self.x_norm["B"])

        self._new_history()
        self._build()
        self._init_params()

    ##################################################################
    #                    physical <-> normalized

    def normalize_data(self, x):
        """Physical -> normalized data (power-affine, then whitened if enabled)."""
        out = _from_coefficients_jnp(x, self._x_alpha, self._x_mean, self._x_half_range, self._x_B)
        if self._x_white is not None:
            out = whiten_forward(out, self._x_white)[0]
        return out

    def denormalize_data(self, x_norm):
        if self._x_white is not None:
            x_norm = whiten_inverse(x_norm, self._x_white)[0]
        return _to_coefficients_jnp(x_norm, self._x_alpha, self._x_mean, self._x_half_range, self._x_B)

    def normalize_context(self, c):
        return _from_coefficients_jnp(c, self._c_alpha, self._c_mean, self._c_half_range, self._c_B)

    def denormalize_context(self, c_norm):
        return _to_coefficients_jnp(c_norm, self._c_alpha, self._c_mean, self._c_half_range, self._c_B)

    def _logdet_normalize_data(self, x):
        ld = _logdet_from_coefficients_jnp(x, self._x_alpha, self._x_mean, self._x_half_range, self._x_B)
        if self._x_white is not None:
            ld = ld + self._x_white["w_logdet"]
        return ld

    def _logdet_denormalize_data(self, x_norm):
        ld = 0.0
        if self._x_white is not None:
            x_norm, ld = whiten_inverse(x_norm, self._x_white)
        return ld + _logdet_to_coefficients_jnp(
            x_norm, self._x_alpha, self._x_mean, self._x_half_range, self._x_B)

    ##################################################################
    #                    model

    def _make_maf_stack(self, x, c, inverse=True):
        """
        Apply all MAF layers (inside a haiku transform).

        inverse=True  : x -> (z, log|det dz/dx|)   (fast)
        inverse=False : z -> x                      (sequential)
        """
        D, num_layers = self.D, self.flow_num_layers
        perm = jnp.arange(D - 1, -1, -1)  # reverse the feature order between layers
        inv_perm = jnp.argsort(perm)
        layer = lambda i: MAFLayer(D, self.C, self.hidden_size, self.num_hidden,
                                   self.num_bins, self.B_float, name=f"maf_{i}")

        if inverse:
            h, total_log_det = x, jnp.zeros(x.shape[0])
            for i in range(num_layers):
                h, ld = layer(i).inverse_and_log_det(h, c)
                total_log_det = total_log_det + ld
                if i < num_layers - 1:
                    h = h[:, perm]
            return h, total_log_det

        h = x
        for i in range(num_layers - 1, -1, -1):
            if i < num_layers - 1:
                h = h[:, inv_perm]
            maf = layer(i)
            if c is not None:
                h = jax.vmap(maf.forward_single)(h, c)
            else:
                h = jax.vmap(lambda z_row: maf.forward_single(z_row))(h)
        return h

    def _build(self):
        """Haiku functions, the optimizer and the jitted functions."""

        def log_prob_norm(x_norm, c_norm):
            z, log_det = self._make_maf_stack(x_norm, c_norm, inverse=True)
            return jnp.sum(-0.5 * z**2 - 0.5 * jnp.log(2 * jnp.pi), axis=-1) + log_det

        ## Normalized space: log p, data -> latent, latent -> data
        self.log_prob_fn = hk.without_apply_rng(hk.transform(log_prob_norm))
        self.inverse_fn = hk.without_apply_rng(
            hk.transform(lambda x, c: self._make_maf_stack(x, c, inverse=True)))
        self.forward_fn = hk.without_apply_rng(
            hk.transform(lambda z, c: self._make_maf_stack(z, c, inverse=False)))
        self.optimizer = optax.adam(self.learning_rate)

        ## Physical space; the weights are always an argument
        def log_prob(params, x, c):
            x_n, c_n = self.normalize_data(x), self.normalize_context(c)
            return self.log_prob_fn.apply(params, x_n, c_n) + self._logdet_normalize_data(x)

        def forward(params, z, c):
            return self.denormalize_data(self.forward_fn.apply(params, z, self.normalize_context(c)))

        def backward(params, x, c):
            z, ld = self.inverse_fn.apply(params, self.normalize_data(x), self.normalize_context(c))
            return z, ld + self._logdet_normalize_data(x)

        def loss_fn(params, x_batch, c_batch):
            return -jnp.mean(self.log_prob_fn.apply(params, x_batch, c_batch))

        @jax.jit
        def update(params, opt_state, x_batch, c_batch, lr_scale=None):
            loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, c_batch)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            updates = _scale_updates(updates, lr_scale)
            return optax.apply_updates(params, updates), opt_state, loss

        self._update = update
        self._log_prob_jit = jax.jit(log_prob)
        self._forward_jit = jax.jit(forward)
        self._backward_jit = jax.jit(backward)

    def _init_params(self):
        self.key, init_key = jr.split(self.key)
        i0 = self.train_rows[:2]
        x0 = self.normalize_data(jnp.asarray(np.asarray(self._data[i0], dtype=np.float64)))
        c0 = jnp.asarray(_from_coefficients_np(np.asarray(self._context[i0], dtype=np.float64), self.c_norm))
        self.params = self.log_prob_fn.init(init_key, x0, c0)
        self.opt_state = self.optimizer.init(self.params)
        self.best_params = self.params

    def _arch_cfg(self):
        return dict(
            D=int(self.D), C=int(self.C), flow_num_layers=int(self.flow_num_layers),
            hidden_size=int(self.hidden_size), num_hidden=int(self.num_hidden),
            num_bins=int(self.num_bins), B=float(self.B_float), whiten=self.whiten,
        )

    ##################################################################
    #                    training

    def fit(
        self,
        mode="pre-loaded",
        num_epochs=100,
        batch_size=256,
        checkpoint=None,
        save_path=None,
        lr_schedule=None,
        lr_min=None,
        warmup_steps=0,
        patience=10,
        normality_alpha=0.01,
        normality_patience=3,
        stop_rule="either",
        stopper=None,
        n_val=50_000,
        n_normality=5000,
        val_every=1,
        verbose=True,
    ):
        """
        Train by maximum likelihood.

        :param mode:
            'pre-loaded' : epochs over the training rows of data / context
            'on-the-fly' : every "epoch" is one batch from a random file of tset_paths
        :param num_epochs: number of epochs
        :param batch_size: rows per batch
        :param checkpoint: path; the flow is saved there after every epoch,
            and training resumes from it if it exists
        :param save_path: path; the flow is saved there whenever the validation log p improves
        :param lr_schedule: None, "cosine", "linear", "exponential" or a callable (see `make_lr_scale`)
        :param lr_min: final learning rate of the decay; default learning_rate / 100
        :param warmup_steps: steps of linear warm-up before the decay
        :param patience, normality_alpha, normality_patience, stop_rule:
            early stopping (see `EarlyStopping`); needs val_frac > 0
        :param stopper: an `EarlyStopping` object; if given, it replaces the four above
        :param n_val: held-out rows used for the validation log p
        :param n_normality: held-out rows (out of n_val) used for the normality test
        :param val_every: validate every this many epochs (useful in 'on-the-fly' mode)
        :param verbose: show a progress bar

        :return: self.history (per-epoch lists: train, val, normality_p, lr)
        """
        self._require_data()
        if mode not in ("pre-loaded", "on-the-fly"):
            raise ValueError("mode must be 'pre-loaded' or 'on-the-fly'")
        steps_per_epoch = -(-len(self.train_rows) // batch_size) if mode == "pre-loaded" else 1
        lr_scale = make_lr_scale(lr_schedule, self.learning_rate, num_epochs * steps_per_epoch,
                                 lr_min, warmup_steps)

        def preloaded_epoch(step):
            self.key, subkey = jr.split(self.key)
            losses = []
            for x_batch, c_batch in _iter_batches(
                self._data, self._context, self.x_norm, self.c_norm, batch_size, subkey,
                x_white=self._x_white_np, rows=self.train_rows,
            ):
                self.params, self.opt_state, loss = self._update(
                    self.params, self.opt_state, x_batch, c_batch, lr_scale(step))
                losses.append(loss)
                step += 1
            return float(jnp.mean(jnp.stack(losses))) if losses else np.nan

        def on_the_fly_step(step):
            training_set = np.load(self.tset_paths[random.randint(0, len(self.tset_paths) - 1)])
            if self.drop_nonfinite:
                training_set = training_set[np.all(np.isfinite(training_set), axis=1)]
            idxs = random.sample(range(training_set.shape[0]), k=min(batch_size, training_set.shape[0]))
            training_set = training_set[np.array(idxs)]
            x_batch = self.normalize_data(training_set[:, : self.last_feature_idx])
            c_batch = self.normalize_context(training_set[:, self.last_feature_idx :])
            self.params, self.opt_state, loss = self._update(
                self.params, self.opt_state, x_batch, c_batch, lr_scale(step))
            return float(loss)

        stopper = stopper or EarlyStopping(patience, normality_alpha, normality_patience, stop_rule)
        return self._train(
            preloaded_epoch if mode == "pre-loaded" else on_the_fly_step,
            num_epochs, steps_per_epoch, lr_scale, checkpoint, save_path, stopper,
            n_val, n_normality, desc=f"Training ConditionalMADEFlow ({mode})", colour="green",
            verbose=verbose, val_every=val_every,
        )

    def _n_held_out(self):
        return len(self.val_rows)

    def _held_out(self, n):
        rng = np.random.default_rng(0)  # same held-out samples in every fit call
        rows = np.sort(rng.choice(self.val_rows, size=min(n, len(self.val_rows)), replace=False))
        order = rng.permutation(len(rows))
        x = np.asarray(self._data[rows], dtype=np.float64)[order]
        c = np.asarray(self._context[rows], dtype=np.float64)[order]
        return x, c

    def _validate(self, x, c, n_normality):
        logp = float(jnp.mean(self.log_prob(x, c)))
        x, c = x[:n_normality], c[:n_normality]
        return logp, self.backward_pass(x, c), self.normalize_context(c)

    ##################################################################
    #                    public API (physical space)

    def log_prob(self, x, c):
        """log p(x | c); x (n, D), c (n, C)."""
        return self._log_prob_jit(self.params, jnp.atleast_2d(x), jnp.atleast_2d(c))

    def sample(self, c, num_samples, chunk_size=None):
        """
        Draw samples from p(x | c). Slow: D sequential steps per layer.

        :param c: (C,), (1, C) or (num_samples, C) physical context
        :param num_samples: number of samples
        :param chunk_size: samples per call, to bound memory
        """
        c = jnp.atleast_2d(jnp.asarray(c))
        if c.shape[0] == 1:
            c = jnp.tile(c, (num_samples, 1))
        chunk_size = num_samples if chunk_size is None else chunk_size
        chunks = []
        for start in range(0, num_samples, chunk_size):
            end = min(start + chunk_size, num_samples)
            self.key, subkey = jr.split(self.key)
            z = jr.normal(subkey, (end - start, self.D))
            chunks.append(self._forward_jit(self.params, z, c[start:end]))
        return jnp.concatenate(chunks, axis=0)

    def forward_pass(self, z, c):
        """Latent z (n, D) -> physical x given physical context c (n, C). Slow."""
        return self._forward_jit(self.params, jnp.atleast_2d(z), jnp.atleast_2d(c))

    def backward_pass(self, x, c):
        """Physical x (n, D) -> latent z given physical context c (n, C). Fast."""
        return self._backward_jit(self.params, jnp.atleast_2d(x), jnp.atleast_2d(c))[0]

    def backward_pass_with_logdet(self, x, c):
        """Physical x -> (z, log|det dz/dx|), so log p(x | c) = log N(z; 0, I) + logdet."""
        return self._backward_jit(self.params, jnp.atleast_2d(x), jnp.atleast_2d(c))

    ##################################################################
    #                    weights only (older .npz format)

    def save_params(self):
        """Save the weights only, to flow_save_path. Prefer `save`."""
        leaves, _ = jax.tree_util.tree_flatten(self.params)
        np.savez(self.flow_save_path, **{f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)})

    def load_params(self):
        """Load weights saved by `save_params` from flow_load_path."""
        data = np.load(self.flow_load_path, allow_pickle=True)
        _, treedef = jax.tree_util.tree_flatten(self.params)
        leaves = [jnp.asarray(data[f"leaf_{i}"]) for i in range(treedef.num_leaves)]
        self.params = self.best_params = jax.tree_util.tree_unflatten(treedef, leaves)


##############################################################################################
#                                         SASFlow
##############################################################################################


def _joint_ctx_forward(c, t, kind):
    """Context columns of a joint row -> normalized, with log|det| (summed)."""
    if kind == "probit":
        u = ndtri((c - t["lo"]) / (t["hi"] - t["lo"]))
        ld = -jnp.log(t["hi"] - t["lo"]) + 0.5 * u**2 + 0.5 * LOG2PI
        return u, jnp.sum(ld, axis=-1)
    if kind == "standardize":
        u = (c - t["mean"]) / t["std"]
        return u, jnp.sum(-jnp.log(t["std"]) * jnp.ones_like(u), axis=-1)
    return sas_forward(c, t)


def _joint_ctx_inverse(u, t, kind):
    if kind == "probit":
        return t["lo"] + (t["hi"] - t["lo"]) * jax.scipy.special.ndtr(u)
    if kind == "standardize":
        return t["mean"] + t["std"] * u
    return sas_inverse(u, t)[0]


class SASFlow(_FlowBase):
    """
    Unconditional RQS coupling flow with a sinh-arcsinh (SAS) preprocessing.

        physical row x = [theta | y]
            theta : context transform ("probit", "standardize" or "sas")
            y     : per-feature SAS map, so every marginal is ~ N(0, 1)
        --> normalized row --(whitening)--> coupling layers --> z ~ N(0, I)

    It can also learn the JOINT density p(theta, y) of a simulation library:
    pass data of shape (n_draws, n_real, n_features) and the per-draw
    parameters as `context` (n_draws, P).  Every (draw, realization) pair is
    one row [theta_i, y_ij] of dimension P + n_features (never built as a
    whole array).  theta is modelled, not conditioned on, so
    forward_pass(z) returns [theta, y].

    The splines act on [-B, B] and are the identity outside, so the tails
    are those of the Gaussian base pulled back through the SAS map.
    "probit" maps a uniform (Latin-hypercube) theta design exactly to N(0, 1)
    and keeps every sampled theta inside its bounds.

    :param data:
        (N, n_features) or (n_draws, n_real, n_features); may be a memmap.
        A 1-D array is treated as (N, 1).
    :param context:
        (n_draws, P) per-draw parameters (library layout), or (N, P) per-row
        columns (flat layout), or None.  Modelled jointly with the data.
    :param context_kind:
        "probit", "standardize" or "sas": transform of the context columns
    :param context_bounds:
        (lo, hi), each (P,), for "probit".  Default: the context range widened
        by 0.1%.  Use your prior ranges so new theta values stay inside.
    :param context_jitter:
        std of Gaussian noise added to the normalized context of every training
        batch.  It stops the flow from putting spikes of density at the finite
        set of library draws.  Try 0.1 to 0.3 and compare the validation log p.
    :param normalizer:
        pre-fitted SAS dict of the data features (`fit_sas_normalizer`); fitted if None
    :param normalizer_max_samples:
        rows used for the SAS and whitening fits
    :param val_frac:
        fraction of draws (library layout) or rows (flat layout) held out for
        early stopping (0 turns it off).  Held-out draws are never trained on,
        so this measures how well the flow interpolates in theta.
    :param flow_num_layers:
        number of coupling layers
    :param hidden_size:
        width of the hidden layers of the conditioner networks
    :param mlp_num_layers:
        number of hidden layers of the conditioner networks
    :param num_bins:
        number of spline bins
    :param B:
        the splines act on [-B, B]
    :param learning_rate:
        learning rate of Adam
    :param grad_clip:
        global-norm gradient clipping; None turns it off
    :param seed:
        random seed
    :param whiten:
        None, "zca", "cholesky" or "pca": decorrelate the whole normalized row
        with a fixed linear map (see `fit_whitener`)
    :param whiten_shrinkage:
        covariance shrinkage of the whitener
    :param drop_nonfinite:
        never train on rows whose features, or whose draw's context, contain
        NaN or inf (nothing is copied)

    Author:
    Nima Laal (11/26/2026)
    """

    _DATA = ("_data", "context_raw", "u_all", "train_rows", "val_rows")
    _BUILT = (
        "log_prob_fn", "forward_fn", "inverse_fn", "optimizer",
        "_update", "_log_prob_jit", "_forward_jit", "_backward_jit",
    )

    def __init__(
        self,
        data,
        context=None,
        context_kind="probit",
        context_bounds=None,
        context_jitter=0.0,
        normalizer=None,
        normalizer_max_samples=200_000,
        val_frac=0.1,
        flow_num_layers=4,
        hidden_size=128,
        mlp_num_layers=2,
        num_bins=8,
        B=5.0,
        learning_rate=1e-4,
        grad_clip=1.0,
        seed=0,
        whiten=None,
        whiten_shrinkage=0.0,
        drop_nonfinite=True,
    ):
        ## Layout of the data
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
        self.N = self.n_groups * self.R

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
        self.context_jitter = float(context_jitter)
        self.key = jr.PRNGKey(seed)
        self.rng = np.random.default_rng(seed)

        ## Rows with NaN / inf in the features or in the draw's context
        self.drop_nonfinite = bool(drop_nonfinite)
        self.n_dropped = 0
        valid = None
        if self.drop_nonfinite:
            mask = finite_row_mask(data, context).reshape(-1)
            self.n_dropped = _report_nonfinite("SASFlow", mask, "(draw, realization) rows")
            valid = mask if self.n_dropped else None

        ## Train / validation split, by draw in the library layout
        self.train_groups, self.val_groups = _split_rows(np.arange(self.n_groups), val_frac, self.rng)
        self._set_rows(valid)

        ## Context transform, fitted on the draws that keep at least one row
        self.context_kind = context_kind if self.P else "none"
        if self.P:
            ctx_fit = context if valid is None else context[np.unique(np.flatnonzero(valid) // self.R)]
            self._set_ctx(self._fit_context(ctx_fit, context_kind, context_bounds,
                                            normalizer_max_samples, seed))
        else:
            self._set_ctx({})

        ## SAS normalizer of the data features, fitted on training rows
        if normalizer is None:
            n = min(len(self.train_rows), normalizer_max_samples)
            idx = np.sort(self.rng.choice(self.train_rows, size=n, replace=False))
            normalizer = fit_sas_normalizer(self._read_y(idx))
        self.set_normalizer(normalizer)

        ## Optional whitening of the normalized rows
        self.whiten = whiten
        self.set_whitener(None)
        if whiten is not None:
            n = min(len(self.train_rows), normalizer_max_samples)
            idx = np.sort(self.rng.choice(self.train_rows, size=n, replace=False))
            self.set_whitener(fit_whitener(np.asarray(self._batch(idx)), whiten,
                                           whiten_shrinkage, seed=seed))

        self._new_history()
        self._build()
        self._init_params()

    ##################################################################
    #                    normalizers

    def _set_rows(self, valid):
        """Training / validation pair indices (draw * R + realization)."""
        rows_of = lambda g: (g[:, None] * self.R + np.arange(self.R)).ravel()
        self.train_rows, self.val_rows = rows_of(self.train_groups), rows_of(self.val_groups)
        if valid is not None:
            self.train_rows = self.train_rows[valid[self.train_rows]]
            self.val_rows = self.val_rows[valid[self.val_rows]]
        if not len(self.train_rows):
            raise ValueError("SASFlow: no training rows left after removing NaN / inf.")

    def _fit_context(self, ctx_fit, kind, bounds, max_samples, seed):
        if kind == "probit":
            if bounds is None:
                lo, hi = ctx_fit.min(0), ctx_fit.max(0)
                pad = 1e-3 * (hi - lo)
                bounds = (lo - pad, hi + pad)
            lo, hi = (np.asarray(b, np.float64).reshape(self.P) for b in bounds)
            if np.any(ctx_fit <= lo) or np.any(ctx_fit >= hi):
                raise ValueError("context values must lie strictly inside context_bounds")
            return dict(lo=lo, hi=hi)
        if kind == "standardize":
            sd = ctx_fit.std(0)
            return dict(mean=ctx_fit.mean(0), std=np.where(sd > 0, sd, 1.0))
        if kind == "sas":
            return fit_sas_normalizer(ctx_fit, max_samples=max_samples, seed=seed)
        raise ValueError("context_kind must be 'probit', 'standardize' or 'sas'")

    def _set_ctx(self, ctx):
        """Set the context transform and the normalized context of every draw."""
        self.ctx_np = {k: np.asarray(v, np.float64) for k, v in ctx.items()}
        self.ctx = {k: jnp.asarray(v) for k, v in self.ctx_np.items()}
        self.u_all = None
        if self.P and self.context_raw is not None:
            c = np.nan_to_num(self.context_raw, nan=0.0, posinf=0.0, neginf=0.0)
            if self.context_kind == "probit":  # keep the unused rows inside the bounds
                bad = ~np.isfinite(self.context_raw)
                c = np.where(bad, 0.5 * (self.ctx_np["lo"] + self.ctx_np["hi"]), c)
            self.u_all = np.asarray(_joint_ctx_forward(jnp.asarray(c), self.ctx, self.context_kind)[0])

    def set_normalizer(self, normalizer):
        """Set the SAS normalizer of the data features."""
        keys = ("loc", "s", "delta", "eps", "mu", "sd")
        self.normalizer_np = {k: np.asarray(normalizer[k], dtype=np.float64).reshape(self.K) for k in keys}
        self.normalizer = {k: jnp.asarray(v) for k, v in self.normalizer_np.items()}

    def set_whitener(self, white):
        """Set (or clear, with None) the whitener of the normalized rows."""
        self.white_np = None if white is None else {k: np.asarray(white[k], np.float64) for k in _WHITE_KEYS}
        self.white = None if white is None else {k: jnp.asarray(v) for k, v in self.white_np.items()}

    def _norms(self):
        return dict(y=self.normalizer, c=self.ctx, w=self.white)

    def _row_forward(self, norms, x):
        """Physical rows [theta | y] -> normalized rows, log|det| (summed)."""
        if not self.P:
            xn, ld = sas_forward(x, norms["y"])
        else:
            u, ld_c = _joint_ctx_forward(x[:, : self.P], norms["c"], self.context_kind)
            yh, ld_y = sas_forward(x[:, self.P :], norms["y"])
            xn, ld = jnp.concatenate([u, yh], axis=-1), ld_c + ld_y
        if norms.get("w") is not None:
            xn, ld_w = whiten_forward(xn, norms["w"])
            ld = ld + ld_w
        return xn, ld

    def _row_inverse(self, norms, xn):
        """Normalized rows -> physical rows, log|det dx/dxn| (summed)."""
        if norms.get("w") is not None:
            xn = whiten_inverse(xn, norms["w"])[0]
        if not self.P:
            x = sas_inverse(xn, norms["y"])[0]
        else:
            x = jnp.concatenate([
                _joint_ctx_inverse(xn[:, : self.P], norms["c"], self.context_kind),
                sas_inverse(xn[:, self.P :], norms["y"])[0],
            ], axis=-1)
        return x, -self._row_forward(norms, x)[1]

    def normalize(self, x):
        """Physical rows -> normalized rows (no Jacobian)."""
        return self._row_forward(self._norms(), jnp.atleast_2d(jnp.asarray(x, jnp.float64)))[0]

    def denormalize(self, y):
        """Normalized rows -> physical rows (no Jacobian)."""
        return self._row_inverse(self._norms(), jnp.atleast_2d(jnp.asarray(y, jnp.float64)))[0]

    def save_normalizer(self, path):
        """Save the data-feature SAS normalizer only."""
        np.savez(path, **self.normalizer_np)

    @staticmethod
    def load_normalizer(path):
        with np.load(path) as f:
            return {k: f[k] for k in f.files}

    ##################################################################
    #                    data access

    def _read_y(self, idx):
        """Data features of the (sorted) pair indices idx, physical units."""
        if self.layout == "library":
            rows = self._data[idx // self.R, idx % self.R]
        else:
            rows = self._data[idx]
        return np.asarray(rows, dtype=np.float64).reshape(len(idx), self.K)

    def rows(self, idx):
        """Physical joint rows [theta | y] of the pair indices idx."""
        idx = np.asarray(idx)
        y = self._read_y(idx)
        if not self.P:
            return y
        return np.concatenate([self.context_raw[idx // self.R], y], axis=1)

    def _batch(self, idx, jitter=0.0):
        """Normalized rows of the pair indices idx."""
        yn = _sas_forward_np(self._read_y(idx), self.normalizer_np)
        if self.P:
            u = self.u_all[idx // self.R]
            if jitter > 0:
                u = u + jitter * self.rng.standard_normal(u.shape)
            yn = np.concatenate([u, yn], axis=1)
        if self.white_np is not None:
            yn = whiten_forward(yn, self.white_np)[0]
        return jnp.asarray(yn)

    def _refresh_data(self):
        self._set_ctx(self.ctx_np)

    ##################################################################
    #                    model

    def _build(self):
        """Haiku functions, the optimizer and the jitted functions."""
        args = (self.event_shape, self.flow_num_layers, self.hidden_sizes,
                self.num_bins, -self.B, self.B)
        model = lambda: make_coupling_flow(*args)

        ## Normalized space: log p, latent -> data, data -> latent
        self.log_prob_fn = hk.without_apply_rng(hk.transform(lambda y: model().log_prob(y)))
        self.forward_fn = hk.without_apply_rng(
            hk.transform(lambda z: model().bijector.forward_and_log_det(z)))
        self.inverse_fn = hk.without_apply_rng(
            hk.transform(lambda y: model().bijector.inverse_and_log_det(y)))
        self.optimizer = _make_optimizer(self.learning_rate, self.grad_clip)

        ## Physical space; weights and normalizers are always arguments
        def log_prob(params, norms, x):
            y, logdet = self._row_forward(norms, x)
            return self.log_prob_fn.apply(params, y) + logdet

        def forward(params, norms, z):
            y, ld_flow = self.forward_fn.apply(params, z)
            x, ld_norm = self._row_inverse(norms, y)
            return x, ld_flow + ld_norm

        def backward(params, norms, x):
            y, ld_norm = self._row_forward(norms, x)
            z, ld_flow = self.inverse_fn.apply(params, y)
            return z, ld_norm + ld_flow

        def loss_fn(params, y):
            return -jnp.mean(self.log_prob_fn.apply(params, y))

        @jax.jit
        def update(params, opt_state, y, lr_scale=None):
            loss, grads = jax.value_and_grad(loss_fn)(params, y)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            updates = _scale_updates(updates, lr_scale)
            return optax.apply_updates(params, updates), opt_state, loss

        self._update = update
        self._log_prob_jit = jax.jit(log_prob)
        self._forward_jit = jax.jit(forward)
        self._backward_jit = jax.jit(backward)

    def _init_params(self):
        self.key, init_key = jr.split(self.key)
        self.params = self.log_prob_fn.init(init_key, self._batch(self.train_rows[:2]))
        self.opt_state = self.optimizer.init(self.params)
        self.best_params = self.params

    def _arch_cfg(self):
        return dict(
            D=int(self.D), P=int(self.P), context_kind=self.context_kind,
            flow_num_layers=int(self.flow_num_layers),
            hidden_sizes=[int(h) for h in self.hidden_sizes], num_bins=int(self.num_bins),
            B=float(self.B), whiten=self.whiten,
        )

    ##################################################################
    #                    training

    def _epoch_batches(self, batch_size, key):
        """Shuffled, lazily read, normalized batches over all training pairs."""
        rows = self.train_rows
        perm = rows[np.asarray(jr.permutation(key, len(rows)))]
        drop_last = len(rows) >= batch_size  # a ragged last batch would recompile
        for start in range(0, len(rows), batch_size):
            idx = np.sort(perm[start : start + batch_size])  # sorted reads are faster on memmaps
            if drop_last and len(idx) < batch_size:
                break
            yield self._batch(idx, jitter=self.context_jitter)

    def fit(
        self,
        num_epochs=100,
        batch_size=256,
        checkpoint=None,
        save_path=None,
        lr_schedule=None,
        lr_min=None,
        warmup_steps=0,
        patience=10,
        normality_alpha=0.01,
        normality_patience=3,
        stop_rule="either",
        stopper=None,
        n_val=50_000,
        n_normality=5000,
        verbose=True,
    ):
        """
        Train by maximum likelihood; every epoch is one pass over the training pairs.

        :param num_epochs: number of epochs
        :param batch_size: rows per batch
        :param checkpoint: path; the flow is saved there after every epoch,
            and training resumes from it if it exists
        :param save_path: path; the flow is saved there whenever the validation log p improves
        :param lr_schedule: None, "cosine", "linear", "exponential" or a callable (see `make_lr_scale`)
        :param lr_min: final learning rate of the decay; default learning_rate / 100
        :param warmup_steps: steps of linear warm-up before the decay
        :param patience, normality_alpha, normality_patience, stop_rule:
            early stopping (see `EarlyStopping`); needs val_frac > 0
        :param stopper: an `EarlyStopping` object; if given, it replaces the four above
        :param n_val: held-out rows used for the validation log p
        :param n_normality: held-out rows (out of n_val) used for the normality test
        :param verbose: show a progress bar

        :return: self.history (per-epoch lists: train, val, normality_p, lr)
        """
        self._require_data()
        n_rows = len(self.train_rows)
        steps_per_epoch = n_rows // batch_size if n_rows >= batch_size else 1
        lr_scale = make_lr_scale(lr_schedule, self.learning_rate, num_epochs * steps_per_epoch,
                                 lr_min, warmup_steps)

        def run_epoch(step):
            self.key, subkey = jr.split(self.key)
            losses = []
            for y in self._epoch_batches(batch_size, subkey):
                self.params, self.opt_state, loss = self._update(
                    self.params, self.opt_state, y, lr_scale(step))
                losses.append(loss)
                step += 1
            return float(jnp.mean(jnp.stack(losses)))

        stopper = stopper or EarlyStopping(patience, normality_alpha, normality_patience, stop_rule)
        return self._train(run_epoch, num_epochs, steps_per_epoch, lr_scale, checkpoint, save_path,
                           stopper, n_val, n_normality, desc="Training SASFlow", verbose=verbose)

    def _n_held_out(self):
        return len(self.val_rows)

    def _held_out(self, n):
        rng = np.random.default_rng(0)  # same held-out samples in every fit call
        idx = np.sort(rng.choice(self.val_rows, size=min(n, len(self.val_rows)), replace=False))
        x = self.rows(idx)
        return x[rng.permutation(len(x))], None

    def _validate(self, x, c, n_normality):
        logp = float(jnp.mean(self.log_prob(x)))
        return logp, self.backward_pass(x[:n_normality]), None

    ##################################################################
    #                    public API (physical rows [theta | y])

    def log_prob(self, x):
        """log p(x); x (n, D) or (D,)."""
        return self._log_prob_jit(self.params, self._norms(), jnp.atleast_2d(x))

    def sample(self, num_samples, key=None):
        """Draw (num_samples, D) rows."""
        if key is None:
            self.key, key = jr.split(self.key)
        return self.forward_pass(jr.normal(key, (num_samples, self.D)))

    def forward_pass(self, z):
        """Latent z -> physical rows. Differentiable (numpyro-ready)."""
        return self._forward_jit(self.params, self._norms(), jnp.atleast_2d(z))[0]

    def forward_pass_with_logdet(self, z):
        """Latent z -> (physical rows, log|det dx/dz|)."""
        return self._forward_jit(self.params, self._norms(), jnp.atleast_2d(z))

    def backward_pass(self, x):
        """Physical rows -> latent z."""
        return self._backward_jit(self.params, self._norms(), jnp.atleast_2d(x))[0]

    def backward_pass_with_logdet(self, x):
        """Physical rows -> (z, log|det dz/dx|), so log p(x) = log N(z; 0, I) + logdet."""
        return self._backward_jit(self.params, self._norms(), jnp.atleast_2d(x))

    ##################################################################
    #                    weights + normalizers (older .npz format)

    def save_params(self, path):
        """Weights, normalizers and context transform in one .npz file. Prefer `save`."""
        leaves, _ = jax.tree_util.tree_flatten(self.params)
        d = {f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)}
        d.update({f"norm_{k}": v for k, v in self.normalizer_np.items()})
        d.update({f"ctx_{k}": v for k, v in self.ctx_np.items()})
        if self.white_np is not None:
            d.update({f"white_{k}": v for k, v in self.white_np.items()})
        d["context_kind"] = np.array(self.context_kind)
        np.savez(path, **d)

    def load_params(self, path):
        """Load a file written by `save_params`."""
        with np.load(path) as f:
            _, treedef = jax.tree_util.tree_flatten(self.params)
            leaves = [jnp.asarray(f[f"leaf_{i}"]) for i in range(treedef.num_leaves)]
            self.params = self.best_params = jax.tree_util.tree_unflatten(treedef, leaves)
            if "norm_loc" in f.files:
                self.set_normalizer({k[5:]: f[k] for k in f.files if k.startswith("norm_")})
            self.set_whitener({k[6:]: f[k] for k in f.files if k.startswith("white_")}
                              if "white_W" in f.files else None)
            if "context_kind" in f.files:
                kind = str(f["context_kind"])
                if kind != self.context_kind:
                    raise ValueError(f"saved context_kind {kind!r} differs from {self.context_kind!r}")
                self._set_ctx({k[4:]: f[k] for k in f.files if k.startswith("ctx_")})


##############################################################################################
#             Conditional SAS flows: shared data handling, training and API
##############################################################################################

_E_LOG_CHI2_1 = -1.2703628454614782  # E[log z^2] for z ~ N(0, 1)


def _scale_apply(params, c_hat):
    """Small tanh network: normalized context (n, C) -> (n, out)."""
    h = c_hat
    for i, p in enumerate(params):
        h = h @ p["W"] + p["b"]
        if i < len(params) - 1:
            h = jnp.tanh(h)
    return h


def _stdz_apply(params, c_hat):
    """Location-scale network: normalized context (n, C) -> (m, log s), each (n, D)."""
    t = params["t_mean"] + params["t_std"] * _scale_apply(params["net"], c_hat)
    D = t.shape[-1] // 2
    return t[..., :D], t[..., D:]


def _init_tanh_net(key, sizes, last_bias):
    """Weights ~ N(0, 1/fan_in); the last layer starts at zero with bias `last_bias`."""
    params = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jr.split(key)
        last = i == len(sizes) - 2
        params.append(dict(
            W=jnp.zeros((a, b)) if last else jr.normal(k, (a, b)) / np.sqrt(a),
            b=jnp.asarray(last_bias) if last else jnp.zeros(b),
        ))
    return params


class _ConditionalSASFlow(_FlowBase):
    """
    What ConditionalFlow and SASNSF share: SAS preprocessing of the data, a
    context transform, memmap-friendly batching, training with early
    stopping, the public API and saving / loading.

        physical y --(context scale / location)--> --SAS--> --(whitening)--> flow(c_hat) --> z ~ N(0, I)
        physical c --context transform--> c_hat

    A subclass defines the flow through four hooks:
        _setup_architecture()          masks, layer lists, haiku functions
        _init_model_params(key)        parameter pytree
        _flow_forward(params, y, c)    normalized data -> (latent, log|det|), one pass
        _flow_inverse(params, z, c)    latent -> normalized data

    Data layouts
        library : data (n_theta, R, D), context (n_theta, C).  Training draws
                  (theta point, realization) pairs uniformly; validation holds
                  out whole theta points.
        flat    : data (N, D), context (N, C) or None.  Validation holds out rows.
    Both accept numpy memmaps.  context=None gives an unconditional flow.

    Context-dependent scale (scale_by_context=True)
        Use it when the context mostly changes the SIZE of the data.  A small
        network learns log sigma_k(c) for every feature by regressing
        log y_k^2 on the context (E[log y^2] = 2 log sigma + E[log chi^2_1]),
        and the flow models y / sigma(c):
            log p(y | c) = log p_flow(SAS(y / sigma(c)) | c) + log|dSAS| - sum_k log sigma_k(c)
        Fitted once, before the SAS map, then frozen; held-out R^2 in `scale_r2`.

    Per-draw standardization (standardize_by_context=True, library layout)
        Use it when the context moves the LOCATION of the data (e.g. log10 hc).
        For every training draw, the median m and the scale s = IQR / 1.349 of
        every feature over its realizations are computed, and a small network
        learns c -> (m(c), log s(c)).  The flow models r = (y - m(c)) / s(c),
        whose distribution is nearly the same for every draw:
            log p(y | c) = log p_flow(SAS(r) | c) + log|dSAS| - sum_k log s_k(c)
        Fitted once, before the SAS map, then frozen; held-out R^2 in
        `standardize_r2`.
    """

    _DATA = ("data", "context_raw", "c_all", "valid", "_pair_cache",
             "train_idx", "val_idx", "loc_all", "logsig_all")
    _BUILT = ("optimizer", "_update", "_phys_lp", "_phys_sample", "_latent")

    def __init__(
        self,
        data,
        context=None,
        context_kind="standardize",
        context_bounds=None,
        val_frac=0.1,
        normalizer=None,
        normalizer_samples=50_000,
        learning_rate=1e-3,
        grad_clip=1.0,
        seed=0,
        dtype=jnp.float64,
        scale_by_context=False,
        scale_hidden=(128, 128),
        scale_steps=3000,
        whiten=None,
        whiten_shrinkage=0.0,
        drop_nonfinite=True,
        standardize_by_context=False,
        standardize_hidden=(128, 128),
        standardize_steps=3000,
        standardize_draws=20_000,
    ):
        ## Layout of the data
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
            self.data, self.layout = data.reshape(-1, 1), "flat"
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
        self.dtype = np.dtype(dtype)
        self.rng = np.random.default_rng(seed)
        self.key = jr.PRNGKey(seed)
        self.seed = seed
        self.learning_rate, self.grad_clip = learning_rate, grad_clip

        ## Rows with NaN / inf in the data or in the group's context
        self.drop_nonfinite = bool(drop_nonfinite)
        self.n_dropped = 0
        self.valid = None  # (n_groups, R) bool, or None when every row is valid
        self._pair_cache = {}
        if self.drop_nonfinite:
            mask = finite_row_mask(self.data, self.context_raw).reshape(self.n_groups, self.R)
            self.n_dropped = _report_nonfinite(type(self).__name__, mask)
            if self.n_dropped:
                self.valid = mask

        ## Train / validation split, by theta point in the library layout
        self.train_idx, self.val_idx = _split_rows(np.arange(self.n_groups), val_frac, self.rng)
        if self.valid is not None:  # groups with no valid row take no part at all
            keep = self.valid.any(axis=1)
            self.val_idx, self.train_idx = self.val_idx[keep[self.val_idx]], self.train_idx[keep[self.train_idx]]
        if not len(self.train_idx):
            raise ValueError("no training rows left after removing NaN / inf.")

        ## Context transform, fitted on the training groups
        if self.C:
            self.ctx_np = fit_context_transform(self.context_raw[self.train_idx], context_kind, context_bounds)
        else:
            self.ctx_np = dict(kind="none")
        self._set_ctx()

        ## Optional context-dependent scale or location-scale (before the SAS map)
        self.scale_by_context = bool(scale_by_context)
        self.scale_hidden = tuple(int(h) for h in scale_hidden)
        self.scale_params = None
        self.standardize_by_context = bool(standardize_by_context)
        self.standardize_hidden = tuple(int(h) for h in standardize_hidden)
        self.stdz_params = None
        if self.scale_by_context and self.standardize_by_context:
            raise ValueError("use either scale_by_context or standardize_by_context, not both")
        if self.scale_by_context:
            if not self.C:
                raise ValueError("scale_by_context needs a context")
            self.scale_params = self._fit_scale(scale_steps)
        if self.standardize_by_context:
            if not self.C or self.layout != "library":
                raise ValueError("standardize_by_context needs the library layout "
                                 "(n_theta, R, D) with a context")
            self.stdz_params = self._fit_standardize(standardize_steps, standardize_draws)
        self._set_logsig_all()

        ## SAS normalizer of the (scaled) data, fitted on the training groups
        self.white_np = None
        if normalizer is None:
            g, r = self._random_pairs(self.train_idx, normalizer_samples)
            normalizer = fit_sas_normalizer(self._read_scaled(g, r))
        self.set_normalizer(normalizer)

        ## Optional whitening of the SAS output, pooled over contexts
        self.whiten = whiten
        if whiten is not None and self.white_np is None:
            g, r = self._random_pairs(self.train_idx, normalizer_samples)
            yh = _sas_forward_np(self._read_scaled(g, r), self.norm_np)
            self.set_normalizer(dict(self.norm_np, **fit_whitener(yh, whiten, whiten_shrinkage, seed=seed)))

        self._new_history()
        self._build()
        self._init_params()

    ##################################################################
    #                    hooks of the subclasses

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

    ##################################################################
    #                    normalizers

    def _set_ctx(self):
        """Context transform as jax arrays, and the normalized context of every group."""
        self.ctx = {k: (v if k == "kind" else jnp.asarray(v)) for k, v in self.ctx_np.items()}
        if not self.C:
            self.c_all = None
            return
        c = self.context_raw
        bad = ~np.isfinite(c)
        if bad.any():  # rows of these groups are never drawn; keep their transform finite
            c = np.where(bad, np.nanmedian(np.where(bad, np.nan, c), axis=0), c)
        self.c_all = np.asarray(context_forward(jnp.asarray(c), self.ctx))

    def set_normalizer(self, n):
        """
        Set the SAS normalizer of the (scaled) data.  If `n` also holds a
        whitener (keys W, Winv, w_mean, w_logdet), it is applied after the SAS map.
        """
        keys = ("loc", "s", "delta", "eps", "mu", "sd")
        _, white = _split_white(n)
        self.norm_np = {k: np.asarray(n[k], np.float64).reshape(self.D) for k in keys}
        self.norm = {k: jnp.asarray(v) for k, v in self.norm_np.items()}
        self.white_np = white
        if white is not None:
            self.norm_np.update(white)
        self.norm["white"] = None if white is None else {k: jnp.asarray(v) for k, v in white.items()}
        self.norm["scale"] = self.scale_params  # None unless scale_by_context
        self.norm["stdz"] = self.stdz_params  # None unless standardize_by_context

    def normalize(self, y):
        """
        Physical data -> SAS-normalized (and whitened) data, no Jacobian.
        The context scaling is not applied: pass y / context_scale(c), or
        (y - m) / s with (m, s) = context_location_scale(c), if it is on.
        """
        yh = sas_forward(jnp.atleast_2d(jnp.asarray(y, jnp.float64)), self.norm)[0]
        if self.norm["white"] is not None:
            yh = whiten_forward(yh, self.norm["white"])[0]
        return yh

    def denormalize(self, y_hat):
        """Inverse of `normalize` (no Jacobian)."""
        y_hat = jnp.atleast_2d(jnp.asarray(y_hat, jnp.float64))
        if self.norm["white"] is not None:
            y_hat = whiten_inverse(y_hat, self.norm["white"])[0]
        return sas_inverse(y_hat, self.norm)[0]

    ##################################################################
    #                    context-dependent scale

    def _fit_scale(self, steps, batch_size=4096, learning_rate=1e-3, verbose=True):
        """Fit log sigma_k(c) by regressing log y_k^2 on the normalized context."""
        ## First, start from the pooled scale
        g, r = self._random_pairs(self.train_idx, 20_000)
        y0 = self._read(g, r)
        self._scale_floor = float(np.median(y0**2) * 1e-12)  # guards log(0)
        floor = self._scale_floor
        target0 = 0.5 * (np.log(y0**2 + floor).mean(0) - _E_LOG_CHI2_1)
        params = _init_tanh_net(jr.PRNGKey(self.seed + 7), (self.C, *self.scale_hidden, self.D), target0)

        ## Second, regress
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

        ## Third, held-out check: fraction of the variance of log y^2 explained
        groups = self.val_idx if len(self.val_idx) else self.train_idx
        g, r = self._random_pairs(groups, 20_000)
        y = self._read(g, r)
        t = np.log(y**2 + floor) - _E_LOG_CHI2_1
        pred = 2.0 * np.asarray(_scale_apply(params, jnp.asarray(self.c_all[g])))
        self.scale_r2 = float(1.0 - np.mean((t - pred) ** 2) / np.mean((t - t.mean(0)) ** 2))
        return params

    ##################################################################
    #                    per-draw standardization

    def _draw_stats(self, groups, chunk=256):
        """Per-draw median and IQR / 1.349 of every feature over its valid realizations."""
        groups = np.sort(np.asarray(groups))
        med = np.empty((len(groups), self.D))
        sc = np.empty((len(groups), self.D))
        for a in range(0, len(groups), chunk):
            gg = groups[a : a + chunk]
            y = np.asarray(self.data[gg], dtype=np.float64)  # (k, R, D)
            ok = np.isfinite(y)
            if self.valid is not None:
                ok &= self.valid[gg][:, :, None]
            y = np.where(ok, y, np.nan)
            q25, q50, q75 = np.nanpercentile(y, [25, 50, 75], axis=1)
            med[a : a + chunk], sc[a : a + chunk] = q50, (q75 - q25) / 1.349
        return groups, med, sc

    def _fit_standardize(self, steps, n_draws, batch_size=4096, learning_rate=1e-3, verbose=True):
        """
        Fit c -> (median, log IQR-scale) of every feature on per-draw statistics.
        The learning rate decays (cosine) from learning_rate to learning_rate / 1000.
        `standardize_r2['location_err_in_s']` is the median |m(c) - median| of
        held-out draws in units of the draw's own s; it should be well below 1.
        """
        D = self.D
        net = _init_tanh_net(jr.PRNGKey(self.seed + 11), (self.C, *self.standardize_hidden, 2 * D),
                             np.zeros(2 * D))
        if steps == 0:  # the parameters will come from `load`
            self.standardize_r2 = None
            return dict(net=net, t_mean=jnp.zeros(2 * D), t_std=jnp.ones(2 * D))
        if self.R < 20:
            warnings.warn(f"standardize_by_context: only {self.R} realizations per draw; "
                          "the per-draw median and IQR will be noisy.")

        ## First, the per-draw statistics of the training draws
        fit_g = self.rng.choice(self.train_idx, size=min(n_draws, len(self.train_idx)), replace=False)
        g, med, sc = self._draw_stats(fit_g)
        floor = 1e-6 * np.median(sc[sc > 0]) if np.any(sc > 0) else 1e-12
        T = np.concatenate([med, np.log(np.maximum(sc, floor))], axis=1)
        keep = np.isfinite(T).all(1)
        g, T = g[keep], T[keep]
        t_mean, t_std = T.mean(0), T.std(0)
        t_std = np.where(t_std > 0, t_std, 1.0)
        Tn = jnp.asarray((T - t_mean) / t_std)
        Cg = jnp.asarray(self.c_all[g])

        ## Second, regress them on the context
        def loss(net, c, t):
            return jnp.mean((_scale_apply(net, c) - t) ** 2)

        opt = optax.adam(optax.cosine_decay_schedule(learning_rate, max(1, steps), alpha=1e-3))
        state = opt.init(net)

        @jax.jit
        def step(net, st, c, t):
            l, gr = jax.value_and_grad(loss)(net, c, t)
            upd, st = opt.update(gr, st)
            return optax.apply_updates(net, upd), st, l

        for _ in trange(steps, desc="Fitting per-draw location and scale", disable=not verbose):
            i = self.rng.integers(0, len(g), size=min(batch_size, len(g)))
            net, state, _ = step(net, state, Cg[i], Tn[i])
        params = dict(net=net, t_mean=jnp.asarray(t_mean), t_std=jnp.asarray(t_std))

        ## Third, held-out check: R^2 of the per-draw median and log scale
        vg = self.val_idx if len(self.val_idx) else fit_g
        vg = self.rng.choice(vg, size=min(2000, len(vg)), replace=False)
        vg, vmed, vsc = self._draw_stats(vg)
        VT = np.concatenate([vmed, np.log(np.maximum(vsc, floor))], axis=1)
        ok = np.isfinite(VT).all(1)
        m, ls = _stdz_apply(params, jnp.asarray(self.c_all[vg[ok]]))
        pred = np.concatenate([np.asarray(m), np.asarray(ls)], axis=1)
        resid = np.mean((VT[ok] - pred) ** 2, 0)
        var = np.var(VT[ok], 0)
        r2 = 1.0 - resid / np.where(var > 0, var, 1.0)
        err_s = np.abs(VT[ok, :D] - pred[:, :D]) / np.exp(VT[ok, D:])
        self.standardize_r2 = dict(location=r2[:D], log_scale=r2[D:],
                                   location_err_in_s=np.median(err_s, 0))
        return params

    def context_location_scale(self, c):
        """(m(c), s(c)) of standardize_by_context for physical contexts c, each (n, D)."""
        if self.stdz_params is None:
            raise RuntimeError("standardize_by_context is off")
        c_hat = context_forward(jnp.atleast_2d(jnp.asarray(c, jnp.float64)), self.ctx)
        m, ls = _stdz_apply(self.stdz_params, c_hat)
        return m, jnp.exp(ls)

    def context_scale(self, c):
        """sigma_k(c) of scale_by_context for physical contexts c, shape (n, D)."""
        if self.scale_params is None:
            raise RuntimeError("scale_by_context is off")
        c_hat = context_forward(jnp.atleast_2d(jnp.asarray(c, jnp.float64)), self.ctx)
        return jnp.exp(_scale_apply(self.scale_params, c_hat))

    def _set_logsig_all(self):
        """Per-group location and log scale applied before the SAS map (None when off)."""
        self.loc_all = None
        if self.stdz_params is not None:
            m, ls = _stdz_apply(self.stdz_params, jnp.asarray(self.c_all))
            self.loc_all, self.logsig_all = np.asarray(m), np.asarray(ls)
        elif self.scale_params is not None:
            self.logsig_all = np.asarray(_scale_apply(self.scale_params, jnp.asarray(self.c_all)))
        else:
            self.logsig_all = None

    def _refresh_data(self):
        self._pair_cache = {}
        self._set_ctx()
        self._set_logsig_all()

    ##################################################################
    #                    data access

    def _valid_pairs(self, groups):
        """Flat indices g * R + r of the valid (group, realization) pairs of `groups`."""
        groups = np.asarray(groups)
        key = groups.tobytes()
        if key not in self._pair_cache:
            if len(self._pair_cache) >= 8:
                self._pair_cache.clear()
            gi, ri = np.nonzero(self.valid[groups])
            self._pair_cache[key] = groups[gi] * self.R + ri
        return self._pair_cache[key]

    def _n_pairs(self, groups):
        """Number of valid (group, realization) pairs in `groups`."""
        if self.valid is None:
            return len(groups) * self.R
        return int(np.count_nonzero(self.valid[np.asarray(groups)]))

    def _sample_pairs(self, groups, n, rng, replace=True):
        """n (group, realization) pairs drawn uniformly from the valid pairs of `groups`."""
        if not replace:
            n = min(n, self._n_pairs(groups))
        if self.valid is None:
            if replace:
                return rng.choice(groups, size=n, replace=True), rng.integers(0, self.R, size=n)
            pos = rng.choice(len(groups) * self.R, size=n, replace=False)
            return np.asarray(groups)[pos // self.R], pos % self.R
        flat = rng.choice(self._valid_pairs(groups), size=n, replace=replace)
        return flat // self.R, flat % self.R

    def _random_pairs(self, groups, n):
        return self._sample_pairs(groups, n, self.rng)

    def _read(self, g, r):
        """Physical data of the pairs (g, r), in the given order."""
        o = np.lexsort((r, g))  # sorted reads are faster on memmaps
        out = np.empty((len(g), self.D))
        rows = self.data[g[o], r[o]] if self.layout == "library" else self.data[g[o]]
        out[o] = np.asarray(rows, dtype=np.float64).reshape(len(g), self.D)
        if not np.all(np.isfinite(out)):
            raise ValueError("Non-finite data values; pass drop_nonfinite=True to ignore "
                             "those rows, or remove them before training.")
        return out

    def _read_scaled(self, g, r):
        y = self._read(g, r)
        if self.loc_all is not None:
            y = y - self.loc_all[g]
        return y if self.logsig_all is None else y * np.exp(-self.logsig_all[g])

    def _batch(self, g, r):
        """Normalized training batch (y_hat, c_hat) of the pairs (g, r)."""
        yh = _sas_forward_np(self._read_scaled(g, r), self.norm_np)
        if self.white_np is not None:
            yh = whiten_forward(yh, self.white_np)[0]
        y = jnp.asarray(yh, self.dtype)
        c = jnp.asarray(self.c_all[g], self.dtype) if self.C else None
        return y, c

    ##################################################################
    #                    model

    def _build(self):
        """Architecture, optimizer and the jitted functions."""
        self._setup_architecture()
        self.optimizer = _make_optimizer(self.learning_rate, self.grad_clip)
        D, dtype = self.D, self.dtype

        def norm_lp(params, y, c):  # log density in normalized space
            z, ladj = self._flow_forward(params, y, c)
            return jnp.sum(-0.5 * z**2, -1) - 0.5 * D * LOG2PI + ladj

        def prep_c(c, ctx):
            return None if c is None else context_forward(c, ctx).astype(dtype)

        def ctx_affine(norm, ch):
            """(location, log scale) applied before the SAS map, or None."""
            if norm["stdz"] is not None:
                return _stdz_apply(norm["stdz"], ch)
            if norm["scale"] is not None:
                return 0.0, _scale_apply(norm["scale"], ch)
            return None

        def to_scaled(norm, ch, y):
            aff = ctx_affine(norm, ch)
            if aff is None:
                return y, 0.0
            m, ls = aff
            return (y - m) * jnp.exp(-ls), -jnp.sum(ls, -1)

        def to_normalized(norm, ch, y):
            """Physical y -> flow input, with log|det|."""
            ys, ld_s = to_scaled(norm, ch, y)
            yh, ld = sas_forward(ys, norm)
            if norm["white"] is not None:
                yh, ld_w = whiten_forward(yh, norm["white"])
                ld = ld + ld_w
            return yh.astype(dtype), ld + ld_s

        def phys_lp(params, norm, ctx, y, c):
            ch = prep_c(c, ctx)
            yh, ld = to_normalized(norm, ch, y)
            return norm_lp(params, yh, ch) + ld

        def phys_sample(params, norm, ctx, z, c):
            ch = prep_c(c, ctx)
            yh = self._flow_inverse(params, z, ch)
            if norm["white"] is not None:
                yh = whiten_inverse(yh, norm["white"])[0]
            ys = sas_inverse(yh, norm)[0]
            aff = ctx_affine(norm, ch)
            if aff is None:
                return ys
            m, ls = aff
            return ys * jnp.exp(ls) + m

        def latent(params, norm, ctx, y, c):
            ch = prep_c(c, ctx)
            return self._flow_forward(params, to_normalized(norm, ch, y)[0], ch)[0]

        def loss(params, y, c):
            return -jnp.mean(norm_lp(params, y, c))

        @jax.jit
        def update(params, opt_state, y, c, lr_scale=None):
            l, g = jax.value_and_grad(loss)(params, y, c)
            upd, opt_state = self.optimizer.update(g, opt_state, params)
            upd = _scale_updates(upd, lr_scale)
            return optax.apply_updates(params, upd), opt_state, l

        self._update = update
        ## ctx holds a python string, so it is closed over (rebuilt on load)
        self._phys_lp = jax.jit(lambda p, n, y, c: phys_lp(p, n, self.ctx, y, c))
        self._phys_sample = jax.jit(lambda p, n, z, c: phys_sample(p, n, self.ctx, z, c))
        self._latent = jax.jit(lambda p, n, y, c: latent(p, n, self.ctx, y, c))

    def _init_params(self):
        self.key, k = jr.split(self.key)
        self.params = self._init_model_params(k)
        self.opt_state = self.optimizer.init(self.params)
        self.best_params = self.params

    ##################################################################
    #                    training

    def fit(
        self,
        num_epochs=100,
        batch_size=1024,
        steps_per_epoch=None,
        checkpoint=None,
        save_path=None,
        lr_schedule=None,
        lr_min=None,
        warmup_steps=0,
        patience=10,
        normality_alpha=0.01,
        normality_patience=3,
        stop_rule="either",
        stopper=None,
        n_val=50_000,
        n_normality=5000,
        verbose=True,
    ):
        """
        Train by maximum likelihood on uniformly drawn (group, realization) pairs.

        :param num_epochs: number of epochs
        :param batch_size: pairs per batch
        :param steps_per_epoch: None -> about one pass over the training pairs
        :param checkpoint: path; the flow is saved there after every epoch,
            and training resumes from it if it exists
        :param save_path: path; the flow is saved there whenever the validation log p improves
        :param lr_schedule: None, "cosine", "linear", "exponential" or a callable (see `make_lr_scale`)
        :param lr_min: final learning rate of the decay; default learning_rate / 100
        :param warmup_steps: steps of linear warm-up before the decay
        :param patience, normality_alpha, normality_patience, stop_rule:
            early stopping (see `EarlyStopping`); needs val_frac > 0.  Validation
            uses held-out groups (theta points in the library layout).
        :param stopper: an `EarlyStopping` object; if given, it replaces the four above
        :param n_val: held-out pairs used for the validation log p
        :param n_normality: held-out pairs (out of n_val) used for the normality test
        :param verbose: show a progress bar

        :return: self.history (per-epoch lists: train, val, normality_p, lr)
        """
        self._require_data()
        if steps_per_epoch is None:
            steps_per_epoch = max(1, self._n_pairs(self.train_idx) // batch_size)
        lr_scale = make_lr_scale(lr_schedule, self.learning_rate, num_epochs * steps_per_epoch,
                                 lr_min, warmup_steps)

        def run_epoch(step):
            losses = []
            for _ in range(steps_per_epoch):
                y, c = self._batch(*self._random_pairs(self.train_idx, batch_size))
                self.params, self.opt_state, loss = self._update(
                    self.params, self.opt_state, y, c, lr_scale(step))
                losses.append(loss)
                step += 1
            return float(jnp.mean(jnp.stack(losses)))

        stopper = stopper or EarlyStopping(patience, normality_alpha, normality_patience, stop_rule)
        return self._train(run_epoch, num_epochs, steps_per_epoch, lr_scale, checkpoint, save_path,
                           stopper, n_val, n_normality, desc=f"Training {type(self).__name__}",
                           verbose=verbose)

    def _n_held_out(self):
        return self._n_pairs(self.val_idx) if len(self.val_idx) else 0

    def _held_out(self, n):
        rng = np.random.default_rng(0)  # same held-out samples in every fit call
        g, r = self._sample_pairs(self.val_idx, n, rng, replace=False)
        return self._read(g, r), (self.context_raw[g] if self.C else None)

    def _validate(self, y, c, n_normality):
        logp = float(jnp.mean(self.log_prob(y, c)))
        y = y[:n_normality]
        c = None if c is None else c[:n_normality]
        c_hat = None if c is None else np.asarray(context_forward(jnp.asarray(c), self.ctx))
        return logp, self.backward_pass(y, c), c_hat

    ##################################################################
    #                    public API (physical units)

    def _prep(self, y, c):
        y = jnp.atleast_2d(jnp.asarray(y, jnp.float64))
        if not self.C:
            return y, None
        if c is None:
            raise ValueError("this flow is conditional; pass the context")
        c = jnp.atleast_2d(jnp.asarray(c, jnp.float64))
        return y, jnp.broadcast_to(c, (y.shape[0], self.C))

    def log_prob(self, y, c=None, chunk=50_000):
        """log p(y | c); y (n, D), c (n, C) or (C,)."""
        y, c = self._prep(y, c)
        parts = [self._phys_lp(self.params, self.norm, y[a : a + chunk],
                               None if c is None else c[a : a + chunk])
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
        Latent z -> physical y given context c (y ~ p(y | c) when z ~ N(0, I)).
        Pure and differentiable in z and c (numpyro-ready).
        z (n, D) or (D,);  c (n, C) or (C,) physical context.
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

    def backward_pass(self, y, c=None, return_logdet=False, chunk=50_000):
        """
        Physical y -> latent z given context c (the inverse of forward_pass).

        :param y: (n, D) or (D,)
        :param c: (n, C) or (C,) physical context
        :param return_logdet: also return log|det dz/dy|, so that
            log p(y | c) = log N(z; 0, I) + logdet
        :param chunk: rows per call, to bound memory
        """
        y, c = self._prep(y, c)
        zs, lds = [], []
        for a in range(0, y.shape[0], chunk):
            yb = y[a : a + chunk]
            cb = None if c is None else c[a : a + chunk]
            z = self._latent(self.params, self.norm, yb, cb)
            zs.append(z)
            if return_logdet:
                log_n = -0.5 * jnp.sum(z**2, axis=-1) - 0.5 * self.D * LOG2PI
                lds.append(self._phys_lp(self.params, self.norm, yb, cb) - log_n)
        z = jnp.concatenate(zs)
        return (z, jnp.concatenate(lds)) if return_logdet else z

    ##################################################################
    #                    diagnostics

    def context_ablation_gap(self, n_pairs=20_000, seed=1):
        """mean[log p(y | c_true) - log p(y | c_shuffled)] on held-out groups; ~0 = context ignored."""
        if not self.C:
            raise RuntimeError("unconditional flow")
        self._require_data()
        rng = np.random.default_rng(seed)
        groups = self.val_idx if len(self.val_idx) else self.train_idx
        g, r = self._sample_pairs(groups, n_pairs, rng)
        y = self._read(g, r)
        c, cs = self.context_raw[g], self.context_raw[rng.permutation(g)]
        return float(jnp.mean(self.log_prob(y, c) - self.log_prob(y, cs)))

    def compare_conditional(self, group, num_samples=None, quantiles=(0.05, 0.5, 0.95)):
        """Library vs flow quantiles at one theta point (library layout)."""
        if self.layout != "library":
            raise RuntimeError("needs the library layout")
        self._require_data()
        lib = np.asarray(self.data[group], dtype=np.float64)
        if self.valid is not None:
            lib = lib[self.valid[group]]
        gen = np.asarray(self.sample(num_samples or max(10 * self.R, 10_000), self.context_raw[group]))
        return np.quantile(lib, quantiles, axis=0), np.quantile(gen, quantiles, axis=0)

    ##################################################################
    #                    loading (also files from the older format)

    def load(self, path):
        """
        Load a saved flow into this one, keeping this flow's training data.
        Also reads files written by the older `save` (a dict of arrays); for
        those, build this flow with scale_steps=0 / standardize_steps=0.
        """
        other = load_flow(path)
        if isinstance(other, dict):
            return self._load_legacy(other)
        return self._load_from(other, path)

    def _load_legacy(self, blob):
        for k, v in blob["cfg"].items():
            if self._arch_cfg().get(k) != v:
                raise ValueError(f"saved model differs on {k}: {v} vs {self._arch_cfg().get(k)}")
        if "orders" in blob and hasattr(self, "layers"):
            if not all(np.array_equal(a, L["order"]) for a, L in zip(blob["orders"], self.layers)):
                raise ValueError("feature orders differ (randperm with a different seed?)")
        if (blob.get("scale_params") is not None) != self.scale_by_context:
            raise ValueError("saved model and this flow differ on scale_by_context")
        if (blob.get("stdz_params") is not None) != self.standardize_by_context:
            raise ValueError("saved model and this flow differ on standardize_by_context")

        self.params = self.best_params = jax.tree_util.tree_map(
            lambda a: jnp.asarray(a, self.dtype), blob["params"])
        self.opt_state = self.optimizer.init(self.params)
        self.ctx_np = blob["ctx"]
        if self.scale_by_context:
            self.scale_params = jax.tree_util.tree_map(jnp.asarray, blob["scale_params"])
        if self.standardize_by_context:
            self.stdz_params = jax.tree_util.tree_map(jnp.asarray, blob["stdz_params"])
        self.val_idx, self.train_idx = blob["val_idx"], blob["train_idx"]
        self._refresh_data()
        self.set_normalizer(blob["norm"])
        self.history = dict(dict(train=[], val=[], normality_p=[], lr=[]), **blob.get("history", {}))
        self._build()
        return self


##############################################################################################
#                                     ConditionalFlow
##############################################################################################


class ConditionalFlow(_ConditionalSASFlow):
    """
    Conditional RQS coupling flow with a SAS preprocessing.

        physical y (D,) --SAS--> y_hat --coupling layers(c_hat)--> z ~ N(0, I)
        context  c (C,) --context transform--> c_hat

        log p(y | c) = log N(z; 0, I) + sum_l log|det dT_l| + log|det dSAS|

    In every layer, the features kept by the mask and c_hat go through an
    MLP that gives the spline parameters of the other features.  Masks
    alternate by feature parity.  The last MLP layer starts at zero, so the
    untrained flow is the identity.  Splines act on [-B, B] and are the
    identity outside, so the tails come from the SAS map.

    Both directions take one pass per layer, so sample, forward_pass and
    log_prob are equally fast: use it when numpyro samples the base variables.

    :param data:
        (n_theta, R, D) library layout, or (N, D) flat layout; may be a memmap
    :param context:
        (n_theta, C) for library, (N, C) for flat, or None (unconditional)
    :param context_kind:
        "standardize", "probit", "sas" or "none" (see `fit_context_transform`)
    :param context_bounds:
        (lo, hi) for "probit"; inferred from the data if None
    :param flow_num_layers:
        number of coupling layers
    :param hidden_size:
        width of the hidden layers of the conditioner networks
    :param mlp_num_layers:
        number of hidden layers of the conditioner networks
    :param num_bins:
        number of spline bins
    :param B:
        the splines act on [-B, B]
    :param val_frac:
        fraction of theta points (library) or rows (flat) held out for early stopping
    :param normalizer:
        pre-fitted SAS dict for the data; fitted if None
    :param normalizer_samples:
        rows used for the SAS and whitening fits
    :param learning_rate:
        learning rate of Adam
    :param grad_clip:
        global-norm gradient clipping; None turns it off
    :param seed:
        random seed
    :param dtype:
        jnp.float64 (default) or jnp.float32 (much faster on GPUs)
    :param scale_by_context:
        model y / sigma(c) with a learned scale (see `_ConditionalSASFlow`).
        Use it when the context mostly changes the size of the data.
    :param scale_hidden, scale_steps:
        hidden layers and training steps of the scale network
    :param whiten:
        None, "zca", "cholesky" or "pca": decorrelate the SAS output with a fixed
        linear map, fitted on the training data pooled over contexts
    :param whiten_shrinkage:
        covariance shrinkage of the whitener
    :param drop_nonfinite:
        never train on rows whose data, or whose group's context, contain NaN
        or inf (nothing is copied); groups with no valid row are dropped
    :param standardize_by_context:
        model (y - m(c)) / s(c) with a learned per-draw median and IQR scale
        (see `_ConditionalSASFlow`). Use it when the context moves the level of
        the data, e.g. log10 hc. Library layout only.
    :param standardize_hidden, standardize_steps:
        hidden layers and training steps of the location-scale network
    :param standardize_draws:
        training draws whose statistics are used to fit it

    Author:
    Nima Laal (11/26/2026)
    """

    _BUILT = _ConditionalSASFlow._BUILT + ("masks", "_hk_fwd", "_hk_inv")

    def __init__(
        self,
        data,
        context=None,
        context_kind="standardize",
        context_bounds=None,
        flow_num_layers=8,
        hidden_size=256,
        mlp_num_layers=2,
        num_bins=8,
        B=5.0,
        val_frac=0.1,
        normalizer=None,
        normalizer_samples=50_000,
        learning_rate=1e-3,
        grad_clip=1.0,
        seed=0,
        dtype=jnp.float64,
        scale_by_context=False,
        scale_hidden=(128, 128),
        scale_steps=3000,
        whiten=None,
        whiten_shrinkage=0.0,
        drop_nonfinite=True,
        standardize_by_context=False,
        standardize_hidden=(128, 128),
        standardize_steps=3000,
        standardize_draws=20_000,
    ):
        self.flow_num_layers = flow_num_layers
        self.hidden_sizes = [int(hidden_size)] * mlp_num_layers
        self.num_bins = num_bins
        self.B = float(B)
        super().__init__(
            data, context, context_kind=context_kind, context_bounds=context_bounds,
            val_frac=val_frac, normalizer=normalizer, normalizer_samples=normalizer_samples,
            learning_rate=learning_rate, grad_clip=grad_clip, seed=seed, dtype=dtype,
            scale_by_context=scale_by_context, scale_hidden=scale_hidden, scale_steps=scale_steps,
            whiten=whiten, whiten_shrinkage=whiten_shrinkage, drop_nonfinite=drop_nonfinite,
            standardize_by_context=standardize_by_context, standardize_hidden=standardize_hidden,
            standardize_steps=standardize_steps, standardize_draws=standardize_draws,
        )

    def _arch_cfg(self):
        return dict(D=self.D, C=self.C, flow_num_layers=self.flow_num_layers,
                    hidden_sizes=list(self.hidden_sizes), num_bins=self.num_bins, B=self.B)

    def _setup_architecture(self):
        D, nparams, B = self.D, 3 * self.num_bins + 1, self.B
        if D == 1:
            self.masks = [np.zeros(1, bool)] * self.flow_num_layers
        else:
            base = np.arange(D) % 2 == 1
            self.masks = [base if l % 2 == 0 else ~base for l in range(self.flow_num_layers)]
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
                return lambda x: net(jnp.concatenate(
                    [x, jnp.broadcast_to(c, x.shape[:-1] + c.shape[-1:])], axis=-1))

            layers = [distrax.MaskedCoupling(
                mask=jnp.asarray(m),
                bijector=lambda p: distrax.RationalQuadraticSpline(p, range_min=-B, range_max=B),
                conditioner=conditioner()) for m in masks]
            return distrax.Chain(layers)

        ## Normalized data -> latent, and latent -> normalized data (one pass each)
        self._hk_fwd = hk.without_apply_rng(hk.transform(lambda y, c: chain(c).forward_and_log_det(y)))
        self._hk_inv = hk.without_apply_rng(hk.transform(lambda z, c: chain(c).inverse(z)))

    def _init_model_params(self, key):
        y0 = jnp.zeros((2, self.D), self.dtype)
        c0 = jnp.zeros((2, self.C), self.dtype) if self.C else None
        return self._hk_fwd.init(key, y0, c0)

    def _flow_forward(self, params, y, c):
        return self._hk_fwd.apply(params, y, c)

    def _flow_inverse(self, params, z, c):
        return self._hk_inv.apply(params, z, c)


##############################################################################################
#                                         SASNSF
##############################################################################################


class SASNSF(_ConditionalSASFlow):
    """
    Conditional neural spline flow (zuko's NSF) with a SAS preprocessing, in JAX.

        physical y (D,) --SAS--> y_hat --T_1 .. T_n--> z ~ N(0, I)
        context  c (C,) --context transform--> c_hat   (fed to every T_i)

        log p(y | c) = log N(z; 0, I) + sum_i log|det dT_i| + log|det dSAS|

    Kept identical to zuko (v1.x); with zuko's weights copied in, log p
    matches to machine precision:
        * MaskedAutoregressiveTransform: hyper-network input [x, c]; feature
          orders alternate ascending / descending (random with randperm=True);
          `passes` groups features (None: fully autoregressive, 2: coupling).
        * MaskedMLP: same masks, ReLU by default, torch's Linear init.
        * MonotonicRQSTransform: bins, bound 5, minimum slope 1e-3, identity
          outside [-bound, bound].
        * Sampling inverts every transform with `passes` fixed-point sweeps.
    Only difference: zero_init_last=True starts every transform at the identity.

    log_prob takes one pass per transform; sample and forward_pass take
    `passes` sweeps per transform (D when fully autoregressive).

    :param data, context, context_kind, context_bounds, val_frac, normalizer,
        normalizer_samples, learning_rate, grad_clip, seed, dtype,
        scale_by_context, scale_hidden, scale_steps, whiten, whiten_shrinkage,
        drop_nonfinite, standardize_by_context, standardize_hidden,
        standardize_steps, standardize_draws:
        as in ConditionalFlow
    :param transforms, hidden_features, bins, passes, randperm, activation, slope, bound:
        zuko NSF arguments (zuko's defaults except hidden_features)
    :param zero_init_last:
        start every transform at the identity

    Author:
    Nima Laal (11/26/2026)
    """

    _BUILT = _ConditionalSASFlow._BUILT + ("layers", "total", "bins", "_activation")

    def __init__(
        self,
        data,
        context=None,
        context_kind="standardize",
        context_bounds=None,
        transforms=5,
        hidden_features=(256, 256),
        bins=8,
        passes=None,
        randperm=False,
        activation="relu",
        slope=1e-3,
        bound=5.0,
        zero_init_last=True,
        val_frac=0.1,
        normalizer=None,
        normalizer_samples=50_000,
        learning_rate=1e-3,
        grad_clip=1.0,
        seed=0,
        dtype=jnp.float64,
        scale_by_context=False,
        scale_hidden=(128, 128),
        scale_steps=3000,
        whiten=None,
        whiten_shrinkage=0.0,
        drop_nonfinite=True,
        standardize_by_context=False,
        standardize_hidden=(128, 128),
        standardize_steps=3000,
        standardize_draws=20_000,
    ):
        self.cfg = dict(
            transforms=transforms, hidden_features=tuple(int(h) for h in hidden_features),
            bins=bins, passes=passes, randperm=randperm, activation=activation,
            slope=slope, bound=float(bound), zero_init_last=zero_init_last,
        )
        super().__init__(
            data, context, context_kind=context_kind, context_bounds=context_bounds,
            val_frac=val_frac, normalizer=normalizer, normalizer_samples=normalizer_samples,
            learning_rate=learning_rate, grad_clip=grad_clip, seed=seed, dtype=dtype,
            scale_by_context=scale_by_context, scale_hidden=scale_hidden, scale_steps=scale_steps,
            whiten=whiten, whiten_shrinkage=whiten_shrinkage, drop_nonfinite=drop_nonfinite,
            standardize_by_context=standardize_by_context, standardize_hidden=standardize_hidden,
            standardize_steps=standardize_steps, standardize_draws=standardize_draws,
        )

    def _arch_cfg(self):
        keys = ("transforms", "hidden_features", "bins", "passes", "activation")
        cfg = dict(D=self.D, C=self.C, **{k: self.cfg[k] for k in keys})
        if self.cfg["randperm"]:
            cfg["seed"] = self.seed  # the feature orders come from the seed
        return cfg

    def _setup_architecture(self):
        c = self.cfg
        self.total = 3 * c["bins"] - 1  # widths K, heights K, inner derivatives K-1
        self.bins = c["bins"]
        orders = [np.arange(self.D), np.arange(self.D)[::-1]]
        prng = np.random.default_rng(self.seed + 12345)
        self.layers = []
        for i in range(c["transforms"]):
            order = prng.permutation(self.D) if c["randperm"] else orders[i % 2]
            adj, p = maf_adjacency(self.D, self.C, order, c["passes"], self.total)
            if adj.any():
                masks = [jnp.asarray(m, self.dtype) for m in masked_mlp_masks(adj, c["hidden_features"])]
            else:  # D == 1 and no context: a free spline
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
        """Hyper-network output of one transform -> spline knots, each (..., D, .)."""
        if L["masks"] is None:
            phi = jnp.broadcast_to(p[0]["b"], x.shape[:-1] + p[0]["b"].shape)
        else:
            inp = x if c is None else jnp.concatenate([x, c], axis=-1)
            phi = masked_mlp_apply(p, L["masks"], inp, self._activation)
        phi = phi.reshape(x.shape[:-1] + (self.D, self.total))
        K = self.bins
        w, h, d = phi[..., :K], phi[..., K : 2 * K], phi[..., 2 * K :]
        return _rqs_knots(w, h, d, self.cfg["bound"], self.cfg["slope"])

    def _flow_forward(self, params, y, c):
        """Normalized data -> latent, summed log|det| (one parallel pass per transform)."""
        x, ladj = y, 0.0
        for p, L in zip(params, self.layers):
            xs, ys, ds = self._spline_params(p, L, x, c)
            x, l = rqs_forward(x, xs, ys, ds)
            ladj = ladj + jnp.sum(l, axis=-1)
        return x, ladj

    def _flow_inverse(self, params, z, c):
        """Latent -> normalized data (`passes` fixed-point sweeps per transform, as zuko)."""
        y = z
        for p, L in reversed(list(zip(params, self.layers))):
            target = y

            def sweep(_, x, p=p, L=L, target=target):
                xs, ys, ds = self._spline_params(p, L, x, c)
                return rqs_inverse(target, xs, ys, ds)

            y = jax.lax.fori_loop(0, L["passes"], sweep, jnp.zeros_like(target))
        return y


##############################################################################################
#                     Flow matching (code unchanged; only moved here)
##############################################################################################

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