"""
Differentiable conditional normalizing flows for astro inference (flax NNX, distrax, optax).

The JAX counterpart of ``zuko_flows.py``: the same flows, written so they can
be jitted, vmapped and differentiated, and so dropped into numpyro models.
Two architectures are available:

* :class:`ZukoNSF` reimplements zuko's ``NSF`` (Pandora's flow) exactly:
  the masked-MLP conditioners and their masks, the alternating orders, the
  two-pass coupling, and zuko's monotonic rational-quadratic spline
  (:class:`ZukoRQS`, a distrax bijector). A trained zuko or Pandora flow
  converts with :meth:`FlaxAstroFlow.from_zuko` and evaluates to the same
  density up to rounding.
* :class:`CouplingRQSFlow` is based on the flow architecture in PELARGIR (Criswell+26): distrax
  ``MaskedCoupling`` layers with ``RationalQuadraticSpline`` bijectors whose
  conditioners start at zero, so an untrained flow is its base distribution.

:class:`FlaxAstroFlow` works in physical units, mirroring
``zuko_flows.ZukoAstroFlow``; :class:`FlowDistribution` wraps it as a numpyro
distribution; :class:`FlaxFlowTrainer` trains either architecture with optax,
drawing the same batches as the zuko trainer (``flow_utils.BatchSampler``)
but gathering them on the device and running the steps between checkpoints
as one ``lax.scan`` (Pelargir's training loop), so GPU training is not bound
by per-step dispatch.

"""

import json
import math
import os
import random
from functools import partial

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm.auto import trange

from ATLAS.experimental.flow_utils import (
    NF_TYPES,
    BatchSampler,
    HellingerMonitor,
    batched_histogram,
    gather,
    hellinger,
)

__all__ = [
    "NF_TYPES",
    "ZukoRQS",
    "ZukoNSF",
    "CouplingRQSFlow",
    "make_flax_flow",
    "FlaxAstroFlow",
    "FlowDistribution",
    "FlaxFlowTrainer",
]

CONFIG_FILE = "flow_config.json"
PARAMS_FILE = "flow_params.npz"
_FORMAT = "atlas-flax-flow-v1"


########################################################################################
# zuko's monotonic rational-quadratic spline
########################################################################################

class ZukoRQS(distrax.Bijector):
    """
    zuko's ``MonotonicRQSTransform`` as an elementwise distrax bijector.

    ``widths``, ``heights`` (``(..., K)``) and ``derivatives`` (``(..., K-1)``)
    are the unconstrained network outputs. They are soft-clipped to the
    minimum slope ``slope``, turned into knots on ``[-bound, bound]`` and
    interior derivatives (the boundary derivatives are 1), exactly as zuko
    does. Outside ``[-bound, bound]`` the bijector is the identity.
    """

    def __init__(self, widths, heights, derivatives, bound=5.0, slope=1e-3):
        super().__init__(event_ndims_in=0)
        ls = math.log(slope)
        widths = widths / (1 + jnp.abs(2 * widths / ls))
        heights = heights / (1 + jnp.abs(2 * heights / ls))
        derivatives = derivatives / (1 + jnp.abs(derivatives / ls))

        pad = [(0, 0)] * (widths.ndim - 1)
        widths = jnp.pad(jax.nn.softmax(widths, axis=-1), pad + [(1, 0)])
        heights = jnp.pad(jax.nn.softmax(heights, axis=-1), pad + [(1, 0)])
        derivatives = jnp.pad(derivatives, pad + [(1, 1)])

        self.horizontal = bound * (2 * jnp.cumsum(widths, axis=-1) - 1)
        self.vertical = bound * (2 * jnp.cumsum(heights, axis=-1) - 1)
        self.derivatives = jnp.exp(derivatives)

    @property
    def bins(self):
        return self.horizontal.shape[-1] - 1

    def _bin(self, k):
        mask = jnp.logical_and(0 <= k, k < self.bins)
        k = k % self.bins

        def take(a, i):
            return jnp.take_along_axis(a, i[..., None], axis=-1)[..., 0]

        x0, x1 = take(self.horizontal, k), take(self.horizontal, k + 1)
        y0, y1 = take(self.vertical, k), take(self.vertical, k + 1)
        d0, d1 = take(self.derivatives, k), take(self.derivatives, k + 1)
        s = (y1 - y0) / (x1 - x0)
        return mask, x0, x1, y0, y1, d0, d1, s

    @staticmethod
    def _searchsorted(seq, value):
        return jnp.sum(seq < value[..., None], axis=-1)

    def forward_and_log_det(self, x):
        k = self._searchsorted(self.horizontal, x) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self._bin(k)
        z = mask * (x - x0) / (x1 - x0)
        y = y0 + (y1 - y0) * (s * z**2 + d0 * z * (1 - z)) / (s + (d0 + d1 - 2 * s) * z * (1 - z))
        jacobian = (s**2 * (2 * s * z * (1 - z) + d0 * (1 - z) ** 2 + d1 * z**2)
                    / (s + (d0 + d1 - 2 * s) * z * (1 - z)) ** 2)
        return jnp.where(mask, y, x), mask * jnp.log(jacobian)

    def inverse(self, y):
        k = self._searchsorted(self.vertical, y) - 1
        mask, x0, x1, y0, y1, d0, d1, s = self._bin(k)
        y_ = mask * (y - y0)
        a = (y1 - y0) * (s - d0) + y_ * (d0 + d1 - 2 * s)
        b = (y1 - y0) * d0 - y_ * (d0 + d1 - 2 * s)
        c = -s * y_
        z = 2 * c / (-b - jnp.sqrt(b**2 - 4 * a * c))
        x = x0 + z * (x1 - x0)
        return jnp.where(mask, x, y)

    def inverse_and_log_det(self, y):
        x = self.inverse(y)
        return x, -self.forward_and_log_det(x)[1]


########################################################################################
# ZukoNSF: zuko's NSF in NNX
########################################################################################

class _Mask(nnx.Variable):
    """A fixed (non-trainable) connectivity mask."""


def zuko_masked_mlp_masks(adjacency, hidden_features):
    """
    The layer masks zuko's ``MaskedMLP`` builds from ``adjacency`` (no residual blocks).

    Returns one ``(out, in)`` boolean mask per linear layer.
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    out_features = adjacency.shape[0]
    unique, inverse = np.unique(adjacency, axis=0, return_inverse=True)
    inverse = np.ravel(inverse)
    a = unique.astype(np.float64)
    precedence = (a @ a.T) == unique.sum(axis=-1)

    masks, indices = [], None
    for i, features in enumerate((*hidden_features, out_features)):
        mask = unique if i == 0 else precedence[:, indices]
        if (~mask).all():
            raise ValueError("The adjacency matrix leads to a null Jacobian.")
        if i < len(hidden_features):
            reachable = np.nonzero(mask.sum(axis=-1))[0]
            indices = reachable[np.arange(features) % len(reachable)]
            mask = mask[indices]
        else:
            mask = mask[inverse]
        masks.append(mask)
    return masks


def zuko_coupling_adjacency(order, features, context, total):
    """The hyper-network adjacency of a coupling (``passes=2``) transform with ``order``."""
    grouped = np.asarray(order) // math.ceil(features / 2)
    adjacency = grouped[:, None] > grouped[None, :]
    if context > 0:
        adjacency = np.concatenate([adjacency, np.ones((features, context), dtype=bool)], axis=1)
    return np.repeat(adjacency, total, axis=0)


class _MaskedLinear(nnx.Module):
    def __init__(self, mask, dtype, *, rngs):
        out_f, in_f = mask.shape
        bound = 1 / math.sqrt(in_f)   # torch's default Linear initialization
        self.weight = nnx.Param(jax.random.uniform(rngs.params(), (out_f, in_f), dtype, -bound, bound))
        self.bias = nnx.Param(jax.random.uniform(rngs.params(), (out_f,), dtype, -bound, bound))
        self.mask = _Mask(jnp.asarray(mask, dtype=dtype))

    def __call__(self, x):
        return x @ (self.mask[...] * self.weight[...]).T + self.bias[...]


class ZukoNSF(nnx.Module):
    """
    zuko's ``NSF(features, context, bins, passes=2, hidden_features, transforms)`` in flax NNX.

    Coupling transforms alternate ascending and descending feature orders.
    In each, the features of the first half of the order get splines
    conditioned on the context alone, and the second half on the first half
    and the context; the spline parameters come from a masked MLP over
    ``[x, c]``. The base is a standard normal.

    ``log_prob`` maps data to latent; ``inverse`` maps latent to data (two
    fixed-point passes per transform, as in zuko).
    """

    def __init__(self, features, context=0, bins=8, hidden_features=(512, 512), transforms=3,
                 bound=5.0, slope=1e-3, dtype="float64", *, rngs):
        if features < 2:
            raise NotImplementedError("ZukoNSF needs at least 2 features "
                                      "(zuko uses an element-wise transform for 1)")
        self.features, self.context = int(features), int(context)
        self.bins = int(bins)
        self.hidden_features = tuple(int(h) for h in hidden_features)
        self.n_transforms = int(transforms)
        self.bound, self.slope = float(bound), float(slope)
        self.dtype = jnp.dtype(dtype)
        self.total = 3 * self.bins - 1
        orders = [np.arange(features), np.arange(features)[::-1]]
        self.orders = tuple(tuple(int(o) for o in orders[i % 2]) for i in range(self.n_transforms))
        layers = []
        for order in self.orders:
            adjacency = zuko_coupling_adjacency(order, self.features, self.context, self.total)
            masks = zuko_masked_mlp_masks(adjacency, self.hidden_features)
            layers.append(nnx.List([_MaskedLinear(m, self.dtype, rngs=rngs) for m in masks]))
        self.layers = nnx.List(layers)

    def _spline(self, t, x, c):
        h = x if c is None else jnp.concatenate([x, jnp.broadcast_to(c, x.shape[:-1] + c.shape[-1:])], -1)
        layers = list(self.layers[t])
        for layer in layers[:-1]:
            h = jax.nn.relu(layer(h))
        phi = layers[-1](h).reshape(h.shape[:-1] + (self.features, self.total))
        K = self.bins
        return ZukoRQS(phi[..., :K], phi[..., K:2 * K], phi[..., 2 * K:],
                       bound=self.bound, slope=self.slope)

    def _base(self):
        return distrax.MultivariateNormalDiag(jnp.zeros(self.features, self.dtype),
                                              jnp.ones(self.features, self.dtype))

    def forward_and_log_det(self, x, c=None):
        """Data to latent: ``(z, log|dz/dx|)``."""
        ladj = jnp.zeros(x.shape[:-1], x.dtype)
        for t in range(self.n_transforms):
            x, l = self._spline(t, x, c).forward_and_log_det(x)
            ladj = ladj + l.sum(-1)
        return x, ladj

    def log_prob(self, x, c=None):
        """``log p(x | c)`` in the flow's (normalized) space."""
        z, ladj = self.forward_and_log_det(x, c)
        return self._base().log_prob(z) + ladj

    def inverse(self, z, c=None):
        """Latent to data."""
        for t in reversed(range(self.n_transforms)):
            x = jnp.zeros_like(z)
            for _ in range(2):
                x = self._spline(t, x, c).inverse(z)
            z = x
        return z

    def sample(self, key, shape=(), c=None):
        z = jax.random.normal(key, tuple(shape) + (self.features,), self.dtype)
        return self.inverse(z, c)

    def architecture(self):
        return dict(kind="zuko_nsf", features=self.features, context=self.context,
                    bins=self.bins, hidden_features=list(self.hidden_features),
                    transforms=self.n_transforms, bound=self.bound, slope=self.slope,
                    dtype=str(self.dtype))

    @classmethod
    def from_zuko(cls, nsf):
        """Copy a zuko ``NSF`` (with coupling transforms) into a :class:`ZukoNSF`."""
        from ATLAS.experimental.zuko_flows import _flow_architecture

        arch = _flow_architecture(nsf)
        model = cls(arch["n_features"], arch["n_context"], bins=arch["bins"],
                    hidden_features=arch["hidden_features"], transforms=arch["transforms"],
                    dtype=arch["dtype"], rngs=nnx.Rngs(0))
        state = {k: v.detach().cpu().numpy() for k, v in nsf.state_dict().items()}
        for t in range(model.n_transforms):
            prefix = f"transform.transforms.{t}"
            divided = np.asarray(model.orders[t]) // math.ceil(model.features / 2)
            if not np.array_equal(state[f"{prefix}.order"], divided):
                raise ValueError(f"transform {t} has an order zuko_nsf does not build")
            for j, layer in enumerate(model.layers[t]):
                key = f"{prefix}.hyper.{2 * j}"
                if not np.array_equal(state[f"{key}.mask"].astype(bool), np.asarray(layer.mask[...]).astype(bool)):
                    raise ValueError(f"mask of {key} differs from zuko's")
                layer.weight[...] = jnp.asarray(state[f"{key}.weight"], model.dtype)
                layer.bias[...] = jnp.asarray(state[f"{key}.bias"], model.dtype)
        return model


########################################################################################
# CouplingRQSFlow: Pelargir's architecture
########################################################################################

class _Conditioner(nnx.Module):
    """MLP ``[x_masked, c] -> (..., D, 3K+1)``; zero-initialized output."""

    def __init__(self, in_features, D, hidden_sizes, num_params, dtype, *, rngs):
        sizes = [in_features, *hidden_sizes]
        self.hidden = nnx.List([nnx.Linear(a, b, dtype=dtype, param_dtype=dtype, rngs=rngs)
                                for a, b in zip(sizes[:-1], sizes[1:])])
        self.out = nnx.Linear(sizes[-1], D * num_params, kernel_init=nnx.initializers.zeros_init(),
                              bias_init=nnx.initializers.zeros_init(), dtype=dtype,
                              param_dtype=dtype, rngs=rngs)
        self.D, self.num_params = D, num_params

    def __call__(self, x_masked, c=None):
        h = x_masked if c is None else jnp.concatenate([x_masked, c], axis=-1)
        for layer in self.hidden:
            h = jax.nn.relu(layer(h))
        return self.out(h).reshape(h.shape[:-1] + (self.D, self.num_params))


class CouplingRQSFlow(nnx.Module):
    """
    Masked-coupling rational-quadratic-spline flow (Pelargir's architecture).

    Layer ``i`` transforms the features where the alternating ``arange(D) % 2``
    mask is False with a ``distrax.RationalQuadraticSpline`` on ``spline_range``,
    whose parameters come from the other features and the context. Splines
    are the identity outside ``spline_range``; the output layers start at zero, so an
    untrained flow is exactly the standard-normal base.
    """

    def __init__(self, features, context=0, num_layers=4, hidden_sizes=(128, 128), num_bins=8,
                 spline_range=(-6.0, 6.0), dtype="float64", *, rngs):
        self.features, self.context = int(features), int(context)
        self.num_layers, self.num_bins = int(num_layers), int(num_bins)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.spline_range = (float(spline_range[0]), float(spline_range[1]))
        self.dtype = jnp.dtype(dtype)
        num_params = 3 * self.num_bins + 1
        self.conditioners = nnx.List([
            _Conditioner(self.features + self.context, self.features, self.hidden_sizes,
                         num_params, self.dtype, rngs=rngs)
            for _ in range(self.num_layers)])

    def _bijector(self, c):
        lo, hi = self.spline_range

        def spline(params):
            return distrax.RationalQuadraticSpline(params, range_min=lo, range_max=hi)

        mask = (jnp.arange(self.features) % 2).astype(bool)
        layers = []
        for conditioner in self.conditioners:
            layers.append(distrax.MaskedCoupling(mask=mask, bijector=spline,
                                                 conditioner=partial(conditioner, c=c)))
            mask = jnp.logical_not(mask)
        return distrax.Inverse(distrax.Chain(layers))

    def _base(self):
        return distrax.MultivariateNormalDiag(jnp.zeros(self.features, self.dtype),
                                              jnp.ones(self.features, self.dtype))

    def _broadcast(self, x, c):
        return None if c is None else jnp.broadcast_to(c, x.shape[:-1] + c.shape[-1:])

    def log_prob(self, x, c=None):
        c = self._broadcast(x, c)
        return distrax.Transformed(self._base(), self._bijector(c)).log_prob(x)

    def inverse(self, z, c=None):
        return self._bijector(self._broadcast(z, c)).forward(z)

    def sample(self, key, shape=(), c=None):
        z = jax.random.normal(key, tuple(shape) + (self.features,), self.dtype)
        return self.inverse(z, c)

    def architecture(self):
        return dict(kind="coupling_rqs", features=self.features, context=self.context,
                    num_layers=self.num_layers, hidden_sizes=list(self.hidden_sizes),
                    num_bins=self.num_bins, spline_range=list(self.spline_range),
                    dtype=str(self.dtype))

_ARCHITECTURES = {"zuko_nsf": ZukoNSF, "coupling_rqs": CouplingRQSFlow}


def make_flax_flow(architecture, seed=0):
    """Build a flow from an ``architecture()`` dict (``kind`` plus constructor arguments)."""
    arch = dict(architecture)
    cls = _ARCHITECTURES[arch.pop("kind")]
    return cls(rngs=nnx.Rngs(seed), **arch)


########################################################################################
# Parameter (de)serialization (Pelargir's)
########################################################################################

def _path_str(path):
    return "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)


def _flat_params(model):
    """Trainable parameters keyed by their path in the module tree, as numpy."""
    pure = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    return {_path_str(p): np.asarray(v) for p, v in jax.tree_util.tree_flatten_with_path(pure)[0]}


def _set_params(model, flat, source):
    """Load path-keyed parameters into ``model``, which must have the same architecture."""
    state = nnx.state(model, nnx.Param)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(state))
    want = {_path_str(p): np.shape(v) for p, v in leaves}
    got = {k: np.shape(v) for k, v in flat.items()}
    if want != got:
        differ = sorted(k for k in want.keys() | got.keys() if want.get(k) != got.get(k))
        raise ValueError(f"{source} holds a different architecture; mismatched parameters: {differ[:6]}")
    new = [jnp.asarray(flat[_path_str(p)], dtype=np.asarray(v).dtype) for p, v in leaves]
    nnx.replace_by_pure_dict(state, jax.tree_util.tree_unflatten(treedef, new))
    nnx.update(model, state)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.ndarray, jax.Array)):
        return np.asarray(obj).tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


########################################################################################
# Physical units (ZukoAstroFlow's interface)
########################################################################################

class FlaxAstroFlow:
    """
    A flax flow evaluated in physical units, with ``ZukoAstroFlow``'s interface.

    Parameters
    ----------
    model : ZukoNSF or CouplingRQSFlow
    nf_type : {'rho|theta', 'theta|rho', 'rho'}
    mapping : dict
        ``B``, ``mean``, ``half_range`` (astrophysical parameters first), as
        from ``TrainingSet.normalization``.
    n_params : int
        Number of astrophysical parameters.
    metadata : dict, optional
        ``param_names``, ``f_conv``, ``tspan``, ...

    ``log_prob`` and ``sample`` are plain JAX: they work under ``jax.jit``,
    ``jax.grad`` and ``jax.vmap``, and inside numpyro models.
    """

    def __init__(self, model, nf_type, mapping, n_params, metadata=None):
        if nf_type not in NF_TYPES:
            raise ValueError(f"nf_type must be one of {NF_TYPES}, got {nf_type!r}")
        self.model = model
        self.nf_type = nf_type
        self.B = float(mapping["B"])
        mean = jnp.asarray(mapping["mean"], jnp.float64)
        half = jnp.asarray(mapping["half_range"], jnp.float64)
        p = int(n_params)
        self.n_params = p
        self.mean_ast, self.mean_gwb = mean[:p], mean[p:]
        self.half_range_ast, self.half_range_gwb = half[:p], half[p:]
        self.n_freqs = int(mean.shape[0]) - p
        self.metadata = dict(metadata or {})

    @property
    def mapping(self):
        return dict(B=self.B, mean=np.concatenate([self.mean_ast, self.mean_gwb]),
                    half_range=np.concatenate([self.half_range_ast, self.half_range_gwb]))

    # ---- physical <-> normalized -------------------------------------------------

    def transform_rho_to_scaled_interval(self, gwb_rho):
        return self.B * (gwb_rho - self.mean_gwb) / self.half_range_gwb

    def transform_params_to_scaled_interval(self, ast_params):
        return self.B * (ast_params - self.mean_ast) / self.half_range_ast

    def transform_rho_to_physical_interval(self, gwb_rho):
        return gwb_rho * self.half_range_gwb / self.B + self.mean_gwb

    def transform_params_to_physical_interval(self, ast_params):
        return ast_params * self.half_range_ast / self.B + self.mean_ast

    def log_jacobian(self):
        """``log |d(scaled)/d(physical)|`` of the modelled variable."""
        half = self.half_range_ast if self.nf_type == "theta|rho" else self.half_range_gwb
        return jnp.sum(jnp.log(self.B / half))

    # ---- density and samples ------------------------------------------------------

    def _log_prob(self, model, log10_rho, astro_params, jacobian):
        rho_s = self.transform_rho_to_scaled_interval(jnp.asarray(log10_rho, jnp.float64))
        if self.nf_type == "rho":
            lp = model.log_prob(rho_s)
        else:
            ast_s = self.transform_params_to_scaled_interval(jnp.asarray(astro_params, jnp.float64))
            x, c = (rho_s, ast_s) if self.nf_type == "rho|theta" else (ast_s, rho_s)
            shape = jnp.broadcast_shapes(x.shape[:-1], c.shape[:-1])
            lp = model.log_prob(jnp.broadcast_to(x, shape + x.shape[-1:]),
                                jnp.broadcast_to(c, shape + c.shape[-1:]))
        return lp + self.log_jacobian() if jacobian else lp

    def log_prob(self, log10_rho, astro_params=None, jacobian=True):
        """
        Log density of the modelled variable (a JAX array).

        ``jacobian=True`` gives the density in physical units; ``False`` the
        density in the normalized space (Pandora's convention).
        """
        return self._log_prob(self.model, log10_rho, astro_params, jacobian)

    def sample(self, key, batch_shape=(), context=None):
        """
        Draw samples in physical units.

        ``context`` is ``theta`` for ``'rho|theta'`` and ``log10_rho`` for
        ``'theta|rho'``; ``batch_shape`` prefixes the context's own batch
        shape. Differentiable in ``context`` at fixed ``key``.
        """
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        if self.nf_type == "rho":
            return self.transform_rho_to_physical_interval(self.model.sample(key, batch_shape))
        if self.nf_type == "rho|theta":
            c = self.transform_params_to_scaled_interval(jnp.asarray(context, jnp.float64))
        else:
            c = self.transform_rho_to_scaled_interval(jnp.asarray(context, jnp.float64))
        shape = tuple(batch_shape) + c.shape[:-1]
        out = self.model.sample(key, shape, jnp.broadcast_to(c, shape + c.shape[-1:]))
        if self.nf_type == "rho|theta":
            return self.transform_rho_to_physical_interval(out)
        return self.transform_params_to_physical_interval(out)

    # ---- functional forms -----------------------------------------------------------

    def split(self):
        """``(graphdef, params, rest)``: the model's structure, trainable and fixed state."""
        return nnx.split(self.model, nnx.Param, ...)

    def apply(self, params, log10_rho, astro_params=None, jacobian=True):
        """``log_prob`` with the trainable parameters given explicitly (for gradients in them)."""
        graphdef, _, rest = self.split()
        return self._log_prob(nnx.merge(graphdef, params, rest), log10_rho, astro_params, jacobian)

    def log_prob_fn(self, jacobian=True):
        """A pure ``(log10_rho, astro_params) -> log p`` with the parameters as constants."""
        return partial(self._log_prob, self.model, jacobian=jacobian)

    # ---- persistence -----------------------------------------------------------------

    def _config(self):
        return dict(format=_FORMAT, architecture=self.model.architecture(), nf_type=self.nf_type,
                    n_params=self.n_params, mapping=self.mapping, **self.metadata)

    def save(self, directory):
        """Write ``flow_params.npz`` and ``flow_config.json``."""
        os.makedirs(directory, exist_ok=True)
        np.savez(os.path.join(directory, PARAMS_FILE), **_flat_params(self.model))
        with open(os.path.join(directory, CONFIG_FILE), "w") as fh:
            json.dump(_jsonable(self._config()), fh, indent=2)

    @classmethod
    def load(cls, directory, params=PARAMS_FILE):
        """Load :meth:`save` output or a :class:`FlaxFlowTrainer` checkpoint (``params='flow_500steps.npz'``)."""
        with open(os.path.join(directory, CONFIG_FILE)) as fh:
            config = json.load(fh)
        if config.pop("format", None) != _FORMAT:
            raise ValueError(f"{directory} does not hold a flax flow")
        model = make_flax_flow(config.pop("architecture"))
        path = os.path.join(directory, params)
        with np.load(path) as data:
            _set_params(model, {k: data[k] for k in data.files}, path)
        mapping = config.pop("mapping")
        nf_type, n_params = config.pop("nf_type"), config.pop("n_params")
        config.pop("B", None)
        return cls(model, nf_type, mapping, n_params, metadata=config)

    @classmethod
    def from_zuko(cls, zuko_flow):
        """Convert a ``zuko_flows.ZukoAstroFlow`` (e.g. from ``ZukoAstroFlow.from_pandora``)."""
        meta = {k: v for k, v in zuko_flow.metadata.items()}
        return cls(ZukoNSF.from_zuko(zuko_flow.flow), zuko_flow.nf_type, zuko_flow.mapping,
                   zuko_flow.n_params, metadata=meta)


########################################################################################
# numpyro
########################################################################################

def _numpyro_distribution():
    import numpyro.distributions as dist
    from numpyro.distributions import constraints

    class FlowDistribution(dist.Distribution):
        """
        A :class:`FlaxAstroFlow` as a numpyro distribution over its modelled variable.

        ``context`` is ``theta`` for ``'rho|theta'`` flows (the distribution is
        over ``log10_rho``), ``log10_rho`` for ``'theta|rho'``, and ``None`` for
        ``'rho'``. Its leading dimensions are the batch shape.
        """

        support = constraints.real_vector

        def __init__(self, flow, context=None, validate_args=None):
            self.flow = flow
            self.context = None if context is None else jnp.asarray(context, jnp.float64)
            event = (flow.n_params,) if flow.nf_type == "theta|rho" else (flow.n_freqs,)
            batch = () if self.context is None else self.context.shape[:-1]
            super().__init__(batch_shape=batch, event_shape=event, validate_args=validate_args)

        def log_prob(self, value):
            if self.flow.nf_type == "rho":
                return self.flow.log_prob(value)
            if self.flow.nf_type == "rho|theta":
                return self.flow.log_prob(value, self.context)
            return self.flow.log_prob(self.context, value)

        def sample(self, key, sample_shape=()):
            if self.context is None:
                return self.flow.sample(key, tuple(sample_shape) + self.batch_shape)
            return self.flow.sample(key, tuple(sample_shape), self.context)

    return FlowDistribution


def __getattr__(name):
    # numpyro is imported on first use of FlowDistribution
    if name == "FlowDistribution":
        cls = _numpyro_distribution()
        globals()["FlowDistribution"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


########################################################################################
# Training (FlowTrainer in JAX)
########################################################################################

@jax.jit
def _train_chunk(model, optimizer, x_data, c_data, x_idx, c_idx):
    """
    Adam steps over a chunk of batches as one ``lax.scan`` on the device (Pelargir's loop).

    ``x_idx`` / ``c_idx`` are ``BatchSampler.indices`` tuples stacked over the
    chunk, ``(n_steps, batch)`` each; rows are gathered from the device-resident
    ``x_data`` / ``c_data`` inside the scan. Returns the updated model and
    optimizer and the per-step losses.
    """
    def step(carry, idx):
        m, opt = carry
        xi, ci = idx
        xb = gather(x_data, xi)
        cb = None if c_data is None else gather(c_data, ci)
        loss, grads = nnx.value_and_grad(lambda mm: -jnp.mean(mm.log_prob(xb, cb)))(m)
        opt.update(m, grads)
        return (m, opt), loss

    (model, optimizer), losses = jax.lax.scan(step, (model, optimizer), (x_idx, c_idx))
    return model, optimizer, losses


def _stack_indices(batches, which):
    """Stack one side (0: inputs, 1: context) of per-step index tuples into ``(n_steps, B)`` arrays."""
    if batches[0][which] is None:
        return None
    return tuple(np.stack([b[which][k] for b in batches]) for k in range(len(batches[0][which])))


@partial(nnx.jit, static_argnums=(3,))
def _sample_rows(model, key, c, ndraws):
    """``ndraws`` samples for each row of ``c`` (or unconditional): ``(rows, F, ndraws)``."""
    if c is None:
        return model.sample(key, (ndraws,)).T[None]
    keys = jax.random.split(key, c.shape[0])
    out = jax.vmap(lambda k, ci: model.sample(k, (ndraws,), jnp.broadcast_to(ci, (ndraws,) + ci.shape)))(keys, c)
    return jnp.swapaxes(out, 1, 2)


class FlaxFlowTrainer:
    """
    Train a flax flow on normalized samples (``zuko_flows.FlowTrainer`` in JAX).

    Parameters
    ----------
    inputs, context, save_dir, B
        As for ``FlowTrainer``.
    architecture : {'zuko_nsf', 'coupling_rqs'}
    arch_kwargs : dict, optional
        Constructor arguments of the architecture (``bins``,
        ``hidden_features``, ``transforms`` for ``zuko_nsf``; ``num_layers``,
        ``hidden_sizes``, ``num_bins``, ``spline_range`` for ``coupling_rqs``).
    model : ZukoNSF or CouplingRQSFlow, optional
        Continue training this model.
    metadata : dict, optional
        Extra ``flow_config.json`` entries (what :class:`FlaxAstroFlow`
        needs to load a checkpoint).
    seed : int
        Initializes the weights and the validation sampling key.
    """

    def __init__(self, inputs, context, save_dir, B, architecture="zuko_nsf", arch_kwargs=None,
                 model=None, metadata=None, seed=0):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.B = B
        # the data live on the device; only batch indices are drawn on the host
        self.inputs = jnp.asarray(inputs, dtype=jnp.float64)
        self.context = None if context is None else jnp.asarray(context, dtype=jnp.float64)
        n_context = 0 if self.context is None else self.context.shape[-1]
        if model is None:
            arch = dict(kind=architecture, features=self.inputs.shape[-1], context=n_context,
                        **(arch_kwargs or {}))
            model = make_flax_flow(arch, seed=seed)
        self.model = model
        self.key = jax.random.key(seed + 1)
        self.config = dict(format=_FORMAT, architecture=model.architecture(), B=B,
                           **(metadata or {}))
        with open(os.path.join(save_dir, CONFIG_FILE), "w") as fh:
            json.dump(_jsonable(self.config), fh, indent=2)

    def _checkpoint(self, step):
        path = os.path.join(self.save_dir, f"flow_{step}steps.npz")
        np.savez(path, **_flat_params(self.model))
        return path

    def _validation_samples(self, val_context, ndraws):
        self.key, key = jax.random.split(self.key)
        out = _sample_rows(self.model, key, val_context, ndraws)
        return np.asarray(out).reshape(-1, ndraws)

    def train(self, steps, batch_size, save_freq, mode="diagonal", repeat_input=False,
              repeat_context=False, validation=None, val_hist_bins=15, val_ndraws=100_000,
              hell_threshold=0.02, hell_q=0.158, patience=3, learning_rate=1e-4,
              seed=None, plot=True, progress_bar=True, steps_per_call=None):
        """
        Adam on ``-mean(log p(x | c))``, with ``FlowTrainer.train``'s arguments and history.

        Steps run on the device in chunks, one jitted ``lax.scan`` per chunk:
        a chunk ends at the next checkpoint (or after ``steps_per_call``
        steps), so the host only syncs at chunk ends. The batches are the
        same as the zuko trainer's: their indices are drawn on the host from
        the same random stream and gathered on the device.

        If a chunk produces a non-finite loss (the JAX analogue of zuko's
        spline ``AssertionError``), the last checkpoint is reloaded, the
        optimizer restarted and the chunk's steps dropped from the history;
        before the first checkpoint it raises. (The torch trainer drops only
        the failing step.)

        The trained model is ``self.model``.
        """
        sampler = BatchSampler(self.inputs, self.context, batch_size, mode=mode,
                               repeat_input=repeat_input, repeat_context=repeat_context,
                               rng=random.Random(seed))
        optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate), wrt=nnx.Param)
        history = dict(loss=[], checkpoints=[], hellinger=[], ll=[], ul=[], decisions=[],
                       stopped_at=None)

        if validation is not None:
            val_input = np.asarray(validation[0])
            val_context = None if validation[1] is None else jnp.asarray(validation[1], jnp.float64)
            lower, upper = val_input.min(axis=-1), val_input.max(axis=-1)
            val_hists = batched_histogram(val_input, lower, upper, val_hist_bins)
            monitor = HellingerMonitor(q=hell_q, threshold=hell_threshold)
            agreements = 0

        def is_checkpoint(s):
            return (not s % save_freq and s) or s == steps - 1

        def next_checkpoint(s):
            nxt = max(s, 1)
            nxt = -(-nxt // save_freq) * save_freq
            return min(nxt, steps - 1)

        cap = steps if steps_per_call is None else max(int(steps_per_call), 1)
        losses, last_path = [], None
        pbar = trange(steps, colour="blue") if progress_bar else None
        step = 0
        while step < steps:
            end = min(next_checkpoint(step), step + cap - 1)
            batches = [sampler.indices() for _ in range(end - step + 1)]
            self.model, optimizer, chunk = _train_chunk(
                self.model, optimizer, self.inputs, self.context,
                _stack_indices(batches, 0), _stack_indices(batches, 1))
            chunk = np.asarray(chunk)
            if pbar is not None:
                pbar.update(end - step + 1)
            if not np.all(np.isfinite(chunk)):
                if last_path is None:
                    raise RuntimeError(f"non-finite loss in steps {step}-{end}, before the first "
                                       "checkpoint; lower the learning rate or checkpoint more often")
                print(f"Non-finite loss in steps {step}-{end}; reloading {os.path.basename(last_path)}")
                with np.load(last_path) as data:
                    _set_params(self.model, {k: data[k] for k in data.files}, last_path)
                optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate), wrt=nnx.Param)
                step = end + 1
                continue
            losses.append(chunk)

            if is_checkpoint(end):
                last_path = self._checkpoint(end)
                history["checkpoints"].append((end, last_path))
                if validation is not None:
                    gen = self._validation_samples(val_context, int(val_ndraws))
                    hell = hellinger(val_hists, batched_histogram(gen, lower, upper, val_hist_bins))
                    decision = monitor.update(hell)
                    history["decisions"].append(decision)
                    if plot:
                        monitor.plot(os.path.join(self.save_dir, "Hell.pdf"))
                    if decision:
                        agreements += 1
                    if agreements > patience:
                        print("Stopping the training early based on Hellinger distances.")
                        history["stopped_at"] = end
                        break
            step = end + 1
        if pbar is not None:
            pbar.close()

        history["loss"] = np.concatenate(losses) if losses else np.empty(0)
        if validation is not None:
            history.update(hellinger=monitor.hell, ll=monitor.ll, ul=monitor.ul)
        return history
