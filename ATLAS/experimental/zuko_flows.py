"""
Conditional normalizing flows for astrophysical GWB inference, on zuko (torch).

This is the port of Pandora's ``nf_dist.py``. The flows are zuko neural
spline flows (``zuko.flows.spline.NSF`` with coupling transforms,
``passes=2``), trained on the normalized training sets from
:mod:`ATLAS.experimental.astro`:

* :func:`make_zuko_flow` builds the flow Pandora builds.
* :class:`~ATLAS.experimental.flow_utils.BatchSampler` (re-exported here)
  draws training batches the way Pandora does: ``'diagonal'`` and ``'mesh'``
  (``NFMaker.sample_from_dist``) or ``'flat'`` (the training loop in
  ``AstroInferenceUpdated.ipynb``).
* :class:`FlowTrainer` is ``NFMaker``: Adam on the negative log likelihood,
  periodic checkpoints, optional early stopping from Hellinger distances
  between flow samples and a validation set (``ValidationHell``).
* :class:`ZukoAstroFlow` is ``NFastroinference``: it evaluates and samples a
  trained flow in physical units.

Given the same seeds, training reproduces Pandora's weights bit for bit.
Stage 2 of the port reimplements these flows in flax (``flax_flows.py``).

Differences from Pandora
------------------------
* ``ZukoAstroFlow.log_prob`` includes the Jacobian of the map to ``[-B, B]``
  by default, so it is the density of the physical variables. Pandora omits
  it (``jacobian=False``). The term is constant, so it only matters when
  comparing evidences between flows.
* Flows are saved as a ``state_dict`` plus a JSON config instead of a pickled
  module; :meth:`ZukoAstroFlow.from_pandora` reads Pandora's pickles.
* When a spline raises ``AssertionError`` during training, Pandora reloads the
  last checkpoint but keeps an optimizer bound to the discarded module, so
  training silently stops updating the flow (and fails with ``NameError``
  before the first checkpoint). Here the checkpoint is loaded in place and
  the optimizer is rebuilt.
* Batch indices come from a seeded ``random.Random`` instead of the global
  ``random`` module, and ``seed=0`` is honoured.
* Validation histograms use numpy instead of JAX, so JAX does not claim GPU
  memory next to torch, and flow samples are drawn without autograd.
* ``nf_type='rho'`` (unconditional) works; Pandora's ``log_prob`` called
  ``self.nf.log_prob``, which zuko's lazy flows do not have.

Importing this module imports torch and zuko (the ``[astro]`` extra).
"""

import json
import os
import random
from contextlib import contextmanager

import numpy as np
import torch
import zuko
from tqdm.auto import trange

from ATLAS.experimental.flow_utils import (
    NF_TYPES,
    BatchSampler,
    HellingerMonitor,
    batched_histogram,
    hellinger,
)

__all__ = [
    "NF_TYPES",
    "make_zuko_flow",
    "BatchSampler",
    "FlowTrainer",
    "ZukoAstroFlow",
]


CONFIG_FILE = "flow_config.json"
STATE_FILE = "flow_state.pt"


########################################################################################
# Flow construction
########################################################################################

@contextmanager
def _default_dtype(dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def make_zuko_flow(n_features, n_context=0, bins=8, hidden_features=(512, 512),
                   transforms=3, dtype="float64"):
    """
    Pandora's flow: ``zuko.flows.spline.NSF`` with coupling transforms.

    Parameters
    ----------
    n_features, n_context : int
        Dimensions of the modelled variable and of the context
        (0 for an unconditional flow).
    bins : int
        Spline bins.
    hidden_features : sequence of int
        Hidden layers of each coupling network. Pandora's ``NFMaker`` uses
        ``[512] * 2``; ``AstroInferenceUpdated.ipynb`` uses ``[512] * 8``.
    transforms : int
        Number of coupling transforms (zuko's default, 3, in Pandora).
    dtype : str
        Parameter dtype. The flow is built with this as torch's default
        dtype, which is how Pandora initializes it, so the same
        ``torch.manual_seed`` gives the same initial weights.
    """
    with _default_dtype(getattr(torch, dtype)):
        return zuko.flows.spline.NSF(
            int(n_features), int(n_context), bins=int(bins), passes=2,
            hidden_features=[int(h) for h in hidden_features],
            transforms=int(transforms),
        )


def _flow_architecture(flow):
    """Recover :func:`make_zuko_flow` arguments from an NSF instance."""
    layers = list(flow.transform.transforms)
    first = layers[0]
    linears = [m for m in first.hyper if isinstance(m, torch.nn.Linear)]
    n_features = len(first.order)
    n_context = linears[0].in_features - n_features
    # each feature gets bins widths, bins heights and bins - 1 derivatives
    bins = (linears[-1].out_features // n_features + 1) // 3
    return dict(n_features=n_features, n_context=n_context, bins=bins,
                hidden_features=[m.out_features for m in linears[:-1]],
                transforms=len(layers),
                dtype=str(linears[0].weight.dtype).replace("torch.", ""))


########################################################################################
# Training (NFMaker)
########################################################################################

class FlowTrainer:
    """
    Train a zuko NSF on normalized samples (Pandora's ``NFMaker``).

    Parameters
    ----------
    inputs : array_like
        Samples of the modelled variable, already mapped to ``[-B, B]``:
        ``(n_draws, n_dim)`` or ``(n_draws, n_real, n_dim)``.
    context : array_like or None
        Context samples on the same footing, 2D or 3D; ``None`` trains an
        unconditional flow.
    save_dir : str
        Checkpoints and ``flow_config.json`` are written here.
    B : float
        Half-width of the normalized interval; recorded in the config.
    device : str or torch.device, optional
        Default: CUDA if available, else CPU.
    spline_bins, hidden_dims, transforms, dtype
        Passed to :func:`make_zuko_flow`.
    flow : zuko flow, optional
        Continue training this flow instead of building a new one.
    metadata : dict, optional
        Extra entries for ``flow_config.json`` (e.g. what
        :class:`ZukoAstroFlow` needs to load a checkpoint).
    """

    def __init__(self, inputs, context, save_dir, B, device=None, spline_bins=8,
                 hidden_dims=(512, 512), transforms=3, dtype="float64", flow=None,
                 metadata=None):
        self.device = torch.device(device if device is not None else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.B = B
        self.inputs = torch.as_tensor(np.asarray(inputs), device=self.device)
        self.context = (None if context is None
                        else torch.as_tensor(np.asarray(context), device=self.device))
        n_context = 0 if self.context is None else self.context.shape[-1]
        if flow is None:
            flow = make_zuko_flow(self.inputs.shape[-1], n_context, bins=spline_bins,
                                  hidden_features=hidden_dims, transforms=transforms,
                                  dtype=dtype)
        self.flow = flow.to(self.device)
        self.config = dict(architecture=_flow_architecture(self.flow), B=B,
                           **(metadata or {}))
        with open(os.path.join(save_dir, CONFIG_FILE), "w") as fh:
            json.dump(_jsonable(self.config), fh, indent=2)

    def _log_prob(self, x, c):
        return self.flow().log_prob(x) if c is None else self.flow(c).log_prob(x)

    def _checkpoint(self, step):
        path = os.path.join(self.save_dir, f"flow_{step}steps.pt")
        torch.save(self.flow.state_dict(), path)
        return path

    @torch.no_grad()
    def _validation_samples(self, val_context, n_dim, ndraws):
        """Flow samples laid out like the validation input: (n_ctx * n_dim, ndraws)."""
        if val_context is None:
            return self.flow().sample((ndraws,)).T.cpu().numpy()
        out = np.empty((val_context.shape[0], n_dim, ndraws))
        for ii in range(val_context.shape[0]):
            out[ii] = self.flow(val_context[ii]).sample((ndraws,)).T.cpu().numpy()
        return out.reshape(val_context.shape[0] * n_dim, ndraws)

    def train(self, steps, batch_size, save_freq, mode="diagonal", repeat_input=False,
              repeat_context=False, validation=None, val_hist_bins=15, val_ndraws=100_000,
              hell_threshold=0.02, hell_q=0.158, patience=3, learning_rate=1e-4,
              seed=None, plot=True, progress_bar=True):
        """
        Run Adam on ``-mean(log p(x | c))`` (``NFMaker.train``).

        Parameters
        ----------
        steps, batch_size : int
        save_freq : int
            A checkpoint ``flow_{step}steps.pt`` is written at every nonzero
            multiple of ``save_freq`` and at the last step.
        mode, repeat_input, repeat_context
            See :class:`BatchSampler`.
        validation : (val_input, val_context) or None
            Enables early stopping. ``val_input`` is ``(n_ctx * n_dim, n_val)``:
            for each validation context, ``n_val`` held-out samples of each
            dimension, normalized. ``val_context`` is ``(n_ctx, n_context)``,
            or ``None`` for an unconditional flow.
        val_hist_bins, val_ndraws
            Histogram bins and flow samples per validation context.
        hell_threshold, hell_q, patience
            Early stopping: training stops once more than ``patience``
            checkpoints (counted cumulatively, as in Pandora) have Hellinger
            quantiles within ``hell_threshold`` of the previous checkpoint.
        learning_rate : float
        seed : int or None
            Seeds the batch indices. Weight initialization follows
            ``torch.manual_seed``, as in Pandora.
        plot : bool
            Write the Hellinger-distance plot ``Hell.pdf`` at each checkpoint.

        Returns
        -------
        dict
            ``loss`` (per step), ``checkpoints`` (``[(step, path)]``),
            ``hellinger``, ``ll``, ``ul``, ``decisions`` and ``stopped_at``
            (``None`` unless training stopped early).
        """
        sampler = BatchSampler(self.inputs, self.context, batch_size, mode=mode,
                               repeat_input=repeat_input, repeat_context=repeat_context,
                               rng=random.Random(seed))
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=learning_rate)
        history = dict(loss=[], checkpoints=[], hellinger=[], ll=[], ul=[], decisions=[],
                       stopped_at=None)

        if validation is not None:
            val_input = np.asarray(validation[0])
            val_context = (None if validation[1] is None else
                           torch.as_tensor(np.asarray(validation[1]), device=self.device))
            lower, upper = val_input.min(axis=-1), val_input.max(axis=-1)
            val_hists = batched_histogram(val_input, lower, upper, val_hist_bins)
            monitor = HellingerMonitor(q=hell_q, threshold=hell_threshold)
            agreements = 0

        losses = []
        last_path = None
        pbar = trange(steps, colour="blue") if progress_bar else range(steps)
        for step in pbar:
            try:
                optimizer.zero_grad()
                x, c = sampler()
                loss = -(self._log_prob(x, c)).mean()
                loss.backward()
                optimizer.step()
            except AssertionError:
                # zuko's spline raises on degenerate bins
                if last_path is None:
                    raise RuntimeError(
                        f"spline transform failed at step {step}, before the first "
                        "checkpoint; lower the learning rate or checkpoint more often")
                print(f"Spline transform failed at step {step}; "
                      f"reloading {os.path.basename(last_path)}")
                self.flow.load_state_dict(torch.load(last_path, map_location=self.device))
                optimizer = torch.optim.Adam(self.flow.parameters(), lr=learning_rate)
                continue
            losses.append(loss.detach())

            if not step % save_freq and step or step == steps - 1:
                last_path = self._checkpoint(step)
                history["checkpoints"].append((step, last_path))

                if validation is not None:
                    gen = self._validation_samples(val_context, self.inputs.shape[-1],
                                                   int(val_ndraws))
                    hell = hellinger(val_hists,
                                      batched_histogram(gen, lower, upper, val_hist_bins))
                    decision = monitor.update(hell)
                    history["decisions"].append(decision)
                    if plot:
                        monitor.plot(os.path.join(self.save_dir, "Hell.pdf"))
                    if decision:
                        agreements += 1
                    if agreements > patience:
                        print("Stopping the training early based on Hellinger distances.")
                        history["stopped_at"] = step
                        break

        history["loss"] = torch.stack(losses).cpu().numpy() if losses else np.empty(0)
        if validation is not None:
            history.update(hellinger=monitor.hell, ll=monitor.ll, ul=monitor.ul)
        return history


########################################################################################
# Evaluation (NFastroinference)
########################################################################################

class ZukoAstroFlow:
    """
    A trained flow, evaluated in physical units (Pandora's ``NFastroinference``).

    Parameters
    ----------
    flow : zuko flow
    nf_type : {'rho|theta', 'theta|rho', 'rho'}
        ``'rho|theta'``: density of ``log10_rho`` given astrophysical
        parameters (the one used for inference). ``'theta|rho'``: the
        reverse. ``'rho'``: unconditional density of ``log10_rho``.
    mapping : dict
        ``B``, ``mean``, ``half_range`` from
        :meth:`ATLAS.experimental.astro.TrainingSet.normalization`: arrays of
        length ``n_params + n_freqs``, astrophysical parameters first.
    n_params : int
        Number of astrophysical parameters.
    device : str, optional
    metadata : dict, optional
        Kept in the saved config (``param_names``, ``f_conv``, ``tspan``, ...).
    """

    def __init__(self, flow, nf_type, mapping, n_params, device="cpu", metadata=None):
        if nf_type not in NF_TYPES:
            raise ValueError(f"nf_type must be one of {NF_TYPES}, got {nf_type!r}")
        self.device = torch.device(device)
        self.flow = flow.to(self.device).eval()
        self.nf_type = nf_type
        self.B = mapping["B"]
        mean, half = np.asarray(mapping["mean"]), np.asarray(mapping["half_range"])
        p = int(n_params)
        self.n_params = p
        self.mean_ast, self.mean_gwb = mean[:p], mean[p:]
        self.half_range_ast, self.half_range_gwb = half[:p], half[p:]
        self.metadata = dict(metadata or {})

    @property
    def mapping(self):
        return dict(B=self.B, mean=np.concatenate([self.mean_ast, self.mean_gwb]),
                    half_range=np.concatenate([self.half_range_ast, self.half_range_gwb]))

    # ---- physical <-> normalized ----------------------------------------------

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
        return float(np.sum(np.log(self.B / half)))

    def _tensor(self, arr):
        return torch.as_tensor(np.asarray(arr, dtype=np.float64), device=self.device)

    # ---- density and samples --------------------------------------------------

    @torch.no_grad()
    def log_prob(self, log10_rho, astro_params=None, jacobian=True):
        """
        Log density of the modelled variable.

        Parameters
        ----------
        log10_rho : (..., n_freqs) array_like
        astro_params : (..., n_params) array_like
            Required unless ``nf_type='rho'``.
        jacobian : bool
            Include the Jacobian of the map to ``[-B, B]``, giving the
            density in physical units. ``False`` reproduces Pandora.

        Returns
        -------
        ndarray
        """
        scaled_gwb = self._tensor(self.transform_rho_to_scaled_interval(np.asarray(log10_rho)))
        if self.nf_type == "rho":
            lp = self.flow().log_prob(scaled_gwb)
        else:
            scaled_ast = self._tensor(self.transform_params_to_scaled_interval(
                np.asarray(astro_params)))
            if self.nf_type == "rho|theta":
                lp = self.flow(scaled_ast).log_prob(scaled_gwb)
            else:
                lp = self.flow(scaled_gwb).log_prob(scaled_ast)
        lp = lp.cpu().numpy()
        return lp + self.log_jacobian() if jacobian else lp

    @torch.no_grad()
    def sample(self, batch_shape, context=None, seed=None):
        """
        Draw samples in physical units.

        ``batch_shape`` is passed to zuko's ``sample`` (an int is treated as
        ``(n,)``). ``context`` is the astrophysical parameters for
        ``'rho|theta'``, ``log10_rho`` for ``'theta|rho'``, and unused for
        ``'rho'``. ``seed`` seeds torch's generator locally.
        """
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(seed)
            if self.nf_type == "rho":
                out = self.flow().sample(batch_shape).cpu().numpy()
                return self.transform_rho_to_physical_interval(out)
            if self.nf_type == "rho|theta":
                c = self._tensor(self.transform_params_to_scaled_interval(np.asarray(context)))
                out = self.flow(c).sample(batch_shape).cpu().numpy()
                return self.transform_rho_to_physical_interval(out)
            c = self._tensor(self.transform_rho_to_scaled_interval(np.asarray(context)))
            out = self.flow(c).sample(batch_shape).cpu().numpy()
            return self.transform_params_to_physical_interval(out)

    # ---- persistence ----------------------------------------------------------

    def _config(self):
        return dict(architecture=_flow_architecture(self.flow), nf_type=self.nf_type,
                    n_params=self.n_params, mapping=self.mapping,
                    torch_version=torch.__version__, zuko_version=zuko.__version__,
                    **self.metadata)

    def save(self, directory):
        """Write ``flow_state.pt`` (``state_dict``) and ``flow_config.json``."""
        os.makedirs(directory, exist_ok=True)
        torch.save(self.flow.state_dict(), os.path.join(directory, STATE_FILE))
        with open(os.path.join(directory, CONFIG_FILE), "w") as fh:
            json.dump(_jsonable(self._config()), fh, indent=2)

    @classmethod
    def load(cls, directory, device="cpu", state=STATE_FILE):
        """
        Load a flow saved by :meth:`save` or a :class:`FlowTrainer` directory.

        ``state`` selects the weights file, e.g. a checkpoint
        ``'flow_500steps.pt'``.
        """
        with open(os.path.join(directory, CONFIG_FILE)) as fh:
            config = json.load(fh)
        flow = make_zuko_flow(**config.pop("architecture"))
        flow.load_state_dict(torch.load(os.path.join(directory, state), map_location=device))
        mapping = config.pop("mapping")
        mapping = dict(B=mapping["B"], mean=np.asarray(mapping["mean"]),
                       half_range=np.asarray(mapping["half_range"]))
        nf_type, n_params = config.pop("nf_type"), config.pop("n_params")
        for key in ("B", "torch_version", "zuko_version"):
            config.pop(key, None)
        return cls(flow, nf_type, mapping, n_params, device=device, metadata=config)

    @classmethod
    def from_pandora(cls, pickle_path, mapping_path, nf_type="rho|theta", device="cpu",
                     metadata=None):
        """
        Load a flow pickled by Pandora.

        Parameters
        ----------
        pickle_path : str
            ``torch.save([flow, B])`` from ``NFMaker`` or ``torch.save([flow])``
            from ``AstroInferenceUpdated.ipynb``. Unpickling runs code from
            the file, so only load files you trust.
        mapping_path : str
            Pandora's ``*_mapping_data.npy.npz`` (``B``, ``mean``, ``half_range``).
        nf_type : str

        The flow is rebuilt from its inferred architecture and the pickled
        weights, so it can be re-saved with :meth:`save`.
        """
        obj = torch.load(pickle_path, map_location=device, weights_only=False)
        pickled = obj[0] if isinstance(obj, (list, tuple)) else obj
        with np.load(mapping_path) as m:
            mapping = dict(B=m["B"][()], mean=m["mean"], half_range=m["half_range"])
        if isinstance(obj, (list, tuple)) and len(obj) > 1 and obj[1] != mapping["B"]:
            raise ValueError(f"pickle has B={obj[1]} but the mapping has B={mapping['B']}")
        arch = _flow_architecture(pickled)
        flow = make_zuko_flow(**arch)
        flow.load_state_dict(pickled.state_dict())
        n_total = len(mapping["mean"])
        n_params = arch["n_features"] if nf_type == "theta|rho" else n_total - arch["n_features"]
        return cls(flow, nf_type, mapping, n_params, device=device,
                   metadata=dict(metadata or {}, source=os.path.abspath(pickle_path)))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
