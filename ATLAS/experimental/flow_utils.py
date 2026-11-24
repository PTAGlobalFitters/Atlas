"""
Torch-free pieces shared by the zuko and flax flow trainers.

* :class:`BatchSampler` draws training batches exactly as Pandora does
  (``NFMaker.sample_from_dist`` and the notebook loop), on numpy, jax or
  torch arrays.
* :func:`batched_histogram`, :func:`hellinger` and :class:`HellingerMonitor`
  implement Pandora's Hellinger-distance early stopping (``ValidationHell``).
"""

import random

import numpy as np

__all__ = ["NF_TYPES", "BatchSampler", "gather", "batched_histogram", "hellinger",
           "HellingerMonitor"]

# 'rho|theta': density of log10_rho given the astrophysical parameters (used for
# inference); 'theta|rho': the reverse; 'rho': unconditional density of log10_rho.
NF_TYPES = ("rho|theta", "theta|rho", "rho")


########################################################################################
# Batches
########################################################################################

class BatchSampler:
    """
    Training batches, drawn as Pandora draws them.

    ``inputs`` and ``context`` are the training tensors. Either may be 3D
    ``(n_draws, n_real, n_dim)`` (holodeck-style: several realizations per
    parameter draw) or 2D ``(n_draws, n_dim)``; ``context`` may be ``None``.

    Modes
    -----
    ``'diagonal'`` and ``'mesh'``
        ``NFMaker.sample_from_dist``. Diagonal draws ``batch_size`` distinct
        draws and ``batch_size`` distinct realizations and pairs them up.
        Mesh takes all ``batch_size**2`` combinations; with a 2D array,
        ``repeat_input`` / ``repeat_context`` repeat each row ``batch_size``
        times to match.
    ``'flat'``
        The loop in ``AstroInferenceUpdated.ipynb``: ``batch_size`` distinct
        rows of the draws x realizations array flattened row-major, without
        building the flattened array.

    Indices come from ``rng``, a ``random.Random``. ``Random(s)`` gives the
    same stream as Pandora's ``random.seed(s)`` on the global generator.
    """

    MODES = ("diagonal", "mesh", "flat")

    def __init__(self, inputs, context, batch_size, mode="diagonal", repeat_input=False,
                 repeat_context=False, rng=None):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.inputs, self.context = inputs, context
        self.batch_size = int(batch_size)
        self.mode = mode
        self.repeat_input, self.repeat_context = repeat_input, repeat_context
        self.rng = random.Random() if rng is None else rng
        if mode == "flat":
            grid = [a for a in (inputs, context) if a is not None and a.ndim == 3]
            if not grid:
                raise ValueError("mode='flat' needs a 3D (n_draws, n_real, dim) array")
            self._n_draws, self._n_real = grid[0].shape[:2]

    def __call__(self):
        """Return ``(inputs, context)`` for one batch (``context`` may be ``None``)."""
        x_idx, c_idx = self.indices()
        return (gather(self.inputs, x_idx),
                None if self.context is None else gather(self.context, c_idx))

    def indices(self):
        """
        Draw one batch's row indices without gathering them.

        Returns ``(x_idx, c_idx)``: tuples of integer arrays that select the
        batch by advanced indexing (see :func:`gather`); ``c_idx`` is ``None``
        without context. Only the arrays' shapes are used, so device arrays
        never leave the device. The random stream is consumed in the order
        Pandora consumes it.
        """
        if self.mode == "flat":
            return self._flat()
        if self.mode == "mesh":
            return self._mesh()
        return self._diagonal()

    def _flat(self):
        rows = self.rng.sample(range(self._n_draws * self._n_real), k=self.batch_size)
        draws, reals = np.divmod(np.asarray(rows), self._n_real)
        pick = lambda a: (draws, reals) if a.ndim == 3 else (draws,)  # noqa: E731
        return pick(self.inputs), None if self.context is None else pick(self.context)

    def _mesh(self):
        b, x, c, rng = self.batch_size, self.inputs, self.context, self.rng
        i0 = np.asarray(rng.sample(range(x.shape[0]), k=b))
        if x.ndim == 3:
            # x[np.ix_(i0, i1)].reshape(b**2, dim), row-major
            i1 = np.asarray(rng.sample(range(x.shape[1]), k=b))
            x_idx = (np.repeat(i0, b), np.tile(i1, b))
        else:
            x_idx = (np.repeat(i0, b),) if self.repeat_input else (i0,)
        if c is None:
            return x_idx, None
        if c.ndim == 3:
            ci = np.asarray(rng.sample(range(c.shape[1]), k=b))
            c_idx = (np.repeat(i0, b), np.tile(ci, b))
        else:
            c_idx = (np.repeat(i0, b),) if self.repeat_context else (i0,)
        return x_idx, c_idx

    def _diagonal(self):
        b, x, c, rng = self.batch_size, self.inputs, self.context, self.rng
        i0 = np.asarray(rng.sample(range(x.shape[0]), k=b))
        if x.ndim == 3:
            i1 = np.asarray(rng.sample(range(x.shape[1]), k=b))
            x_idx = (i0, i1)
        else:
            x_idx = (i0,)
        if c is None:
            return x_idx, None
        if c.ndim == 3:
            ci = np.asarray(rng.sample(range(c.shape[1]), k=b))
            return x_idx, (i0, ci)
        return x_idx, (i0,)


def gather(data, idx):
    """Rows of ``data`` (numpy, jax or torch) selected by an index tuple from ``BatchSampler.indices``."""
    return data[tuple(idx)]


########################################################################################
# Hellinger-distance validation (ValidationHell)
########################################################################################

def batched_histogram(samples, lower, upper, bins):
    """Row-wise ``np.histogram`` of ``samples`` (rows, n) on per-row ranges."""
    samples = np.asarray(samples)
    return np.stack([np.histogram(row, bins=bins, range=(lo, hi))[0]
                     for row, lo, hi in zip(samples, np.asarray(lower), np.asarray(upper))])


def hellinger(hist1, hist2):
    """Hellinger distance between matching rows of two histogram arrays."""
    p = hist1 / hist1.sum(axis=-1)[:, None]
    q = hist2 / hist2.sum(axis=-1)[:, None]
    return 1 / np.sqrt(2) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2, axis=-1))


class HellingerMonitor:
    """
    ``ValidationHell``'s bookkeeping and decision rule.

    Each checkpoint adds the Hellinger distances between flow samples and
    the validation set, and their ``q`` and ``1 - q`` quantiles rounded to two
    decimals. From the third checkpoint on, a checkpoint "agrees" when both
    quantiles moved by less than ``threshold`` since the previous one.
    """

    def __init__(self, q=0.158, threshold=0.02):
        self.q, self.threshold = q, threshold
        self.hell, self.ll, self.ul = [], [], []

    def update(self, hell):
        self.hell.append(np.asarray(hell))
        self.ll.append(np.round(np.quantile(hell, q=self.q), 2))
        self.ul.append(np.round(np.quantile(hell, q=1 - self.q), 2))
        if len(self.hell) <= 2:
            return None
        return bool(np.abs(self.ll[-1] - self.ll[-2]) < self.threshold
                    and np.abs(self.ul[-1] - self.ul[-2]) < self.threshold)

    def plot(self, path):
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for ct, hell in enumerate(self.hell):
            ax.hist(hell, range=(0, 1), bins=40, histtype="step", lw=3, density=True,
                    label=f"Checkpoint {ct + 1}; ll = {self.ll[ct]}; ul = {self.ul[ct]}")
        ax.legend()
        ax.set_ylabel("Count")
        ax.set_xlabel("Hellinger Distances")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
