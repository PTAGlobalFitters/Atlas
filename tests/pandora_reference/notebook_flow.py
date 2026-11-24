"""Pandora's notebook flow-training loop, copied for identity tests.

From ``examples/AstroInferenceUpdated.ipynb`` (pandora commit ebd614f), cells
34 and 37. Module-level settings became arguments, ``.cuda()`` became a
``device`` argument, the step count is an argument instead of ``int(1e4)``,
the progress printing is dropped, and the flow is returned instead of
pickled. Nothing else is changed.

The notebook sets torch's default dtype to float64 before this runs (cell
33); so does importing ``pandora.nf_dist``, which the tests do first.
"""
import random

import numpy as np
import torch
import zuko


def train_notebook_flow(chain_rho, chain_ast, steps, batch_size, lr, hidden, device="cpu"):
    NUM_FREQS = chain_rho.shape[-1]
    n_pars = chain_ast.shape[-1]

    # ---- cell 34
    total_sample_size_data = chain_rho.shape[1]
    total_sample_size_cont = chain_ast.shape[0]

    data = torch.tensor(chain_rho.reshape(total_sample_size_data * total_sample_size_cont, NUM_FREQS))
    contx = torch.tensor(np.repeat(chain_ast[:, None], total_sample_size_data, axis = 1).reshape(data.shape[0], n_pars))

    num_features = data.shape[-1]
    context_features = contx.shape[-1]

    # ---- cell 37
    flow = zuko.flows.spline.NSF(
        num_features,
        context_features,
        bins=8,  # Number of bins for the spline
        passes=2,  # Number of passes (2 for coupling)
        hidden_features = hidden
    ).to(device)

    total_size = data.shape[0]
    data, contx = data.to(device), contx.to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    for epoch in range(steps):
        optimizer.zero_grad()

        rand_sample = random.sample(range(total_size), k = batch_size)

        dist = flow(contx[rand_sample])
        loss = -dist.log_prob(data[rand_sample]).mean()
        loss.backward()
        optimizer.step()
    return flow
