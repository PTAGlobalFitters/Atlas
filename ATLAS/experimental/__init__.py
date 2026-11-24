"""Code that is not reachable from any entry point, kept deliberately.

These modules are complete and substantial but wired to nothing. They are
parked here so that reading the live package does not mean reading them, and
so that their undeclared dependencies (torch, haiku, distrax, optax -- see the
``experimental`` extra in pyproject.toml) are not implied to be requirements of
ATLAS proper.

``flows.py`` and ``cw_astro.py`` together are ~4,100 lines: normalising flows
over GWB Fourier coefficients, SMBH-binary population draws, CW antenna
patterns.
Stage 7 of the development plan promotes them back deliberately, by replacing
the Gaussian ``-0.5 g^T phi_G^-1 g`` prior term inside
``partial_marg_lnposterior`` with ``log p_flow(g | theta_pop)``.

``inverse_wishart.py`` is an alternative prior on the cross-pulsar covariance.

``astro.py`` is the port of Pandora's hierarchical astrophysical inference
(holodeck training sets so far; needs the ``astro`` extra).

Nothing here is imported by the sampling path, and nothing here should be.
"""
