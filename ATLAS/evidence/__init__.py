"""Bayesian evidence for ATLAS models.

``ti``  thermodynamic integration with fixed-beta chains on the partially
        marginalised likelihood (``model_maker(..., beta=...)``): estimate
        ln Z, its errors, and an adaptive beta schedule (n(beta) ~ sigma(beta),
        Malnati, Kralj & Olum 2026, arXiv:2610.24354).
"""
from ATLAS.evidence.ti import (TIPoint, TIResult, ti_estimate, propose_betas,
                               initial_betas, bayes_factor, run_fixed_beta)

__all__ = ["TIPoint", "TIResult", "ti_estimate", "propose_betas", "initial_betas",
           "bayes_factor", "run_fixed_beta"]
