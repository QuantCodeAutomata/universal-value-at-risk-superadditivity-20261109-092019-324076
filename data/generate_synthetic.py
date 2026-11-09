"""Synthetic sample generation for the Value-at-Risk superadditivity experiments.

Every experiment in this repository is purely simulation-based (the paper
"Universal Value-at-Risk superadditivity" uses no market data for these
examples); this module is the single place where random draws are produced,
via the inverse-CDF method U -> F^{-1}(U) with the closed-form quantiles in
src/distributions.py.
"""

from __future__ import annotations

import numpy as np

from src.distributions import dclass_quantile, pareto_var, stpetersburg_var


def generate_pareto_samples(n: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """n iid draws from Pareto(alpha) via X = U^(-1/alpha), U ~ Uniform(0,1) (paper Section 2, Example 1)."""
    assert n > 0, "n must be positive"
    u = rng.uniform(0.0, 1.0, size=n)
    return pareto_var(u, alpha)


def generate_stpetersburg_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    """n iid draws from the St. Petersburg lottery via X = 2^ceil(-log2(U)) (paper Section 2, Example 2)."""
    assert n > 0, "n must be positive"
    u = rng.uniform(0.0, 1.0, size=n)
    return stpetersburg_var(u)


def generate_dclass_samples(n: int, L: float, c: float, rng: np.random.Generator) -> np.ndarray:
    """n iid draws from F_{L,c} via the closed-form inverse CDF (paper Section 5, Example 7)."""
    assert n > 0, "n must be positive"
    u = rng.uniform(0.0, 1.0, size=n)
    return dclass_quantile(u, L, c)
