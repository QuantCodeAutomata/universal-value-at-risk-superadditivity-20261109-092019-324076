"""Shared Value-at-Risk computation and comparison utilities.

VaR_p(X) is the left-continuous inverse of the CDF (paper Section 2):
VaR_p(X) = inf{x in R : F(x) >= p}. For a Monte Carlo sample of X, VaR_p(X)
is estimated by the empirical p-quantile of the sample (numpy.quantile),
which is the standard simulation-based estimator and the one the paper's
implementation notes point to ("use numpy.quantile or numpy.percentile").
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def empirical_var(samples: np.ndarray, p_grid: np.ndarray) -> np.ndarray:
    """Empirical VaR_p of ``samples`` at every level in ``p_grid`` (linear interpolation)."""
    samples = np.asarray(samples, dtype=float)
    p_grid = np.asarray(p_grid, dtype=float)
    assert samples.size > 0, "samples must be non-empty"
    assert np.all((p_grid > 0) & (p_grid < 1)), "p_grid must lie in (0, 1)"
    return np.quantile(samples, p_grid)


def check_superadditivity(var_sum: np.ndarray, var_components_sum: np.ndarray) -> np.ndarray:
    """Boolean array: True where VaR_p(sum) >= sum of VaR_p(components), i.e. UVS holds (Eq. 1)."""
    return np.asarray(var_sum) >= np.asarray(var_components_sum)


def build_comparison_table(
    p_grid: np.ndarray,
    var_sum: np.ndarray,
    var_components_sum: np.ndarray,
    panel: np.ndarray,
) -> pd.DataFrame:
    """Tidy comparison table with columns p, panel, VaR_sum, VaR_components_sum, superadditive, gap."""
    p_grid = np.asarray(p_grid, dtype=float)
    var_sum = np.asarray(var_sum, dtype=float)
    var_components_sum = np.asarray(var_components_sum, dtype=float)
    superadditive = check_superadditivity(var_sum, var_components_sum)
    return pd.DataFrame({
        "p": p_grid,
        "panel": panel,
        "VaR_sum": var_sum,
        "VaR_components_sum": var_components_sum,
        "superadditive": superadditive,
        "gap": var_sum - var_components_sum,
    })
