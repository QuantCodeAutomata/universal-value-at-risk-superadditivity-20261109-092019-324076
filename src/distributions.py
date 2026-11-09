"""Closed-form distribution and quantile (VaR) formulas used by the paper's examples.

Each function implements the exact analytical formula given in the paper
"Universal Value-at-Risk superadditivity"; no numerical library implements
these bespoke distributions, so they are written from scratch
(custom implementation, paper Section 2 / Section 5 as cited per function).

The symbol VaR_p(X) throughout is the left-continuous inverse of the CDF,
VaR_p(X) = F^{-1}(p) = inf{x in R : F(x) >= p}, as defined in Section 2 of the
paper.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Pareto(alpha) -- paper Section 2, Example 1
# ---------------------------------------------------------------------------

def pareto_cdf(x: np.ndarray, alpha: float) -> np.ndarray:
    """CDF F(x) = 1 - (1/x)^alpha for x >= 1, else 0 (paper Section 2, Example 1)."""
    x = np.asarray(x, dtype=float)
    assert alpha > 0, "alpha must be positive"
    out = np.where(x >= 1.0, 1.0 - np.power(1.0 / np.clip(x, 1.0, None), alpha), 0.0)
    return out


def pareto_var(p: np.ndarray, alpha: float) -> np.ndarray:
    """Analytical VaR_p(X) = (1-p)^(-1/alpha) for Pareto(alpha), p in (0,1).

    Derived by inverting F(x) = 1 - (1/x)^alpha = p for x: x = (1-p)^(-1/alpha).
    """
    p = np.asarray(p, dtype=float)
    assert alpha > 0, "alpha must be positive"
    assert np.all((p > 0) & (p < 1)), "p must lie in the open interval (0, 1)"
    return np.power(1.0 - p, -1.0 / alpha)


# ---------------------------------------------------------------------------
# St. Petersburg lottery -- paper Section 2, Example 2
# ---------------------------------------------------------------------------

def stpetersburg_pmf(k: np.ndarray) -> np.ndarray:
    """P(X = 2^k) = 2^-k for k = 1, 2, 3, ... (paper Section 2, Example 2)."""
    k = np.asarray(k, dtype=float)
    assert np.all(k >= 1), "k must be a positive integer"
    return np.power(2.0, -k)


def stpetersburg_var(p: np.ndarray) -> np.ndarray:
    """Analytical VaR_p(X) = 2^k with k = ceil(-log2(1-p)) for the St. Petersburg lottery.

    The CDF is F(2^m) = sum_{k=1}^m 2^-k = 1 - 2^-m for m >= 1 (a step function),
    so the left-continuous inverse at level p is the smallest m with 1-2^-m >= p,
    i.e. m = ceil(-log2(1-p)) (paper Section 2, Example 2).
    """
    p = np.asarray(p, dtype=float)
    assert np.all((p > 0) & (p < 1)), "p must lie in the open interval (0, 1)"
    k = np.ceil(-np.log2(1.0 - p))
    k = np.maximum(k, 1.0)  # the lottery's smallest outcome is 2^1
    return np.power(2.0, k)


# ---------------------------------------------------------------------------
# D-class counterexample F_{L,c} -- paper Section 5, Example 7
# ---------------------------------------------------------------------------

def dclass_h(t: np.ndarray, L: float, c: float) -> np.ndarray:
    """h_{L,c}(t) = c*k + min(t - k*L, c), k = floor(t/L), for 0 < c < L, t > 0.

    Equal to inf_{m in N_0} {m*c + (t - m*L)_+} (paper Section 5, Example 7);
    piecewise linear, continuous, non-decreasing, and subadditive in t.
    """
    t = np.asarray(t, dtype=float)
    assert 0 < c < L, "the paper requires 0 < c < L"
    assert np.all(t > 0), "t must be positive"
    k = np.floor(t / L)
    return c * k + np.minimum(t - k * L, c)


def dclass_cdf(x: np.ndarray, L: float, c: float) -> np.ndarray:
    """CDF F_{L,c}(x) = exp(-h_{L,c}(1/x)) for x > 0 (paper Section 5, Example 7)."""
    x = np.asarray(x, dtype=float)
    assert np.all(x > 0), "x must be positive"
    return np.exp(-dclass_h(1.0 / x, L, c))


def dclass_quantile(p: np.ndarray, L: float, c: float) -> np.ndarray:
    """Closed-form left-continuous inverse of F_{L,c} at level p in (0, 1).

    F_{L,c}(x) = p  <=>  h_{L,c}(1/x) = -log(p) = v. Since h_{L,c} rises with
    slope 1 from level k*c to (k+1)*c on t in [k*L, k*L + c) and is flat at
    level (k+1)*c on t in [k*L + c, (k+1)*L], the smallest t with h(t) >= v is
    obtained on the rising segment: with k = ceil(v/c) - 1,
    t = k*L + (v - k*c). Then VaR_p(X) = 1/t (custom implementation, paper
    Section 5, Example 7).
    """
    p = np.asarray(p, dtype=float)
    assert 0 < c < L, "the paper requires 0 < c < L"
    assert np.all((p > 0) & (p < 1)), "p must lie in the open interval (0, 1)"
    v = -np.log(p)
    k = np.ceil(v / c) - 1.0
    k = np.maximum(k, 0.0)
    t = k * L + (v - k * c)
    t = np.maximum(t, 1e-300)  # guard against p -> 1 (v -> 0, t -> 0)
    return 1.0 / t
