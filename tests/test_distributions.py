"""Unit tests for src/distributions.py (closed-form CDFs and VaR formulas)."""

import numpy as np
import pytest

from src.distributions import (
    dclass_cdf,
    dclass_h,
    dclass_quantile,
    pareto_cdf,
    pareto_var,
    stpetersburg_pmf,
    stpetersburg_var,
)


def test_pareto_cdf_boundary_and_monotone():
    x = np.linspace(1.0, 100.0, 50)
    f = pareto_cdf(x, alpha=0.8)
    assert f[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(f) >= 0)
    assert np.all((f >= 0) & (f <= 1))


def test_pareto_var_matches_cdf_inverse():
    alpha = 0.8
    p = np.array([0.1, 0.5, 0.9, 0.99])
    x = pareto_var(p, alpha)
    f = pareto_cdf(x, alpha)
    assert np.allclose(f, p, atol=1e-8)


def test_pareto_var_increasing_in_p():
    p = np.linspace(0.01, 0.99, 50)
    var = pareto_var(p, alpha=0.7)
    assert np.all(np.diff(var) > 0)


def test_pareto_var_rejects_bad_p():
    with pytest.raises(AssertionError):
        pareto_var(np.array([0.0, 0.5]), alpha=0.8)
    with pytest.raises(AssertionError):
        pareto_var(np.array([1.0]), alpha=0.8)


def test_stpetersburg_pmf_sums_to_one():
    k = np.arange(1, 60)
    total = stpetersburg_pmf(k).sum()
    assert total == pytest.approx(1.0, abs=1e-9)


def test_stpetersburg_var_known_values():
    # F(2^m) = 1 - 2^-m; VaR_{0.55}(X) = 4 because 1-2^-1=0.5 < 0.55 <= 1-2^-2=0.75
    assert stpetersburg_var(np.array([0.55]))[0] == pytest.approx(4.0)
    # VaR just above 0.5 should jump to the next power of two (step function)
    assert stpetersburg_var(np.array([0.5 + 1e-9]))[0] == pytest.approx(4.0)
    assert stpetersburg_var(np.array([0.5]))[0] == pytest.approx(2.0)


def test_stpetersburg_var_is_step_function_of_powers_of_two():
    p = np.linspace(0.001, 0.999, 500)
    var = stpetersburg_var(p)
    log2_var = np.log2(var)
    assert np.allclose(log2_var, np.round(log2_var))


def test_dclass_h_subadditive_small_grid():
    L, c = 1.0, 1.0 / 500.0
    grid = np.linspace(0.01, 20.0, 60)
    t_mesh, s_mesh = np.meshgrid(grid, grid)
    lhs = dclass_h(t_mesh + s_mesh, L, c)
    rhs = dclass_h(t_mesh, L, c) + dclass_h(s_mesh, L, c)
    assert np.all(lhs <= rhs + 1e-9)


def test_dclass_h_increasing():
    L, c = 10.0, 0.2
    t = np.linspace(0.01, 100.0, 500)
    h = dclass_h(t, L, c)
    assert np.all(np.diff(h) >= -1e-12)


def test_dclass_cdf_is_valid_cdf():
    L, c = 1.0, 1.0 / 500.0
    x = np.geomspace(1e-3, 1e4, 200)
    f = dclass_cdf(x, L, c)
    assert np.all((f >= 0) & (f <= 1))
    assert np.all(np.diff(f) >= -1e-12)


def test_dclass_quantile_matches_cdf_inverse():
    L, c = 10.0, 1.0 / 5.0
    p = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    x = dclass_quantile(p, L, c)
    f = dclass_cdf(x, L, c)
    assert np.allclose(f, p, atol=1e-6)


def test_dclass_quantile_rejects_invalid_params():
    with pytest.raises(AssertionError):
        dclass_quantile(np.array([0.5]), L=1.0, c=2.0)  # requires c < L
