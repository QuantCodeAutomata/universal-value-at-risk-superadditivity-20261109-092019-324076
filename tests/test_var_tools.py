"""Unit tests for src/var_tools.py (empirical VaR and comparison tables)."""

import numpy as np
import pytest

from src.var_tools import build_comparison_table, check_superadditivity, empirical_var


def test_empirical_var_matches_known_distribution():
    rng = np.random.default_rng(0)
    # Uniform(0,1): VaR_p = p
    samples = rng.uniform(0.0, 1.0, size=2_000_000)
    p_grid = np.array([0.1, 0.5, 0.9])
    var = empirical_var(samples, p_grid)
    assert np.allclose(var, p_grid, atol=5e-3)


def test_empirical_var_monotone_in_p():
    rng = np.random.default_rng(1)
    samples = rng.standard_normal(100_000)
    p_grid = np.linspace(0.01, 0.99, 20)
    var = empirical_var(samples, p_grid)
    assert np.all(np.diff(var) > 0)


def test_empirical_var_single_point():
    samples = np.array([5.0])
    var = empirical_var(samples, np.array([0.5]))
    assert var[0] == pytest.approx(5.0)


def test_empirical_var_rejects_empty_samples():
    with pytest.raises(AssertionError):
        empirical_var(np.array([]), np.array([0.5]))


def test_check_superadditivity_flags_violations():
    var_sum = np.array([10.0, 5.0, 8.0])
    var_components = np.array([9.0, 6.0, 8.0])
    flags = check_superadditivity(var_sum, var_components)
    assert list(flags) == [True, False, True]


def test_build_comparison_table_columns_and_gap():
    p = np.array([0.5, 0.9])
    var_sum = np.array([10.0, 20.0])
    var_components = np.array([8.0, 25.0])
    table = build_comparison_table(p, var_sum, var_components, panel=np.array(["A", "A"]))
    assert list(table.columns) == ["p", "panel", "VaR_sum", "VaR_components_sum", "superadditive", "gap"]
    assert table["gap"].tolist() == pytest.approx([2.0, -5.0])
    assert table["superadditive"].tolist() == [True, False]
