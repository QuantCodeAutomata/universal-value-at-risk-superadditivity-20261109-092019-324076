"""Unit tests for data/generate_synthetic.py sample generators."""

import numpy as np
import pytest

from data.generate_synthetic import (
    generate_dclass_samples,
    generate_pareto_samples,
    generate_stpetersburg_samples,
)
from src.distributions import pareto_var, stpetersburg_var


def test_generate_pareto_samples_support_and_quantile():
    rng = np.random.default_rng(42)
    samples = generate_pareto_samples(200_000, alpha=0.8, rng=rng)
    assert np.all(samples >= 1.0)
    empirical = np.quantile(samples, 0.9)
    theoretical = pareto_var(np.array([0.9]), alpha=0.8)[0]
    assert empirical == pytest.approx(theoretical, rel=0.05)


def test_generate_stpetersburg_samples_are_powers_of_two():
    rng = np.random.default_rng(1)
    samples = generate_stpetersburg_samples(100_000, rng=rng)
    log2_samples = np.log2(samples)
    assert np.allclose(log2_samples, np.round(log2_samples))
    assert np.all(samples >= 2.0)


def test_generate_stpetersburg_samples_quantile_matches_formula():
    rng = np.random.default_rng(2)
    samples = generate_stpetersburg_samples(500_000, rng=rng)
    empirical = np.quantile(samples, 0.55)
    theoretical = stpetersburg_var(np.array([0.55]))[0]
    assert empirical == pytest.approx(theoretical, rel=0.2)


def test_generate_dclass_samples_positive_and_finite():
    rng = np.random.default_rng(3)
    samples = generate_dclass_samples(100_000, L=1.0, c=1.0 / 500.0, rng=rng)
    assert np.all(samples > 0)
    assert np.all(np.isfinite(samples))


def test_generators_reject_nonpositive_n():
    rng = np.random.default_rng(4)
    with pytest.raises(AssertionError):
        generate_pareto_samples(0, alpha=0.8, rng=rng)
