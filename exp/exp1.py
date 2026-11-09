"""Experiment 1: VaR superadditivity for independent non-identically distributed
Pareto risks (paper Section 2, Example 1; Section 3.2, Example 4).

Simulates X_1 ~ Pareto(0.7), X_2 ~ Pareto(0.8), X_3 ~ Pareto(0.9), independent,
and compares VaR_p(X_1+X_2+X_3) to VaR_p(X_1)+VaR_p(X_2)+VaR_p(X_3) for a grid
of p in (0, 0.99), reproducing Figure 1 of the paper.

Run: python -m exp.exp1
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import generate_pareto_samples
from src import config
from src.distributions import pareto_var
from src.var_tools import build_comparison_table, check_superadditivity

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run(n_simulations: int = config.N_SIMULATIONS_RUN) -> dict:
    rng = np.random.default_rng(config.RANDOM_SEED)
    alphas = config.EXP1_ALPHAS

    samples = [generate_pareto_samples(n_simulations, alpha, rng) for alpha in alphas]
    total = np.sum(samples, axis=0)

    p_left = np.linspace(config.EXP1_P_LEFT_MIN, config.EXP1_P_LEFT_MAX, config.EXP1_P_GRID_LEFT_N)
    p_right = np.linspace(config.EXP1_P_RIGHT_MIN, config.EXP1_P_RIGHT_MAX, config.EXP1_P_GRID_RIGHT_N)

    tables = []
    for p_grid, panel_name in ((p_left, "(0, 0.95)"), (p_right, "(0.95, 0.99)")):
        var_sum = np.quantile(total, p_grid)
        var_components_sum = sum(pareto_var(p_grid, alpha) for alpha in alphas)
        tables.append(build_comparison_table(p_grid, var_sum, var_components_sum, panel_name))
    table = pd.concat(tables, ignore_index=True)

    table.to_csv(RESULTS_DIR / "exp1_pareto_var_table.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, panel_name in zip(axes, ("(0, 0.95)", "(0.95, 0.99)")):
        sub = table[table["panel"] == panel_name]
        ax.plot(sub["p"], sub["VaR_sum"], label=r"$VaR_p(X_1+X_2+X_3)$", color="tab:blue")
        ax.plot(sub["p"], sub["VaR_components_sum"], label=r"$VaR_p(X_1)+VaR_p(X_2)+VaR_p(X_3)$",
                color="tab:orange", linestyle="--")
        ax.set_xlabel("p")
        ax.set_ylabel("VaR")
        ax.set_title(f"p in {panel_name}")
        ax.legend()
    fig.suptitle("Experiment 1: VaR superadditivity, independent Pareto(0.7, 0.8, 0.9) risks")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_pareto_var.png", dpi=150)
    plt.close(fig)

    violations = int((~table["superadditive"]).sum())
    at_095 = table[np.isclose(table["p"], 0.95, atol=1e-6)]
    at_099 = table[np.isclose(table["p"], 0.99, atol=1e-6)]
    summary = {
        "experiment": "exp1_pareto_uvs",
        "alphas": list(alphas),
        "n_simulations": int(n_simulations),
        "p_grid_points": int(len(table)),
        "violations_of_uvs": violations,
        "uvs_holds_everywhere": violations == 0,
        "var_sum_at_p0.95": float(at_095["VaR_sum"].mean()) if len(at_095) else None,
        "var_components_sum_at_p0.95": float(at_095["VaR_components_sum"].mean()) if len(at_095) else None,
        "var_sum_at_p0.99": float(at_099["VaR_sum"].mean()) if len(at_099) else None,
        "var_components_sum_at_p0.99": float(at_099["VaR_components_sum"].mean()) if len(at_099) else None,
        "paper_table_reference": config.EXP1_PAPER_REFERENCE,
    }
    with open(RESULTS_DIR / "exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
