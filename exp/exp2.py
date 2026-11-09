"""Experiment 2: UVS without WUVS for iid St. Petersburg lotteries
(paper Section 2, Example 2; Figure 2).

Top panels: VaR_p(X_1+X_2) vs VaR_p(X_1)+VaR_p(X_2) for iid St. Petersburg
lotteries X_1, X_2 (UVS should hold for all p in (0, 0.99)).

Bottom panels: VaR_p(0.1*X_1 + 0.9*X_2) vs VaR_p(0.1*X_1)+VaR_p(0.9*X_2)
(WUVS fails, in particular at p = 0.55).

Run: python -m exp.exp2
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import generate_stpetersburg_samples
from src import config
from src.distributions import stpetersburg_var
from src.var_tools import build_comparison_table

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _panel_tables(total: np.ndarray, var_components_sum_fn, left_n: int, right_n: int,
                   left_bounds: tuple[float, float], right_bounds: tuple[float, float]) -> pd.DataFrame:
    p_left = np.linspace(left_bounds[0], left_bounds[1], left_n)
    p_right = np.linspace(right_bounds[0], right_bounds[1], right_n)
    tables = []
    for p_grid, panel_name in ((p_left, "(0, 0.95)"), (p_right, "(0.95, 0.99)")):
        var_sum = np.quantile(total, p_grid)
        var_components_sum = var_components_sum_fn(p_grid)
        tables.append(build_comparison_table(p_grid, var_sum, var_components_sum, panel_name))
    return pd.concat(tables, ignore_index=True)


def run(n_simulations: int = config.N_SIMULATIONS_RUN) -> dict:
    rng = np.random.default_rng(config.RANDOM_SEED)
    theta1, theta2 = config.EXP2_WEIGHTS

    x1 = generate_stpetersburg_samples(n_simulations, rng)
    x2 = generate_stpetersburg_samples(n_simulations, rng)

    unweighted_sum = x1 + x2
    weighted_sum = theta1 * x1 + theta2 * x2

    top_table = _panel_tables(
        unweighted_sum,
        lambda p: 2.0 * stpetersburg_var(p),
        config.EXP2_P_GRID_LEFT_N, config.EXP2_P_GRID_RIGHT_N,
        (config.EXP2_P_LEFT_MIN, config.EXP2_P_LEFT_MAX),
        (config.EXP2_P_RIGHT_MIN, config.EXP2_P_RIGHT_MAX),
    )
    top_table.insert(0, "part", "unweighted_sum")

    bottom_table = _panel_tables(
        weighted_sum,
        lambda p: theta1 * stpetersburg_var(p) + theta2 * stpetersburg_var(p),
        config.EXP2_P_GRID_LEFT_N, config.EXP2_P_GRID_RIGHT_N,
        (config.EXP2_P_LEFT_MIN, config.EXP2_P_LEFT_MAX),
        (config.EXP2_P_RIGHT_MIN, config.EXP2_P_RIGHT_MAX),
    )
    bottom_table.insert(0, "part", "weighted_sum")

    table = pd.concat([top_table, bottom_table], ignore_index=True)
    table.to_csv(RESULTS_DIR / "exp2_stpetersburg_table.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for col, panel_name in enumerate(("(0, 0.95)", "(0.95, 0.99)")):
        sub = top_table[top_table["panel"] == panel_name]
        ax = axes[0, col]
        ax.step(sub["p"], sub["VaR_sum"], where="post", label=r"$VaR_p(X_1+X_2)$", color="tab:blue")
        ax.step(sub["p"], sub["VaR_components_sum"], where="post",
                label=r"$VaR_p(X_1)+VaR_p(X_2)$", color="tab:orange", linestyle="--")
        ax.set_title(f"Top: unweighted sum, p in {panel_name}")
        ax.set_xlabel("p")
        ax.set_ylabel("VaR")
        ax.legend()

        sub = bottom_table[bottom_table["panel"] == panel_name]
        ax = axes[1, col]
        ax.step(sub["p"], sub["VaR_sum"], where="post",
                label=r"$VaR_p(0.1X_1+0.9X_2)$", color="tab:blue")
        ax.step(sub["p"], sub["VaR_components_sum"], where="post",
                label=r"$VaR_p(0.1X_1)+VaR_p(0.9X_2)$", color="tab:orange", linestyle="--")
        ax.set_title(f"Bottom: weighted sum, p in {panel_name}")
        ax.set_xlabel("p")
        ax.set_ylabel("VaR")
        ax.legend()
    fig.suptitle("Experiment 2: UVS holds (top) but WUVS fails (bottom), iid St. Petersburg lotteries")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_stpetersburg.png", dpi=150)
    plt.close(fig)

    # Analytic check at p = 0.55 (paper Section 2, Example 2)
    p_check = config.EXP2_CHECK_P
    var_weighted_sum_mc = round(float(np.quantile(weighted_sum, p_check)), 6)
    var_weighted_components = float(theta1 * stpetersburg_var(np.array([p_check]))[0]
                                     + theta2 * stpetersburg_var(np.array([p_check]))[0])
    # Exact dyadic values of theta1*X1 + theta2*X2 (e.g. 0.1*2 + 0.9*4 = 3.8)
    # can round to 3.8000000000000003 in float64, which would wrongly count
    # as part of the strict tail event {W > 3.8}; a small tolerance restores
    # the exact equality implied by the paper's analytic computation.
    tail_prob_mc = float(np.mean(weighted_sum > config.EXP2_CHECK_THRESHOLD + 1e-9))

    top_violations = int((~top_table["superadditive"]).sum())
    bottom_violations = int((~bottom_table["superadditive"]).sum())

    summary = {
        "experiment": "exp2_stpetersburg_uvs_without_wuvs",
        "n_simulations": int(n_simulations),
        "uvs_top_holds_everywhere": top_violations == 0,
        "wuvs_bottom_violations": bottom_violations,
        "wuvs_fails_somewhere": bottom_violations > 0,
        "check_p": p_check,
        "var_weighted_sum_mc_at_check_p": var_weighted_sum_mc,
        "var_weighted_components_at_check_p": var_weighted_components,
        "paper_threshold_var_weighted_sum": config.EXP2_CHECK_THRESHOLD,
        "paper_benchmark_var_weighted_components": config.EXP2_CHECK_BENCHMARK,
        "tail_prob_mc_P(W>3.8)": tail_prob_mc,
        "paper_tail_prob_P(W>3.8)": config.EXP2_CHECK_TAIL_PROB,
        "wuvs_fails_at_check_p": var_weighted_sum_mc < var_weighted_components,
    }
    with open(RESULTS_DIR / "exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
