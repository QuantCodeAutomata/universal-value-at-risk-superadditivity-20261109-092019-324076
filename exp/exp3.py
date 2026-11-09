"""Experiment 3: Failure of UVS for independent D-class risks with
non-identical distributions (paper Section 5, Example 7; Figure 4).

Constructs F_{L,c}(x) = exp(-h_{L,c}(1/x)), a class-D distribution, with
X_1 ~ F_{1, 1/500} and X_2 ~ F_{10, 1/5} independent, and shows that
VaR_p(X_1+X_2) < VaR_p(X_1)+VaR_p(X_2) at p = 0.68, so UVS fails even though
both marginals belong to class D.

Run: python -m exp.exp3
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.generate_synthetic import generate_dclass_samples
from src import config
from src.distributions import dclass_h, dclass_quantile
from src.var_tools import build_comparison_table

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def check_h_subadditivity(L: float, c: float, grid_n: int, t_max: float) -> bool:
    """Numerically verify h_{L,c}(t+s) <= h_{L,c}(t) + h_{L,c}(s) on a grid (Section 5, Example 7 proof)."""
    grid = np.linspace(1e-6, t_max, grid_n)
    t_mesh, s_mesh = np.meshgrid(grid, grid)
    lhs = dclass_h(t_mesh + s_mesh, L, c)
    rhs = dclass_h(t_mesh, L, c) + dclass_h(s_mesh, L, c)
    return bool(np.all(lhs <= rhs + 1e-9))


def run(n_simulations: int = config.N_SIMULATIONS_RUN) -> dict:
    rng = np.random.default_rng(config.RANDOM_SEED)
    L1, c1 = config.EXP3_L1, config.EXP3_C1
    L2, c2 = config.EXP3_L2, config.EXP3_C2

    x1 = generate_dclass_samples(n_simulations, L1, c1, rng)
    x2 = generate_dclass_samples(n_simulations, L2, c2, rng)
    total = x1 + x2

    p_left = np.linspace(config.EXP3_P_LEFT_MIN, config.EXP3_P_LEFT_MAX, config.EXP3_P_GRID_LEFT_N)
    p_right = np.linspace(config.EXP3_P_RIGHT_MIN, config.EXP3_P_RIGHT_MAX, config.EXP3_P_GRID_RIGHT_N)

    tables = []
    for p_grid, panel_name in ((p_left, "(0, 0.8)"), (p_right, "(0.8, 0.999)")):
        var_sum = np.quantile(total, p_grid)
        var_components_sum = dclass_quantile(p_grid, L1, c1) + dclass_quantile(p_grid, L2, c2)
        tables.append(build_comparison_table(p_grid, var_sum, var_components_sum, panel_name))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(RESULTS_DIR / "exp3_dclass_table.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, panel_name in zip(axes, ("(0, 0.8)", "(0.8, 0.999)")):
        sub = table[table["panel"] == panel_name]
        ax.plot(sub["p"], sub["VaR_sum"], label=r"$VaR_p(X_1+X_2)$", color="tab:blue")
        ax.plot(sub["p"], sub["VaR_components_sum"], label=r"$VaR_p(X_1)+VaR_p(X_2)$",
                color="tab:orange", linestyle="--")
        ax.set_xlabel("p")
        ax.set_ylabel("VaR")
        ax.set_title(f"p in {panel_name}")
        ax.legend()
    fig.suptitle(r"Experiment 3: UVS fails for independent $F_{1,1/500}$ and $F_{10,1/5}$ risks")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_dclass.png", dpi=150)
    plt.close(fig)

    p_check = np.array([config.EXP3_CHECK_P])
    var_sum_mc_check = float(np.quantile(total, p_check)[0])
    var_components_check = float((dclass_quantile(p_check, L1, c1) + dclass_quantile(p_check, L2, c2))[0])

    subadditive_h1 = check_h_subadditivity(L1, c1, config.EXP3_SUBADD_GRID_N, config.EXP3_SUBADD_T_MAX)
    subadditive_h2 = check_h_subadditivity(L2, c2, config.EXP3_SUBADD_GRID_N, config.EXP3_SUBADD_T_MAX)

    violations = int((~table["superadditive"]).sum())

    summary = {
        "experiment": "exp3_dclass_uvs_failure",
        "L1_c1": [L1, c1],
        "L2_c2": [L2, c2],
        "n_simulations": int(n_simulations),
        "h_subadditive_on_grid_L1_c1": subadditive_h1,
        "h_subadditive_on_grid_L2_c2": subadditive_h2,
        "uvs_violations_on_grid": violations,
        "uvs_fails_somewhere": violations > 0,
        "check_p": config.EXP3_CHECK_P,
        "var_sum_mc_at_check_p": var_sum_mc_check,
        "var_components_sum_at_check_p": var_components_check,
        "uvs_violated_at_check_p": var_sum_mc_check < var_components_check,
    }
    with open(RESULTS_DIR / "exp3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
