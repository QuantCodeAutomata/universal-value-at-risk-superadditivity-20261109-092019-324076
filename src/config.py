"""Parameters for the three Value-at-Risk superadditivity experiments.

All parameters are module-level constants, each commented with its meaning,
unit and source (a paper section/example, or "not specified in the paper").
No external market data is used anywhere in this repository: every
experiment reproduces a fully analytical / simulation-based example from
the paper "Universal Value-at-Risk superadditivity".
"""

# ---------------------------------------------------------------------------
# General simulation settings (shared by all three experiments)
# ---------------------------------------------------------------------------

RANDOM_SEED = 20261109  # seed for numpy's default_rng; not specified in the paper, fixed for reproducibility

# Monte Carlo sample size used to estimate VaR of a sum by the empirical
# quantile of simulated draws. The paper states that Figures 1, 2 and 4 are
# produced "by simulations" (Section 2, Example 1) but does not give the
# number of draws; the implementation notes suggest N >= 10,000,000 for
# stable estimates at high quantiles, which this repository uses for every
# experiment (full paper configuration; cheap enough to always run in full).
N_SIMULATIONS_PAPER = 10_000_000  # paper: "not specified"; implementation note floor
N_SIMULATIONS_RUN = 10_000_000  # executed value; equal to the paper configuration (no time-budget reduction needed)

# ---------------------------------------------------------------------------
# Experiment 1 - Pareto example (Section 2, Example 1; Section 3.2, Example 4)
# ---------------------------------------------------------------------------

EXP1_ALPHAS = (0.7, 0.8, 0.9)  # Pareto tail parameters alpha_1, alpha_2, alpha_3; paper Section 2, Example 1
# Pareto CDF: F(x) = 1 - (1/x)^alpha for x >= 1 (paper Section 2, Example 1).
# For alpha <= 1 the mean is infinite, which the paper uses to obtain strict UVS.

EXP1_P_GRID_LEFT_N = 400  # number of grid points for the p in (0, 0.95) panel; not specified in the paper, chosen for a smooth curve
EXP1_P_GRID_RIGHT_N = 400  # number of grid points for the p in (0.95, 0.99) panel; not specified in the paper
EXP1_P_LEFT_MIN = 1e-4  # smallest p plotted in the left panel (p=0 is excluded by the open interval (0,1) in the UVS definition, Section 2)
EXP1_P_LEFT_MAX = 0.95
EXP1_P_RIGHT_MIN = 0.95
EXP1_P_RIGHT_MAX = 0.99

# Table points read off Figure 1 of the paper (Section 2, Example 1), used only
# to report the paper's own grid of p values alongside the simulated curve.
EXP1_TABLE_P_LEFT = (0.00, 0.20, 0.40, 0.60, 0.80, 0.95)
EXP1_TABLE_P_RIGHT = (0.95, 0.96, 0.97, 0.98, 0.99)

# Approximate values read off Figure 1 of the paper (Section 2, Example 1) at
# p = 0.95 (left panel) and p = 0.99 (right panel), used only as a reference
# for the comparison printed in the results, not as inputs to the simulation.
EXP1_PAPER_REFERENCE = {
    "p=0.95": {"VaR_sum": 195, "VaR_components_sum": 135},
    "p=0.99": {"VaR_sum": 1550, "VaR_components_sum": 1150},
}

# ---------------------------------------------------------------------------
# Experiment 2 - St. Petersburg lotteries (Section 2, Example 2; Figure 2)
# ---------------------------------------------------------------------------

# St. Petersburg lottery: P(X = 2^k) = 2^-k for k = 1, 2, 3, ... (paper Section 2, Example 2)
EXP2_WEIGHTS = (0.1, 0.9)  # theta_1, theta_2 used for the weighted sum in Example 2

EXP2_P_GRID_LEFT_N = 400  # grid points for p in (0, 0.95); not specified in the paper
EXP2_P_GRID_RIGHT_N = 400  # grid points for p in (0.95, 0.99); not specified in the paper
EXP2_P_LEFT_MIN = 1e-4
EXP2_P_LEFT_MAX = 0.95
EXP2_P_RIGHT_MIN = 0.95
EXP2_P_RIGHT_MAX = 0.99

EXP2_CHECK_P = 0.55  # probability level at which the paper proves WUVS fails (Section 2, Example 2)
EXP2_CHECK_THRESHOLD = 3.8  # VaR_0.55(0.1*X1 + 0.9*X2) <= 3.8 shown analytically in the paper
EXP2_CHECK_BENCHMARK = 4.0  # VaR_0.55(0.1*X1) + VaR_0.55(0.9*X2) = VaR_0.55(X1) = 4 (paper Section 2, Example 2)
EXP2_CHECK_TAIL_PROB = 13 / 32  # P(0.1*X1 + 0.9*X2 > 3.8) = 13/32, paper Section 2, Example 2

# Table points read off Figure 2 of the paper, top panels (unweighted sum)
EXP2_TABLE_TOP_P_LEFT = (0.00, 0.20, 0.40, 0.60, 0.80, 0.95)
EXP2_TABLE_TOP_P_RIGHT = (0.95, 0.96, 0.97, 0.98, 0.985, 0.99)
# Table points for the bottom panels (weighted sum, theta = (0.1, 0.9))
EXP2_TABLE_BOTTOM_P_LEFT = (0.00, 0.20, 0.40, 0.60, 0.80, 0.95)
EXP2_TABLE_BOTTOM_P_RIGHT = (0.95, 0.96, 0.97, 0.98, 0.985, 0.99)

# ---------------------------------------------------------------------------
# Experiment 3 - D-class counterexample (Section 5, Example 7; Figure 4)
# ---------------------------------------------------------------------------

# h_{L,c}(t) = c*floor(t/L) + min(t - L*floor(t/L), c), for 0 < c < L (paper Section 5, Example 7)
# F_{L,c}(x) = exp(-h_{L,c}(1/x)) for x > 0 is the resulting class-D distribution.
EXP3_L1, EXP3_C1 = 1.0, 1.0 / 500.0  # parameters of X_1 ~ F_{1, 1/500}, paper Section 5, Example 7
EXP3_L2, EXP3_C2 = 10.0, 1.0 / 5.0  # parameters of X_2 ~ F_{10, 1/5}, paper Section 5, Example 7

EXP3_P_GRID_LEFT_N = 400  # grid points for p in (0, 0.8); not specified in the paper
EXP3_P_GRID_RIGHT_N = 400  # grid points for p in (0.8, 0.999); not specified in the paper
EXP3_P_LEFT_MIN = 1e-4
EXP3_P_LEFT_MAX = 0.8
EXP3_P_RIGHT_MIN = 0.8
EXP3_P_RIGHT_MAX = 0.999

EXP3_CHECK_P = 0.68  # the paper states VaR_p(X1+X2) < VaR_p(X1)+VaR_p(X2) at this level (Section 5, Example 7)

# Table points read off Figure 4 of the paper
EXP3_TABLE_P_LEFT = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8)
EXP3_TABLE_P_RIGHT = (0.80, 0.81, 0.85, 0.90, 0.95, 0.98, 0.99, 1.00)

# Subadditivity check grid for h_{L,c}: a grid of (t, s) pairs on which
# h(t+s) <= h(t) + h(s) is verified numerically (Section 5, Example 7 proof).
EXP3_SUBADD_GRID_N = 200  # not specified in the paper; a diagnostic grid size
EXP3_SUBADD_T_MAX = 50.0  # upper bound of the (t, s) grid used for the subadditivity check
