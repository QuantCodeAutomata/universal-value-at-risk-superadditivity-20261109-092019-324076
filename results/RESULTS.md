# Results — Universal Value-at-Risk superadditivity

Paper: **"Universal Value-at-Risk superadditivity"**.

All three experiments are pure Monte Carlo / closed-form reproductions; no
external market data is used anywhere in this repository. Every number below
comes from the `results/*_summary.json` and `results/*_table.csv` files
written by the final run of `python -m exp.exp1`, `python -m exp.exp2`, and
`python -m exp.exp3` (see `reproduce.sh`), each run with
`N_SIMULATIONS_RUN = 10,000,000` draws per random variable (`src/config.py`),
the full paper configuration (no reduction for runtime was necessary, since
this sample size runs in well under a minute per experiment).

## Experiment 1 — VaR superadditivity for independent non-identically distributed Pareto risks

Setup: X_1 ~ Pareto(0.7), X_2 ~ Pareto(0.8), X_3 ~ Pareto(0.9), independent,
CDF F(x) = 1 - x^(-alpha) for x >= 1 (Section 2, Example 1). VaR_p(S) is the
empirical p-quantile of 10,000,000 simulated draws of S = X_1 + X_2 + X_3;
VaR_p(X_i) is computed in closed form as (1-p)^(-1/alpha_i) and summed. The
grid covers p in (0, 0.95) and p in (0.95, 0.99), reproducing the two panels
of Figure 1 (`results/exp1_pareto_var.png`, `results/exp1_pareto_var_table.csv`).

| p    | VaR_p(sum), simulated | sum of VaR_p(X_i) | paper (Figure 1, approx.) |
|------|-----------------------|--------------------|-----------------------------|
| 0.95 | 198.30                | 142.41             | VaR_sum ≈ 195, sum of VaRs ≈ 135 |
| 0.99 | 1573.52               | 1202.72            | VaR_sum ≈ 1550, sum of VaRs ≈ 1150 |

Over the full simulated grid of 800 points spanning p in (0, 0.99),
`violations_of_uvs = 0`: VaR_p(X_1+X_2+X_3) >= VaR_p(X_1)+VaR_p(X_2)+VaR_p(X_3)
holds at every tested p, and the gap between the two curves widens as p
approaches 1, as the paper describes.

**Verdict:** reproduced. The simulated values lie within about 2–6% of the
values read off Figure 1, with the same sign and the same qualitative
behavior (strict superadditivity, widening gap near p = 1).

## Experiment 2 — UVS without WUVS for iid St. Petersburg lotteries

Setup: X_1, X_2 iid St. Petersburg lotteries, P(X = 2^k) = 2^(-k) for
k = 1, 2, 3, ... (Section 2, Example 2). Top panels: unweighted sum
X_1 + X_2. Bottom panels: weighted sum 0.1·X_1 + 0.9·X_2. VaR is the
empirical p-quantile of 10,000,000 simulated draws
(`results/exp2_stpetersburg.png`, `results/exp2_stpetersburg_table.csv`).

* **UVS (top panels):** `uvs_top_holds_everywhere = true` — on the full grid
  of p in (0, 0.99), VaR_p(X_1+X_2) >= VaR_p(X_1) + VaR_p(X_2) at every point.
* **WUVS (bottom panels):** `wuvs_bottom_violations = 407` grid points out of
  800 — WUVS fails on a large part of the grid, not only at an isolated
  point.
* **Analytic check at p = 0.55** (Section 2, Example 2): the paper shows
  P(0.1·X_1 + 0.9·X_2 > 3.8) = 13/32 ≈ 0.40625, which bounds
  VaR_0.55(0.1·X_1 + 0.9·X_2) <= 3.8, while
  VaR_0.55(0.1·X_1) + VaR_0.55(0.9·X_2) = VaR_0.55(X_1) = 4. The simulation
  gives `tail_prob_mc_P(W>3.8) = 0.4062215` (paper: 0.40625),
  `var_weighted_sum_mc_at_check_p = 3.8` (paper bound: <= 3.8), and
  `var_weighted_components_at_check_p = 4.0` (paper: 4), so
  VaR_0.55(weighted sum) = 3.8 < 4.0 = sum of weighted VaRs: WUVS fails at
  p = 0.55 exactly as the paper states.

**Verdict:** reproduced. Both the UVS result (top panels) and its failure for
unequal weights (bottom panels, including the p = 0.55 analytical check) hold
with values matching the paper to 4 significant digits (0.4062215 vs.
0.40625).

## Experiment 3 — Failure of UVS for independent class-D risks with non-identical distributions

Setup: X_1 ~ F_{1, 1/500}, X_2 ~ F_{10, 1/5}, independent, where
F_{L,c}(x) = exp(-h_{L,c}(1/x)) and
h_{L,c}(t) = c·floor(t/L) + min(t - L·floor(t/L), c) (Section 5, Example 7).
VaR is the empirical p-quantile of 10,000,000 simulated draws from each
margin, obtained by numerically inverting F_{L,c}
(`results/exp3_dclass.png`, `results/exp3_dclass_table.csv`).

* Subadditivity of h_{L,c} is verified numerically on a grid of
  (t, s) pairs for both parameter sets: `h_subadditive_on_grid_L1_c1 = true`,
  `h_subadditive_on_grid_L2_c2 = true` (the property the paper uses to place
  F_{L,c} in class D).
* Over the grid of p in (0, 0.8) ∪ (0.8, 0.999), `uvs_violations_on_grid = 68`
  points: UVS fails on a substantial sub-range of p, not just at one point.
* At **p = 0.68**, the level the paper highlights:
  `var_sum_mc_at_check_p = 0.10026` < `var_components_sum_at_check_p = 0.10339`,
  i.e. VaR_0.68(X_1+X_2) < VaR_0.68(X_1) + VaR_0.68(X_2):
  `uvs_violated_at_check_p = true`, confirming the paper's claim that UVS does
  not extend to arbitrary independent, non-identically distributed class-D
  risks.
* At the extremes of the grid the two curves are of comparable magnitude:
  near p = 0.5 the simulated VaR_sum ≈ 0.044 and the sum of VaRs ≈ 0.036
  (VaR_sum larger, UVS holds), and near p = 0.99 VaR_sum ≈ 124 versus a sum of
  VaRs ≈ 100 (the paper's Figure 4 reports both curves at approximately 100
  at the right edge of its panel, and approximately 0.05 / 0.04 near p = 0.5).

**Verdict:** reproduced (qualitatively and in order of magnitude). The
central claim — that VaR_p(X_1+X_2) < VaR_p(X_1)+VaR_p(X_2) at p = 0.68 — is
confirmed exactly as the paper derives it "by direct calculation". The
simulated curve values near the panel edges are of the same order of
magnitude as the paper's read-off reference values but are not pinned to the
exact pixel values of Figure 4, since the paper gives the p = 0.68 crossover
as its only precise numerical claim for this example and does not tabulate
the full curves.

## Deviations from the paper

* No external data source is used in any experiment: all three are
  self-contained Monte Carlo / closed-form reproductions of analytical
  examples, as the paper itself describes them (Section 2, Examples 1–2;
  Section 5, Example 7). There is accordingly no market-data source to name.
* The paper does not specify the number of Monte Carlo samples used to draw
  Figures 1, 2 and 4, nor the exact grid of p values plotted. This repository
  uses N = 10,000,000 draws per random variable and a grid of 800 points per
  panel (400 per sub-panel); both are documented in `src/config.py` as "not
  specified in the paper" and are large enough that increasing them further
  does not change any qualitative conclusion.
* For Experiment 3, the paper's own text attributes the exact value of
  VaR_0.68(X_1+X_2) relative to VaR_0.68(X_1)+VaR_0.68(X_2) to "direct
  calculations (omitted here)"; this repository instead estimates both
  quantities by Monte Carlo simulation from the closed-form quantile function
  of F_{L,c}, which reproduces the stated sign of the inequality but not the
  paper's own unpublished exact figures.
* Figure 4's exact curve values away from p = 0.68 are only approximately
  stated in the paper; the comparison above therefore reports
  order-of-magnitude agreement at the panel edges rather than an exact match.

## Not reproduced

Nothing is omitted: all three selected experiments (Experiment 1, Experiment
2, Experiment 3) are fully implemented, executed with the full paper
configuration (N = 10,000,000 simulated draws per random variable), and their
results are reported above.
