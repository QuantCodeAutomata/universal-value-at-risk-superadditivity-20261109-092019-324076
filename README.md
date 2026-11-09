# Universal Value-at-Risk superadditivity

Reproduction of three simulation-based examples from the paper **"Universal
Value-at-Risk superadditivity"**.

This repository contains no market data: every experiment is a Monte Carlo /
closed-form reproduction of an analytical example from the paper. All random
variables (Pareto, St. Petersburg lottery, and the class-D distribution
`F_{L,c}`) are generated from closed-form quantile functions implemented in
`src/distributions.py` and sampled in `data/generate_synthetic.py`.

## Experiments

* **exp1** — VaR superadditivity for independent, non-identically
  distributed, infinite-mean Pareto risks X_1 ~ Pareto(0.7), X_2 ~
  Pareto(0.8), X_3 ~ Pareto(0.9) (Section 2, Example 1; Section 3.2, Example
  4; Figure 1).
* **exp2** — Universal VaR superadditivity (UVS) without weighted UVS (WUVS)
  for two iid St. Petersburg lotteries (Section 2, Example 2; Figure 2).
* **exp3** — Failure of UVS for independent class-D risks with non-identical
  marginals F_{1,1/500} and F_{10,1/5} (Section 5, Example 7; Figure 4).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No API keys or external data are required for this repository.

## Running the tests

```bash
python -m pytest
```

## Running the experiments

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
```

or simply:

```bash
./reproduce.sh
```

Each command overwrites its artifacts directly in `results/` (tables as
`.csv`, figures as `.png`, and a machine-readable `*_summary.json` with the
reported metrics). The numbers and verdicts in `results/RESULTS.md` are
written from these files and must be re-checked by hand after a rerun.

All three experiments use `N_SIMULATIONS_RUN = 10,000,000` draws per random
variable (`src/config.py`), the full configuration; this sample size is cheap
enough to always run in full, so no reduction for runtime was necessary.

## Repository layout

```
src/     distribution/quantile formulas, VaR comparison utilities, config
data/    synthetic sample generation (inverse-CDF sampling)
exp/     one script per experiment, calling src/ and data/
tests/   unit tests for src/ and data/
results/ generated tables, figures, summaries, and RESULTS.md
```
