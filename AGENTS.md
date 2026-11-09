# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this implements

Three simulation-based examples from the paper **"Universal Value-at-Risk
superadditivity"**, demonstrating when Value-at-Risk is, and is not,
superadditive for sums of independent (but not necessarily identically
distributed) heavy-tailed risks:

* **exp1** — VaR superadditivity (UVS) for independent, non-identically
  distributed, infinite-mean Pareto risks (paper Section 2, Example 1;
  Section 3.2, Example 4; Figure 1).
* **exp2** — UVS without weighted UVS (WUVS) for two iid St. Petersburg
  lotteries (paper Section 2, Example 2; Figure 2).
* **exp3** — Failure of UVS for independent class-D risks with non-identical
  marginals F_{1,1/500} and F_{10,1/5} (paper Section 5, Example 7; Figure 4).

All three experiments are self-contained Monte Carlo / closed-form
reproductions. No market data or other external dataset is used anywhere in
the repository.

## Layout

```
src/     distribution/quantile formulas (src/distributions.py), VaR
         comparison utilities (src/var_tools.py), all parameters
         (src/config.py)
data/    synthetic sample generation by inverse-CDF sampling
         (data/generate_synthetic.py) — the only place that draws random
         variables for the experiments
exp/     one script per experiment (exp/exp1.py, exp/exp2.py, exp/exp3.py),
         each calling src/ and data/ only
tests/   pytest unit tests for src/ and data/
results/ generated tables (.csv), figures (.png), machine-readable summaries
         (*_summary.json), and results/RESULTS.md
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests (all unit tests; there are no market-data tests in this repository
since no experiment loads market data):

```bash
python -m pytest -m "not market_data"
```

(equivalently `python -m pytest`, since no test is marked `market_data`.)

Experiments, run from the repository root:

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
```

or `./reproduce.sh`, which runs the tests and all three experiments with the
paper's full configuration (`N_SIMULATIONS_RUN = 10,000,000` draws per random
variable; see `src/config.py`). This sample size already corresponds to the
paper's full configuration — no reduction for runtime was needed.

## Conventions

* Every experiment parameter lives in `src/config.py` as a module-level
  UPPER_CASE variable with a comment stating its meaning, unit, and source
  (a paper section, or "not specified in the paper"). No parameter is
  hardcoded inside `exp/` or `src/` modules.
* Random sampling is done only in `data/generate_synthetic.py`, using the
  closed-form quantile functions in `src/distributions.py`; no `exp/` script
  draws samples directly.
* No API keys are used by this repository (there is no market-data path);
  if one is ever added, keys must be read from an uncommitted `.env` and
  never hardcoded.
* After rerunning an experiment, its command overwrites the corresponding
  files in `results/`; the numbers and verdicts in `results/RESULTS.md` are
  not regenerated automatically and must be checked against the new files
  and updated by hand.
