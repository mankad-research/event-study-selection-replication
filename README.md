# Replication package

**Sample Selection on the Outcome with an Application to Event Studies**
Shawn Mankad (NC State University) and William Schmidt (Emory University)

## Overview

This package reproduces every table, figure, and number in the paper:

- Propositions 1–2 and Lemma 1, checked against Monte Carlo (`verify_propositions.py`).
- The simulation study of Section 3: Figure 1 and Tables 1–2.
- The application of Section 4: Figure 2, Table 3, and the winsorized footnote.

All results are written to `output/`. The file `output/numbers_for_text.md` lists every number quoted in the text, grouped by the section that uses it.

On 8 cores, the full run takes about 3–4 hours, almost all of it spent on the likelihood estimator in the simulation study. Both run scripts save each replication as it finishes, so an interrupted run resumes when the same command is relaunched.

## Data availability

The application uses `data/working_data_dedup.pkl`. It contains 3,103 executive appointments announced by press release, of which 1,574 were covered by the Wall Street Journal. See `data/README.md` for the variables and their provenance. The simulation study uses no external data.

## Computational requirements

- Python 3.10 or later. The package was tested with Python 3.12.3; package versions are in `requirements.txt`.
- Install with `pip install -r requirements.txt`. JAX is needed only for the likelihood estimator.
- Any multi-core CPU; no GPU is needed.
- Runtime with 8 workers:
  - `verify_propositions.py`: about 2 minutes.
  - `run_simulations.py`: about 3 hours, or about 15 minutes with `--no-mle`.
  - `run_realdata.py`: under 1 hour.
  - `make_outputs.py`: under 1 minute.
- Random seeds are fixed (`--seed 2026`). Each replication draws from its own random stream, so results do not depend on the number of workers.

## Programs (`code/`)

| File | Purpose |
|---|---|
| `estimators.py` | The proposed estimator: the Q-weighted mean and its standard error (Section 2.6, Proposition 3), including the optional variance term for an estimated Q. Also the textbook Heckman two-step with corrected standard errors (Section 3) and the uncorrected mean. |
| `smle.py` | The likelihood estimator for unknown Q (Section 2.8, Lemma 2, Proposition 4). Student-t or normal errors, Wald and likelihood-ratio tests; uses JAX for exact derivatives. |
| `verify_propositions.py` | Checks Propositions 1–2, as typeset, against Monte Carlo. Also checks Lemma 1 (invariance to ρ) and the β₂ → 0 limit. |
| `run_simulations.py` | Monte Carlo study of Section 3: nine designs, supplements of 0–50%, normal and Student-t errors. |
| `run_realdata.py` | Section 4: supplement draws from the press-release-only pool at 5–90% of the WSJ-covered sample, full-sample benchmarks, and the coverage Probit. |
| `make_outputs.py` | Builds all tables and figures and `numbers_for_text.md`. |

## Instructions

```
pip install -r requirements.txt
bash run_all.sh 8          # 8 = number of worker processes
```

Or run the steps one at a time from `code/`:

```
python verify_propositions.py
python run_simulations.py --workers 8      # add --quick for a short smoke test
python run_realdata.py --workers 8
python make_outputs.py                     # add --quick to build from the smoke-test run
```

## Tables and figures

| Paper | Output file | Program |
|---|---|---|
| Figure 1 | `output/fig_main_simulation_band.pdf` | `run_simulations.py`, then `make_outputs.py` |
| Table 1 | `output/tab_reject.tex` | `run_simulations.py`, then `make_outputs.py` |
| Table 2 | `output/tab_coverage.tex` | `run_simulations.py`, then `make_outputs.py` |
| Table 3 | `output/tab_realprobit.tex` | `run_realdata.py`, then `make_outputs.py` |
| Figure 2, left | `output/fig_realdata.pdf` | `run_realdata.py`, then `make_outputs.py` |
| Figure 2, right | `output/fig_realdata_zoom.pdf` | `run_realdata.py`, then `make_outputs.py` |
| Section 4 footnote (winsorized) | `output/fig_realdata_winsor.pdf` and `numbers_for_text.md` | `run_realdata.py`, then `make_outputs.py` |
| Numbers in Sections 2.6, 3, 4 | `output/numbers_for_text.md` | `make_outputs.py` |
| Propositions 1–2, Lemma 1 | console output (`output/verify_propositions.log` via `run_all.sh`) | `verify_propositions.py` |

The `.tex` tables are included in the manuscript with `\input{...}`.

## Implementation notes

- **Outcome.** In the application, the outcome for every event is the press-release-date abnormal return (`abret_10`). See `data/README.md` for why the `returns` column is not used.
- **Supplement design.** The supplementary sample is a random sample of uncovered events. In the simulations, Q is the population coverage rate. In the application, Q = 1,574/3,103, which is known because the press-release pool is enumerated.
- **Likelihood estimator.** It evaluates P(C=1|x) by Gauss–Legendre quadrature split at the kink. If the t degrees of freedom reach their upper bound, it refits and reports the normal model.
- **Heckman comparator.** Probit of C on (1, x) in the combined sample, then OLS of AR on the inverse Mills ratio in the covered sample.
