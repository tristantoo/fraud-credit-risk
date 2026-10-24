# fraud-credit-risk

Fraud and credit-risk test (P3). Pre-registered plan and deviations: `docs/PLAN.md`. Results: `outputs/part_a_*` (tables, curves, run log).

- **Part a:** ULB credit-card fraud (OpenML 1597). Train earlier, validate, test later on `Time`.
  Logistic regression vs HistGradientBoosting: precision@0.5% review capacity, PR-AUC, expected cost
  at a validation-chosen cutoff, calibration (Brier, ECE, reliability). Paired bootstrap, Holm.
- **Part b:** Freddie Mac loan-level sample: vintage default curves, roll rates, out-of-time 12-month
  PD. **BLOCKED-ON-TRISTAN** (needs registration); the pipeline runs on a SYNTHETIC fixture in tests.

## Run

```
uv sync
uv run pytest                          # includes leakage audits that must fail on leaky fixtures
uv run python scripts/run_part_a.py    # ~25 min (3 x 5000 bootstrap resamples); writes outputs/part_a_*
uv run python scripts/run_part_b.py    # exits BLOCKED-ON-TRISTAN until data/raw/freddie/ has the files
```

Part a reads the cached OpenML ARFF at `data/raw/openml_cache/...` (gitignored). `fetch_openml`
drops `Time`, so `src/frc/data.py` parses the ARFF directly and checks its md5. To recreate the
cache: `sklearn.datasets.fetch_openml(data_id=1597, data_home="data/raw/openml_cache")`.

**Fresh clone:** `data/raw/` is gitignored, so the 5 real-data leakage tests are SKIPPED until you download it.
`pytest` prints each skip with the reason (`-rs` is set in `pyproject.toml`); a run showing `5 skipped` has NOT
audited the real split. Download, then rerun `uv run pytest`:

```
uv run python -c "from sklearn.datasets import fetch_openml; fetch_openml(data_id=1597, data_home='data/raw/openml_cache')"
```

## Authorship

Built with AI coding assistance (Claude Code). I set the scope and questions and reviewed the methods, code and results.
