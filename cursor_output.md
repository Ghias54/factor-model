# Cursor output — chore/lock-decisions

Date: 2026-10-20 · Branch: `chore/lock-decisions` (from updated `main` @ `38d4811`) · Pushed, not merged.

## 1. Discarded working-tree diffs (FE-Factor-Model)

Recorded before `git restore` on 2026-10-20. These were superseded by
`docs/design-decisions-2026-10-20.md` and contained the incorrect survivorship claim
(later softened in the working tree, but still discarded as directed).

### `configs/default.yaml` (discarded)

```diff
-  min_market_cap_pctile: 0.0 # TBD — e.g. 0.2 drops the smallest 20% each month
-  top_n_by_market_cap: null  # TBD — e.g. 1000 for a Russell-1000-style universe
+  min_market_cap_pctile: 0.0
+  top_n_by_market_cap: 1000  # DECIDED 2026-10-06 — re-ranked every month from all qualifying stocks
+  robustness_top_n: 500      # secondary check, S&P-500-like
```

### `docs/decisions.md` (discarded)

```diff
+| 2026-10-06 | Universe = top 1000 US common stocks by market cap, re-ranked monthly (top 500 as robustness check) | Russell-1000-like; avoids micro-cap noise; larger cross-section for ML than S&P 500; least exposed of the candidate universes to the vendor's missing pre-2016 delistings (see docs/data-audit.md). Re-ranking at t is survivorship-free only where dead names exist in the database — roughly from 2021; earlier years are partly survivorship-biased. Today's S&P 500 list rejected (fully survivorship-biased) |
```

### Other uncommitted items at task start (not discarded)

- `M cursor_task.md` — new task file for this run (kept / committed on the chore branch).
- `?? docs/design-decisions-2026-10-20.md` — locked decisions source (kept / committed).

No other dirty files. Did not stop; these two are required inputs for this task.

## 2. Merges into main (Approved by Rehan 2026-10-20)

### FE-Factor-Model

| | |
|---|---|
| Merge commit | `38d4811` Merge docs/data-audit into main. Approved by Rehan 2026-10-20. |
| Strategy | `--no-ff`, ort; **no conflicts** |
| Pushed | `origin/main` `601502c..38d4811` |
| Confirmed on main | `src/factor_model/data/db.py`, `loaders.py`, `audit.py`, `__init__.py` |
| Tests | `pytest`: **9 passed** (after merge, before chore branch) |

### stock-database

| | |
|---|---|
| Merge commit | `35559e8` Merge load/gap-pull-1016 into main. Approved by Rehan 2026-10-20. |
| Strategy | `--no-ff`, ort; **no conflicts** |
| Pushed | `origin/main` `c1d83a0..35559e8` |
| Brought in | `src/load/gap_pull.py`, `scripts/load_gap_pull.py`, `scripts/load_sec_q3_statements.py`, migration `049_ownership_incomplete.sql`, instrument_guard fix, cursor_output |
| Tests | `pytest`: **486 passed, 3 skipped** in 46.9s |
| Not done | No loads, crawls, repairs, or migrations run; database untouched |

## 3. Branch `chore/lock-decisions` commits

```
fc48c82 Add cursor_task for chore/lock-decisions
2264d90 Assert locked config keys in test_config
9ef23df Lock decisions log and plan to 2026-10-20 design decisions
8e15d2c Lock configs/default.yaml to 2026-10-20 design decisions
(+ final commit for this cursor_output.md)
```

Tests on branch: `pytest` **12 passed**.

## 4. Config keys changed (`configs/default.yaml`)

| Area | Change |
|---|---|
| `sample` | start → 2006-01-31; end kept 2026-08-31; holdout kept 2024-01-31; **added** `first_oos: 2011-01-31` |
| `universe` | `top_n_by_market_cap: 1000`; `price_field: close_raw`; `market_cap: close_x_shares_diluted`; membership / delisting_return notes; **removed** any `robustness_top_n` |
| `target` | `type: rank`; `risk_free: month3`; `return_field: close_adj` |
| `features` | **new** block: 26 named features, staleness 548, earnings ≤120 days, 16:00 ET cutoff |
| `preprocessing` | rank to [-1,1]; `missing_to: 0`; `missing_indicator: false`; winsorize null |
| `walk_forward` | `retrain: january`; `validation_years: 3` |
| `models` | **new** list of 7 models with placeholder tune grids |
| `portfolio` | `long_only: true`; top decile; equal weight; benchmark EW universe; cost 10 bps/side |
| `evaluation` | **new** regimes (covid, ratehike, survivorship_biased/clean) + metrics |

## 5. Docs

- `docs/decisions.md` rewritten as a date/decision/reason/source table covering every section of the 2026-10-20 design doc. Explicit: universe is **NOT survivorship-free before ~2021**.
- `docs/plan.md` rebuild order matches "Build order after cleanup" (§1 panel → §2 classic+FF → §3 models → §4 portfolios/regimes → §5 holdout once).
- `docs/design-decisions-2026-10-20.md` committed as the source-of-truth review note.

## 6. Conflicts

None on either merge.

## 7. NOT done / skipped

- Did **not** build universe, panel, features, or models.
- Did **not** merge `chore/lock-decisions` into main (human merges).
- Did **not** run any stock-database loads, crawls, repairs, or migrations.
- Did **not** change production or dev databases.
- Did **not** apply the stock-database zero-lag fundamentals repair (loaders still guard).
- Did **not** refetch pre-2005 prices for the 602 truncated names (explicitly skipped per decisions).
