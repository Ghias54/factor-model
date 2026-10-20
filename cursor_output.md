# Cursor output — feat/panel

Date: 2026-10-20 · Branch: `feat/panel` (from `main` @ `1373836`) · Pushed, **not** merged.

## 0. Merge `chore/lock-decisions` → main (Approved by Rehan 2026-10-20)

| | |
|---|---|
| Merge commit | `1373836` Merge chore/lock-decisions into main. Approved by Rehan 2026-10-20. |
| Strategy | `--no-ff` |
| Tests after merge | `pytest`: **12 passed** (pre-panel); after panel work: **16 passed** |
| Then | `git checkout -b feat/panel` |

## Branch commits (`feat/panel` not in `main`)

```
0edbcdd Add cursor_task for feat/panel
479f8cb Extend loaders for daily prices, index, treasury, and TTM as-of
e75e6d8 Add monthly top-N universe builder
191063c Add 26-feature panel compute, preprocess, CLI, lookahead tests
c7a50e0 Fix symbol_asof: interval join and index-aligned assignment
273cb2e Speed up asof_last_n_quarters with per-security index
320c6f0 Chunk daily price loads for vol/Amihud to fit in RAM
(+ final: panel-report + GSPC target-sanity corr fix)
```

## Universe

| Metric | Value |
|---|---|
| Months | 248 (2006-01-31 → 2026-08-31) |
| Rows | 248,000 (exactly 1000 every month; none below) |
| Distinct securities | 2,650 |
| Vendor mcap fallback | 14,109 rows (5.69%) when diluted shares missing |
| Min `close_raw` | $5.0 |
| mcap | `close × weighted_average_shares_diluted` (PIT + zero-lag guard) |

## Feature coverage

Full table in `docs/panel-report.md`. **No feature below 70% in any year.**

Weakest early years (still ≥70%): `reversal_36_13` ~75% in 2006–07; `beta_60m` ~77% in 2006; `delta_roe_4q` / quality-ish fundamentals mid-70s–80s early sample. Coverage rises through 2025–26.

## Lookahead tests

`tests/test_panel_lookahead.py` — sample of 200 `(security_id, month)` rows:

- Fundamentals: `date_available` ≤ 16:00 ET on month-end `t`
- Earnings: `report_date` < `t` (SUE)

`pytest`: **16 passed** (including 4 lookahead).

## Spot checks / mcap / target

| Check | Result |
|---|---|
| AAPL | 248 months, 2006-01-31 → 2026-08-31 |
| SIVB | 201 months through **2023-02-28**; present Feb-2023, **absent** Mar-2023 |
| mcap vs vendor `market_cap_daily` | median ratio **1.008**; 94.6% within ±20% (n=246,217) |
| Mean EW universe monthly return | **0.860%** |
| Corr vs ^GSPC (same t→t+1 window) | **0.935** (n=247) |

Note: an earlier report draft compared forward stock returns to *trailing* GSPC pct_change (~0.05 corr). Fixed in `write_report` to use GSPC forward return aligned with `ret_1m`.

## Artifacts (gitignored under `data/`)

| File | Size |
|---|---|
| `data/panel/universe.parquet` | 12 MB |
| `data/panel/features_raw.parquet` | 50 MB |
| `data/panel/panel.parquet` | 27 MB |
| plus `*_meta.json`, `report_extras.json` | |

Processed panel: Gu–Kelly–Xiu rank → [-1,1], missing → 0, plus `ret_1m`, `excess_ret_1m`, `target_rank`.

## Performance (tradinghost)

| | |
|---|---|
| Total wall | **877.6 s** (~14.6 min) |
| Universe | 128.5 s |
| Features | 746.2 s (TTM as-of 340 s; daily load 217 s; daily features 284 s) |
| Peak RSS | **2784 MB** |
| Parallelism | batched DB reads; daily prices in chunks of 150 securities (avoided ~8 GB OOM) |

## Judgment calls

1. Vendor mcap used only for **ranking** when diluted shares missing (counted); size feature still uses computed mcap where available.
2. Delisting: return to last non-suspect `close_adj` through next month-end when possible.
3. TTM requires 4 consecutive quarters; else NaN (not partial sum).
4. Did **not** use `stock-database` monthly_panel builder.
5. Daily price load chunked to stay under RAM; not full `n_jobs=-1` across months for that stage (memory-bound).

## NOT done / skipped

- Classic single-factor portfolios / FF check
- Any ML models, CV, SHAP, backtests
- Merging `feat/panel` into `main` (human merges)
- Holdout fitting/tuning (holdout untouched)
- Writing to the stock database / schema changes
- Zero-lag fundamentals DB repair (loaders still guard)
- Refetch of pre-2005 truncated prices
