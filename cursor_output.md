# Cursor output — data audit (docs/data-audit)

Date: 2026-10-06 · Branch: `docs/data-audit` (from `main` @ `601502c`) · Pushed, not merged.

## Read this first

1. **The database is survivorship-biased before about 2021.** FMP's delisted-companies list has
   almost no delistings before 2016 (~70 in total) and few in 2016–2020. Inside the universe, the
   share of names that delist in the next 12 months is 0.0% every year 1990–2014, 0.6–2.8% in
   2015–2019, 5.2% in 2020, and a realistic 9.5–13.1% only from December 2021. This is a vendor
   limit, not a loading bug.
2. **Action needed in stock-database while the FMP subscription is still live (~mid-October):**
   602 delisted securities that listed before 2005 have no prices before 2005-09-19. The crawler's
   first EOD window (2005-09-17 → 2026-09-17) returned fewer than 5,000 rows because each name
   stopped trading, so it never walked back to older windows (e.g. MXIM, listed 1994, delisted
   2021). Once the subscription lapses, this cannot be refetched.
3. **Lookahead in `fundamental_quarterly`:** 44,801 rows have `date_available` = 00:00 ET on
   `fiscal_period_end`, labelled `fmp_accepted`. The earlier stock-database repair only caught
   UTC-midnight placeholders. The loaders here re-impute them (+45 days for Q1–Q3, +90 days for
   Q4/FY, at 16:00 ET). The repair should also be made in stock-database.
4. **`analyst_estimate` is not point-in-time** (current consensus per target period, `date_available`
   NULL). It cannot be a feature.

## Commits

```
ee6fb53 Add data audit with coverage numbers and config proposals
c63b18f Add data dictionary for stock-database tables
6050c87 Add data audit module and CLI
08e1994 Guard zero-lag fundamentals and drop partial final month
548cc02 Add batched read-only data loaders and dev-DB smoke tests
4afe56a Anchor data/ and outputs/ ignores to repo root
```
plus:
```
3268141 Add cursor_output.md and task file for the data audit
e6e9e76 Revert unrelated config/decisions edits swept in by git add -A
(final) Update cursor_output.md
```

**Workflow note.** While this task was running, `configs/default.yaml` and `docs/decisions.md`
were edited in the working tree at 16:48. The edits set `top_n_by_market_cap: 1000`, add
`robustness_top_n: 500`, and add a decisions row. The task's `git add -A` swept them into
`3268141`. Since that commit was already pushed, I did not rewrite history:
- `e6e9e76` reverts both files, so the branch's net diff for them versus `main` is empty.
- Your edits are restored in the working tree as **uncommitted** changes. Commit them on their
  own branch.

**Conflict with the audit.** The new decisions row says the top-1000 universe is
"survivorship-free by construction". The audit shows it is not before ~2021. Re-ranking at `t`
only helps if dead names exist in the database, and pre-2016 deaths do not. The top-1000
12-month-forward delisting rate is 0.0% in 2006–2014, versus 3–4% from 2021. Please reword that
reason before committing it.

## What was done

- **`.env`**: created (gitignored, mode 600) with `STOCKDB_DSN` → `stockdb` and `STOCKDB_DEV_DSN` →
  `stockdb_dev`, using the stock-database credentials. Not committed.
- **`.gitignore` fix**: the unanchored `data/` rule also matched `src/factor_model/data/`, so
  **`db.py` and the package `__init__.py` were never committed on `main`**. Changing it to `/data/`
  and `/outputs/` fixes that; both files are now committed on this branch.
- **`src/factor_model/data/loaders.py`** (read-only, one batched query per table):
  - Readers: `month_ends`, `securities`, `symbol_history`, `prices_on_dates`, `price_span`,
    `fundamentals`, `earnings_events`, `analyst_estimates` (audit only), `vendor_market_cap_on_dates`.
  - Point-in-time helpers: `asof_cutoff` (16:00 ET), `asof_fundamentals` (merge-asof on
    `date_available`, with a staleness limit), `asof_earnings` (`report_date < t`),
    `guard_zero_lag`, `listed_on`.
- **`src/factor_model/data/audit.py`** and **`scripts/run_data_audit.py`**: the audit itself.
  Independent queries run in an 8-thread pool, and outputs go to `data/audit/` (gitignored).
- **Tests**:
  - `tests/test_loaders_dev.py`: 5 smoke tests against `stockdb_dev`, skipped unless
    `STOCKDB_DEV_DSN` is set.
  - `tests/test_loaders_unit.py`: 3 offline tests (zero-lag guard, 16:00 ET cutoff, a lookahead
    shift test, same-day earnings exclusion).
  - Result: `pytest` 9 passed; `ruff` clean.
- **`docs/data-dictionary.md`** and **`docs/data-audit.md`**.

## Schema summary

- **Identity**:
  - `security` has 8,527 qualifying US common stocks, 4,092 of them delisted (join on
    `security_id`).
  - `symbol_history` holds ticker ranges.
  - `company` sector/industry are today's values.
  - `index_membership` is empty.
- **Prices**: `price_daily` has 29.0M rows, 1970 → 2026-09-17.
  - `close`, OHLC and `volume` are split-adjusted to today's basis.
  - `close_raw` is the as-traded price.
  - `close_adj` is split- and dividend-adjusted.
  - 60,833 rows are flagged `price_quality='suspect'`.
- **Fundamentals**: `fundamental_quarterly` has 617k rows (Q1–Q4 and FY as separate rows), joined
  on `date_available` (SEC acceptance). Share counts and EPS are restated to today's split basis.
- **Market cap**: compute as `close × weighted_average_shares_diluted`. It matches the vendor
  series in `market.market_cap_daily` (median ratio 1.00 every year). Do **not** use
  `close_adj × shares`, which is what the stock-database `monthly_panel` builder does.
- **Shares outstanding**: no table exists; only statement weighted-average shares.
- **Earnings**:
  - `earnings_event` (1985+, usable from ~1993) gives actual and consensus EPS at each report. It
    is PIT-caution: the vendor rewrote it in 2023–2026.
  - `analyst_estimate` is not PIT.
  - `analyst_action` starts 2011, `analyst_consensus_daily` 2016, and price targets 2021.
- **Beta / volatility / turnover**: `close_adj` returns, `^GSPC` in `market.index_price_daily`
  (from 1970), `treasury_rate` (`month3` from 1990, `month1` from 2001), `volume` (split-adjusted).

## Key numbers

Stocks per month-end (December):

| | 1995 | 2000 | 2005 | 2010 | 2015 | 2020 | 2022 | 2025 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| listed | 1,503 | 2,103 | 2,639 | 3,219 | 4,268 | 5,549 | 6,091 | 4,668 |
| priced | 1,105 | 1,497 | 2,426 | 2,981 | 4,022 | 5,298 | 5,886 | 4,571 |
| ≥ $5 (`close_raw`) | 972 | 1,225 | 2,100 | 2,393 | 3,268 | 4,364 | 4,376 | 3,507 |
| top 1000 by mcap | 896 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 |
| delist within 12m (all) | 0.0% | 0.0% | 0.0% | 0.1% | 1.5% | 5.2% | 13.1% | 10.4% |

Coverage (yearly mean, % non-null):

| | price | mcap | book eq. | NI | revenue | GP | assets | debt | EPS consensus / surprise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all priced, 2010 | 92 | 88 | 88 | 88 | 88 | 88 | 88 | 88 | 63 |
| ≥ $5, 2010 | 100 | 87 | 87 | 87 | 87 | 87 | 87 | 87 | 70 |
| top 1000, 2010 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 89 |
| all priced, 2024 | 96 | 94 | 95 | 94 | 94 | 94 | 95 | 95 | 78 |

- **Earliest month with >80% coverage** of price, mcap, book equity and net income (sustained):
  2005-09-30 for all names (limited by the price crawl gap), 1991-04-30 for ≥ $5, and 1990-04-30
  for the top 1000.
- **Median filing lag** (zero-lag rows excluded): 43 days for quarterly reports, 73 days for annual.
- **SIVB** is in the universe every month from 2003-09 to 2023-02 (229 months; ≥ $5 and top 1000
  in all of them) and drops out after its 2023-03-09 last trade. It is missing for 1987–2003 only
  because of the crawl gap.
- **Delistings**:
  - No delisting returns or reasons exist.
  - A final price and a partial last-month return are derivable for 4,073 of 4,092 names.
  - The vendor `last_trade_date` is earlier than the last price for 1,318 names, so use the later
    of the two.

## Proposed config values (not applied; `configs/default.yaml` unchanged)

- `sample.start: 2006-01-31`.
  - It is the first full year after the 2005-09 price floor, and all-names core coverage stays
    above 80% from then on.
  - Before 2005 the universe is ~90% today's survivors.
- `sample.end: 2026-07-31`, not 2026-08-31.
  - 2026-08-31 has no realised t+1 return, because prices end 2026-09-17.
- `sample.holdout_start: 2024-01-31` (keep).
  - This gives 31 holdout months, all in the era where delisting coverage looks normal.
  - Training and validation get 216 months (2006-01 to 2023-12).
- Universe: `top_n_by_market_cap: 1000` with `min_price: 5.0` applied to `close_raw`.
  - There are ≥1,000 eligible names every month since 1997.
  - Fundamentals coverage is ~100% and earnings-surprise coverage 86–98%.
  - It is the set least exposed to the missing deaths.
  - Run all ≥ $5 names as a robustness check.
- `target.type: rank` for training, with portfolios reported in excess of the 3-month T-bill.
- **Caveat:** 2006–2020 still lacks most of the deaths that happened in those years. Report
  walk-forward results separately for 2006–2020 and 2021–2023. The holdout is entirely in the
  clean era.

## Point-in-time hazards (details in docs/data-audit.md §6)

1. Pre-2016 delistings are missing from the vendor list (survivorship). Not fixable from FMP.
2. 602 delisted names have no pre-2005-09-19 prices. A crawl gap, fixable while FMP is live.
3. 44,801 zero-lag fundamentals rows. Guarded in the loaders; the stock-database repair is pending.
4. `analyst_estimate` is a current snapshot. Never use it.
5. `earnings_event` was rewritten by the vendor and EPS is on today's split basis. PIT-caution.
6. Shares and EPS are on today's split basis. Pair them with `close`, never with `close_raw` or
   `close_adj`.
7. Statement values may include later restatements while keeping the original filing date. This
   cannot be measured.
8. `trading_day` flags 2026-09-17 as a month-end. `month_ends()` drops it.
9. Listing dates disagree with prices for ~2.5k securities (`first_trade_date` too late for 1,202,
   `last_trade_date` too early for 1,318).
10. Sector and industry are today's values.
11. `corporate_action` has 219 future ex-dates and no declaration dates.
12. `earnings_event` has 3,804 future rows and 25 junk rows dated 1970-01-01.
13. The stock-database `monthly_panel` builder has a market-cap error (`close_adj × shares`) and
    no zero-lag guard. Do not reuse it.
14. Weighted-average shares are not the same as period-end shares outstanding.

## Wall time

- **Full audit:** 153 s wall, 3.6 GB peak RSS, on 8 threads (one batched query per table). It is
  database-bound; the Python process used ~23% CPU.
- **Table stats:** 112 s. The 37M-row 13F holder table took 111 s and `market_cap_daily` 56 s,
  running in parallel.
- **Monthly loads:** 30 s in parallel (month-end prices 29.6 s, fundamentals 23 s, vendor mcap 23 s).
- **Pandas joins:** 9 s.

## Judgment calls

- **Information cutoff of 16:00 ET** on the month-end session. A filing accepted after the close on
  `t` is not usable at `t`. The stock-database builder used end of day, which is looser.
- **Fundamentals use quarterly rows only**, so quarterly and annual flows are never mixed. Adding
  an FY fallback gains only ~1 point of coverage.
- **Fundamentals staleness limit of 548 days**, the same as the stock-database builder. Earnings
  events must be ≤120 days old.
- **The zero-lag guard is on by default** (`fundamentals(guard=True)`). It changes the data the
  loaders return, as opposed to only reporting the problem. The alternative was shipping a
  loader with known lookahead.
- **Universe counts use listing dates.** Price-only membership would add up to 29k name-months.
  Left for the panel task to decide.
- **A light venv** (pandas, sqlalchemy, psycopg, pyarrow, pytest, ruff) was installed instead of
  the full `pyproject` dependencies (torch, xgboost, …), which are not needed here.
- **`cursor_task.md` is committed** with this branch, because the task's final step is
  `git add -A`.

## Possible design flaws

- **`db.py` was never committed on `main`.** This came from the `.gitignore` issue above and is
  fixed on this branch. Anyone cloning `main` gets no database module.
- **The stock-database panel builder computes market cap as `close_adj × shares`.** That
  understates historical market cap by the cumulative dividend factor (about 17% for AAPL in 2010)
  and inflates book-to-market for older dates.
- **The `fundamental_quarterly_pit` view** only filters `accepted_date_placeholder` (0 rows today).
  It does not catch the ET-midnight zero-lag rows, so it gives a false sense of safety.

## NOT done / skipped

- **No universe, panel, factors, or models were built** (per the task).
- **`configs/default.yaml` was not changed.** The proposals are only in `docs/data-audit.md`.
- **Nothing was written to either database, and nothing was changed in `~/stock-database`.** The
  crawl-gap refetch and the zero-lag repair belong there and need a separate stock-database task.
- **No external benchmark comparison** (e.g. CRSP counts). The survivorship evidence is internal:
  the forward delisting rate and the vendor delisted-list dates.
- **Not audited:** `esm.*`, `options_*`, `text.*`, `funds.*`, and most of `altdata.*` (ESG,
  congressional, COT). They were not needed for the monthly equity panel.
- **Not measured:**
  - **Restatements in statement values** (no as-reported data was crawled).
  - **Whether `earnings_event.eps_estimated` matches the true pre-announcement consensus.**
- **No timing-adjusted (BMO/AMC) earnings availability** beyond the conservative
  `report_date < t`. Timing is known for only 49% of events.
