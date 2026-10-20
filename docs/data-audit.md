# Data audit — stock-database for the monthly factor model

Date: 2026-10-06 · Branch `docs/data-audit` · Read-only against `stockdb`.
Reproduce with `python scripts/run_data_audit.py` (≈2.5 min). Raw outputs go to `data/audit/`
(gitignored): `monthly_counts.parquet`, `coverage_*.parquet`, `delist_rate_12m.parquet`,
`table_stats.csv`, `lag_stats.csv`, `summary.json`, `report.md`.
Table semantics are in `docs/data-dictionary.md`.

## Summary

1. **The universe is survivorship-biased before ~2021.** Two independent causes:
   - **Vendor:** FMP's delisted-companies list (13,949 symbols) contains almost no delistings before
     2016 (about 70 in total), and few in 2016–2020. Within the universe, the share of names that delist
     in the next 12 months is **0.0% every year 1990–2014**, 0.6–2.8% in 2015–2019, 5.2% in 2020, and
     only reaches a realistic **9.5–13.1% from December 2021**.
   - **Crawl (fixable):** 602 delisted securities that listed before 2005 have **no prices before
     2005-09-19**. For a delisted name, the crawler's first window (2005-09-17 → 2026-09-17) returned
     fewer than 5,000 rows, so it never walked back (e.g. MXIM, listed 1994, delisted 2021). This
     produces the jump in the universe from 1,738 priced names (Dec 2004) to 2,426 (Dec 2005).
2. **SIVB passes:** it is in the universe every month from 2003-09-30 to 2023-02-28, passes the $5
   filter and is top-1000 in all 229 months, and drops out after its 2023-03-09 last trade. It is
   absent before 2003-09 only because of the crawl gap above (listed 1987).
3. **Fundamentals are point-in-time on `date_available`**, with one exception: **44,801 rows have
   `date_available` = midnight ET on the fiscal period end.** The loaders re-impute these.
   Median filing lag is **43 days for quarters and 73 days for annual reports**.
4. **Analyst estimates (`analyst_estimate`) are NOT point-in-time** and cannot be used. Earnings
   surprises (`earnings_event`) are usable with care, with coverage of 60–70% of all names and
   86–98% of the top 1000.
5. **Market cap must be `close × shares`**, not `close_adj × shares`. The computed series matches
   the vendor's historical mcap (median ratio 1.00 every year).
6. **No delisting returns** exist. A final price and a partial last-month return can be derived for
   4,073 of 4,092 delisted names. The return realised after delisting cannot.

## 1. Stocks per month-end

Definitions, all applied at month-end `t` using only information at `t`:
- **listed**: `first_trade_date ≤ t ≤ last_trade_date` (null `last_trade_date` = still listed).
- **priced**: listed, with a `price_daily` close on the month-end session.
- **≥ $5**: priced and `close_raw ≥ 5` (as-traded price; `close_raw` is never missing).
- **≥ $5 with mcap**: also has PIT market cap = `close` × latest-available quarterly
  `weighted_average_shares_diluted` (fallback basic), filed within 548 days.
- **top 1000**: largest 1,000 of the previous set by that market cap.

Survivorship columns: `later delisted` = share of priced names that are delisted today.
`delist ≤12m` = share of the set that delists within the next 12 months (a normal US rate is
roughly 6–10% for all names).

| month-end | listed | priced | ≥ $5 | ≥ $5 with mcap | top 1000 | later delisted | delist ≤12m (all) | delist ≤12m (top 1000) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1990-12 | 960 | 724 | 591 | 561 | 561 | 6.1% | 0.0% | 0.0% |
| 1993-12 | 1,266 | 942 | 848 | 787 | 787 | 7.4% | 0.0% | 0.0% |
| 1995-12 | 1,503 | 1,105 | 972 | 896 | 896 | 7.3% | 0.0% | 0.0% |
| 2000-12 | 2,103 | 1,497 | 1,225 | 1,154 | 1,000 | 9.1% | 0.0% | 0.0% |
| 2004-12 | 2,515 | 1,738 | 1,528 | 1,419 | 1,000 | 10.7% | 0.0% | 0.0% |
| 2005-12 | 2,639 | 2,426 | 2,100 | 1,852 | 1,000 | 33.2% | 0.0% | 0.0% |
| 2006-12 | 2,758 | 2,548 | 2,239 | 1,993 | 1,000 | 33.6% | 0.0% | 0.0% |
| 2008-12 | 3,001 | 2,773 | 1,986 | 1,761 | 1,000 | 34.9% | 0.0% | 0.0% |
| 2010-12 | 3,219 | 2,981 | 2,393 | 2,087 | 1,000 | 36.4% | 0.1% | 0.1% |
| 2015-12 | 4,268 | 4,022 | 3,268 | 2,878 | 1,000 | 41.5% | 1.5% | 1.5% |
| 2018-12 | 4,760 | 4,497 | 3,528 | 3,083 | 1,000 | 40.2% | 0.6% | 0.5% |
| 2020-12 | 5,549 | 5,298 | 4,364 | 3,760 | 1,000 | 42.7% | 5.2% | 1.4% |
| 2021-12 | 6,380 | 6,152 | 5,068 | 4,488 | 1,000 | 44.7% | 9.5% | 4.2% |
| 2022-12 | 6,091 | 5,886 | 4,376 | 3,931 | 1,000 | 40.3% | 13.1% | 3.6% |
| 2024-12 | 5,019 | 4,876 | 3,640 | 3,470 | 1,000 | 22.0% | 12.6% | 3.3% |
| 2025-12 | 4,668 | 4,571 | 3,507 | 3,405 | 1,000 | 10.7% | 10.4% | 3.6% |
| 2026-08 | 4,450 | 4,436 | 3,475 | 3,304 | 1,000 | 0.5% | 0.5% | 0.3% |

All 440 month-ends from 1990-01 to 2026-08 are in `data/audit/monthly_counts.parquet`. The top-1000
set has fewer than 1,000 names until 1997, so the cap does not bind before then.

How to read it:
- **Before 2005-09**, the universe is mostly today's survivors: 6–11% of names are "later delisted",
  versus 33–45% afterwards, and none of them delist within 12 months.
- **2005-09 to 2020**: dead names are present (they delisted after 2016) but deaths that happened
  in this window are almost entirely missing. Lehman, Enron-era and 2008–2015 bankruptcies are not
  in the database at all.
- **2021 onward**: the forward delisting rate looks realistic, and the SPAC boom and bust is visible
  (6,380 listed names at end-2021).
- **Benchmark:** CRSP has roughly 6–7k NYSE/AMEX/NASDAQ common stocks in the late 1990s and roughly
  3.5–4k in 2015–2020. The 1990s counts here (724–1,437 priced) are a small fraction of that.

**Other listing anomalies:**
- 32,192 month-end prices fall outside a security's listing dates (1,202 securities): 29,376
  before `first_trade_date` (vendor IPO date later than real trading) and 2,816 after
  `last_trade_date`.
- The counts above use listing dates. The panel task should decide whether a month-end price
  (non-suspect) is enough for membership.

## 2. Coverage by month

Each cell is the percentage of the set with a non-null value at month-end, averaged over that
year's months.
- `price` in the all-names table is priced ÷ listed. In the other two tables it is 100% by
  construction.
- Fundamentals are the latest quarterly row with `date_available` ≤ 16:00 ET on `t`, after the
  zero-lag guard, filed within 548 days.
- `EPS cons.` / `surprise` come from the latest `earnings_event` strictly before `t` and within
  120 days.
- `EPS est. (NOT PIT)` asks whether `analyst_estimate` has any target period in the next 400 days.
  It is shown only to size what is missing; it is not usable.
- Top-1000 mcap is 100% by construction (it is the ranking variable).

### All priced names

| year | price | mcap | book eq. | NI | revenue | GP | assets | debt | EPS cons. | surprise | EPS est. (NOT PIT) | vendor mcap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1990 | 76 | 94 | 71 | 94 | 94 | 94 | 71 | 71 | 0 | 0 | 0 | 99 |
| 1993 | 74 | 93 | 84 | 93 | 93 | 93 | 84 | 84 | 52 | 52 | 12 | 99 |
| 1995 | 74 | 91 | 88 | 91 | 91 | 91 | 88 | 88 | 56 | 56 | 27 | 99 |
| 2000 | 71 | 94 | 93 | 94 | 94 | 94 | 93 | 93 | 59 | 59 | 64 | 100 |
| 2005 | 77 | 91 | 91 | 91 | 91 | 91 | 91 | 91 | 62 | 62 | 64 | 100 |
| 2006 | 92 | 89 | 89 | 89 | 89 | 89 | 89 | 89 | 58 | 58 | 56 | 100 |
| 2010 | 92 | 88 | 88 | 88 | 88 | 88 | 88 | 88 | 63 | 63 | 57 | 100 |
| 2015 | 94 | 88 | 89 | 89 | 89 | 89 | 89 | 89 | 64 | 64 | 60 | 100 |
| 2020 | 95 | 87 | 88 | 87 | 87 | 87 | 88 | 88 | 68 | 68 | 65 | 100 |
| 2022 | 96 | 89 | 90 | 90 | 90 | 90 | 90 | 90 | 68 | 68 | 67 | 100 |
| 2024 | 96 | 94 | 95 | 94 | 94 | 94 | 95 | 95 | 78 | 78 | 87 | 100 |
| 2026 | 99 | 96 | 97 | 97 | 97 | 97 | 97 | 97 | 82 | 82 | 86 | 99 |

### Price ≥ $5

| year | mcap | book eq. | NI | revenue | GP | assets | debt | EPS cons. | surprise | EPS est. (NOT PIT) | vendor mcap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1990 | 95 | 74 | 95 | 95 | 95 | 74 | 74 | 0 | 0 | 0 | 100 |
| 1993 | 93 | 84 | 93 | 93 | 93 | 84 | 84 | 57 | 57 | 13 | 99 |
| 1995 | 92 | 90 | 93 | 93 | 93 | 90 | 90 | 62 | 62 | 29 | 99 |
| 2000 | 94 | 93 | 94 | 94 | 94 | 93 | 93 | 65 | 65 | 71 | 100 |
| 2005 | 91 | 91 | 91 | 91 | 91 | 91 | 91 | 67 | 67 | 69 | 100 |
| 2006 | 89 | 89 | 89 | 89 | 89 | 89 | 89 | 63 | 63 | 60 | 100 |
| 2010 | 87 | 87 | 87 | 87 | 87 | 87 | 87 | 70 | 70 | 64 | 100 |
| 2015 | 88 | 89 | 89 | 89 | 89 | 89 | 89 | 68 | 68 | 65 | 100 |
| 2020 | 86 | 88 | 87 | 87 | 87 | 88 | 88 | 71 | 71 | 68 | 100 |
| 2022 | 89 | 90 | 89 | 89 | 89 | 90 | 90 | 67 | 67 | 68 | 100 |
| 2024 | 94 | 95 | 94 | 94 | 94 | 95 | 95 | 80 | 80 | 87 | 100 |
| 2026 | 96 | 96 | 97 | 97 | 97 | 96 | 96 | 85 | 84 | 86 | 99 |

### Top 1000 by market cap (≥ $5)

| year | book eq. | NI | revenue | GP | assets | debt | EPS cons. | surprise | EPS est. (NOT PIT) | vendor mcap |
|---|---|---|---|---|---|---|---|---|---|---|
| 1990 | 78 | 100 | 100 | 100 | 78 | 78 | 0 | 0 | 0 | 99 |
| 1993 | 91 | 100 | 100 | 100 | 91 | 91 | 60 | 60 | 13 | 99 |
| 1995 | 97 | 100 | 100 | 100 | 97 | 97 | 66 | 66 | 31 | 99 |
| 2000 | 100 | 100 | 100 | 100 | 100 | 100 | 77 | 77 | 82 | 100 |
| 2005 | 100 | 100 | 100 | 100 | 100 | 100 | 87 | 87 | 86 | 100 |
| 2010 | 100 | 100 | 100 | 100 | 100 | 100 | 89 | 89 | 81 | 100 |
| 2015 | 100 | 100 | 100 | 100 | 100 | 100 | 90 | 90 | 85 | 100 |
| 2020 | 100 | 100 | 100 | 100 | 100 | 100 | 93 | 93 | 89 | 100 |
| 2024 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 98 | 99 | 100 |
| 2026 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 98 | 99 | 100 |

Observations:
- **~11–13% of priced names have no fundamentals** in 2006–2020, falling to 3% by 2026. The top
  1000 have essentially complete fundamentals from 1995 onward.
- In 1990–1994, balance-sheet items (book equity, assets, debt) lag income items by 10–23 points:
  early quarterly rows often lack a balance sheet.
- Using FY rows as a fallback adds only ~1 point of book-equity coverage (`book_equity_any_period`
  in the parquet).
- Earnings consensus and surprise data starts in 1993 (some 1985–1992 events exist but rarely
  carry an estimate).
- Vendor mcap covers ~100% but is only a fallback (data dictionary).

### Earliest month with > 80% coverage of price, mcap, book equity, and net income

| set | first month | sustained from (never drops below afterwards) |
|---|---|---|
| all priced names (price = priced ÷ listed) | 2005-09-30 | 2005-09-30 |
| price ≥ $5 | 1991-04-30 | 1991-04-30 |
| top 1000 | 1990-04-30 | 1990-04-30 |

For the all-names set, price coverage is the binding item, and it is the crawl gap in §Summary.
For the filtered sets, coverage alone would allow a 1990–1991 start. Survivorship does not.

## 3. Filing lag (fiscal period end → `date_available`)

Lag is measured as the ET calendar date of `date_available` minus `fiscal_period_end`.

| period | rows (excl. zero-lag) | median | p10 | p90 |
|---|---:|---:|---:|---:|
| quarterly (Q1–Q4) | 455,025 | **43 days** | 32 | 86 |
| annual (FY) | 117,558 | **73 days** | 51 | 91 |

Including the zero-lag rows, the medians would be 41 and 70 days.

By source (all rows):

| source | Q median | FY median | rows |
|---|---:|---:|---:|
| `fmp_accepted` (vendor SEC acceptance) | 40 | 62 | 530,581 |
| `imputed_lag` (+45 / +90 days) | 45 | 90 | 71,554 |
| `sec_filing` | 43 | 74 | 10,460 |
| `edgar` | 44 | 75 | 4,784 |

The median quarterly lag falls from 45 days (1985–2004 period ends) to 38–40 days (2005+),
consistent with the SEC's accelerated-filer deadlines.

**Zero-lag hazard:** 44,801 rows (33,454 quarterly) have `date_available` = 00:00 ET on the period
end, with `filing_date` = period end and `date_source = 'fmp_accepted'`. Quarterly rows by
period-end bucket:

| 1985–89 | 1990–94 | 1995–99 | 2000–04 | 2005–09 | 2010–14 | 2015–19 | 2020–24 | 2025+ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3,729 | 9,515 | 2,563 | 1,754 | 1,875 | 2,931 | 4,650 | 5,562 | 875 |

The stock-database repair targeted UTC-midnight placeholders, and these are ET-midnight, so they
slipped through. `loaders.fundamentals()` re-imputes them to period end + 45 days (Q1–Q3) or
+ 90 days (Q4/FY) at 16:00 ET, which is the stock-database's own `imputed_lag` convention.
The repair belongs in stock-database (see hazards).

## 4. Delistings

There is no delisting-return table, no delisting reason, and no post-delisting price.

| check | count |
|---|---:|
| delisted securities (`last_trade_date` not null) | 4,092 |
| with any price | 4,073 |
| last price date = `last_trade_date` | 1,836 |
| last price within 0–5 days before `last_trade_date` | 2,600 |
| last price **after** `last_trade_date` | 1,318 |
| gap `last_trade_date − last price`: median / p90 / p99 | 0 / 1 / 371 days |
| last raw close < $1 (likely distress delisting) | 584 |
| delisting year: ≤2015 / 2016–20 / 2021–26 | 25 / 407 / 3,660 |

What can be derived:
- **Final price**: `close` / `close_raw` / `close_adj` on the last price date, for 4,073 names.
- **Partial last-month return**: `close_adj(last day) / close_adj(previous month-end) − 1`, for every
  delisted name with a prior month-end price.
- Use **`max(last_trade_date, last price date)`** as the effective delisting date. The vendor date is
  earlier than the last price for 1,318 names.

What cannot be derived:
- The return between the last exchange price and what holders actually received (merger
  consideration, OTC trading, bankruptcy recovery).
- Merger vs. performance delisting (`archive.merger_acquisition` has only 4.7k recent rows).

**Proposal for the panel task** (a decision, not done here):
- Target for the delisting month = partial-month return to the last price.
- As a sensitivity, apply a Shumway-style penalty of −30% to likely performance delistings
  (last raw close < $1, or a final-month drawdown greater than 50%).

## 5. Proposed config values (not applied — `configs/default.yaml` is unchanged)

| key | current | proposed | why |
|---|---|---|---|
| `sample.start` | 2000-01-31 (TBD) | **2006-01-31** | First full year after the 2005-09-19 crawl floor. All-names core coverage is >80% and stays there from 2005-09. Before 2005 the universe is ~90% today's survivors. Momentum (12-1) needs 12 months of prices, so the first fully-populated month is 2006-09. |
| `sample.end` | 2026-08-31 | **2026-07-31** | Last month-end with a *realised* 1-month forward return. Prices end 2026-09-17, so 2026-08-31 has no t+1 close. |
| `sample.holdout_start` | 2024-01-31 (TBD) | **2024-01-31** (keep) | 31 holdout months (as-of 2024-01 to 2026-07) in the cleanest era (delisting coverage normal since 2021). Training/validation runs 2006-01 to 2023-12: 216 months, enough for `min_train_months: 60` plus about 13 walk-forward years. |
| `universe.min_price` | 5.0 | **5.0** on `close_raw` | Applied to the as-traded price, not the split-adjusted `close`. |
| `universe.top_n_by_market_cap` | null | **1000** | ≥1,000 eligible names every month since 1997. ~100% fundamentals, 86–98% earnings-surprise coverage. Least exposed to survivorship (top-1000 12-month delisting rate is 3–4% even in the clean era, mostly acquisitions). Run all ≥$5 names (~2,000–3,500) as a robustness check. |
| `universe.min_market_cap_pctile` | 0.0 | 0.0 | Redundant with top-N. |
| `target.type` | excess_return (TBD) | **rank** for training; report portfolios in excess of the 3-month T-bill | Cross-sectional rank is robust to the 2008/2020 return-scale shifts. `treasury_rate.month3` covers 1990+; `month1` starts only 2001-03. |
| `portfolio.weighting` | equal | equal (primary) + value (secondary) | Value weights on `close × shares` mcap are available ~100% in the top 1000. |

**About survivorship in the training years.** Choosing 2006 does not remove the bias in 2006–2020:
names that died in those years are absent. The bias is unavoidable with this database short of
starting in 2021, which leaves too little history. The proposal is therefore to:
1. Train from 2006 and state the bias explicitly.
2. Report walk-forward results separately for 2006–2020 (biased) and 2021–2023 (clean), with the
   holdout entirely in the clean era.
3. Prefer the top-1000 universe, where the missing deaths are rarer and mostly acquisitions.

**About an earlier start.** It would need both of these:
- **Refetch the pre-2005 price windows for the 602 truncated names.** This is a stock-database task,
  possible only while the FMP subscription is active, which is likely until about mid-October 2026.
- **Find a source for pre-2016 delistings.** FMP does not have them.

## 6. Point-in-time hazards found

| # | Hazard | Severity | Handling |
|---|---|---|---|
| 1 | Pre-2016 delistings absent from the vendor list; 2016–2020 under-covered | **High** (survivorship) | Start 2006, report the 2021+ era separately, prefer top 1000; needs another data source to fix |
| 2 | 602 delisted names have no prices before 2005-09-19 (crawl stopped walking back) | **High** before 2005; **fixable** | Start ≥ 2005-10; ask stock-database to refetch while FMP is live |
| 3 | 44,801 fundamentals rows have `date_available` = period-end midnight ET | **High** (40–90 days of lookahead) | Re-imputed in `loaders.fundamentals()`; should also be repaired in stock-database |
| 4 | `analyst_estimate` is a current snapshot (`date_available` NULL, one row per target period) | **High** if used | Never a feature; no revision or dispersion features possible |
| 5 | `earnings_event` rewritten by the vendor in 2023–2026 (`last_updated`); EPS on today's split basis | Medium | Use surprise only after `report_date`, scaled by split-adjusted `close`; treat as PIT-caution |
| 6 | Statement share counts / EPS restated to today's split basis | Medium (wrong mcap if mixed) | Pair with `close`, never `close_raw`; never `close_adj × shares` |
| 7 | Statement values are the vendor's latest version (restatements possible), dated at the original filing | Medium, unmeasurable | Known FMP limitation; no as-reported table was crawled |
| 8 | `trading_day` flags 2026-09-17 as a month-end | Low | `month_ends(complete_only=True)` drops it |
| 9 | `first_trade_date` later than real trading for 1,202 names; vendor `last_trade_date` earlier than the last price for 1,318 | Low–Medium | Use price presence plus listing dates; effective delist = max of the two |
| 10 | `company` / `company_attribute_history` sector is today's value | Low | Only for industry-neutral ranking, stated as an assumption |
| 11 | `corporate_action` has 219 future ex-dates and no declaration date | Low | Only matters for dividend-yield features |
| 12 | `earnings_event` has 3,804 future scheduled rows and 25 junk 1970-01-01 rows | Low | `asof_earnings` uses `report_date < t`; rows with null actuals give no surprise |
| 13 | The stock-database `monthly_panel` builder uses `close_adj × shares` and has no zero-lag guard | Medium if reused | Do not reuse; the table is empty anyway |
| 14 | Weighted-average shares ≠ period-end shares outstanding (lag and averaging) | Low | No shares-outstanding table exists; document it |

## 7. Wall time and resources (tradinghost, 8 cores, 11 GB RAM)

Full run: **153 s wall, 3.6 GB peak RSS.** All per-table queries run in an 8-thread pool, one
batched query per table.

| step | wall |
|---|---:|
| Table stats (26 tables, in parallel) | 112 s, dominated by `altdata.institutional_holder_position` (37M rows, 111 s) and `market.market_cap_daily` (31M rows, 56 s) |
| Filing-lag stats (one SQL) | 2 s |
| Monthly audit loads (8 in parallel) | 30 s; the slowest are month-end prices (29.6 s), quarterly fundamentals (23 s) and vendor mcap (23 s) |
| Pandas joins and coverage (440 months × up to 6.4k names) | 9 s |

The run is database-bound: the Python process used ~23% CPU, and Postgres parallel workers do the
scanning.
