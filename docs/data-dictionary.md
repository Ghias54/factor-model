# Data dictionary — stock-database tables used by FE-Factor-Model

Source: the `stockdb` Postgres on tradinghost (schema from `~/stock-database/migrations/sql/`,
column comments in the database itself). Snapshot taken 2026-10-06; the vendor crawl ended
2026-09-17. Row counts and date ranges come from `scripts/run_data_audit.py`
(`data/audit/table_stats.csv`). Numbers behind every claim are in `docs/data-audit.md`.

Access is only through `src/factor_model/data/db.py` (read-only session) and the batched
loaders in `src/factor_model/data/loaders.py`.

## How to read the PIT column

- **PIT-safe**: has an availability clock that can be compared with month-end `t`.
- **PIT-caution**: dated, but the vendor value may have been revised after the date (no
  archive of the original). Usable with a stated assumption.
- **NOT PIT — never a feature**: a "now" snapshot or a table with no as-of date. Audit use only.

## Global conventions that matter for every join

| Topic | Rule |
|---|---|
| Identity | Join on `security_id`, never on ticker. Tickers are recycled (`symbol_history`, half-open `[valid_from, valid_to)`). |
| Month-end grain | `trading_day.is_month_end` (last XNYS session). **The last loaded session (2026-09-17) is also flagged** — drop it (`loaders.month_ends(complete_only=True)`). |
| Information cutoff | A value is usable at month-end `t` if public by **16:00 ET on `t`** (`loaders.asof_cutoff`). |
| Price basis | `close`, OHLC, `volume` are split-adjusted to **today's** share basis. `close_raw` is the price as traded. `close_adj` is split- and dividend-adjusted. |
| Share basis | Statement share counts and per-share items (EPS) are restated by FMP to **today's** split basis (AAPL FY2010 shows 25.5B shares, not 0.9B as filed). |
| Market cap | `close × weighted_average_shares_diluted` (both on today's split basis). **Not** `close_adj × shares`: dividend adjustment shrinks historical mcap. |
| Price filter | Use `close_raw` (as traded). `close` would turn AAPL 2010 ($322) into $11.52. |
| Returns | Use `close_adj`; exclude rows with `price_quality = 'suspect'`. |

## Identity and listing history

### `public.security` — 8,527 rows · PIT-safe (for listing dates)
| | |
|---|---|
| PK | `security_id` |
| Key columns | `company_id`, `share_class` (CUSIP), `is_active` (audit only, never a filter), `first_trade_date`, `last_trade_date` (non-null = delisted) |
| Range | `first_trade_date` 1912-06-26 … ; `last_trade_date` up to 2026-09-17 |
| Notes | 4,092 delisted, 4,425 active, 10 inactive with no `last_trade_date`, 50 with no `first_trade_date`. Every row is a qualifying US-listed common stock (all 8,527 map to `universe_member.qualifies`). **Delisted names are almost all 2016+** (vendor delisted list; see audit §1). `first_trade_date` is the vendor IPO date and is sometimes *later* than real trading (29,376 month-end prices fall before it). |

### `public.symbol_history` — 8,528 rows · PIT-safe
PK `(security_id, symbol, valid_from)`. Ticker ranges, half-open. Exclusion constraint forbids two
securities holding a ticker at once. Display / external-join only.

### `public.company` — 7,180 rows · NOT PIT for `sector`/`industry`/`name`
PK `company_id`; `cik` unique. Sector/industry/name are today's profile values.

### `public.company_attribute_history` — 21,420 rows · PIT-caution
PK `(company_id, attr_name, valid_from)`. SCD-2 for name/sector/industry, but **every row is the
single open version from today's profile** (0 closed versions). Today's sector projected backward:
acceptable for industry-neutral ranking if stated, never as a signal.

### `public.universe_member` / `public.universe_criteria`
Classification log (108,546 rows) of every vendor symbol against the `us_listed_common` rule
(NYSE/NASDAQ/NYSE American common; no ETFs, funds, ADRs, units, warrants, rights, preferreds).
Already applied: `security` holds the qualifiers. No dates — not a time-varying universe.

### `public.symbol_rename` — 101 rows
Ticker-change reconciliation (FMP, SEC, NASDAQ Trader) with confidence/conflict flags. Not needed if joining on `security_id`.

### `public.index_membership` — **0 rows**
Schema exists (historical S&P 500 etc.) but nothing is loaded. Cannot build an S&P 500 universe from it.

## Prices

### `public.price_daily` — 28,959,170 rows · PIT-safe
| | |
|---|---|
| PK | `(security_id, date)`; partitioned by year; `date` FK → `trading_day` |
| Join on | `date` (= session `t`; close is known at `t`) |
| Range | 1970-01-02 … 2026-09-17; 8,501 securities |
| `close`, `open`, `high`, `low`, `vwap`, `volume` | From `/historical-price-eod/full`: **split-adjusted** (volume too). Not dividend-adjusted. |
| `close_split_adj` | Same as `close` (kept for clarity). |
| `close_adj` | From `/dividend-adjusted`: **split + dividend adjusted**. Use for returns. Never null where `close` is. |
| `close_raw` | From `/non-split-adjusted` (name is misleading): **raw as-traded** close. Use for $ price filters. |
| `price_quality` / `_reason` | `suspect` on 60,833 rows (`adj_factor_unstable`, `adj_step_no_action`, `unconfirmed_return`, `close_flip`, `close_split_break`, `ticker_reuse_other_instrument`). Exclude from return calcs. |
| `split_repair_factor` / `close_vendor` / `close_adj_vendor` | Audit trail of stock-database repairs (81,514 rows rescaled for unapplied splits). |
| Hazards | **Delisted names' history starts 2005-09-19** for 602 securities that listed earlier (crawl gap, audit §1). 2.15M rows have zero/null volume. |

### `public.trading_day` — 14,299 rows
PK `date`. XNYS calendar 1970-01-02 … 2026-09-17, `is_month_end` flags. See month-end caveat above.

### `archive.price_daily_closed_day` — 7,515 rows
Vendor prints on days the exchange was closed, quarantined out of `price_daily`. Ignore.

### `public.corporate_action` — 255,584 rows · PIT-caution
PK `action_id`. `(security_id, ex_date, action_type, ratio_or_amount)`; 10,524 splits, the rest
dividends (`dividend_quarterly`, `_monthly`, `_special`, …). Range 1962 … 2029-06-15
(**219 future ex-dates**). No declaration date, so a dividend is only knowable for certain at its
ex-date. Already reflected in `close_adj`; only needed for dividend-yield-type features.

### `market.market_cap_daily` — 31,233,427 rows · PIT-caution (reconciliation / fallback only)
PK `(security_id, date)`; `date_available = date`. Vendor `/historical-market-capitalization`,
1970 … 2026-09-18, ~100% coverage of priced names. Agrees with computed `close × shares`
(median ratio 1.00 every year; 85–94% of rows within ±20%). The vendor's share count is not
auditable, so prefer computed mcap; vendor mcap may fill gaps for size *ranking* only.

### `market.index_price_daily` / `market.index_instrument` — 1,663,390 rows / 427 indexes · PIT-safe
PK `(symbol, date)`. `^GSPC` from 1970-01-02. `close_adj` from the dividend-adjusted endpoint.
Market factor for beta. Not linked to `security` (by design).

### `public.treasury_rate` — 9,187 rows · PIT-safe
PK `date`. 1990-01-02 … 2026-09-18; `month1`, `month3`, … `year30` (percent). Risk-free rate for excess returns.

## Fundamentals

### `public.fundamental_quarterly` — 617,384 rows · PIT-safe **after the zero-lag guard**
| | |
|---|---|
| PK | `(security_id, fiscal_period_end, fiscal_period)` — Q1–Q4 and FY are separate rows (488,479 quarterly, 128,905 annual) |
| Join on | **`date_available`** (timestamptz; SEC acceptance). Never `fiscal_period_end`. |
| Range | `date_available` 1983-09-28 … 2026-09-28; `fiscal_period_end` 1983-06-30 … 2026-08-31; 7,643 securities |
| Clock provenance | `date_source`: `fmp_accepted` (530,581), `imputed_lag` (71,554: period end +45d Q / +90d FY), `sec_filing`, `edgar`, `fmp_filing`; `accepted_at`, `filing_date`, `filer_type` |
| Income statement | `revenue`, `cost_of_revenue`, `gross_profit`, `research_and_development`, `selling_general_and_admin`, `operating_income`, `interest_expense`, `ebitda`, `income_before_tax`, `income_tax_expense`, `net_income`, `eps`, `eps_diluted`, `weighted_average_shares`, `weighted_average_shares_diluted` |
| Balance sheet | `cash_and_equivalents`, `short_term_investments`, `accounts_receivable`, `inventory`, `total_current_assets`, `ppe_net`, `goodwill`, `intangible_assets`, `total_assets`, `accounts_payable`, `short_term_debt`, `total_current_liabilities`, `long_term_debt`, `total_liabilities`, `total_equity` (book equity), `total_debt` |
| Cash flow | `depreciation_and_amortization`, `stock_based_compensation`, `change_in_working_capital`, `cash_from_operations`, `capex`, `cash_from_investing`, `dividends_paid`, `cash_from_financing`, `free_cash_flow` |
| Other | `raw_payload` (full vendor JSON); `accepted_date_placeholder` (0 rows true today) |
| Hazards | (1) **44,801 rows have `date_available` = 00:00 ET on the period-end day** (filing_date = period end, labelled `fmp_accepted`): 40–90 days of lookahead if used raw. `loaders.fundamentals()` re-imputes them (33,454 quarterly rows). (2) One row per period = the vendor's current version; values may include later restatements while keeping the original filing date. (3) Shares/EPS are on today's split basis. (4) Flow items on Q rows are quarterly, on FY rows annual — never mix; TTM must be summed from four Q rows. |

### `public.fundamental_quarterly_pit` (view)
`fundamental_quarterly WHERE NOT accepted_date_placeholder`. Lacks `date_source`/`filer_type`/
`accepted_at`, and does **not** catch the zero-lag rows. The loaders read the base table with the
same filter plus the guard.

### `public.owner_earnings` — 222,610 rows · PIT-safe (inherited clock)
PK `(security_id, fiscal_period_end, period)`. Vendor owner earnings / maintenance vs growth capex.
`date_available` copied from the matching statement row. 1991-08 … 2026-09.

### `public.revenue_segment` — 704,573 rows · PIT-safe (inherited clock)
Product/geographic segment revenue, 2009-08 … 2026-09. Same inherited clock.

### `public.employee_count_history` — 96,889 rows · PIT-safe
`date_available` = SEC acceptance of the 10-K. 1994-01 … 2026-09.

## Shares outstanding

There is **no shares-outstanding table** (the `shares_history` table from migration 003 is not
present; vendor float is archive-only and `/historical/shares-float` 404s). Options:

1. `fundamental_quarterly.weighted_average_shares_diluted` (fallback `weighted_average_shares`),
   as-of `date_available`. Period-average not period-end, and up to ~3 months stale; on today's split basis.
2. Implied shares = `market.market_cap_daily.market_cap / close` (vendor, PIT-caution).

## Analyst estimates and earnings

### `public.analyst_estimate` — 347,999 rows · **NOT PIT — never a feature**
PK `(security_id, target_period, period_type)`. One row per target fiscal period = the vendor's
**current** consensus (`eps_avg/low/high`, `revenue_*`, `analyst_count`); `date_available` is NULL
on all rows (the parser sets it to None; the vendor has no as-of date). Target periods 1992 … 2034.
No revision history, so no revisions or dispersion-through-time features.

### `public.earnings_event` — 460,672 rows · PIT-caution
| | |
|---|---|
| PK | `event_id` (no unique on `(security_id, report_date)`) |
| Join on | `report_date` — usable strictly **after** the report (`loaders.asof_earnings` uses `report_date < t`) |
| Columns | `eps_actual`, `eps_estimated` (consensus at the report), `revenue_actual`, `revenue_estimated`, `last_updated` |
| Range | Real coverage from 1985 (1,106 events) and ~5k/yr by 1995; 25 junk 1970-01-01 rows; 3,804 future scheduled events (null actuals) |
| Hazards | `last_updated` is 2023–2026 on 99% of rows: the vendor rewrote history (EPS on today's split basis). The consensus is the vendor's reconstruction, not an archived snapshot. Surprise = actual − estimated, scaled by split-adjusted `close` at `t`, is consistent. |

### `public.earnings_event_timing` / `earnings_event_timing_best` (view) — 225,873 rows
BMO/AMC/during from 8-K acceptance times (one source today). Known for 49% of events; when unknown, treat the event as public the session after `report_date`.

### `public.earnings_calendar_event` — 2,430,300 rows · PIT-caution
Vendor `/earnings-calendar` rows (1985 … 2026-09) with estimates/actuals and `raw_payload`; overlaps `earnings_event`. Not needed for v1.

### `public.analyst_action` — 492,832 rows · PIT-safe (dated)
Firm-level upgrades/downgrades from `/grades`, `action_date` 2011-12 … 2026-09. Too short for a 2006 start.

### `public.analyst_consensus_daily` — 349,611 rows · PIT-caution
Daily buy/hold/sell counts from `/grades-historical`, 2016-05 … 2026-09.

### `public.analyst_price_target` — 90,800 rows · PIT-safe
Published targets with publish timestamp (`date_available`), 2021-03 … 2026-09. Too short.

## Ownership, insiders, filings

| Table | Rows | Clock | Range | PIT |
|---|---:|---|---|---|
| `public.ownership_snapshot` | 239,896 | `date_available` (median period end + 45d) | 1998-03 … 2026-06 | PIT-safe |
| `altdata.institutional_holder_position` | 37,130,408 | `date_available` = 13F filing date | 2018-09 … 2026-06 | PIT-safe, short |
| `public.insider_transaction` | 8,078,338 | `filing_date` (not trade date) | 2003-05 … 2026-09 | PIT-safe |
| `altdata.sec_filing` | 8.1M | `date_available` = acceptance | — | PIT-safe |
| `public.transcript_date` / `text.transcript` | 196,300 | `event_date` | 2001-01 … 2026-09 | PIT-safe |

## Beta, volatility, turnover, liquidity

All from PIT-safe tables:

- **Daily returns**: `price_daily.close_adj` (drop `suspect` rows). Market: `^GSPC` in `market.index_price_daily`. Risk-free: `treasury_rate.month1`/`month3`.
- **Turnover**: `volume / weighted_average_shares` — both on today's split basis, so consistent.
- **Dollar volume / Amihud**: `close × volume` (both split-adjusted, so equal to raw $ volume) and `|ret| / (close × volume)`.

## Tables that must NOT be used as features

| Table | Why |
|---|---|
| `public.analyst_estimate` | Current consensus per target period; `date_available` NULL everywhere |
| `public.company.sector/industry/name` | Today's profile |
| `archive.key_metrics_history`, `archive.ratios_history`, `archive.enterprise_value_history` | Vendor ratios built with prices; not PIT |
| `archive.*_snapshot` (`key_metrics_ttm`, `ratios_ttm`, `vendor_float`, `price_target_consensus`, `stock_peers`, `key_executive`, `market_risk_premium`, `exchange_hours`, ETF snapshots) | "Now" snapshots, no as-of date |
| `archive.ratings_historical` | Vendor scores with price components |
| `archive.*_backup_*` | Pre-repair copies |
| `public.monthly_panel` | Empty; its builder (`stock-database/src/panel/builder.py`) computes mcap as `close_adj × shares` (biased low) and does not guard zero-lag rows |
| `esm.*` | Derived tables from a separate earnings-event project; not audited here |
| `public.options_*` | Recent intraday options data; out of scope |
