# Cursor Output — Per-share / sub-3× split audit

Date: 2026-11-04  
Branch: `feat/panel-ms02-rebuild`  
Panel audited: current MS-02 panel (post mcap-sanity). Backup kept at `data/panel_ms02_pre_split_audit/`.  
Raw numbers: `logs/split_audit/`. Stockdb was read-only.

## Verdict

**Yes — per-share features are a real panel problem today; tightening the mcap gate to 1.5 is not.** About **19%** of panel rows have a post-fundamentals split that leaves `earnings_yield = TTM EPS / close` on mixed bases; of those with non-null EY, **88%** move by >10% and **68%** shift at least one cross-sectional decile after a mechanical EPS restatement. `sue` shows a similar footprint, but `earnings_event` EPS may already be vendor-restated, so that impact is less certain. The `[1.5, 3]` mcap band the current gate leaves open is mostly **not** split-driven (~24% split-explained); tightening to 1.5 swaps ~1.3k membership slots/month-rows and mostly hits preferreds / dual-class noise. Prefer waiting for `fix/split-restatement` in stock-database for the real fix; do **not** change `max_ratio` to 1.5 now.

## Step 1 — Feature / split-basis table

| Feature | Formula | ÷ by | Same split basis? |
|---|---|---|---|
| **earnings_yield** | `sum(q0..q3 eps_diluted) / close` | split-adj `close` | **No** — EPS from `fundamental_quarterly` not restated |
| **sue** | `(eps_actual − eps_estimated) / close` | split-adj `close` | **Uncertain** — surprise from `earnings_event` (vendor often rewrites to today basis); not YoY/std-dev scaled |
| **turnover** | `volume / WAS_diluted` | WAS (as-of; sanity may restate) | **No** unless `mcap_vendor_sanity` fired |
| value_bm | `total_equity / mcap` | mcap | OK **if** mcap OK ($-over-$) |
| size_ln_mcap | `log(mcap)` | — | OK if mcap OK |
| sales_to_price | `TTM revenue / mcap` | mcap | OK if mcap OK |
| cashflow_to_price | `TTM CFO / mcap` | mcap | OK if mcap OK |
| dollar_volume_ln | `log(close × volume)` | — | **Yes** (both split-adj) |

No dividends-per-share or book-value-per-share features in `compute.py`.  
Momentum / reversal / vol / beta / Amihud use prices only (same basis).  
ROE / margins / growth use $-fundamentals only.

**sue detail:** latest `earnings_event` with `report_date < month-end`, age ≤ 120d; surprise = actual − estimate, scaled by **`close`**, not by historical EPS stdev.

**earnings_yield detail:** TTM = sum of four quarterly `eps_diluted`, divided by month-end **`close`** (not net income / mcap).

## Step 2 — Exposure and restatement impact

Method: for each row, `cum_split_factor = ∏ ratios` with `ex_date > fund_fiscal_period_end` (EY) or `> report_date` (sue). Restated feature = feature / factor (equivalent to restating EPS onto the price basis). “Exposed” = factor ≠ 1.

### Counts

| Feature | Exposed rows | Share of panel | >10% change | ≥1 decile move |
|---|---:|---:|---:|---:|
| earnings_yield | 48,322 | 19.5% | 39,377 | 30,662 |
| sue | 44,773 | 18.1% | 34,795 | 22,384 |

Lookback-window-only (split between lookback start and month-end): EY **12,436** (5.0%); sue ≈ full post-split set (report→month-end is short, but any later split still restates `close`).

### EY exposed share by year (selected)

| year | n exposed (non-null impact set) | share of year panel | >10% | decile |
|---:|---:|---:|---:|---:|
| 2006 | 4,373 | 36.4% | 4,028 | 3,171 |
| 2010 | 3,337 | 27.8% | 2,965 | 2,394 |
| 2015 | ~2.4k | ~20% | — | — |
| 2020 | ~1.5k | ~12% | — | — |
| 2024 | 522 | 4.4% | 446 | 388 |
| 2026 | 105 | 1.3% | 87 | 74 |

Older months are worse: more subsequent splits have already restated the entire price history.

Among exposed EY rows, the most common post-period factors are **2.0** (9,849), **4.0**, **3.0**, **1.5** — i.e. ordinary forward splits the 3× mcap gate never sees.

### Worst 30 (by relative change) — top of list

Full list in `logs/split_audit/step2_pershare_impact.json`. Extreme tails are mega-cumulative reverse-split names (CDT, APDN, CETX, GNLN for EY; CYCC, PSTV for sue) where `factor` is tiny and EY/sue near zero → huge relative changes. More representative mid-list movers are ordinary 2:1 / 3:1 names.

## Step 3 — Computed / vendor mcap ratio

### Histogram (universe rows with both mcaps; n=232,116)

| bin | count |
|---|---:|
| <0.1 | 809 |
| 0.1–0.2 | 633 |
| 0.2–1/3 | 873 |
| **1/3–1/1.5** | **2,159** |
| 1/1.5–1/1.2 | 2,480 |
| 1/1.2–1 | 45,829 |
| 1–1.2 | 171,839 |
| 1.2–1.5 | 4,056 |
| **1.5–2** | **1,538** |
| **2–3** | **1,228** |
| 3–4 | 80 |
| 4–5 | 42 |
| 5–10 | 122 |
| >10 | 418 |

- Median ratio **1.009**; **94.2%** within ±20% of 1.
- Inside current gate but outside a 1.5 gate: **4,925** rows (`[1.5,3]∪[1/3,1/1.5]`).
- Of those, split after fiscal period end explains **1,199 (24%)**; **3,726 (76%)** have **no** such split → other issues (preferreds, share-class, vendor gaps).

### max_ratio = 1.5 test (config untouched; in-memory only)

| | max_ratio 3.0 (current) | max_ratio 1.5 (test) |
|---|---:|---:|
| `n_mcap_sanity_replaced` | 2,977 | **8,064** |
| Membership dropped / added vs current | — | **1,261 / 1,261** |

Top dropped (months): MDU (126), VLYPO (99), ESRT (56), SNX (48), WOR, POST, LBTYB, OGCP, MLI, BKI, VLYPN, XPO, AGNCO/AGNCM/AGNCN (preferreds), …  
Top added: SJI, PEB, VIRT, NSS, ZIONO, RAVN, …  

Pattern: tightening pulls in **preferred / special share classes** and mid-cap boundary noise, not cleanly “2:1 split victims.”

## Recommendations

1. **Per-share fix now vs wait**  
   - **earnings_yield: real bug; prefer wait for stockdb `fix/split-restatement`** if it lands before the next model train. If training starts first, temporary options (not applied here): (a) switch EY to `TTM net_income / mcap` (same basis as sales_to_price), or (b) restate TTM EPS by `∏ split ratios after each quarter FPE` inside the panel build.  
   - **sue: do not panel-patch yet** without proving `earnings_event` EPS is unadjusted; vendor rewrite notes suggest it may already match `close`. Re-check after stockdb fix.  
   - **turnover:** already patched on sanity-replaced rows; residual risk is sub-3× WAS mismatches — stockdb restatement is the right fix.

2. **Tighten max_ratio to 1.5?**  
   - **No.** Only ~24% of the newly caught band is split-explained; collateral damage is large (preferreds, +5k sanity rows, 1.3k membership swaps). Keep **3.0** until stockdb restates WAS; then re-measure and consider lowering.

## Artifacts

| Path | Contents |
|---|---|
| `scripts/audit_split_pershare.py` | Audit CLI |
| `logs/split_audit/step1_feature_basis.json` | Step 1 inventory |
| `logs/split_audit/step2_pershare_impact.json` | Exposure, by-year, worst 30 |
| `logs/split_audit/step3_mcap_ratio_hist.json` | Ratio histogram + in-gate split share |
| `logs/split_audit/step3_max_ratio_1_5_test.json` | 1.5 universe test |
| `logs/split_audit/summary.json` | Combined |
| `data/panel_ms02_pre_split_audit/` | Panel backup (untracked) |

## Commits

| SHA | Message |
|---|---|
| `dd60c50` | Add cursor_task for per-share / sub-3x split audit. |
| *(final)* | Add split/per-share audit script, logs, cursor_output. |

## Not done / skipped

- Did **not** change `configs/default.yaml` or feature code.
- Did **not** write to stockdb.
- Did **not** rebuild the production panel (1.5 test wrote only under `logs/split_audit/`).
- Did **not** merge to `main`.
- Did **not** delete prior backups.
- Worst-30 relative-change list is dominated by near-zero EY/sue ÷ tiny factors; interpret tails cautiously.
