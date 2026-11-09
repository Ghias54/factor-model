# Cursor Output — earnings_yield formula + post-backfill panel rebuild

Date: 2026-11-08  
Branch: `feat/panel-ms02-rebuild`  
Baseline backup: `data/panel_ms02_pre_ey_fix/` (pre-backfill panel, old EY formula)

## Summary

Switched `earnings_yield` from TTM diluted EPS / month-end close to TTM net income / mcap (same dollar basis as sales_to_price and cashflow_to_price), added a 2:1 split-invariance unit test, and rebuilt the panel on post–split-restatement prod. Full suite: **21 passed** before and after. Panel still **248 months × 1,000 names = 248,000 rows**. Backfill moved liquidity/size features on ~143 securities (turnover, dollar volume, Amihud, mcap-linked yields); the formula change re-ranks ~10% of EY rows by at least one decile while Spearman vs old EY stays ~0.98. Vendor sanity replacements: **2,977 → 3,009**. Peak RSS ~7.2 GB; rebuild wall ~304 s.

## Step 0 — Backup and tests

- Copied `data/panel/` → `data/panel_ms02_pre_ey_fix/`.
- Pre-change suite: **20 passed** in 1.4 s (then +1 new test → 21).

## Step 1 — Old earnings_yield mixed-basis exposure (today’s DB, before code change)

Audit-style “any split after latest fiscal period end” still flags **48,343 / 248,000 = 19.49%** of panel rows (vs audit 19.48%). That count is almost unchanged because it only looks at corporate_action dates, not whether EPS was already on the price basis.

On post-backfill fundamentals for the same panel keys:

| Measure | Rows | Share |
|---|---:|---:|
| Audit-exposed (post-FPE split) | 48,343 | 19.5% |
| Any TTM quarter with stockdb `split_factor` set | 1,297 | 0.52% |
| Actually mixed (exposed and EY_pershare vs EY_dollar disagree >20%) | 5,028 | 2.0% |
| Actually mixed v2 (exposed, no same-basis evidence) | 2,697 | 1.1% |

Among audit-exposed rows, median `(TTM EPS / close) / (TTM NI / mcap)` ≈ **1.00** (p10/p90 ≈ 0.93 / 1.08). Spot checks: AAPL/NVDA/TSLA/GOOGL are heavily audit-exposed but have **no** `split_factor` rows and EY ratio ≈ 1 — vendor already restated ordinary forward-split EPS onto today’s share basis. Stockdb’s backfill only touched the `needs_adjust` set (~227 securities with non-null `split_factor` in FQ; ~86 of those appear in stockdb’s panel-cell summary).

**Why 19.5% ≠ true mixed:** the 11/04 audit treated every post-filing split as proof that EPS and close were on different bases. Ordinary forward splits (NVDA, AAPL, TSLA, GOOGL, …) were already vendor-aligned, so most of that 19.5% was never mixed. True residual mixed basis after the backfill is about **1–2%**. The formula change to NI/mcap still stands (split-proof by construction).

Artifacts: `logs/ey_remeasure/step1_remeasure.json`.

## Step 2 — Feature change

In `src/factor_model/features/compute.py`:

```text
earnings_yield = TTM net_income (q0..q3, same asof/staleness as before) / mcap
```

Null if fewer than four quarters or mcap missing/≤0; negatives kept. Name unchanged. Docs/config one-liners updated (`configs/default.yaml`, `docs/decisions.md`, `docs/design-decisions-2026-10-20.md`).

## Step 3 — Tests

- Added `tests/test_earnings_yield_split.py`: same NI, price halves, shares double, mcap unchanged → EY unchanged; old EPS/close doubles under unrestated EPS.
- Full suite: **21 passed**.

## Step 4 — Rebuild and old vs new comparison

Rebuild: wall **304.2 s**, peak RSS **~7.2 GB**, n_jobs=-1. Confirmed **248 months, 248,000 rows, every month 1,000 names**.

Universe membership (vs backup): **135 dropped, 135 added** across 119 months (≤2 slots/month). Sanity replacements **2,977 → 3,009**; vendor mcap fallback rows **17,088 → 17,113**.

### Backfill effect (every feature except earnings_yield)

Inner join 247,865 rows. Changes concentrated in volume/shares/mcap paths; price-only features unchanged.

| Feature | Rows changed | Corr | Share ≥1 decile move |
|---|---:|---:|---:|
| amihud | 5,966 | 1.000 | 3.8% |
| dollar_volume_ln | 5,914 | 0.999 | 3.8% |
| turnover | 5,820 | 0.972 | 4.2% |
| size_ln_mcap | 1,353 | 1.000 | 0.6% |
| cashflow_to_price | 1,315 | 0.999 | 0.8% |
| value_bm | 1,313 | 1.000 | 0.7% |
| sales_to_price | 1,234 | 1.000 | 0.7% |
| sue | 234 | 1.000 | ~0% |
| (other fund features) | ≤160 | ~1.0 | ≤0.2% |
| momentum / reversal / vol / beta / max_daily_ret | 0 | 1.0 | 0 |

**143 securities** with any non-EY change. Top 10 by rows changed: SCCO (248), WRB (223), CNQ (222), IBKR (215), BN (204), RJF (189), RTX (174), TT (170), FAST (161), PANW (150).

### Formula effect (earnings_yield old vs new)

| Metric | Value |
|---|---|
| Rows changed | 220,446 (89%) |
| Non-null coverage | 220,457 → 220,526 |
| Pearson (levels) | ~0 (outlier-driven; not useful) |
| Spearman | **0.978** |
| Share moving ≥1 cross-sectional decile | **10.2%** |

Spearman by year stays 0.97–0.99 throughout; decile-move share by year ~8–13%.

Artifacts: `logs/ey_remeasure/step4_compare.json`, `logs/ey_remeasure/rebuild.log`.

## Step 5 — Open items from 11/04 audit

### sue — verdict: fine as is

Sampled NVDA/AAPL/TSLA/GOOGL (ordinary forward splits), KNDI/GP (`needs_adjust`, not in top-1000 panel), plus 10 random `needs_adjust` names. For named forward splits, `earnings_event` EPS across the ex-date is closer to a restated history (levels do not drop by ~1/split ratio) than to an unrestated series. On panel months with SUE, `(eps_actual − eps_estimated) / close` matches the dollar form `(surprise × WAS) / mcap` with median ratio **1.0** and **100%** within 20% for NVDA/AAPL/TSLA/GOOGL/WCN/VIAV/DINO. No code change proposed.

### turnover — verdict: fine as is (monitor needs_adjust)

`volume / WAS_diluted` vs `volume / (vendor mcap / close)`:

| Universe | Median ratio | Within ±20% |
|---|---:|---:|
| Old (all) | 0.991 | 95.5% |
| New (all) | 0.991 | 95.4% |
| Old needs_adjust only | 0.992 | 84.5% |
| New needs_adjust only | 0.990 | 79.9% |

Overall alignment is unchanged and strong. The `needs_adjust` subset still has a fatter mismatch tail after the backfill (within-20% slipped ~5 pp). **Propose (do not implement):** optional turnover fallback to vendor-implied shares when `|WAS / (vendor_mcap/close) − 1|` exceeds a config threshold (mirrors the mcap sanity path). Not blocking.

### Vendor mcap gate (max_ratio 3.0) — still looks right

Computed/vendor histogram on universe `mcap_computed` (same bins as `logs/split_audit/step3`):

| Band | Old (11/04 panel) | New (post-backfill) |
|---|---:|---:|
| within [1/1.2, 1.2] | 94.2% | 94.0% |
| median ratio | 1.009 | 1.009 |
| outside 3× (= sanity fires) | 2,977 | 3,009 |
| 1.5–3 plus 1/3–1/1.5 | 4,923 | 5,331 |

Extremes (`>10`, `<0.1`) barely moved. Backfill did not clear the dual-class / preferred / vendor-noise tail the prior audit called out. **Keep max_ratio at 3.0.**

Artifacts: `logs/ey_remeasure/step5_checks.json`.

## Commits

1. `f5fe912` — Switch earnings_yield to TTM net income / mcap (split-proof).
2. *(this push)* — Panel rebuild report + Step 4/5 measurement logs + `cursor_output.md`.

## Not done / skipped

- Did not implement turnover vendor-implied fallback (proposal only).
- Did not change `max_ratio` (decision locked at 3.0).
- Did not merge to `main`.
- KNDI/GP not in the top-1000 panel in either build (0 months); SUE/turnover panel checks for those two are N/A (fund `split_factor` rows present).
- No model training, no holdout evaluation.

## Surprises / judgment calls

- Audit 19.5% was a large overstatement of true mixed-basis EY once vendor-adjusted forward splits are recognized; used both dollar-vs-pershare disagreement and statement/mcap alignment as the “actually mixed” gauges (~1–2%).
- EY Pearson ≈ 0 overall despite Spearman 0.98 — a few extreme level outliers; reported Spearman and decile moves as the ranking-relevant stats.
- Post-backfill, `needs_adjust` turnover vs vendor-implied got slightly worse, not better — worth a stockdb follow-up, not a panel blocker.
- Sanity replacement count rose slightly after backfill (2,977 → 3,009); outside-3× mass was not eliminated.
