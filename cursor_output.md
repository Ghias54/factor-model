# Cursor Output — KNDI/GP mcap investigation + sanity fix

Date: 2026-11-04  
Branch: `feat/panel-ms02-rebuild` (continued)  
Host: `ms02`

## Verdict

| Name | security_id | Verdict | Evidence |
|---|---:|---|---|
| **KNDI** | 10012 | **Data error** | 10:1 reverse split `2026-10-16` restated `close` (~10× `close_raw`) but left WAS ~37–47M unadjusted. Computed mcap ≈ **9–10×** vendor. Vendor-implied shares ≈ WAS/10. |
| **GP** | 9909 | **Data error** | Reverse splits `2020-08-28` (0.142857), `2025-09-08` (0.1), `2026-10-19` (0.2); product → ~350× price scale historically. In flagged window `close/close_raw` ≈ 50 and computed/vendor ≈ **3.7–4.9×**. |

Neither name was in the tradinghost backup universe (0 months). They only entered the first MS-02 rebuild because MS-02 stockdb has post-split price restatement without matching WAS restatement.

## Per-name tables (flagged months; price / shares / mcap / vendor / rank)

### KNDI (10012) — excerpt (full window in investigation log)

| date | close | close_raw | WAS diluted | mcap_computed | mcap_vendor | ratio | rank (pre-fix) | rank backup |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2013-12-31 | 118.04 | 11.79 | 37.02M | 4.37B | 0.46B | 9.47 | 795 | — |
| 2014-02-28 | 167.10 | 16.69 | 37.02M | 6.19B | 0.65B | 9.47 | 652 | — |
| 2014-08-29 | 178.82 | 17.86 | 41.26M | 7.38B | 0.77B | 9.56 | 607 | — |
| 2016-03-31 | 72.09 | 7.20 | 46.97M | 3.39B | 0.34B | 9.88 | 997 | — |

- Ratio range in universe months: **8.91 – 10.01**
- `close/close_raw` median over 2013–2021: **10.012** (= 1 / 0.1 split ratio)
- Split: `corporate_action` ex_date **2026-10-16**, ratio **0.1**
- After fix: **0 months** in universe (vendor mcap ~0.3–0.8B < top-1000 cutoff ~3B)

### GP (9909) — excerpt

| date | close | close_raw | WAS diluted | mcap_computed | mcap_vendor | ratio | rank (pre-fix) | rank backup |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2020-08-31 | 986.08 | 19.75 | 15.49M | 15.27B | 3.31B | 4.61 | 494 | — |
| 2020-12-31 | 1453.41 | 29.11 | 15.49M | 22.51B | 5.66B | 3.98 | 436 | — |
| 2021-06-30 | 980.10 | 19.63 | 20.75M | 20.34B | 4.13B | 4.92 | 551 | — |
| 2021-12-31 | 473.32 | 9.48 | 20.75M | 9.82B | 2.11B | 4.66 | 895 | — |

- Ratio range in universe months: **3.67 – 4.92**
- After fix: **0 months** in universe

## SEC comparison

Source: `fundamental_original` where `line_item='shares_outstanding'` (`CommonStockSharesOutstanding`).

- **KNDI:** Almost all rows are `fmp_only` (no `sec_value`) through 2023; recent **20-F** rows have SEC shares ~82–85M — same unadjusted order of magnitude as WAS. **SEC gate does not catch the bug** because cover-page shares and WAS share the unadjusted basis; only vendor mcap reflects the post-split share count.
- **GP:** **No** `shares_outstanding` rows in `fundamental_original`.

## What changed in stockdb vs tradinghost panel

- Backup universe: KNDI 0, GP 0 months.
- Pre-fix MS-02 panel: KNDI 25 months (ranks 607–997), GP 17 months (436–895).
- Root cause is **not** a point edit of two rows — it is a **global split-adjustment inconsistency** after reverse splits landed in MS-02 `price_daily` without restating `fundamental_quarterly` WAS. Tradinghost extract predates / does not show this mismatch for these names.

## Panel-wide offender scan

### A. MoM mcap jump >5× (or <0.2×) without similar price move — 50 events

Top by severity include: PEB (629831×), IVR, NYRT, SIX, RAAS, CETX, LIXT, PMC, SLH, ARVL, … (many are penny-stock / reverse-split / data-break cases).

### B. Computed mcap / vendor mcap outside [1/3, 3] inside the pre-fix universe

- **2,809 rows · 224 securities** (pre-fix universe)
- Includes KNDI (25) and GP (17)
- Extreme tails: MULN (~10¹¹), HSDT, ELAB, MLVF, CSC, BEAV, PEB, GWAV, NUWE, UAVS, LIXT, …

### C. WAS vs SEC shares_outstanding >3×

- Noisy (624 names / 20k rows): many preferred shares, dual-class, missing SEC tags, or zero WAS. **Not used as the fix gate.** KNDI/GP show **0** SEC-disagree rows for the reason above.

## Fix implemented (factor-model only; stockdb untouched)

Config (`configs/default.yaml`):

```yaml
universe:
  mcap_vendor_sanity:
    enabled: true
    max_ratio: 3.0
```

Behavior (`apply_mcap_vendor_sanity` in `src/factor_model/universe/build.py`):

1. If both computed and vendor mcap exist and `computed/vendor` ∉ `[1/r, r]`, **use vendor mcap** for ranking and features.
2. On those rows, **restate WAS** to `mcap_vendor / close` so turnover stays on the same basis.
3. Flag: `mcap_sanity_replaced` / counted in `universe_meta.json`.

### Rebuild results (post-fix)

| Metric | Pre-fix MS-02 | Post-fix |
|---|---:|---:|
| Rows / months / features | 248000 / 248 / 26 | same |
| Securities | 2650 | 2651 |
| Vendor mcap fallback rows | 14107 | 17088 (incl. sanity) |
| `n_mcap_sanity_replaced` | 0 | **2977** |
| KNDI / GP in universe | 25 / 17 | **0 / 0** |
| Membership Δ vs pre-fix | — | 2121 dropped + 2121 added |
| Wall | 294s | **278.6s** (rebuild) |
| Peak RSS | 7.3 GiB | 7.3 GiB |

Feature mean shifts vs pre-fix (>1% relative) are from **universe composition change**, not feature-definition changes: cashflow_to_price, reversal_1m, momentum_6m, amihud, gross_margin, roa, delta_roe_4q, accruals, debt_to_assets, sue. Null rates unchanged (<0.2 pp). Turnover mean restored (~0.010) after WAS restatement.

### Tests

```
pytest -q  →  20 passed
```

(includes new `tests/test_mcap_vendor_sanity.py` + lookahead suite)

## Recommended stock-database fix

1. When applying a `corporate_action` **split** to `price_daily.close`, also multiply historical `fundamental_quarterly.weighted_average_shares` / `_diluted` (and EPS) by the same cumulative split factor so `close × WAS` stays coherent.
2. Backfill `fundamental_original.shares_outstanding` from DEI / cover page for 10-K/10-Q/20-F/40-F (GP currently empty; KNDI mostly FMP-only).
3. Add a DB QA check: `|close / close_raw - 1/∏split_ratios|` small ⇒ expect WAS ≈ `market_cap_daily / close` within ~3×.

## Commits (this task)

| SHA | Message |
|---|---|
| `44f5001` | Add cursor_task for KNDI/GP mcap investigation. |
| `42347b4` | Add read-only KNDI/GP mcap outlier investigation script. |
| `fafd600` | Prefer vendor mcap when computed/vendor ratio exceeds config gate. |
| `70c21d1` | Also restate WAS from vendor-implied shares when mcap sanity fires. |
| *(final)* | Panel report + cursor_output for mcap sanity |

## Files changed

- `configs/default.yaml` — `mcap_vendor_sanity`
- `src/factor_model/universe/build.py` — sanity + implied-shares restatement
- `tests/test_mcap_vendor_sanity.py`
- `scripts/investigate_mcap_outliers.py`
- `docs/panel-report.md` — regenerated
- `cursor_task.md` / `cursor_output.md`

Untracked keeps: `data/panel/`, `data/panel_tradinghost_backup/`, `data/panel_ms02_pre_mcap_fix/`

## Not done / skipped

- Did **not** edit any stockdb rows.
- Did **not** change feature definitions, top-N rule, sample window, or holdout.
- Did **not** build classic factors / FF validation / models.
- Did **not** merge to `main`.
- Did **not** delete backup or pre-fix panel folders.
- Did **not** use SEC shares as the live gate (unreliable / same-basis as WAS for these names).
- Panel-wide MoM jump list not turned into an additional filter (vendor-ratio gate already covers KNDI/GP and 224 similar names).
