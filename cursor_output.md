# Cursor Output — feat/panel-ms02-rebuild

Date: 2026-11-04
Host: `ms02` · cores: **24** · RAM: **30 GiB**

## Resolved DB

| Field | Value |
|---|---|
| DSN key | `STOCKDB_DSN` |
| Host | `127.0.0.1` |
| Port | `5432` |
| Database | `stockdb` |
| Server addr | `127.0.0.1` (local Postgres 18.6) |
| Tradinghost / Tailscale? | **No** — already pointed at MS-02 local |

`.env.example` comment updated (was still saying “on tradinghost”). No secrets changed.

## Branch

Branched from **`feat/panel`** (not `main`): panel compute/universe/loaders are not on `main` yet. Per workspace rules, when a task depends on an unmerged branch, branch from that branch.

```
feat/panel-ms02-rebuild
```

### Commits on this branch (beyond `feat/panel`)

| SHA | Message |
|---|---|
| `672ff5e` | Add cursor_task for MS-02 panel rebuild from local stockdb. |
| `4afae47` | Raise MS-02 panel compute: larger daily chunks, full-core TTM/daily threads. |
| `5db4628` | Normalize loader datetimes to datetime64[ns] for Postgres 18. |
| `3dc7334` | Harden remaining date casts to datetime64[ns] in panel compute. |
| *(final)* | Panel report + compare script + cursor_output |

## What was done

1. Confirmed DB → MS-02 local `127.0.0.1/stockdb`.
2. Moved `data/panel/` → `data/panel_tradinghost_backup/` (untracked; kept).
3. Raised compute knobs in `configs/default.yaml`:
   - `daily_price_chunk: 800` (was hardcoded 150)
   - `daily_feature_n_jobs: -1` (was capped at 8)
   - Parallelized `asof_last_n_quarters` with `n_jobs=-1` (threads)
4. Fixed Postgres-18 `datetime64[s]` vs `[ns]` `merge_asof` breakage via `_to_dt64ns` in loaders.
5. Rebuilt panel with `python scripts/build_panel.py` (universe + features + report).
6. Compared new vs tradinghost backup (`scripts/compare_panels.py`).
7. Ran full pytest (incl. `tests/test_panel_lookahead.py`).
8. Regenerated `docs/panel-report.md`.

## Wall times

| Stage | Tradinghost backup | MS-02 rebuild |
|---|---:|---:|
| Universe | 128.45s | **32.07s** |
| Features | 746.22s | **260.68s** |
| **Total** | **877.6s** | **294.1s** (`/usr/bin/time`: 5:00) |

Feature stage walls (MS-02): prices_me 12.7 · fund 15.5 · earn 4.4 · macro 2.3 · px_feat 0.3 · **ttm_asof 117.1** · sue 0.1 · daily_load 76.2 · daily_feat 107.8 · beta 0.4

Peak RSS: **7329 MB** (`/usr/bin/time` Max RSS 7505208 KB ≈ 7.3 GiB). Cores available: 24. Joblib threads are GIL-bound on the Python TTM/daily loops, so wall CPU ≈ 99% of one process rather than 24×; larger chunks + more threads still cut wall ~3× vs tradinghost.

## New panel counts

| Metric | Backup | New |
|---|---:|---:|
| Rows | 248,000 | 248,000 |
| Months | 248 | 248 |
| Features | 26 | 26 (identical list) |
| Securities | 2,650 | 2,650 |
| Vendor mcap fallback rows | 14,109 | 14,107 |

Per-month universe size: all 248 months have exactly 1000 names; max \|Δn\| vs backup = 0.

## New vs backup comparison

**No feature flagged at >1%** (null-rate absolute or mean relative). Aggregate mean/std/null are essentially identical.

Not bitwise-identical (DB is close but not a frozen snapshot of the tradinghost build):

| Issue | Detail |
|---|---|
| 42 universe membership swaps | Same months, different rank-1000 boundary names. MS-02 includes **KNDI** (`security_id=10012`, 25 months 2013-12→2016-03) and **GP** (`9909`, 17 months 2020-08→2021-12) at mid-universe ranks, displacing the previous rank-1000 names. Likely share-count / mcap history updates on MS-02 after the tradinghost panel was built (vendor fallback count also −2). |
| Inner-join rows | 247,958 / 248,000 |
| Exact float diffs | `beta_60m` ~58k rows differ at float noise only (`np.isclose` OK). Real content diffs: **sue** 11 rows (DHC, AYRO, CRH — earnings estimate revisions), and **delta_roe_4q / revenue_growth_yoy / asset_growth_yoy** filled for one previously-null row (`security_id=7088`, 2006-01-31). |

**Judgment:** Exact equality was expected only if the DB were a faithful frozen copy. MS-02 stockdb has small PIT updates vs the Oct 20 tradinghost extract; aggregate stats and universe size match; no >1% feature drift.

## Tests

```
pytest -q  →  16 passed in 0.89s
```

Includes `tests/test_panel_lookahead.py`.

## Files changed

- `cursor_task.md` / `cursor_output.md`
- `configs/default.yaml` — MS-02 compute knobs
- `.env.example` — host comment (MS-02 local)
- `src/factor_model/data/loaders.py` — `_to_dt64ns`, parallel TTM asof
- `src/factor_model/features/compute.py` — config-driven chunk/threads; ns casts
- `scripts/compare_panels.py` — new compare CLI
- `docs/panel-report.md` — regenerated from MS-02 build

Untracked (intentionally): `data/panel/`, `data/panel_tradinghost_backup/`

## Not done / skipped

- Did **not** change feature definitions, universe rule, sample window, or holdout.
- Did **not** build classic factors, Fama-French validation, or any models.
- Did **not** delete `data/panel_tradinghost_backup/`.
- Did **not** merge to `main`.
- Did **not** re-parallelize with processes (would copy large fund/price frames); threads remain GIL-limited.
- Did **not** investigate root cause of KNDI/GP mcap divergence in stock-database (flag for DB audit if needed).
- README / docs/plan still mention “tradinghost” in prose; left alone (task said config only).
