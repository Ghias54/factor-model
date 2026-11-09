# Cursor Output — classic factors vs Ken French

Date: 2026-11-08  
Branch: `feat/classic-factors`

## Summary

Built Ken French monthly factors (FF5 + UMD) into `data/external/ff/`, computed our classic four long-short returns (decile EW/VW and universe-breakpoint 2×3 VW) from `features_raw.parquet` through formation **2023-11** / return month **2023-12**, and ran the long-only top-decile baseline plus classic-4 composite. **29 tests passed**, including holdout guards. Equal-weight decile spreads clear the 0.6 bar vs HML/SMB/UMD/RMW; several value-weighted and 2×3 variants miss — expected given top-1000 vs all-exchange FF universe and our own breakpoints (no NYSE flag). Wall ~10 s, peak RSS ~0.8 GB.

## Correlation vs Fama–French (n = 215 months)

Breakpoints: **our top-1000 universe** (no NYSE flag). Pass = corr ≥ 0.6.

| Factor | Version | FF | Corr | 2006–10 | 2011–20 | 2021–23 | ≥0.6? |
|---|---|---|---:|---:|---:|---:|---|
| value | decile EW | HML | 0.705 | 0.700 | 0.706 | 0.740 | pass |
| value | decile VW | HML | 0.375 | 0.333 | 0.621 | 0.670 | **miss** |
| value | 2×3 VW | HML | 0.621 | 0.459 | 0.866 | 0.863 | pass |
| size | decile EW | SMB | 0.690 | 0.851 | 0.628 | 0.676 | pass |
| size | decile VW | SMB | 0.316 | 0.204 | 0.439 | 0.262 | **miss** |
| size | 2×3 VW | SMB | 0.518 | 0.308 | 0.723 | 0.782 | **miss** |
| momentum | decile EW | UMD | 0.886 | 0.931 | 0.838 | 0.866 | pass |
| momentum | decile VW | UMD | 0.584 | 0.472 | 0.746 | 0.748 | **miss** |
| momentum | 2×3 VW | UMD | 0.591 | 0.542 | 0.680 | 0.616 | **miss** |
| ROE | decile EW | RMW | 0.665 | 0.559 | 0.555 | 0.799 | pass |
| ROE | decile VW | RMW | 0.558 | 0.359 | 0.555 | 0.759 | **miss** |
| ROE | 2×3 VW | RMW | 0.615 | 0.497 | 0.486 | 0.877 | pass |

**Why misses (not tuned):**  
- **Top-1000 large-cap** vs FF’s all NYSE/AMEX/NASDAQ — size/value VW spreads are dominated by mega-caps that barely move in our universe.  
- **Universe breakpoints** (not NYSE 30/70) compress the characteristic spread, especially for SMB 2×3.  
- **VW vs EW:** EW matches FF better here because equal weight restores small/mid names inside the top 1000; VW collapses toward the largest names.  
- **ROE vs RMW:** we use TTM net income / equity; RMW is operating profitability — definition gap shows in early subperiods.

12-month rolling correlation PNGs: `reports/classic_factors/rolling_corr_*.png`.  
Corr matrices: `corr_matrix_ours_2x3.csv`, `corr_matrix_ff.csv`.

## Long-only baseline (top decile, EW, 1-month hold)

vs equal-weight universe; RF = FF RF. Formation ≤ 2023-11.

| Portfolio | Ann ret | Ann vol | Sharpe | Max DD | Turnover | Excess vs EW univ | IR | Rank IC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| long value | 8.1% | 22.7% | 0.40 | −68% | 0.11 | −0.1% | 0.04 | −0.002 |
| long size (small) | 8.1% | 19.7% | 0.43 | −53% | 0.35 | −0.4% | −0.04 | −0.001 |
| long momentum | 8.4% | 21.5% | 0.43 | −58% | 0.30 | −0.2% | 0.04 | 0.010 |
| long ROE | 11.3% | 16.9% | 0.65 | −48% | 0.08 | +2.0% | 0.40 | 0.023 |
| classic-4 composite | 8.7% | 18.7% | 0.47 | −51% | 0.29 | −0.0% | 0.03 | 0.014 |
| EW universe | 8.7% | 17.9% | 0.49 | −52% | — | — | — | — |

Top-minus-bottom EW spread (ann. mean of monthly LS): value 4.7%, size 1.9%, momentum 4.7%, ROE 6.1%, composite 6.7%.

### Regimes

| Portfolio | COVID 2020-02→12 ann | Rate-hike 2022 ann |
|---|---:|---:|
| long value | +3.2% | −8.0% |
| long size | +11.5% | −23.6% |
| long momentum | +39.9% | −12.3% |
| long ROE | +22.4% | −15.5% |
| classic-4 | +12.9% | −10.9% |
| EW universe (implied via excess) | — | — |

## Saved where

| Path | Contents |
|---|---|
| `data/external/ff/ff_factors.parquet` + `meta.json` | Ken French monthly (decimal); source URLs + download date |
| `data/factors/classic_factor_returns.parquet` | Decile EW/VW + 2×3 spreads |
| `data/factors/classic_long_only_returns.parquet` | Long-only + EW universe |
| `data/factors/classic_long_only_turnover.parquet` | Monthly turnover |
| `reports/classic_factors/` | CSVs, rolling-corr PNGs, `summary.json` |
| `src/factor_model/external/ff.py` | Loader |
| `src/factor_model/factors/classic.py` | Sorts + holdout caps |
| `scripts/run_classic_factors.py` | End-to-end runner |

## Tests / merge gate

- Full suite: **29 passed** (holdout tests assert no formation after 2023-11 / no return month after 2023-12 on saved series).  
- Step 5 holdout check: **pass**.  
- Merge to main: authorized and performed after green tests (see commits).

## Not done / skipped

- No hyperparameter / breakpoint tuning to raise FF correlations.  
- Did not evaluate or peek at holdout formation months 2023-12+.  
- FF parquet/meta live under gitignored `data/` (local only); fixtures under `tests/fixtures/ff/` are committed.
