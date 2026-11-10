# cursor_output — Lasso 40-seed leakage + validation alpha check

**Verdict:** shuffle gate **passed** · validation alpha **changed (2011)** · merged **no**

**Branch:** `feat/nonlinear-models` (unmerged)

## 1. Lasso 40-seed shuffle (corrected IC)

Seeds 0–39, full window 2011-01…2023-11, same Lasso grid. Wall ≈ 854 s (new seeds) + reuse of 0–4.

| Stat | Value |
|---|---|
| Pooled mean IC (across 40 seed means) | **0.00073** |
| SD across seed means | 0.00947 |
| SE (sd/√40) | 0.00150 |
| 95% CI | **[−0.00221, 0.00366]** |
| Pooled monthly t (6200 month×seed ICs) | **0.66** |
| Constant-pred months (total) | 2373 |
| Seed mean min / max | −0.0172 / 0.0283 |
| Seeds with t > 2 | 3 (expect ~1 under null; also 5 with \|t\| > 2) |
| Gate \|mean\| < 0.005 and \|t\| < 2 | **PASS** |

Five seeds were underpowered (SE ≈ 0.006 > gate); forty seeds put the mean firmly near zero with a CI inside ±0.005. File: `reports/models/leakage_lasso_40seed.csv`.

## 2. Validation constant months / alpha choice (real Lasso)

Refit each year’s train/val split; score every alpha with old (drop constant months) vs new (IC=0). **Do not retune production.**

| Year | Alpha (old drop) | Alpha (new =0) | Changed? | Max dropped months (any α) |
|---|---:|---:|---|---:|
| **2011** | **1e-5** | **10** | **yes** | 36 |
| 2012–2023 | (unchanged each year) | same | no | ≤36 |

**2011 flips from near-unregularized to fully regularized.** Under the new metric, alphas that zero predictions get IC=0 every month; the old metric dropped those months and let α=1e-5 win on the remaining variation. Per Step 2: **stop — do not merge.**

Files: `lasso_validation_ic_metric_compare.csv`, `.json`.

## 3. Tests

**45 passed** (`pytest -q`, ~380 s).

## 4. Not done / skipped

- **Merge** blocked solely by the 2011 validation alpha change (shuffle gate is clean).
- No production retune / retrain despite the 2011 discrepancy.
- Other models not re-shuffled this task.
