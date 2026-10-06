# Decisions log

Record every research decision here with the date and the reason. Newest first.

| Date | Decision | Reason |
|------|----------|--------|
| 2026-10-06 | Universe = top 1000 US common stocks by market cap, re-ranked monthly (top 500 as robustness check) | Survivorship-free by construction (ranks what traded at t, incl. later-delisted names); Russell-1000-like; avoids micro-cap noise; larger cross-section for ML than S&P 500. Today's S&P 500 list rejected due to survivorship bias |
| 2026-10-06 | Separate repo from `stock-database` | Keep the data pipeline and the research code independent; this repo reads the DB only |
| 2026-10-06 | Monthly frequency, walk-forward validation | Required by the FE Club project spec |
