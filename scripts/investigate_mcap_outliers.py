"""Investigate inflated mcap names (KNDI/GP) and scan the panel for similar offenders.

Read-only vs stockdb. Writes JSON under data/panel/ (gitignored).

    python scripts/investigate_mcap_outliers.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_model.config import load_config
from factor_model.data import loaders
from factor_model.data.db import read_sql

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "panel"
BACKUP = ROOT / "data" / "panel_tradinghost_backup"
TARGETS = {10012: "KNDI", 9909: "GP"}
# Flagged windows ± 12 months
WINDOWS = {
    10012: ("2012-12-31", "2017-03-31"),
    9909: ("2019-08-31", "2022-12-31"),
}


def _month_ends_between(start: str, end: str) -> pd.DatetimeIndex:
    me = loaders.month_ends(start, end, complete_only=True)
    return pd.to_datetime(me)


def per_name_table(sid: int) -> pd.DataFrame:
    start, end = WINDOWS[sid]
    me = _month_ends_between(start, end)
    px = loaders.prices_on_dates(me)
    px = px[px["security_id"] == sid].copy()
    fund = loaders.fundamentals(
        columns=(
            "weighted_average_shares",
            "weighted_average_shares_diluted",
        ),
        quarterly_only=True,
        guard=True,
    )
    fund = fund[fund["security_id"] == sid]
    asof = loaders.asof_fundamentals(px[["security_id", "date"]], fund, max_staleness_days=548)
    vendor = loaders.vendor_market_cap_on_dates(me)
    vendor = vendor[vendor["security_id"] == sid].rename(columns={"market_cap": "mcap_vendor"})

    # Current panel ranks if present
    univ = pd.read_parquet(OUT / "universe.parquet")
    univ = univ[univ["security_id"] == sid][["date", "rank", "mcap", "mcap_computed", "mcap_from_vendor"]]
    univ["date"] = pd.to_datetime(univ["date"])

    # Backup panel
    bak = pd.read_parquet(BACKUP / "universe.parquet")
    bak = bak[bak["security_id"] == sid][["date", "rank", "mcap"]].rename(
        columns={"rank": "rank_backup", "mcap": "mcap_backup"}
    )
    bak["date"] = pd.to_datetime(bak["date"])

    dil = asof["weighted_average_shares_diluted"].where(
        asof["weighted_average_shares_diluted"] > 0, asof["weighted_average_shares"]
    )
    out = px.merge(asof, on=["security_id", "date"], how="left", suffixes=("", "_f"))
    out["shares_was"] = out["weighted_average_shares"]
    out["shares_diluted"] = out["weighted_average_shares_diluted"]
    out["shares_used"] = dil
    out["mcap_computed"] = out["close"] * out["shares_used"].where(out["shares_used"] > 0)
    out = out.merge(vendor[["security_id", "date", "mcap_vendor"]], on=["security_id", "date"], how="left")
    out["implied_shares_vendor"] = out["mcap_vendor"] / out["close"].where(out["close"] > 0)
    out["mcap_ratio_vs_vendor"] = out["mcap_computed"] / out["mcap_vendor"].where(out["mcap_vendor"] > 0)
    out = out.merge(univ, on="date", how="left", suffixes=("", "_univ"))
    out = out.merge(bak, on="date", how="left")
    out["symbol"] = TARGETS[sid]
    cols = [
        "date",
        "symbol",
        "security_id",
        "close",
        "close_raw",
        "shares_was",
        "shares_diluted",
        "shares_used",
        "mcap_computed",
        "mcap_vendor",
        "implied_shares_vendor",
        "mcap_ratio_vs_vendor",
        "rank",
        "rank_backup",
        "mcap_backup",
        "fiscal_period_end",
        "date_available",
    ]
    return out[cols].sort_values("date")


def sec_shares(sid: int) -> pd.DataFrame:
    df = read_sql(
        """
        SELECT security_id, fiscal_period_end, period_type, date_available,
               original_value, sec_value, fmp_value, standard_value,
               sec_concept, sec_form, sec_filed, sec_accession, match_status
        FROM fundamental_original
        WHERE security_id = :sid AND line_item = 'shares_outstanding'
        ORDER BY fiscal_period_end, date_available
        """,
        {"sid": sid},
    )
    for c in ("fiscal_period_end", "sec_filed"):
        if c in df:
            df[c] = pd.to_datetime(df[c])
    if "date_available" in df.columns and len(df):
        df["date_available"] = pd.to_datetime(df["date_available"], utc=True)
    return df


def fund_shares_history(sid: int) -> pd.DataFrame:
    df = read_sql(
        """
        SELECT security_id, fiscal_period_end, fiscal_period, date_available,
               date_source, weighted_average_shares, weighted_average_shares_diluted,
               accepted_date_placeholder
        FROM fundamental_quarterly
        WHERE security_id = :sid AND fiscal_period LIKE 'Q%'
        ORDER BY fiscal_period_end, date_available
        """,
        {"sid": sid},
    )
    df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"])
    df["date_available"] = pd.to_datetime(df["date_available"], utc=True)
    return df


def splits(sid: int) -> pd.DataFrame:
    return read_sql(
        """
        SELECT security_id, ex_date, action_type, ratio_or_amount,
               action_type_vendor, ratio_or_amount_vendor, correction
        FROM corporate_action
        WHERE security_id = :sid AND action_type = 'split'
        ORDER BY ex_date
        """,
        {"sid": sid},
    )


def panel_wide_mcap_jumps(univ: pd.DataFrame) -> pd.DataFrame:
    """Flag MoM mcap jump >5x or drop <0.2x without similar price move."""
    u = univ.sort_values(["security_id", "date"]).copy()
    u["date"] = pd.to_datetime(u["date"])
    u["mcap_prev"] = u.groupby("security_id")["mcap"].shift(1)
    u["close_prev"] = u.groupby("security_id")["close"].shift(1)
    u["mcap_mult"] = u["mcap"] / u["mcap_prev"]
    u["price_mult"] = u["close"] / u["close_prev"]
    # Jump not explained by price: |log(mcap_mult/price_mult)| large
    shares_mult = u["mcap_mult"] / u["price_mult"]
    u["implied_shares_mult"] = shares_mult
    bad = (u["mcap_mult"] > 5) | (u["mcap_mult"] < 0.2)
    # price move of similar size would have |price_mult| in same ballpark
    price_explains = (u["price_mult"] > 3) | (u["price_mult"] < 1 / 3)
    suspects = u[bad & ~price_explains].copy()
    suspects["severity"] = np.where(
        suspects["mcap_mult"] > 5,
        suspects["mcap_mult"],
        1 / suspects["mcap_mult"],
    )
    return suspects.sort_values("severity", ascending=False)


def shares_vs_sec_scan(univ: pd.DataFrame, max_sids: int | None = None) -> pd.DataFrame:
    """Compare as-of WAS diluted vs latest SEC shares_outstanding (as-of)."""
    sids = sorted(univ["security_id"].unique().tolist())
    if max_sids:
        sids = sids[:max_sids]
    sec = read_sql(
        """
        SELECT security_id, fiscal_period_end, date_available,
               COALESCE(sec_value, standard_value, original_value, fmp_value) AS sec_shares
        FROM fundamental_original
        WHERE line_item = 'shares_outstanding'
          AND security_id = ANY(:ids)
          AND COALESCE(sec_value, standard_value, original_value, fmp_value) > 0
        """,
        {"ids": [int(x) for x in sids]},
    )
    if sec.empty:
        return pd.DataFrame()
    sec["date_available"] = pd.to_datetime(sec["date_available"], utc=True)
    sec["fiscal_period_end"] = pd.to_datetime(sec["fiscal_period_end"])
    sec = sec.sort_values(["security_id", "date_available", "fiscal_period_end"])

    keys = univ[["security_id", "date"]].drop_duplicates().copy()
    keys["date"] = pd.to_datetime(keys["date"])
    keys["_cutoff"] = loaders.asof_cutoff(keys["date"]).astype("datetime64[ns, UTC]")
    keys = keys.sort_values("_cutoff")

    # asof merge per security would be slow; do globally with merge_asof by cutoff after sort
    parts = []
    sec_by = {sid: g for sid, g in sec.groupby("security_id", sort=False)}
    for sid, gkeys in keys.groupby("security_id", sort=False):
        gs = sec_by.get(sid)
        if gs is None or gs.empty:
            continue
        right = gs.rename(columns={"date_available": "_avail"}).sort_values("_avail")
        right["_avail"] = pd.to_datetime(right["_avail"], utc=True).astype("datetime64[ns, UTC]")
        left = gkeys.sort_values("_cutoff")
        left["_cutoff"] = pd.to_datetime(left["_cutoff"], utc=True).astype("datetime64[ns, UTC]")
        m = pd.merge_asof(
            left,
            right[["_avail", "sec_shares", "fiscal_period_end"]],
            left_on="_cutoff",
            right_on="_avail",
            direction="backward",
        )
        parts.append(m)
    if not parts:
        return pd.DataFrame()
    asof_sec = pd.concat(parts, ignore_index=True)
    u = univ.copy()
    u["date"] = pd.to_datetime(u["date"])
    dil = u["weighted_average_shares_diluted"].where(
        u["weighted_average_shares_diluted"] > 0, u["weighted_average_shares"]
    )
    u["shares_used"] = dil
    m = u.merge(asof_sec[["security_id", "date", "sec_shares"]], on=["security_id", "date"], how="left")
    m["ratio"] = m["shares_used"] / m["sec_shares"]
    bad = m[m["sec_shares"].notna() & ((m["ratio"] > 3) | (m["ratio"] < 1 / 3))].copy()
    bad["severity"] = np.where(bad["ratio"] > 1, bad["ratio"], 1 / bad["ratio"])
    return bad.sort_values("severity", ascending=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    result: dict = {"targets": {}, "windows": {str(k): v for k, v in WINDOWS.items()}}

    for sid, sym in TARGETS.items():
        print(f"\n======== {sym} ({sid}) ========")
        tbl = per_name_table(sid)
        sec = sec_shares(sid)
        fq = fund_shares_history(sid)
        sp = splits(sid)
        print(tbl.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        print("\n--- SEC shares_outstanding ---")
        print(sec.to_string(index=False) if len(sec) else "(none)")
        print("\n--- fundamental_quarterly WAS ---")
        print(
            fq[
                [
                    "fiscal_period_end",
                    "fiscal_period",
                    "date_available",
                    "weighted_average_shares",
                    "weighted_average_shares_diluted",
                ]
            ].to_string(index=False)
            if len(fq)
            else "(none)"
        )
        print("\n--- splits ---")
        print(sp.to_string(index=False) if len(sp) else "(none)")

        # Compare WAS vs SEC around flagged window
        if len(sec) and len(fq):
            sec_m = sec.rename(columns={"fiscal_period_end": "sec_fpe"})
            cmp = fq.merge(
                sec_m[
                    [
                        "sec_fpe",
                        "sec_value",
                        "fmp_value",
                        "standard_value",
                        "sec_form",
                        "sec_filed",
                    ]
                ],
                left_on="fiscal_period_end",
                right_on="sec_fpe",
                how="left",
            )
            cmp["was_over_sec"] = cmp["weighted_average_shares_diluted"] / cmp["sec_value"]
            print("\n--- WAS diluted / SEC shares (matched FPE) ---")
            print(
                cmp[
                    [
                        "fiscal_period_end",
                        "fiscal_period",
                        "weighted_average_shares_diluted",
                        "sec_value",
                        "was_over_sec",
                        "sec_form",
                    ]
                ]
                .dropna(subset=["sec_value"])
                .to_string(index=False)
            )

        result["targets"][str(sid)] = {
            "symbol": sym,
            "monthly": json.loads(tbl.to_json(orient="records", date_format="iso")),
            "sec_shares": json.loads(sec.to_json(orient="records", date_format="iso")),
            "fund_quarterly": json.loads(fq.to_json(orient="records", date_format="iso")),
            "splits": json.loads(sp.to_json(orient="records", date_format="iso")),
        }

    # Backup membership
    bak = pd.read_parquet(BACKUP / "universe.parquet")
    bak["date"] = pd.to_datetime(bak["date"])
    new = pd.read_parquet(OUT / "universe.parquet")
    new["date"] = pd.to_datetime(new["date"])
    for sid, sym in TARGETS.items():
        b = bak[bak["security_id"] == sid]
        n = new[new["security_id"] == sid]
        print(f"\n{sym} backup months in univ: {len(b)} ranks={b['rank'].tolist()[:5]}...")
        print(f"{sym} new months in univ: {len(n)} rank range {n['rank'].min() if len(n) else None}-{n['rank'].max() if len(n) else None}")

    print("\n======== panel-wide mcap jumps ========")
    jumps = panel_wide_mcap_jumps(new)
    print(f"jump events: {len(jumps)}")
    top_j = (
        jumps.groupby("security_id")
        .agg(
            symbol=("symbol", "first"),
            n=("mcap_mult", "size"),
            max_severity=("severity", "max"),
            max_mcap_mult=("mcap_mult", "max"),
            min_mcap_mult=("mcap_mult", "min"),
            example_date=("date", "first"),
        )
        .sort_values("max_severity", ascending=False)
        .head(30)
        .reset_index()
    )
    print(top_j.to_string(index=False))

    print("\n======== shares vs SEC (>3x) ========")
    sec_bad = shares_vs_sec_scan(new)
    print(f"row-level disagreements: {len(sec_bad)}")
    if len(sec_bad):
        top_s = (
            sec_bad.groupby("security_id")
            .agg(
                symbol=("symbol", "first"),
                n=("ratio", "size"),
                max_severity=("severity", "max"),
                max_ratio=("ratio", "max"),
                min_ratio=("ratio", "min"),
                med_ratio=("ratio", "median"),
                example_date=("date", "first"),
            )
            .sort_values("max_severity", ascending=False)
            .head(40)
            .reset_index()
        )
        print(top_s.to_string(index=False))
        result["sec_disagreement_top"] = json.loads(top_s.to_json(orient="records", date_format="iso"))
        # spotlight targets
        for sid in TARGETS:
            sub = sec_bad[sec_bad["security_id"] == sid]
            print(f"\n{TARGETS[sid]} SEC-disagree rows: {len(sub)}")
            if len(sub):
                print(
                    sub[
                        ["date", "symbol", "shares_used", "sec_shares", "ratio", "mcap", "rank"]
                    ]
                    .head(20)
                    .to_string(index=False)
                )
    else:
        result["sec_disagreement_top"] = []

    result["mcap_jump_top"] = json.loads(top_j.to_json(orient="records", date_format="iso"))
    result["n_mcap_jump_events"] = int(len(jumps))
    result["n_sec_disagree_rows"] = int(len(sec_bad)) if len(sec_bad) else 0

    path = OUT / "mcap_outlier_investigation.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
