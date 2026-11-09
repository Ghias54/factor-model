"""Unit tests for Ken French CSV parsing (offline fixture)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from factor_model.external.ff import parse_ff_monthly_csv

FIX = Path(__file__).resolve().parent / "fixtures" / "ff"


def test_parse_ff5_sample_percent_to_decimal():
    text = (FIX / "ff5_sample.csv").read_text(encoding="utf-8")
    df = parse_ff_monthly_csv(text)
    assert list(df.columns) == ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert df.index.min() == pd.Timestamp("1963-07-31")
    # First row Mkt-RF was -0.39 percent → -0.0039
    assert abs(df.iloc[0]["Mkt_RF"] - (-0.0039)) < 1e-10
    assert abs(df.iloc[0]["RF"] - 0.0027) < 1e-10


def test_parse_mom_sample_renames_umd():
    text = (FIX / "mom_sample.csv").read_text(encoding="utf-8")
    df = parse_ff_monthly_csv(text)
    assert "UMD" in df.columns
    assert df.index.min() == pd.Timestamp("1927-01-31")
    assert abs(df.iloc[0]["UMD"] - 0.0057) < 1e-10
