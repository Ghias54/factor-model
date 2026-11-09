"""Ken French Data Library monthly factors (local parquet cache)."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from factor_model.config import ROOT

FF_DIR = ROOT / "data" / "external" / "ff"

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)


def parse_ff_monthly_csv(text: str, value_cols: list[str] | None = None) -> pd.DataFrame:
    """Parse a Ken French monthly CSV (stops before the Annual section).

    Values are converted from percent to decimal. Index is month-end Timestamp
    named ``date``.
    """
    lines = text.splitlines()
    # Find header row (starts with comma or contains Mkt-RF / Mom)
    header_i = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        # Header rows look like ",Mkt-RF,SMB,..." or ",Mom" — not prose mentioning Mom.
        if s.startswith(",") and ("Mkt-RF" in s or s == ",Mom" or s.startswith(",Mom")):
            header_i = i
            break
        if s.startswith("Mkt-RF,") or s == "Mom" or s.startswith("Mom,"):
            header_i = i
            break
    if header_i is None:
        raise ValueError("could not find Ken French monthly header row")

    body: list[str] = []
    for line in lines[header_i:]:
        if line.strip().lower().startswith("annual"):
            break
        if line.strip() == "" and body:
            # blank line after monthly block
            break
        body.append(line)

    raw = "\n".join(body)
    df = pd.read_csv(io.StringIO(raw), skipinitialspace=True)
    # First column is YYYYMM (may be unnamed)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "yyyymm"})
    df["yyyymm"] = pd.to_numeric(df["yyyymm"], errors="coerce")
    df = df.dropna(subset=["yyyymm"])
    df["yyyymm"] = df["yyyymm"].astype(int)
    # Drop annual rows that snuck through (4-digit years)
    df = df[df["yyyymm"] >= 100000]
    df["date"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.drop(columns=["yyyymm"]).set_index("date").sort_index()
    df.columns = [c.strip().replace("-", "_") for c in df.columns]
    # Momentum file uses Mom → UMD
    rename = {"Mom": "UMD", "Mkt_RF": "Mkt_RF"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "Mom" in df.columns:
        df = df.rename(columns={"Mom": "UMD"})
    # Missing sentinels
    df = df.replace([-99.99, -999, -999.0], pd.NA)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    if value_cols is not None:
        missing = [c for c in value_cols if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns {missing}; have {list(df.columns)}")
        df = df[value_cols]
    return df


def download_ff(dest: Path | None = None, *, force: bool = False) -> Path:
    """Download FF5 + Momentum zips, parse to parquet, write meta.json."""
    dest = dest or FF_DIR
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "ff_factors.parquet"
    meta_path = dest / "meta.json"
    if out_path.exists() and meta_path.exists() and not force:
        return out_path

    sources = {"ff5": FF5_URL, "mom": MOM_URL}
    texts: dict[str, str] = {}
    for key, url in sources.items():
        raw = urlopen(url, timeout=120).read()
        (dest / f"{key}.zip").write_bytes(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            texts[key] = zf.read(name).decode("latin-1")
            (dest / f"{key}_raw.csv").write_text(texts[key], encoding="utf-8")

    ff5 = parse_ff_monthly_csv(texts["ff5"])
    mom = parse_ff_monthly_csv(texts["mom"])
    # Ensure UMD column name
    if "UMD" not in mom.columns:
        # after rename Mom→UMD; if still Mom
        if "Mom" in mom.columns:
            mom = mom.rename(columns={"Mom": "UMD"})
        else:
            raise ValueError(f"momentum columns: {list(mom.columns)}")
    out = ff5.join(mom[["UMD"]], how="outer").sort_index()
    out.to_parquet(out_path, compression="zstd")
    meta = {
        "downloaded": str(date.today()),
        "sources": sources,
        "columns": list(out.columns),
        "start": str(out.index.min().date()),
        "end": str(out.index.max().date()),
        "units": "decimal (percent / 100)",
        "path": str(out_path.relative_to(ROOT)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_path


def load_ff_factors(path: Path | None = None) -> pd.DataFrame:
    """Load cached FF monthly factors (decimal). Downloads once if missing."""
    path = path or (FF_DIR / "ff_factors.parquet")
    if not path.exists():
        download_ff(path.parent)
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index).astype("datetime64[ns]")
    df.index.name = "date"
    return df
