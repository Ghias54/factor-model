"""Read-only access to the stock-database Postgres.

All database reads in this repo go through this module. Never write to the
stock database from here.
"""
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def get_engine(dev: bool = False):
    key = "STOCKDB_DEV_DSN" if dev else "STOCKDB_DSN"
    dsn = os.environ.get(key)
    if not dsn:
        raise RuntimeError(f"{key} is not set. Copy .env.example to .env and fill it in.")
    # Read-only at the session level so an accidental write fails loudly.
    return create_engine(
        dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        connect_args={"options": "-c default_transaction_read_only=on"},
    )


def read_sql(query: str, params: dict | None = None, dev: bool = False) -> pd.DataFrame:
    with get_engine(dev).connect() as conn:
        return pd.read_sql(text(query), conn, params=params)
