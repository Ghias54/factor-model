"""Unit tests for reverse-split mcap sanity (vendor ratio gate)."""
from __future__ import annotations

import pandas as pd

from factor_model.universe.build import apply_mcap_vendor_sanity


def test_sanity_replaces_inflated_computed_mcap():
    computed = pd.Series([1e10, 5e8, None, 2e9])  # KNDI-like 10x, ok, missing, 4x
    vendor = pd.Series([1e9, 5e8, 3e8, 5e8])
    mcap, replaced = apply_mcap_vendor_sanity(computed, vendor, enabled=True, max_ratio=3.0)
    assert bool(replaced.iloc[0]) is True
    assert float(mcap.iloc[0]) == 1e9
    assert bool(replaced.iloc[1]) is False
    assert float(mcap.iloc[1]) == 5e8
    assert bool(replaced.iloc[2]) is False
    assert float(mcap.iloc[2]) == 3e8  # fell back to vendor
    assert bool(replaced.iloc[3]) is True
    assert float(mcap.iloc[3]) == 5e8


def test_sanity_disabled_keeps_computed():
    computed = pd.Series([1e10])
    vendor = pd.Series([1e9])
    mcap, replaced = apply_mcap_vendor_sanity(computed, vendor, enabled=False, max_ratio=3.0)
    assert bool(replaced.iloc[0]) is False
    assert float(mcap.iloc[0]) == 1e10


def test_sanity_catches_deflated_computed():
    computed = pd.Series([1e8])
    vendor = pd.Series([1e9])  # computed 0.1x vendor
    mcap, replaced = apply_mcap_vendor_sanity(computed, vendor, enabled=True, max_ratio=3.0)
    assert bool(replaced.iloc[0]) is True
    assert float(mcap.iloc[0]) == 1e9


def test_implied_shares_from_vendor_mcap():
    close = pd.Series([10.0, 20.0])
    vendor = pd.Series([1e9, 2e9])
    implied = vendor / close
    assert float(implied.iloc[0]) == 1e8
    assert float(implied.iloc[1]) == 1e8
