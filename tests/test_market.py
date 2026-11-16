"""Imbalance-price collector: a reply without prices must not be stored, and a stored empty file must be refetched."""
import datetime as dt

import pandas as pd
import pytest

from foretec_live import market

D = dt.date(2026, 11, 11)
NO_DATA = (b'<?xml version="1.0" encoding="UTF-8"?><Acknowledgement_MarketDocument><Reason><code>999</code>'
           b'<text>No matching data found</text></Reason></Acknowledgement_MarketDocument>')


def _prices(n: int) -> bytes:
    pts = "".join(f"<Point><position>{i}</position><imbalance_Price.amount>{i}</imbalance_Price.amount>"
                  f"<imbalance_Price.category>{c}</imbalance_Price.category></Point>"
                  for i in range(1, n + 1) for c in ("A04", "A05"))
    return (f'<Balancing_MarketDocument xmlns="urn:x"><TimeSeries><Period><timeInterval><start>2026-11-10T23:00Z</start>'
            f"<end>2026-11-11T23:00Z</end></timeInterval><resolution>PT15M</resolution>{pts}</Period></TimeSeries>"
            f"</Balancing_MarketDocument>").encode()


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTSOE_TOKEN", "test")
    return {"_root": str(tmp_path), "timezone": "Europe/Brussels", "zones": {"NL": {"entsoe": "10YNL----------L"}}}


def test_no_data_reply_is_not_stored(cfg, monkeypatch):
    monkeypatch.setattr(market, "_request", lambda params: NO_DATA)
    df = market.fetch_imbalance(cfg, "NL", D)
    assert list(df.columns) == ["long", "short"] and df.isna().all(axis=None)
    assert market.update_imbalance(cfg, D, D) == []
    assert not market._path(cfg, "NL", D).exists()


def test_empty_file_is_refetched_and_ignored_on_load(cfg, monkeypatch):
    f = market._path(cfg, "NL", D)
    f.parent.mkdir(parents=True)
    pd.DataFrame(index=pd.DatetimeIndex([])).to_parquet(f)         # the 11-11 / 11-12 NL files: no columns at all
    assert market.load_imbalance(cfg, "NL", D) is None
    monkeypatch.setattr(market, "_request", lambda params: _prices(96))
    assert market.update_imbalance(cfg, D, D) == [f"{D} NL"]
    got = market.load_imbalance(cfg, "NL", D)
    assert len(got) == 96 and got[["long", "short"]].notna().all(axis=None)
