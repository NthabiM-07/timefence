"""A minimal online store: materialise point-in-time features to SQLite for lookup.

Materialisation runs the *same* compiled bitemporal query as training, with every
customer on the spine at one ``as_of``. So there is one definition of a feature,
used both offline and online: the usual source of training/serving skew is gone
by construction, and the parity test checks it anyway.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config
from .store import FeatureStore


def materialise(store: FeatureStore, as_of: pd.Timestamp, db_path: Path = config.ONLINE_DB,
                purpose: str = "credit_decisioning") -> int:
    customers = pd.read_parquet(store.raw_dir / "customers.parquet", columns=["customer_id", "onboarded_at"])
    customers = customers[customers["onboarded_at"] <= as_of]
    spine = pd.DataFrame({"request_id": customers["customer_id"].values,
                          "customer_id": customers["customer_id"].values,
                          "decision_time": pd.Timestamp(as_of)})
    frame = store.build(spine, "bitemporal", purpose=purpose, row_id="request_id", with_manifest=False)
    out = frame.values.drop(columns=["request_id", "decision_time"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        out.to_sql("online_features", con, if_exists="replace", index=False)
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_customer ON online_features(customer_id)")
        con.execute("DROP TABLE IF EXISTS online_meta")
        con.execute("CREATE TABLE online_meta (k TEXT PRIMARY KEY, v TEXT)")
        meta = {"as_of": str(pd.Timestamp(as_of)), "registry_sha256": store.registry.sha256,
                "purpose": purpose, "features": json.dumps(list(frame.features)),
                "materialised_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        con.executemany("INSERT INTO online_meta VALUES (?, ?)", meta.items())
    return len(out)


def lookup(customer_id: str, db_path: Path = config.ONLINE_DB,
           expected_registry_sha256: str | None = None) -> dict:
    """Serve one customer's features. Refuses to serve if the registry has moved on."""
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        meta = dict(con.execute("SELECT k, v FROM online_meta").fetchall())
        if expected_registry_sha256 and meta["registry_sha256"] != expected_registry_sha256:
            raise RuntimeError("Online store was materialised from a different feature registry; re-materialise")
        row = con.execute("SELECT * FROM online_features WHERE customer_id = ?", (customer_id,)).fetchone()
    if row is None:
        raise KeyError(customer_id)
    return {"as_of": meta["as_of"], "registry_sha256": meta["registry_sha256"], "features": dict(row)}
