"""A minimal online store: materialise point-in-time features to SQLite for lookup.

Materialisation runs the *same* compiled bitemporal query as training, with every
customer on the spine at one ``as_of``. So there is one definition of a feature,
used both offline and online: the usual source of training/serving skew is gone
by construction, and the parity test checks it anyway.

A refresh stages the new snapshot beside the live one and swaps both tables in a
single transaction, so a failed refresh leaves the previous snapshot serving.
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
    meta = {"as_of": str(pd.Timestamp(as_of)), "registry_sha256": store.registry.sha256,
            "purpose": purpose, "features": json.dumps(list(frame.features)),
            "materialised_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, isolation_level=None)  # explicit transactions only
    try:
        # 1. Stage the new snapshot next to the live one. Readers still see the old one.
        con.execute("DROP TABLE IF EXISTS online_features_next")
        con.execute("DROP TABLE IF EXISTS online_meta_next")
        out.to_sql("online_features_next", con, index=False)
        con.execute("CREATE TABLE online_meta_next (k TEXT PRIMARY KEY, v TEXT)")
        con.executemany("INSERT INTO online_meta_next VALUES (?, ?)", meta.items())
        _before_swap()
        # 2. Swap features and metadata together, or not at all.
        con.execute("BEGIN IMMEDIATE")
        try:
            con.execute("DROP TABLE IF EXISTS online_features")  # also drops its index
            con.execute("DROP TABLE IF EXISTS online_meta")
            con.execute("ALTER TABLE online_features_next RENAME TO online_features")
            con.execute("ALTER TABLE online_meta_next RENAME TO online_meta")
            con.execute("CREATE UNIQUE INDEX ux_online_customer ON online_features(customer_id)")
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
    finally:
        con.close()
    return len(out)


def _before_swap() -> None:
    """Test hook: lets a test fail a refresh after staging, before the swap."""


def lookup(customer_id: str, db_path: Path = config.ONLINE_DB,
           expected_registry_sha256: str | None = None) -> dict:
    """Serve one customer's features. Refuses to serve if the registry has moved on."""
    con = sqlite3.connect(db_path, isolation_level=None)
    try:
        con.row_factory = sqlite3.Row
        # One read transaction, so metadata and row come from the same snapshot even if
        # a refresh commits in between the two statements.
        con.execute("BEGIN")
        meta = dict(con.execute("SELECT k, v FROM online_meta").fetchall())
        row = con.execute("SELECT * FROM online_features WHERE customer_id = ?", (customer_id,)).fetchone()
        con.execute("COMMIT")
    finally:
        con.close()
    if expected_registry_sha256 and meta["registry_sha256"] != expected_registry_sha256:
        raise RuntimeError("Online store was materialised from a different feature registry; re-materialise")
    if row is None:
        raise KeyError(customer_id)
    return {"as_of": meta["as_of"], "registry_sha256": meta["registry_sha256"], "features": dict(row)}
