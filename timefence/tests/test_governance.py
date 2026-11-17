"""Registry validation, permitted use, label maturity, labels and the online store."""
from __future__ import annotations

import pandas as pd
import pytest
import yaml

from conftest import REGISTRY
from timefence import config
from timefence.online import lookup, materialise
from timefence.registry import PermittedUseError, RegistryError, load_registry
from timefence.store import LabelMaturityError, attach_labels


def _write(tmp_path, mutate):
    raw = yaml.safe_load(REGISTRY.read_text())
    mutate(raw)
    p = tmp_path / "reg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_registry_loads_and_hash_is_stable():
    a, b = load_registry(REGISTRY), load_registry(REGISTRY)
    assert a.sha256 == b.sha256 and len(a.features) >= 15


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r["features"].append(dict(r["features"][0])), "Duplicate"),
    (lambda r: r["features"][0].update(source="nope"), "unknown source"),
    (lambda r: r["features"][0].pop("owner"), "governance"),
    (lambda r: r["features"][0].update(window_days=-5), "positive"),
    (lambda r: r["features"].append({"name": "loop_a", "derived": "loop_b + 1", "description": "x",
                                     "owner": "o", "permitted_use": ["analytics"]}) or
     r["features"].append({"name": "loop_b", "derived": "loop_a + 1", "description": "x",
                           "owner": "o", "permitted_use": ["analytics"]}), "Cycle"),
])
def test_registry_rejects_bad_definitions(tmp_path, mutate, message):
    with pytest.raises(RegistryError, match=message):
        load_registry(_write(tmp_path, mutate))


def test_permitted_use_is_enforced(store, apps):
    names = [f.name for f in store.registry.select(purpose="credit_decisioning")]
    assert "province" not in names
    assert "province" in [f.name for f in store.registry.select(purpose="analytics")]
    with pytest.raises(PermittedUseError):
        store.build(apps.head(5), features=["worst_dpd_3m", "province"], purpose="credit_decisioning")


def test_label_maturity_guard(store, apps, raw_dir):
    labels = pd.read_parquet(raw_dir / "labels.parquet")
    fr = store.build(apps, "bitemporal", features=["worst_dpd_3m"], with_manifest=False)
    with pytest.raises(LabelMaturityError):
        attach_labels(fr, labels, training_cutoff=pd.Timestamp("2024-06-30"))
    kept = attach_labels(fr, labels, training_cutoff=pd.Timestamp("2024-06-30"), drop_immature=True)
    assert (kept["label_matured_at"] <= pd.Timestamp("2024-06-30")).all() and len(kept) > 0
    assert len(attach_labels(fr, labels, training_cutoff=config.DATA_END)) == len(apps)


def test_labels_agree_with_the_latest_snapshot_versions(store, raw_dir):
    """The generator's labels must be reproducible from the data a modeller would see."""
    labels = pd.read_parquet(raw_dir / "labels.parquet")
    store.con.register("lab", labels)
    apps = pd.read_parquet(raw_dir / "applications.parquet")
    store.con.register("apps_t", apps)
    derived = store.con.execute(f"""
        WITH latest AS (
            SELECT * FROM src_account_snapshots
            QUALIFY row_number() OVER (PARTITION BY account_id, snapshot_at
                                       ORDER BY ingested_at DESC, record_version DESC) = 1)
        SELECT a.application_id,
               coalesce(max(s.dpd) >= {config.BAD_DPD}, false)::INT AS bad_sql
        FROM apps_t a
        JOIN lab USING (application_id)
        LEFT JOIN latest s ON s.customer_id = a.customer_id
             AND s.snapshot_at > a.decision_time AND s.snapshot_at <= lab.label_matured_at
        GROUP BY a.application_id
    """).df()
    store.con.unregister("lab")
    store.con.unregister("apps_t")
    m = labels.merge(derived, on="application_id")
    assert len(m) == len(labels)
    assert (m["bad"] == m["bad_sql"]).all()


def test_online_store_serves_the_offline_definition(store, tmp_path):
    as_of = pd.Timestamp("2025-06-30 12:00")
    db = tmp_path / "online.sqlite"
    n = materialise(store, as_of, db_path=db)
    assert n > 0
    cust = "C000042"
    served = lookup(cust, db_path=db, expected_registry_sha256=store.registry.sha256)["features"]
    offline = store.build(pd.DataFrame({"application_id": ["x"], "customer_id": [cust],
                                        "decision_time": [as_of]}), "bitemporal", with_manifest=False)
    for f in offline.features:
        a, b = served[f], offline.values[f].iloc[0]
        assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), f
    with pytest.raises(RuntimeError):
        lookup(cust, db_path=db, expected_registry_sha256="0" * 64)
