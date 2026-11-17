"""Registry validation, permitted use, label maturity, labels and the online store."""
from __future__ import annotations

import pandas as pd
import pytest
import yaml

from conftest import REGISTRY
from timefence import config
from timefence.online import lookup, materialise
from timefence.registry import PermittedUseError, RegistryError, load_registry
from timefence.store import FeatureStore, LabelMaturityError, attach_labels


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


def test_string_literals_in_derived_features_are_not_dependencies(tmp_path):
    def add(r):
        expr = "CASE WHEN enquiries_3m > 0 THEN 'enquiries_12m' END"
        r["features"].append({"name": "flag", "derived": expr,
                              "description": "x", "owner": "o", "permitted_use": ["analytics"]})
    reg = load_registry(_write(tmp_path, add))
    assert reg.features["flag"].depends_on == ("enquiries_3m",)


def test_online_store_can_be_refreshed_repeatedly(store, tmp_path):
    db = tmp_path / "online.sqlite"
    for as_of in ("2025-03-31 12:00", "2025-06-30 12:00", "2025-09-30 12:00"):
        materialise(store, pd.Timestamp(as_of), db_path=db)
        assert lookup("C000042", db_path=db)["as_of"] == str(pd.Timestamp(as_of))


def test_failed_refresh_keeps_the_previous_snapshot_serving(store, tmp_path, monkeypatch):
    from timefence import online
    db = tmp_path / "online.sqlite"
    materialise(store, pd.Timestamp("2025-03-31 12:00"), db_path=db)
    before = lookup("C000042", db_path=db)

    def boom():
        raise RuntimeError("simulated crash after staging")

    monkeypatch.setattr(online, "_before_swap", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        materialise(store, pd.Timestamp("2025-09-30 12:00"), db_path=db)
    assert lookup("C000042", db_path=db) == before
    monkeypatch.undo()
    materialise(store, pd.Timestamp("2025-09-30 12:00"), db_path=db)  # recovers cleanly
    assert lookup("C000042", db_path=db)["as_of"] == "2025-09-30 12:00:00"


# ------------------------------------------------------------- config mistakes fail loudly
def _feature(r, name):
    return next(f for f in r["features"] if f["name"] == name)


def _rename_key(d, old, new):
    d[new] = d.pop(old)


@pytest.mark.parametrize("mutate, message", [
    (lambda r: _rename_key(_feature(r, "worst_dpd_3m"), "window_days", "windows_days"),
     r"unknown key 'windows_days' \(did you mean 'window_days'\?\)"),
    (lambda r: r["sources"]["payments"].update(record_key="payment_id"),
     "record_key must be a non-empty list"),
    (lambda r: _feature(r, "worst_dpd_12m").update(permitted_use="credit_decisioning"),
     "permitted_use must be a non-empty list"),
    (lambda r: _feature(r, "worst_dpd_12m").update(name="decision_time"), "reserved"),
    (lambda r: _feature(r, "worst_dpd_12m").update(name="customer_id"), "reserved"),
    (lambda r: _feature(r, "worst_dpd_3m").update(default="zero"), "default must be a number"),
    (lambda r: _feature(r, "enquiries_3m").update(where="1=1; DROP TABLE x"), "may not contain ';'"),
    (lambda r: _feature(r, "reversal_rate_6m").update(source="payments"), "unknown key 'source'"),
    (lambda r: r["sources"]["payments"].update(event_tme="paid_at"), "did you mean 'event_time'"),
])
def test_config_mistakes_are_rejected_with_a_clear_message(tmp_path, mutate, message):
    with pytest.raises(RegistryError, match=message):
        load_registry(_write(tmp_path, mutate))


def _copy_raw(raw_dir, dest):
    for f in raw_dir.glob("*.parquet"):
        (dest / f.name).write_bytes(f.read_bytes())


def test_source_missing_a_declared_column_is_named(raw_dir, tmp_path):
    _copy_raw(raw_dir, tmp_path)
    enq = pd.read_parquet(tmp_path / "bureau_enquiries.parquet").drop(columns="ingested_at")
    enq.to_parquet(tmp_path / "bureau_enquiries.parquet", index=False)
    with pytest.raises(ValueError, match=r"bureau_enquiries.*missing columns \['ingested_at'\]"):
        FeatureStore(REGISTRY, tmp_path)


def test_null_clock_values_are_rejected(raw_dir, tmp_path):
    """A NULL ingested_at would vanish from bitemporal builds but leak into event-time ones."""
    _copy_raw(raw_dir, tmp_path)
    pays = pd.read_parquet(tmp_path / "payments.parquet")
    pays.loc[pays.index[:3], "ingested_at"] = pd.NaT
    pays.to_parquet(tmp_path / "payments.parquet", index=False)
    with pytest.raises(ValueError, match=r"NULLs in key or clock columns.*'ingested_at': 3"):
        FeatureStore(REGISTRY, tmp_path)


def test_spine_rows_need_an_id_entity_and_decision_time(store, apps):
    bad = apps.head(5).copy()
    bad.loc[bad.index[0], "decision_time"] = pd.NaT
    with pytest.raises(ValueError, match="missing values in \\['decision_time'\\]"):
        store.build(bad, "bitemporal", with_manifest=False)
    clash = apps.head(5).rename(columns={"application_id": "enquiries_3m"})
    with pytest.raises(ValueError, match="clashes with a feature"):
        store.build(clash, "bitemporal", row_id="enquiries_3m", with_manifest=False)


def test_cli_reports_its_version(capsys):
    from timefence import __version__
    from timefence.cli import main
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out
