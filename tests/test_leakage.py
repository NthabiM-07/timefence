"""Leakage guarantees on generated data: audit, canary poisoning, replay parity, golden values."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import REGISTRY
from timefence.replay import parity, replay
from timefence.store import FeatureStore, frame_hash


def test_audit_passes_for_bitemporal_and_catches_the_other_strategies(store, apps):
    audits = {s: store.build(apps, s, with_manifest=False).audit()
              for s in ("naive", "event_time", "bitemporal")}
    assert (audits["bitemporal"]["status"] == "PASS").all()
    assert audits["event_time"]["late_knowledge_rows"].sum() > 0
    assert audits["event_time"]["future_event_rows"].sum() == 0
    assert audits["naive"]["future_event_rows"].sum() > 0


def test_replay_parity(store, apps):
    """Logical bitemporal filtering == physical truncation of the warehouse."""
    month = apps[(apps["decision_time"] >= "2024-03-01") & (apps["decision_time"] < "2024-04-01")]
    assert len(month) > 30
    prod = replay(store, month)
    bt = store.build(month, "bitemporal", with_manifest=False)
    et = store.build(month, "event_time", with_manifest=False)
    feats = list(bt.features)
    assert parity(bt.values, prod.values, feats)["mismatched_rows"].sum() == 0
    assert parity(et.values, prod.values, feats)["mismatched_rows"].sum() > 0  # the test can fail


def test_canary_poison_never_reaches_a_bitemporal_frame(raw_dir, apps, tmp_path):
    """Plant absurd values that happened before each decision but landed after it."""
    rng = np.random.default_rng(0)
    targets = (apps.drop_duplicates("customer_id").sample(40, random_state=1)
               .sort_values("application_id").reset_index(drop=True))
    for f in raw_dir.glob("*.parquet"):
        (tmp_path / f.name).write_bytes(f.read_bytes())

    snaps = pd.read_parquet(tmp_path / "account_snapshots.parquet")
    enq = pd.read_parquet(tmp_path / "bureau_enquiries.parquet")
    coll = pd.read_parquet(tmp_path / "collections.parquet")
    lands = targets["decision_time"] + pd.Timedelta(hours=1)
    happened = targets["decision_time"] - pd.to_timedelta(rng.integers(1, 20, len(targets)), unit="D")
    poison_snaps = pd.DataFrame({
        "account_id": "POISON-" + targets["application_id"], "customer_id": targets["customer_id"],
        "product": "card", "snapshot_at": happened, "balance": 9e9, "credit_limit": 1.0, "dpd": 999,
        "ingested_at": lands, "record_version": 1})
    poison_enq = pd.DataFrame({
        "enquiry_id": "POISON-" + targets["application_id"], "customer_id": targets["customer_id"],
        "enquired_at": happened, "lender_type": "micro_lender", "ingested_at": lands, "record_version": 1})
    poison_coll = pd.DataFrame({
        "referral_id": "POISON-" + targets["application_id"], "account_id": "X",
        "customer_id": targets["customer_id"], "effective_at": happened, "referral_reason": "poison",
        "ingested_at": lands, "record_version": 1})
    pd.concat([snaps, poison_snaps]).to_parquet(tmp_path / "account_snapshots.parquet", index=False)
    pd.concat([enq, poison_enq]).to_parquet(tmp_path / "bureau_enquiries.parquet", index=False)
    pd.concat([coll, poison_coll]).to_parquet(tmp_path / "collections.parquet", index=False)

    clean = FeatureStore(REGISTRY, raw_dir).build(targets, "bitemporal", with_manifest=False).values
    poisoned_store = FeatureStore(REGISTRY, tmp_path)
    poisoned = poisoned_store.build(targets, "bitemporal", with_manifest=False).values
    pd.testing.assert_frame_equal(clean, poisoned)

    leaky = poisoned_store.build(targets, "event_time", with_manifest=False).values
    assert (leaky["worst_dpd_12m"] == 999).all()  # proves the poison was reachable


def test_golden_values_match_a_hand_written_pandas_implementation(raw_dir, store, apps):
    sample = apps.sample(60, random_state=3).reset_index(drop=True)
    built = store.build(sample, "bitemporal", with_manifest=False).values.set_index("application_id")
    snaps = pd.read_parquet(raw_dir / "account_snapshots.parquet")
    pays = pd.read_parquet(raw_dir / "payments.parquet")
    enq = pd.read_parquet(raw_dir / "bureau_enquiries.parquet")

    def as_known(df, key, t):
        known = df[df["ingested_at"] <= t]
        return known.sort_values(["ingested_at", "record_version"]).groupby(key, sort=False).tail(1)

    for row in sample.itertuples():
        t, c = row.decision_time, row.customer_id
        s = as_known(snaps[snaps["customer_id"] == c], ["account_id", "snapshot_at"], t)
        s = s[(s["snapshot_at"] <= t) & (s["snapshot_at"] > t - pd.Timedelta(days=92))]
        assert built.loc[row.application_id, "worst_dpd_3m"] == (s["dpd"].max() if len(s) else 0)

        p = as_known(pays[pays["customer_id"] == c], ["payment_id"], t)
        p = p[(p["paid_at"] <= t) & (p["paid_at"] > t - pd.Timedelta(days=183))]
        assert built.loc[row.application_id, "payments_reversed_6m"] == int((p["status"] == "reversed").sum())

        e = as_known(enq[enq["customer_id"] == c], ["enquiry_id"], t)
        e = e[(e["enquired_at"] <= t) & (e["enquired_at"] > t - pd.Timedelta(days=92))]
        assert built.loc[row.application_id, "enquiries_3m"] == len(e)


def test_builds_are_reproducible_and_manifested(store, apps):
    a = store.build(apps.head(500), "bitemporal")
    b = store.build(apps.head(500), "bitemporal")
    assert frame_hash(a.values) == frame_hash(b.values)
    assert a.manifest["output_sha256"] == b.manifest["output_sha256"]
    assert a.manifest["registry_sha256"] == store.registry.sha256
    assert set(a.manifest["sources"]) == set(store.registry.sources)


def test_duplicate_version_keys_are_rejected(raw_dir, tmp_path):
    for f in raw_dir.glob("*.parquet"):
        (tmp_path / f.name).write_bytes(f.read_bytes())
    enq = pd.read_parquet(tmp_path / "bureau_enquiries.parquet")
    pd.concat([enq, enq.head(1)]).to_parquet(tmp_path / "bureau_enquiries.parquet", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        FeatureStore(REGISTRY, tmp_path)


def test_replay_of_an_empty_spine_returns_an_empty_frame(store, apps):
    out = replay(store, apps.iloc[0:0])
    assert out.values.empty and out.manifest["knowledge_epochs"] == 0
    assert list(out.features) == [f.name for f in store.registry.select(purpose="credit_decisioning")]


def test_paths_and_identifiers_are_quoted(raw_dir, apps, tmp_path):
    odd = tmp_path / "Mo'Nique's data"
    odd.mkdir()
    for f in raw_dir.glob("*.parquet"):
        (odd / f.name).write_bytes(f.read_bytes())
    a = FeatureStore(REGISTRY, odd).build(apps.head(200), "bitemporal", with_manifest=False).values
    b = FeatureStore(REGISTRY, raw_dir).build(apps.head(200), "bitemporal", with_manifest=False).values
    pd.testing.assert_frame_equal(a, b)


def test_lazy_parquet_views_give_identical_results(raw_dir, store, apps):
    lazy = FeatureStore(REGISTRY, raw_dir, in_memory=False)
    a = lazy.build(apps, "bitemporal", with_manifest=False).values
    assert frame_hash(a) == frame_hash(store.build(apps, "bitemporal", with_manifest=False).values)
    month = apps[(apps["decision_time"] >= "2024-05-01") & (apps["decision_time"] < "2024-06-01")]
    feats = [f.name for f in store.registry.select(purpose="credit_decisioning")]
    offline = store.build(month, "bitemporal", with_manifest=False).values
    assert parity(replay(lazy, month).values, offline, feats)["mismatched_rows"].sum() == 0
