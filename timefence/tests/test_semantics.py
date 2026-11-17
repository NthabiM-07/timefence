"""Exact as-of semantics on a hand-built world where every answer is known."""
from __future__ import annotations

import pandas as pd

from conftest import T, spine


def snap(dpd, at, ingested, version=1, account="AC1", balance=500.0):
    return dict(account_id=account, customer_id="C1", product="card", snapshot_at=T(at), balance=balance,
                credit_limit=1000.0, dpd=dpd, ingested_at=T(ingested), record_version=version)


def enquiry(eid, at, ingested, version=1, lender="bank"):
    return dict(enquiry_id=eid, customer_id="C1", enquired_at=T(at), lender_type=lender,
                ingested_at=T(ingested), record_version=version)


def value(store, strategy, decision, feature):
    v = store.build(spine(decision), strategy, features=[feature], with_manifest=False).values[feature].iloc[0]
    return None if pd.isna(v) else v


def test_fact_is_unknown_until_it_lands_and_known_from_the_load_instant(mini_world):
    fs = mini_world(account_snapshots=[snap(30, "2024-01-31 23:59:59", "2024-02-01 02:00")])
    assert value(fs, "bitemporal", "2024-02-01 01:59", "worst_dpd_1m") is None
    assert value(fs, "event_time", "2024-02-01 01:59", "worst_dpd_1m") == 30  # the leak
    assert value(fs, "bitemporal", "2024-02-01 02:00", "worst_dpd_1m") == 30  # inclusive boundary


def test_restatement_is_invisible_before_it_lands(mini_world):
    fs = mini_world(account_snapshots=[
        snap(0, "2024-01-31 23:59:59", "2024-02-01 02:00", version=1),
        snap(60, "2024-01-31 23:59:59", "2024-02-10 02:00", version=2),
    ])
    assert value(fs, "bitemporal", "2024-02-05 10:00", "worst_dpd_1m") == 0
    assert value(fs, "event_time", "2024-02-05 10:00", "worst_dpd_1m") == 60
    assert value(fs, "bitemporal", "2024-02-11 10:00", "worst_dpd_1m") == 60


def test_same_batch_versions_break_ties_on_version_not_row_order(mini_world):
    # Written in the "wrong" order on purpose.
    fs = mini_world(account_snapshots=[
        snap(60, "2024-01-31 23:59:59", "2024-02-01 02:00", version=2),
        snap(30, "2024-01-31 23:59:59", "2024-02-01 02:00", version=1),
    ])
    for _ in range(3):
        assert value(fs, "bitemporal", "2024-02-02 10:00", "worst_dpd_1m") == 60


def test_window_is_applied_after_choosing_the_known_version(mini_world):
    # v2 corrects the enquiry date backwards, out of the 3-month window.
    fs = mini_world(bureau_enquiries=[
        enquiry("E1", "2024-01-20 10:00", "2024-02-12 02:00", version=1),
        enquiry("E1", "2023-06-01 10:00", "2024-02-20 02:00", version=2),
    ])
    assert value(fs, "bitemporal", "2024-02-15 10:00", "enquiries_3m") == 1  # only v1 known
    assert value(fs, "bitemporal", "2024-03-01 10:00", "enquiries_3m") == 0  # v2 moved it out
    assert value(fs, "bitemporal", "2024-03-01 10:00", "enquiries_12m") == 1


def test_window_lower_bound_is_exclusive(mini_world):
    fs = mini_world(account_snapshots=[snap(90, "2024-01-31 23:59:59", "2024-02-01 02:00")])
    edge = T("2024-01-31 23:59:59") + pd.Timedelta(days=92)
    assert value(fs, "bitemporal", str(edge), "worst_dpd_3m") == 0  # default when nothing in window
    assert value(fs, "bitemporal", str(edge), "worst_dpd_12m") == 90


def test_backdated_collections_referral(mini_world):
    fs = mini_world(collections=[dict(referral_id="R1", account_id="AC1", customer_id="C1",
                                      effective_at=T("2024-01-31 23:59:59"), referral_reason="60_dpd",
                                      ingested_at=T("2024-03-10 02:00"), record_version=1)])
    assert value(fs, "bitemporal", "2024-03-01 10:00", "collections_referrals_12m") == 0
    assert value(fs, "event_time", "2024-03-01 10:00", "collections_referrals_12m") == 1
    fr = fs.build(spine("2024-03-01 10:00"), "event_time", features=["collections_referrals_12m"],
                  with_manifest=False)
    row = fr.audit().iloc[0]
    assert row["late_knowledge_rows"] == 1 and row["median_days_late"] > 8


def test_naive_is_flagged_even_when_the_value_is_zero(mini_world):
    fs = mini_world(bureau_enquiries=[enquiry("E1", "2024-06-01 10:00", "2024-07-12 02:00")])
    fr = fs.build(spine("2024-01-10 10:00"), "naive", features=["enquiries_3m", "enquiries_12m"],
                  with_manifest=False)
    assert (fr.audit()["status"] == "FAIL").all()


def test_derived_features_inherit_lineage(mini_world):
    fs = mini_world(payments=[
        dict(payment_id="P1", account_id="AC1", customer_id="C1", paid_at=T("2024-01-25 09:00"),
             amount_due=100.0, amount_paid=100.0, status="cleared", ingested_at=T("2024-01-26 02:00"),
             record_version=1),
        dict(payment_id="P1", account_id="AC1", customer_id="C1", paid_at=T("2024-01-25 09:00"),
             amount_due=100.0, amount_paid=100.0, status="reversed", ingested_at=T("2024-02-02 02:00"),
             record_version=2),
    ])
    assert value(fs, "bitemporal", "2024-01-30 10:00", "reversal_rate_6m") == 0
    assert value(fs, "event_time", "2024-01-30 10:00", "reversal_rate_6m") == 1
    fr = fs.build(spine("2024-01-30 10:00"), "event_time", features=["reversal_rate_6m"], with_manifest=False)
    assert fr.audit().iloc[0]["status"] == "FAIL"
