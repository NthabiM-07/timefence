from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from timefence import config
from timefence.generate import generate, write
from timefence.store import FeatureStore

REGISTRY = config.ROOT / "features" / "credit_features.yaml"


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("raw")
    write(generate(n_customers=1500, seed=11), d)
    return d


@pytest.fixture(scope="session")
def store(raw_dir) -> FeatureStore:
    return FeatureStore(REGISTRY, raw_dir)


@pytest.fixture(scope="session")
def apps(raw_dir) -> pd.DataFrame:
    return pd.read_parquet(raw_dir / "applications.parquet")


# ------------------------------------------------------------------ mini world
T = pd.Timestamp


def _frame(rows: list[dict], columns: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(columns))
    return df.astype(columns)


SCHEMAS = {
    "customers": {"customer_id": "str", "onboarded_at": "datetime64[ns]", "income_band": "int64",
                  "age_band": "str", "province": "str", "ingested_at": "datetime64[ns]",
                  "record_version": "int64"},
    "account_snapshots": {"account_id": "str", "customer_id": "str", "product": "str",
                          "snapshot_at": "datetime64[ns]", "balance": "float64", "credit_limit": "float64",
                          "dpd": "int64", "ingested_at": "datetime64[ns]", "record_version": "int64"},
    "payments": {"payment_id": "str", "account_id": "str", "customer_id": "str", "paid_at": "datetime64[ns]",
                 "amount_due": "float64", "amount_paid": "float64", "status": "str",
                 "ingested_at": "datetime64[ns]", "record_version": "int64"},
    "bureau_enquiries": {"enquiry_id": "str", "customer_id": "str", "enquired_at": "datetime64[ns]",
                         "lender_type": "str", "ingested_at": "datetime64[ns]", "record_version": "int64"},
    "collections": {"referral_id": "str", "account_id": "str", "customer_id": "str",
                    "effective_at": "datetime64[ns]", "referral_reason": "str",
                    "ingested_at": "datetime64[ns]", "record_version": "int64"},
}


@pytest.fixture
def mini_world(tmp_path):
    """Write hand-made source tables and return a FeatureStore over them."""

    def make(**tables: list[dict]) -> FeatureStore:
        base = {"customers": [dict(customer_id="C1", onboarded_at=T("2020-01-15 09:00"), income_band=3,
                                   age_band="26-35", province="Gauteng",
                                   ingested_at=T("2020-01-16 02:00"), record_version=1)]}
        for name, schema in SCHEMAS.items():
            rows = tables.get(name, base.get(name, []))
            _frame(rows, schema).to_parquet(tmp_path / f"{name}.parquet", index=False)
        return FeatureStore(REGISTRY, tmp_path)

    return make


def spine(*decision_times: str, customer: str = "C1") -> pd.DataFrame:
    return pd.DataFrame({"application_id": [f"A{i}" for i in range(len(decision_times))],
                         "customer_id": customer,
                         "decision_time": [T(t) for t in decision_times]})
