"""Synthetic South African retail-credit data with realistic *arrival* behaviour.

The point of this generator is not the credit risk itself, it is *when* each
fact becomes known. Every record carries two clocks:

* ``event_time`` (valid time): when the fact was true in the world.
* ``ingested_at`` (knowledge time): when the warehouse first held it.

Four mechanisms make the two clocks disagree, and each one is a leak path that
an event-time-only point-in-time join will walk straight through:

1. **Monthly bureau file.** Enquiries made at other lenders in month M arrive in
   the bureau extract that lands on the 12th of month M+1 (8% slip a month).
2. **Debit-order reversals.** A payment lands as ``cleared`` the next night and
   is re-issued as ``reversed`` 3-10 days later, same ``payment_id``.
3. **Arrears restatements.** If a reversal lands after the month-end snapshot,
   the snapshot is restated (version 2) with the true days-past-due.
4. **Backdated collections referrals.** When an account reaches 60 DPD, a
   referral is recorded with its effective date set back to the start of the
   arrears, so its ``event_time`` is roughly 45 days before it lands.

Distressed customers generate more of all four, and also apply for credit more
often, so the late-arriving facts are exactly the ones that predict default.
No real person, account or bank data is used. Identifiers are synthetic.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PROVINCES = np.array(
    ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Free State",
     "Limpopo", "Mpumalanga", "North West", "Northern Cape"]
)
PROVINCE_P = np.array([0.27, 0.12, 0.19, 0.11, 0.05, 0.10, 0.08, 0.06, 0.02])
CARD_LIMIT_BY_BAND = np.array([8_000, 15_000, 30_000, 55_000, 90_000])
LOAN_PRINCIPAL_BY_BAND = np.array([20_000, 40_000, 80_000, 140_000, 250_000])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _macro(month_idx: int) -> float:
    """Rate-cycle stress: ramps from mid-2022, peaks 2023, eases through 2025."""
    if month_idx < 17:
        return 0.0
    if month_idx < 29:
        return 0.45 * (month_idx - 17) / 12
    if month_idx < 38:
        return 0.45
    if month_idx < 53:
        return 0.45 - 0.30 * (month_idx - 38) / 15
    return 0.15


def _load_time(day: pd.Series | np.ndarray) -> np.ndarray:
    """Floor to midnight and add the 02:00 batch load hour."""
    d = pd.to_datetime(day).astype("datetime64[ns]")
    return (d.values.astype("datetime64[D]") + np.timedelta64(config.LOAD_HOUR, "h")).astype(
        "datetime64[ns]"
    )


@dataclass
class GeneratedData:
    customers: pd.DataFrame
    accounts: pd.DataFrame
    account_snapshots: pd.DataFrame
    payments: pd.DataFrame
    bureau_enquiries: pd.DataFrame
    collections: pd.DataFrame
    applications: pd.DataFrame
    labels: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def generate(n_customers: int = 10_000, seed: int = 7) -> GeneratedData:
    rng = np.random.default_rng(seed)
    months = pd.date_range(config.HISTORY_START, periods=config.N_MONTHS, freq="MS")
    month_ends = months + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59, seconds=59)

    # ---------------------------------------------------------------- customers
    C = n_customers
    z = rng.normal(0, 1, C)  # latent, persistent credit quality (higher = riskier)
    income_band = np.clip(np.round(3 - 0.6 * z + rng.normal(0, 1, C)), 1, 5).astype(int)
    onboard_days = rng.integers(0, (pd.Timestamp("2023-06-30") - pd.Timestamp("2016-01-01")).days, C)
    onboarded_at = (
        pd.Timestamp("2016-01-01")
        + pd.to_timedelta(onboard_days, unit="D")
        + pd.to_timedelta(rng.integers(8 * 60, 18 * 60, C), unit="m")
    )
    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(C)],
            "onboarded_at": onboarded_at,
            "income_band": income_band,
            "age_band": rng.choice(["18-25", "26-35", "36-45", "46-55", "56+"], C,
                                   p=[0.14, 0.31, 0.26, 0.17, 0.12]),
            "province": rng.choice(PROVINCES, C, p=PROVINCE_P),
        }
    )
    customers["ingested_at"] = _load_time(customers["onboarded_at"] + pd.Timedelta(days=1))
    customers["record_version"] = 1

    # ----------------------------------------------------------------- accounts
    has_loan = rng.random(C) < 0.55
    card_cust = np.arange(C)
    loan_cust = np.flatnonzero(has_loan)
    acc_cust = np.concatenate([card_cust, loan_cust])
    acc_product = np.array(["card"] * C + ["personal_loan"] * len(loan_cust))
    A = len(acc_cust)
    loan_offset = pd.to_timedelta(rng.integers(0, 3 * 365, len(loan_cust)), unit="D")
    opened_at = np.concatenate(
        [onboarded_at.values, (onboarded_at[loan_cust] + loan_offset).values]
    ).astype("datetime64[ns]")
    band = income_band[acc_cust] - 1
    credit_limit = np.where(
        acc_product == "card",
        CARD_LIMIT_BY_BAND[band] * rng.lognormal(0, 0.25, A),
        LOAN_PRINCIPAL_BY_BAND[band] * rng.lognormal(0, 0.35, A),
    ).round(-2)
    term = rng.choice([24, 36, 48, 60, 72], A)
    instalment = np.where(acc_product == "personal_loan", credit_limit / term * 1.35, 0.0).round(2)
    accounts = pd.DataFrame(
        {
            "account_id": [f"AC{i:07d}" for i in range(A)],
            "customer_id": customers["customer_id"].values[acc_cust],
            "product": acc_product,
            "opened_at": opened_at,
            "credit_limit": credit_limit,
        }
    )

    # ------------------------------------------------------------ monthly loop
    s = rng.normal(0, 0.5, C)  # transient financial stress, AR(1)
    dpd = np.zeros(A, dtype=int)
    util = np.clip(_sigmoid(-0.5 + 0.9 * z[acc_cust]) + rng.normal(0, 0.1, A), 0, 1)
    balance = np.where(acc_product == "card", util * credit_limit, credit_limit)
    arrears_start = np.full(A, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    dpd_hist = np.zeros((A, config.N_MONTHS), dtype=int)

    snaps, pays, enqs, colls, apps = [], [], [], [], []
    pay_seq = enq_seq = coll_seq = app_seq = 0
    is_card = acc_product == "card"

    for m, (ms, me) in enumerate(zip(months, month_ends, strict=True)):
        me64 = np.datetime64(me, "ns")
        s = 0.8 * s + rng.normal(0, 0.55, C)
        risk_c = 0.9 * z + s + _macro(m)
        risk = risk_c[acc_cust]
        active = opened_at <= me64

        # -- payments due this month -------------------------------------------
        p_miss = _sigmoid(-5.0 + 1.0 * risk + 1.8 * (dpd > 0) + 1.5 * (dpd >= 90))
        missed = active & (rng.random(A) < p_miss)
        paid = active & ~missed
        reversed_ = paid & (rng.random(A) < _sigmoid(-4.6 + 0.9 * risk))
        true_miss = missed | reversed_

        amount_due = np.where(is_card, np.maximum(0.05 * balance, 200.0), instalment).round(2)
        card_mult = np.clip(np.exp(rng.normal(0.9 - 0.7 * risk, 0.4)), 1.0,
                            np.maximum(balance / np.maximum(amount_due, 1), 1.0))
        partial = rng.random(A) < _sigmoid(-2.5 + risk)
        loan_mult = np.where(partial, rng.uniform(0.5, 0.95, A), 1.0)
        amount_paid = (amount_due * np.where(is_card, card_mult, loan_mult)).round(2)

        pay_day = rng.integers(20, 29, A)
        pay_ts = (
            np.datetime64(ms, "ns")
            + (pay_day - 1).astype("timedelta64[D]")
            + rng.integers(6 * 60, 22 * 60, A).astype("timedelta64[m]")
        ).astype("datetime64[ns]")
        pay_load = _load_time(pay_ts + np.timedelta64(1, "D"))
        rev_load = _load_time(pay_ts + rng.integers(3, 11, A).astype("timedelta64[D]"))

        idx = np.flatnonzero(paid)
        ids = np.array([f"P{pay_seq + i:08d}" for i in range(len(idx))])
        pay_seq += len(idx)
        base = dict(
            payment_id=ids,
            account_id=accounts["account_id"].values[idx],
            customer_id=accounts["customer_id"].values[idx],
            paid_at=pay_ts[idx],
            amount_due=amount_due[idx],
            amount_paid=amount_paid[idx],
        )
        pays.append(pd.DataFrame({**base, "status": "cleared", "ingested_at": pay_load[idx],
                                  "record_version": 1}))
        r = reversed_[idx]
        pays.append(pd.DataFrame({k: v[r] for k, v in base.items()} | {
            "status": "reversed", "ingested_at": rev_load[idx][r], "record_version": 2}))

        # -- arrears ------------------------------------------------------------
        dpd_prev = dpd.copy()
        cure = (dpd_prev == 0) | (rng.random(A) < 0.6)
        dpd = np.where(true_miss, np.minimum(dpd_prev + 30, 180),
                       np.where(paid & cure, 0, dpd_prev))
        dpd = np.where(active, dpd, 0)
        dpd_hist[:, m] = dpd

        newly_in_arrears = (dpd_prev == 0) & (dpd > 0)
        arrears_start = np.where(newly_in_arrears, me64, arrears_start)
        to_collections = active & (dpd_prev == 30) & (dpd == 60)
        arrears_start = np.where(dpd == 0, np.datetime64("NaT", "ns"), arrears_start)

        cidx = np.flatnonzero(to_collections)
        if len(cidx):
            colls.append(pd.DataFrame({
                "referral_id": [f"R{coll_seq + i:07d}" for i in range(len(cidx))],
                "account_id": accounts["account_id"].values[cidx],
                "customer_id": accounts["customer_id"].values[cidx],
                # Effective date is back-dated to the start of the arrears.
                "effective_at": arrears_start[cidx],
                "referral_reason": "60_dpd",
                "ingested_at": _load_time(me64 + rng.integers(4, 16, len(cidx)).astype("timedelta64[D]")),
                "record_version": 1,
            }))
            coll_seq += len(cidx)

        # -- balances -----------------------------------------------------------
        target = _sigmoid(-0.5 + 0.9 * risk)
        util = np.clip(0.8 * util + 0.2 * target + rng.normal(0, 0.05, A), 0, 1.1)
        cleared = paid & ~reversed_
        balance = np.where(is_card, util * credit_limit,
                           np.maximum(balance - np.where(cleared, amount_paid * 0.8, 0), 0)).round(2)

        # -- month-end snapshot (v1) and restatement (v2) -------------------------
        batch_delay = 0 if rng.random() < 0.85 else int(rng.integers(1, 6))
        snap_load = _load_time(np.array([me64 + np.timedelta64(1 + batch_delay, "D")]))[0]
        late_rev = reversed_ & (rev_load > snap_load)
        dpd_v1 = np.where(late_rev, dpd_prev, dpd)  # reversal not yet processed
        aidx = np.flatnonzero(active)
        snap_base = dict(
            account_id=accounts["account_id"].values[aidx],
            customer_id=accounts["customer_id"].values[aidx],
            product=acc_product[aidx],
            snapshot_at=np.full(len(aidx), me64),
            balance=balance[aidx],
            credit_limit=credit_limit[aidx],
        )
        snaps.append(pd.DataFrame({**snap_base, "dpd": dpd_v1[aidx],
                                   "ingested_at": np.full(len(aidx), snap_load), "record_version": 1}))
        lr = late_rev[aidx]
        snaps.append(pd.DataFrame({k: v[lr] for k, v in snap_base.items()} | {
            "dpd": dpd[aidx][lr],
            "ingested_at": rev_load[aidx][lr] + np.timedelta64(1, "D"),
            "record_version": 2}))

        # -- bureau enquiries at other lenders -------------------------------------
        cust_active = onboarded_at.values <= me64
        n_enq = np.minimum(rng.poisson(np.exp(-2.4 + 0.75 * risk_c)), 6) * cust_active
        ec = np.repeat(np.arange(C), n_enq)
        if len(ec):
            days_in_month = me.day
            enq_ts = (np.datetime64(ms, "ns")
                      + rng.integers(0, days_in_month * 24 * 60, len(ec)).astype("timedelta64[m]"))
            micro = rng.random(len(ec)) < _sigmoid(-1.6 + 0.9 * risk_c[ec])
            lender = np.where(micro, "micro_lender",
                              np.where(rng.random(len(ec)) < 0.35, "retailer", "bank"))
            file_month = ms + pd.DateOffset(months=1)
            slip = rng.random(len(ec)) < 0.08
            file_day = np.where(slip, np.datetime64(file_month + pd.DateOffset(months=1), "ns"),
                                np.datetime64(file_month, "ns")) + np.timedelta64(11, "D")
            enqs.append(pd.DataFrame({
                "enquiry_id": [f"E{enq_seq + i:08d}" for i in range(len(ec))],
                "customer_id": customers["customer_id"].values[ec],
                "enquired_at": enq_ts,
                "lender_type": lender,
                "ingested_at": _load_time(file_day),
                "record_version": 1,
            }))
            enq_seq += len(ec)

        # -- credit applications at our bank ----------------------------------------
        if config.APPS_START <= ms <= config.APPS_END:
            eligible = (onboarded_at + pd.Timedelta(days=90)).values <= np.datetime64(ms, "ns")
            p_app = 1 - np.exp(-np.exp(-3.1 + 0.5 * risk_c))
            ac = np.flatnonzero(eligible & (rng.random(C) < p_app))
            if len(ac):
                dec_ts = (np.datetime64(ms, "ns")
                          + rng.integers(0, me.day, len(ac)).astype("timedelta64[D]")
                          + rng.integers(8 * 60, 20 * 60, len(ac)).astype("timedelta64[m]"))
                apps.append(pd.DataFrame({
                    "application_id": [f"A{app_seq + i:07d}" for i in range(len(ac))],
                    "customer_id": customers["customer_id"].values[ac],
                    "decision_time": dec_ts.astype("datetime64[ns]"),
                    "product": rng.choice(["personal_loan", "limit_increase"], len(ac)),
                    "requested_amount": (LOAN_PRINCIPAL_BY_BAND[income_band[ac] - 1]
                                         * rng.lognormal(-0.7, 0.5, len(ac))).round(-2),
                    "_month_idx": m,
                }))
                app_seq += len(ac)

    applications = pd.concat(apps, ignore_index=True)

    # ------------------------------------------------------------------- labels
    # bad = any account of the customer at 90+ DPD at any of the 12 month-ends
    # starting with the decision month (all of which fall after the decision).
    cust_index = pd.Series(np.arange(C), index=customers["customer_id"].values)
    worst = np.zeros((C, config.N_MONTHS), dtype=int)
    np.maximum.at(worst, acc_cust, dpd_hist)
    ci = cust_index[applications["customer_id"]].values
    mi = applications["_month_idx"].values
    k = config.PERFORMANCE_MONTHS
    window = np.stack([worst[ci, mi + j] for j in range(k)], axis=1)
    labels = pd.DataFrame({
        "application_id": applications["application_id"],
        "bad": (window.max(axis=1) >= config.BAD_DPD).astype(int),
        "label_matured_at": month_ends[mi + k - 1].values,
    })
    applications = applications.drop(columns="_month_idx")

    def _cat(frames: list[pd.DataFrame], key: list[str]) -> pd.DataFrame:
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values([*key, "record_version"], kind="stable").reset_index(drop=True)

    return GeneratedData(
        customers=customers,
        accounts=accounts,
        account_snapshots=_cat(snaps, ["account_id", "snapshot_at"]),
        payments=_cat(pays, ["payment_id"]),
        bureau_enquiries=_cat(enqs, ["enquiry_id"]),
        collections=_cat(colls, ["referral_id"]),
        applications=applications,
        labels=labels,
    )


def write(data: GeneratedData, out_dir: Path = config.RAW_DIR) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, df in data.tables().items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        counts[name] = len(df)
    return counts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate synthetic credit data")
    p.add_argument("--customers", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=config.RAW_DIR)
    a = p.parse_args(argv)
    counts = write(generate(a.customers, a.seed), a.out)
    for k, v in counts.items():
        print(f"{k:>18}: {v:>9,} rows")


if __name__ == "__main__":
    main()
