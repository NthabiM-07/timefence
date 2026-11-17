"""Production replay: what would the live system have served at each decision?

A production scorer does not do bitemporal filtering. It queries whatever its
tables hold at the moment of the decision, taking the latest version of each
record. So to know what production *actually* saw, we do not filter logically,
we truncate physically: for each decision, rebuild every source table from only
the rows that had landed by then, and run the plain event-time query against it.

Decisions are grouped into *knowledge epochs*: the interval between two loads.
Every decision inside an epoch saw an identical warehouse, so one truncation
serves them all. This is exact for any load schedule, not just nightly batches.

If ``bitemporal(full log, t) == event_time(log truncated at t, t)`` for every
row, the offline store is provably free of knowledge leakage for that spine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .store import FeatureFrame, FeatureStore


def knowledge_epochs(store: FeatureStore, decision_times: pd.Series) -> pd.Series:
    """For each decision, the timestamp of the last load at or before it."""
    parts = [
        f'SELECT DISTINCT "{s.ingested_at}" AS t FROM src_{n}'
        for n, s in store.registry.sources.items()
    ]
    loads = store.con.execute(" UNION ".join(parts) + " ORDER BY t").df()["t"].values
    loads = loads.astype("datetime64[ns]")
    dt = pd.to_datetime(decision_times).values.astype("datetime64[ns]")
    pos = np.searchsorted(loads, dt, side="right") - 1
    epoch = np.where(pos >= 0, loads[np.maximum(pos, 0)], np.datetime64("NaT", "ns"))
    return pd.Series(epoch, index=decision_times.index, name="epoch")


def replay(store: FeatureStore, spine: pd.DataFrame, features: list[str] | None = None,
           purpose: str = "credit_decisioning", row_id: str = "application_id",
           progress: bool = False) -> FeatureFrame:
    sources = store.registry.sources
    ent = store.registry.entity
    tables = {n: f"trunc_{n}" for n in sources}
    spine = spine.copy()
    spine["__epoch"] = knowledge_epochs(store, spine["decision_time"])
    if spine["__epoch"].isna().any():
        raise ValueError("Some decisions precede the first load; nothing was known yet")

    frames, lineages = [], []
    groups = list(spine.groupby("__epoch", sort=True))
    for i, (epoch, grp) in enumerate(groups):
        ts = pd.Timestamp(epoch).isoformat(sep=" ")
        # Rebuild each table as production held it at this epoch, for the customers
        # being scored (other customers cannot affect their features).
        store.con.register("epoch_entities", grp[[ent]].drop_duplicates())
        for n, s in sources.items():
            store._run(
                f"CREATE OR REPLACE TEMP TABLE {tables[n]} AS "
                f"SELECT r.* FROM src_{n} r SEMI JOIN epoch_entities e ON r.\"{ent}\" = e.\"{ent}\" "
                f"WHERE r.\"{s.ingested_at}\" <= TIMESTAMP '{ts}'"
            )
        store.con.unregister("epoch_entities")
        fr = store.build(grp.drop(columns="__epoch"), strategy="event_time", features=features,
                         purpose=purpose, row_id=row_id, table_for_source=tables, with_manifest=False)
        frames.append(fr.values)
        lineages.append(fr.lineage)
        if progress and (i + 1) % 50 == 0:
            print(f"  replayed {i + 1}/{len(groups)} knowledge epochs")
    for n in sources:
        store._run(f"DROP TABLE IF EXISTS {tables[n]}")

    values = pd.concat(frames, ignore_index=True).sort_values(row_id).reset_index(drop=True)
    lineage = pd.concat(lineages, ignore_index=True).sort_values(row_id).reset_index(drop=True)
    return FeatureFrame(values=values, lineage=lineage, features=fr.features, strategy="replay",
                        manifest={"strategy": "replay", "knowledge_epochs": len(groups)})


def parity(a: pd.DataFrame, b: pd.DataFrame, features: list[str], row_id: str = "application_id",
           rtol: float = 0.0) -> pd.DataFrame:
    """Per feature: share of rows where two frames disagree (NaN == NaN counts as equal).

    Exact by default: builds are deterministic, so any difference is a real one.
    """
    m = a[[row_id, *features]].merge(b[[row_id, *features]], on=row_id, suffixes=("_a", "_b"),
                                     validate="one_to_one")
    rows = []
    for f in features:
        x, y = m[f"{f}_a"], m[f"{f}_b"]
        both_nan = x.isna() & y.isna()
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            same = both_nan | np.isclose(x.astype(float), y.astype(float), rtol=rtol, atol=0)
        else:
            same = both_nan | (x == y)
        rows.append({"feature": f, "rows": len(m), "mismatched_rows": int((~same).sum()),
                     "mismatch_pct": round(100 * float((~same).mean()), 2)})
    return pd.DataFrame(rows)
