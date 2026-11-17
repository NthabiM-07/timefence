"""Compile registry features into a single DuckDB query over an entity spine.

The spine is a table with one row per prediction moment:
``__row_id``, ``<entity>``, ``decision_time`` and ``__as_of``.

Three strategies are compiled from the same feature definitions. They differ
only in which clock they trust:

``naive``
    Every row is evaluated at one extraction timestamp (``__as_of`` is a
    constant). This is the "join the current customer table" mistake.

``event_time``
    Evaluated at ``decision_time`` using valid time only, and the latest version
    of each record. This is a textbook point-in-time join, and it still leaks:
    anything that happened before the decision but landed after it is used.

``bitemporal``
    Evaluated at ``decision_time`` using only record versions whose
    ``ingested_at <= decision_time``, then the newest of those per record key.
    This reconstructs exactly what the warehouse knew at the decision.

Order of operations matters and is fixed here: filter on knowledge time,
de-duplicate versions (deterministically), *then* apply the valid-time window.
Windowing before de-duplicating can resurrect a stale version whose valid time
was later corrected. Ties on ``ingested_at`` break on the version column, and
sums/averages accumulate in exact decimals, so a build is bit-for-bit
reproducible regardless of thread scheduling.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .registry import Feature, Registry, derived_layers

STRATEGIES = ("naive", "event_time", "bitemporal")


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class CompiledQuery:
    sql: str
    features: tuple[str, ...]
    strategy: str


def compile_features(
    registry: Registry,
    features: list[Feature],
    strategy: str,
    spine_table: str = "spine",
    table_for_source: dict[str, str] | None = None,
) -> CompiledQuery:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    feats = registry.closure(features)
    ent = _q(registry.entity)
    by_source: dict[str, list[Feature]] = defaultdict(list)
    for f in feats:
        if not f.is_derived:
            by_source[f.source].append(f)

    ctes = []
    joins = []
    for src_name, src_feats in by_source.items():
        src = registry.sources[src_name]
        tbl = (table_for_source or {}).get(src_name, f"src_{src_name}")
        knowledge = (
            f"AND r.{_q(src.ingested_at)} <= sp.__as_of" if strategy == "bitemporal" else ""
        )
        partition = ", ".join(["sp.__row_id"] + [f"r.{_q(k)}" for k in src.record_key])
        ctes.append(f"""
known_{src_name} AS (
    SELECT sp.__row_id, sp.__as_of, r.*
    FROM {spine_table} sp
    JOIN {tbl} r ON r.{ent} = sp.{ent} {knowledge}
    QUALIFY row_number() OVER (
        PARTITION BY {partition}
        ORDER BY r.{_q(src.ingested_at)} DESC, r.{_q(src.version_col)} DESC
    ) = 1
)""")
        evt, ing = _q(src.event_time), _q(src.ingested_at)
        cols = []
        for f in src_feats:
            cond = [f"{evt} <= __as_of"]
            if f.window_days:
                cond.append(f"{evt} > __as_of - INTERVAL {int(f.window_days)} DAY")
            if f.where:
                cond.append(f"({f.where})")
            flt = " AND ".join(cond)
            if f.agg == "count":
                value = f"count(*) FILTER (WHERE {flt})"
            elif f.agg in ("sum", "avg"):
                # Exact decimal accumulation: parallel float sums are order-dependent
                # in the last bit, which would break bit-for-bit reproducibility.
                value = (f"CAST({f.agg}(CAST({f.column} AS DECIMAL(38, 12))) "
                         f"FILTER (WHERE {flt}) AS DOUBLE)")
            else:
                value = f"{f.agg}({f.column}) FILTER (WHERE {flt})"
            cols.append(f"{value} AS {_q(f.name)}")
            cols.append(f"max({ing}) FILTER (WHERE {flt}) AS {_q('__ing__' + f.name)}")
            cols.append(f"max({evt}) FILTER (WHERE {flt}) AS {_q('__evt__' + f.name)}")
        ctes.append(f"""
agg_{src_name} AS (
    SELECT __row_id,
        {(','+chr(10)+'        ').join(cols)}
    FROM known_{src_name}
    GROUP BY __row_id
)""")
        joins.append(f"LEFT JOIN agg_{src_name} USING (__row_id)")

    base_cols = []
    for f in feats:
        if f.is_derived:
            continue
        val = _q(f.name)
        if f.default is not None:
            val = f"coalesce({val}, {f.default!r})"
        base_cols += [f"{val} AS {_q(f.name)}", _q("__ing__" + f.name), _q("__evt__" + f.name)]
    ctes.append(f"""
layer_0 AS (
    SELECT sp.__row_id, sp.{ent}, sp.decision_time, sp.__as_of,
        {(','+chr(10)+'        ').join(base_cols)}
    FROM {spine_table} sp
    {chr(10).join('    ' + j for j in joins)}
)""")

    for i, layer in enumerate(derived_layers(feats), start=1):
        cols = []
        for f in layer:
            deps_ing = ", ".join(_q("__ing__" + d) for d in f.depends_on)
            deps_evt = ", ".join(_q("__evt__" + d) for d in f.depends_on)
            cols += [f"({f.derived}) AS {_q(f.name)}",
                     f"greatest({deps_ing}) AS {_q('__ing__' + f.name)}",
                     f"greatest({deps_evt}) AS {_q('__evt__' + f.name)}"]
        ctes.append(f"""
layer_{i} AS (
    SELECT *, {', '.join(cols)}
    FROM layer_{i - 1}
)""")
    last = f"layer_{len(ctes) - 2 * len(by_source) - 1}"
    sql = "WITH " + ",".join(ctes) + f"\nSELECT * FROM {last} ORDER BY __row_id"
    return CompiledQuery(sql=sql, features=tuple(f.name for f in features), strategy=strategy)
