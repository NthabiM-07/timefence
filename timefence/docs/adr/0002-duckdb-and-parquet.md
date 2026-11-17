# ADR-0002: DuckDB over Parquet for the reference implementation

**Status:** accepted

## Context

The project has to run anywhere with `pip install`, in CI and on a laptop, with no cluster, credentials
or services. It also has to be credible as a design for a bank-scale lakehouse.

## Decision

Use DuckDB as the engine and Parquet files as the storage format. The compiler emits standard SQL with
a few constructs that most lakehouse engines also support: `QUALIFY`, `FILTER (WHERE ...)` on aggregates
and interval arithmetic.

## Mapping to a production lakehouse

| Here | At scale |
|---|---|
| `ingested_at` column on each row | The same column, written by the ingestion job. Alternatively, an Iceberg/Delta snapshot or commit timestamp per load. |
| Replay truncates tables to `ingested_at <= epoch` | Time travel to the snapshot or commit current at the epoch (for example `TIMESTAMP AS OF` on an Iceberg table in Spark SQL, or a pinned Nessie commit) |
| Knowledge epochs from distinct load times | Snapshot or commit history of each source table |
| DuckDB SQL | Spark SQL or Dremio. `QUALIFY` may need rewriting as a subquery on engines without it. |
| SQLite online store | A key-value store (Redis, DynamoDB or similar) materialised from the same compiled query |

Row-level `ingested_at` is the more portable option. It survives table compaction and snapshot expiry,
and copying data between systems keeps it. Snapshot time travel is a good *independent* check of it, in
the same way that the replay here checks the bitemporal filter.

## Consequences

- A full experiment (20,000 applications, 1.7 million source rows, 306 replayed knowledge epochs) runs in
  about a minute on one machine.
- Sources are loaded into memory once per store. Past a few tens of millions of rows, swap the
  `CREATE TEMP TABLE` for a view over Parquet, or move to a distributed engine. The SQL does not change.
