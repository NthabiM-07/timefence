# ADR-0003: Exact decimal accumulation for sums and averages

**Status:** accepted

## Context

`test_builds_are_reproducible_and_manifested` builds the same frame twice and compares content hashes.
It failed. `card_utilisation_3m` and `payment_to_due_ratio_3m` differed in a few hundred rows by up to
4e-15. DuckDB aggregates in parallel, and floating-point addition is not associative. The summation
order depends on thread scheduling, so the last bit of a sum can change from run to run.

No model would notice. But the manifest promises that a training set can be rebuilt and verified by
hash. With these differences, any rebuild would "fail" verification for no real reason, and auditors
would learn to ignore the check.

## Decision

`sum` and `avg` features accumulate in `DECIMAL(38, 12)` and cast the result back to `DOUBLE`. Integer
decimal addition is associative, so the result does not depend on order. Inputs are rounded to 12
decimal places before summing, which is far below anything a credit feature can resolve.

The replay parity check now compares exactly (`rtol = 0`). Any difference is real.

## Consequences

- Four identical builds of 20,000 rows produce one hash.
- Decimal aggregation is slightly slower than float aggregation. That cost does not show at this scale.
- `max`, `min` and `count` are order-independent already and are unchanged.
