# ADR-0001: Bitemporal as-of semantics for every feature

**Status:** accepted

## Context

A training row for a credit decision must contain only what the decision system could have known at
`decision_time`. The common point-in-time join keeps source rows with `event_time <= decision_time` and
takes the latest version of each record. That answers "what was true before the decision?". The question
that matters is "what did we *know* at the decision?". The two diverge whenever data lands late or is
corrected after the fact. In retail credit that happens all the time: monthly bureau files, debit-order
reversals, arrears restatements and back-dated collections referrals.

## Decision

1. Every source declares a **valid-time** column (`event_time`), a **knowledge-time** column
   (`ingested_at`), a **record key** that identifies one real-world fact across its versions, and a
   **version column** for deterministic tie-breaks.
2. A feature evaluated at time `t` uses this sequence, in this order:
   1. keep versions with `ingested_at <= t` (inclusive: a load at 02:00 is known at 02:00);
   2. keep the newest known version per record key (`ORDER BY ingested_at DESC, version DESC`);
   3. apply the valid-time window `t - N days < event_time <= t`.
3. Every feature value carries the max `event_time` and max `ingested_at` of the rows behind it, so it
   can be audited after the fact without recomputing it.
4. The guarantee is tested against an independent mechanism, not just reasserted. The replay truncates
   each source to the rows that had landed by each load and runs the plain event-time query. The two
   must agree exactly.

## Consequences

- Sources without a trustworthy `ingested_at` cannot be onboarded until they have one. A load timestamp
  or a table snapshot id is enough. This is deliberate.
- Corrections that move a record's valid time are handled correctly, because the window is applied
  after the known version is chosen. Windowing first would silently use the stale version.
- Frames are a little slower to build than a valid-time-only join. Pre-filtering on valid time is unsafe
  unless a source promises that versions never change their valid time. That optimisation is left out on
  purpose.
- The naive and event-time strategies stay in the code base as compiled options. They exist so the
  audit, the experiment and the tests can show what the bitemporal rule prevents.
