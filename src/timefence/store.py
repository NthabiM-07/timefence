"""Offline feature store: build point-in-time feature frames over a spine."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import config
from ._sql import lit_str, q
from .compiler import STRATEGIES, compile_features
from .registry import Registry, load_registry


class LabelMaturityError(ValueError):
    """A label's performance window had not closed by the training cut-off."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    """Order-sensitive content hash of a frame (used for reproducibility checks)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


@dataclass
class FeatureFrame:
    """Feature values plus the lineage needed to prove they were knowable."""

    values: pd.DataFrame
    lineage: pd.DataFrame
    features: tuple[str, ...]
    strategy: str
    manifest: dict = field(default_factory=dict)

    def audit(self) -> pd.DataFrame:
        """Per feature: how many rows used information unavailable at decision time.

        ``future_event_rows``: a contributing fact happened after the decision, or the
        feature was evaluated at a moment after the decision (so even a zero count
        describes the future).
        ``late_knowledge_rows``: a contributing fact happened before the decision
        but had not landed in the warehouse yet.
        """
        dt = self.lineage["decision_time"]
        evaluated_late = self.lineage["__as_of"] > dt  # the whole window sits after the decision
        rows = []
        for f in self.features:
            evt, ing = self.lineage[f"__evt__{f}"], self.lineage[f"__ing__{f}"]
            future = (evt > dt) | evaluated_late
            late = (ing > dt) & ~future
            lateness = (ing[late] - dt[late]).dt.total_seconds() / 86400
            rows.append({
                "feature": f,
                "rows": len(dt),
                "future_event_rows": int(future.sum()),
                "late_knowledge_rows": int(late.sum()),
                "contaminated_pct": round(100 * float((future | late).mean()), 2),
                "median_days_late": round(float(lateness.median()), 1) if late.any() else 0.0,
                "status": "PASS" if not (future | late).any() else "FAIL",
            })
        return pd.DataFrame(rows)

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.values.to_parquet(out_dir / "features.parquet", index=False)
        self.lineage.to_parquet(out_dir / "lineage.parquet", index=False)
        audit = self.audit()
        self.manifest["audit"] = {
            "status": "PASS" if (audit["status"] == "PASS").all() else "FAIL",
            "failing_features": audit.loc[audit["status"] == "FAIL", "feature"].tolist(),
        }
        (out_dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2, default=str))
        return out_dir


class FeatureStore:
    """Point-in-time feature builds over the registry's sources.

    ``in_memory=True`` (default) loads each source into DuckDB once, which keeps
    the production replay fast (it re-truncates every source per knowledge epoch).
    ``in_memory=False`` registers lazy Parquet views instead, for sources larger
    than RAM: builds take the same time here, but the replay is about 40% slower.
    """

    def __init__(self, registry_path: Path = config.REGISTRY_PATH, raw_dir: Path = config.RAW_DIR,
                 in_memory: bool = True):
        self.registry: Registry = load_registry(registry_path)
        self.raw_dir = Path(raw_dir)
        self.con = duckdb.connect()
        self._run("SET TimeZone = 'UTC'")  # timestamps are tz-naive SAST; never shift them
        for name, src in self.registry.sources.items():
            path = self.raw_dir / src.path
            if not path.exists():
                raise FileNotFoundError(f"Source '{name}' expects {path}. Run `timefence generate` first.")
            kind = "TABLE" if in_memory else "VIEW"
            self._run(f"CREATE OR REPLACE TEMP {kind} {q(f'src_{name}')} AS "
                      f"SELECT * FROM read_parquet({lit_str(path.as_posix())})")
        self.check_sources()

    def _run(self, sql: str) -> None:
        """Execute a statement and drain it (DuckDB 2 keeps unfetched results open)."""
        self.con.execute(sql).fetchall()

    # ------------------------------------------------------------------ contracts
    def check_sources(self) -> None:
        """Data contract: version keys unique, and no version lands before its fact."""
        for name, s in self.registry.sources.items():
            tbl = q(f"src_{name}")
            cols = [*s.record_key, s.version_col]
            dup = self.con.execute(
                f"SELECT count(*) FROM (SELECT {', '.join(map(q, cols))} FROM {tbl} "
                "GROUP BY ALL HAVING count(*) > 1)"
            ).fetchall()[0][0]
            if dup:
                raise ValueError(f"Source '{name}': {dup} duplicate ({', '.join(cols)}) keys; "
                                 "tie-breaking would be undefined")
            early = self.con.execute(
                f"SELECT count(*) FROM {tbl} WHERE {q(s.ingested_at)} < {q(s.event_time)}"
            ).fetchall()[0][0]
            if early:
                raise ValueError(f"Source '{name}': {early} rows ingested before they happened")

    # ------------------------------------------------------------------ building
    def _register_spine(self, spine: pd.DataFrame, strategy: str, as_of: pd.Timestamp | None,
                        row_id: str) -> None:
        ent = self.registry.entity
        missing = {row_id, ent, "decision_time"} - set(spine.columns)
        if missing:
            raise ValueError(f"Spine is missing columns {sorted(missing)}")
        if spine[row_id].duplicated().any():
            raise ValueError(f"Spine column '{row_id}' must be unique")
        sp = spine[[row_id, ent, "decision_time"]].rename(columns={row_id: "__row_id"}).copy()
        sp["decision_time"] = pd.to_datetime(sp["decision_time"]).astype("datetime64[ns]")
        if strategy == "naive":
            sp["__as_of"] = pd.Timestamp(as_of or config.EXTRACT_TIME)
        else:
            sp["__as_of"] = sp["decision_time"]
        sp["__as_of"] = sp["__as_of"].astype("datetime64[ns]")
        self.con.register("spine_df", sp)
        self._run("CREATE OR REPLACE TEMP TABLE spine AS SELECT * FROM spine_df")
        self.con.unregister("spine_df")

    def build(
        self,
        spine: pd.DataFrame,
        strategy: str = "bitemporal",
        features: list[str] | None = None,
        purpose: str = "credit_decisioning",
        row_id: str = "application_id",
        as_of: pd.Timestamp | None = None,
        table_for_source: dict[str, str] | None = None,
        with_manifest: bool = True,
    ) -> FeatureFrame:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        chosen = self.registry.select(features, purpose)
        q = compile_features(self.registry, chosen, strategy, table_for_source=table_for_source)
        self._register_spine(spine, strategy, as_of, row_id)
        out = self.con.execute(q.sql).df()
        out = out.rename(columns={"__row_id": row_id})
        names = list(q.features)
        values = out[[row_id, self.registry.entity, "decision_time", *names]].copy()
        lineage = out[[row_id, "decision_time", "__as_of"] + [c for c in out.columns if c.startswith(("__ing__", "__evt__"))
                                                   and c[7:] in names]].copy()
        manifest = self._manifest(spine, strategy, names, purpose, values, row_id) if with_manifest else {}
        return FeatureFrame(values=values, lineage=lineage, features=tuple(names), strategy=strategy,
                            manifest=manifest)

    def sql(self, strategy: str = "bitemporal", features: list[str] | None = None,
            purpose: str = "credit_decisioning") -> str:
        return compile_features(self.registry, self.registry.select(features, purpose), strategy).sql

    def _manifest(self, spine, strategy, names, purpose, values, row_id) -> dict:
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True,
                                    text=True, timeout=5).stdout.strip() or None
        except Exception:  # pragma: no cover - git is optional
            commit = None
        return {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "strategy": strategy,
            "purpose": purpose,
            "features": names,
            "registry_sha256": self.registry.sha256,
            "sources": {
                n: {"file": s.path, "sha256": _sha256_file(self.raw_dir / s.path),
                    "event_time": s.event_time, "ingested_at": s.ingested_at}
                for n, s in self.registry.sources.items()
            },
            "spine": {"rows": len(spine), "sha256": frame_hash(spine[[row_id, "decision_time"]]),
                      "min_decision_time": str(spine["decision_time"].min()),
                      "max_decision_time": str(spine["decision_time"].max())},
            "output_sha256": frame_hash(values),
            "code_commit": commit,
            "versions": {"python": platform.python_version(), "duckdb": duckdb.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__},
        }


def attach_labels(frame: FeatureFrame, labels: pd.DataFrame, training_cutoff: pd.Timestamp,
                  row_id: str = "application_id", drop_immature: bool = False) -> pd.DataFrame:
    """Join labels, refusing any whose performance window was still open at the cut-off."""
    df = frame.values.merge(labels, on=row_id, how="left", validate="one_to_one")
    if df["bad"].isna().any():
        raise ValueError(f"{int(df['bad'].isna().sum())} rows have no label")
    immature = df["label_matured_at"] > pd.Timestamp(training_cutoff)
    if immature.any():
        if not drop_immature:
            raise LabelMaturityError(
                f"{int(immature.sum())} labels mature after the training cut-off {training_cutoff}; "
                "pass drop_immature=True to exclude them"
            )
        df = df.loc[~immature].reset_index(drop=True)
    return df
