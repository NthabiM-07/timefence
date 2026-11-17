"""timefence command line.

    timefence generate [--customers N] [--seed S]
    timefence build --strategy bitemporal [--out data/frames/bitemporal]
    timefence audit data/frames/event_time
    timefence sql --strategy bitemporal
    timefence experiment
    timefence materialise --as-of "2025-12-31 23:00"
    timefence lookup C000123
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__, config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="timefence", description="Point-in-time correct credit feature store")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate synthetic credit data")
    g.add_argument("--customers", type=int, default=10_000)
    g.add_argument("--seed", type=int, default=7)

    b = sub.add_parser("build", help="Build a feature frame for all applications")
    b.add_argument("--strategy", choices=["naive", "event_time", "bitemporal"], default="bitemporal")
    b.add_argument("--purpose", default="credit_decisioning")
    b.add_argument("--out", type=Path)

    a = sub.add_parser("audit", help="Audit a saved frame for leakage (exit 1 on failure)")
    a.add_argument("frame_dir", type=Path)

    s = sub.add_parser("sql", help="Print the compiled SQL")
    s.add_argument("--strategy", choices=["naive", "event_time", "bitemporal"], default="bitemporal")

    sub.add_parser("experiment", help="Run the reported-vs-realised leakage experiment")

    m = sub.add_parser("materialise", help="Materialise features to the online store")
    m.add_argument("--as-of", required=True)

    lk = sub.add_parser("lookup", help="Look up one customer in the online store")
    lk.add_argument("customer_id")

    args = p.parse_args(argv)

    if args.cmd == "generate":
        from .generate import generate, write
        for k, v in write(generate(args.customers, args.seed)).items():
            print(f"{k:>18}: {v:>9,} rows")
        return 0

    from .store import FeatureFrame, FeatureStore

    if args.cmd == "build":
        fs = FeatureStore()
        apps = pd.read_parquet(fs.raw_dir / "applications.parquet")
        fr = fs.build(apps, args.strategy, purpose=args.purpose)
        out = fr.save(args.out or config.FRAMES_DIR / args.strategy)
        print(f"Wrote {len(fr.values):,} rows x {len(fr.features)} features to {out} "
              f"(audit {fr.manifest['audit']['status']})")
        return 0

    if args.cmd == "audit":
        d = args.frame_dir
        manifest = json.loads((d / "manifest.json").read_text())
        lineage = pd.read_parquet(d / "lineage.parquet")
        fr = FeatureFrame(values=pd.DataFrame(), lineage=lineage, features=tuple(manifest["features"]),
                          strategy=manifest["strategy"])
        report = fr.audit()
        print(report.to_string(index=False))
        failed = (report["status"] == "FAIL").any()
        print("\nAUDIT", "FAILED" if failed else "PASSED", f"({manifest['strategy']})")
        return 1 if failed else 0

    if args.cmd == "sql":
        print(FeatureStore().sql(args.strategy))
        return 0

    if args.cmd == "experiment":
        from .experiment import run
        run()
        return 0

    if args.cmd == "materialise":
        from .online import materialise
        n = materialise(FeatureStore(), pd.Timestamp(args.as_of))
        print(f"Materialised {n:,} customers as of {args.as_of} to {config.ONLINE_DB}")
        return 0

    if args.cmd == "lookup":
        from .online import lookup
        print(json.dumps(lookup(args.customer_id), indent=2, default=str))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
