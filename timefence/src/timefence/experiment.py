"""The leakage experiment: same model, same applications, three ways of joining.

For each strategy we do what a model developer would do: build features for
2022-2023 applications, train, then validate out-of-time on 2024 applications
with features built *the same way*. That validation number goes in the model
document ("reported").

Then we score the same 2024 applications with the features production would
really have served, from the physical replay ("realised"). The gap between the
two is the leak, expressed in the currency a credit committee cares about.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from . import config
from .replay import parity, replay
from .store import FeatureStore, attach_labels

STRATEGY_LABELS = {
    "naive": "Naive (current-state join)",
    "event_time": "Event-time PIT join",
    "bitemporal": "Bitemporal PIT join (timefence)",
}
APPROVAL_RATE = 0.80
SEED = 20260101

# Chart tokens (reference palette, light mode).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
REPORTED, REALISED = "#2a78d6", "#eb6834"


def gini(y: np.ndarray, p: np.ndarray) -> float:
    return 2 * roc_auc_score(y, p) - 1


def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=False, random_state=0,
    )


def _approved_bad_rate(y: np.ndarray, p: np.ndarray, cutoff: float) -> float:
    approved = p <= cutoff
    return float(y[approved].mean())


def _paired_bootstrap(y, p_rep, p_real, n=500, seed=SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(gini(y[i], p_rep[i]) - gini(y[i], p_real[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def run(out_dir: Path = config.REPORTS_DIR, frames_dir: Path = config.FRAMES_DIR,
        store: FeatureStore | None = None, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    fs = store or FeatureStore()
    apps = pd.read_parquet(fs.raw_dir / "applications.parquet")
    labels = pd.read_parquet(fs.raw_dir / "labels.parquet")

    # Policy exclusion: customers already in default at the decision are not scored.
    # The exclusion itself must be point-in-time, or it leaks too.
    known = fs.build(apps, "bitemporal", features=["worst_dpd_1m"], with_manifest=False).values
    excluded = known["worst_dpd_1m"].fillna(0) >= config.BAD_DPD
    spine = apps.loc[~excluded.values].reset_index(drop=True)
    is_oot = spine["decision_time"] >= config.OOT_START
    is_train = spine["decision_time"] <= config.TRAIN_END
    log(f"Spine: {len(spine):,} applications ({int(excluded.sum()):,} excluded as already in default); "
        f"train {int(is_train.sum()):,}, out-of-time {int(is_oot.sum()):,}")

    frames = {}
    for s in STRATEGY_LABELS:
        frames[s] = fs.build(spine, s)
        frames[s].save(frames_dir / s)
        log(f"Built {s:<10} frame: audit {frames[s].manifest['audit']['status']}")
    features = list(frames["bitemporal"].features)

    log("Replaying production for the out-of-time year (physical truncation per knowledge epoch)...")
    oot_spine = spine.loc[is_oot]
    prod = replay(fs, oot_spine, progress=verbose)
    oot_ids = set(oot_spine["application_id"])

    def oot(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["application_id"].isin(oot_ids)].sort_values("application_id").reset_index(drop=True)

    bt_parity = parity(oot(frames["bitemporal"].values), prod.values, features)
    if bt_parity["mismatched_rows"].sum():
        raise AssertionError("Bitemporal frame differs from production replay:\n" + bt_parity.to_string())
    log(f"Parity: bitemporal == production replay on all {len(prod.values):,} rows x {len(features)} features")

    y_lab = labels.set_index("application_id")
    prod_vals = prod.values.sort_values("application_id").reset_index(drop=True)
    y_oot = y_lab.loc[prod_vals["application_id"], "bad"].values

    results, audits, drifts, importances = {}, {}, {}, {}
    for s, fr in frames.items():
        train = attach_labels(fr, labels, training_cutoff=config.DATA_END)
        train = train[train["application_id"].isin(set(spine.loc[is_train, "application_id"]))]
        model = _model().fit(train[features], train["bad"])

        rep = oot(fr.values)
        assert (rep["application_id"].values == prod_vals["application_id"].values).all()
        p_rep = model.predict_proba(rep[features])[:, 1]
        p_real = model.predict_proba(prod_vals[features])[:, 1]
        cutoff = float(np.quantile(p_rep, APPROVAL_RATE))
        lo, hi = _paired_bootstrap(y_oot, p_rep, p_real)
        results[s] = {
            "label": STRATEGY_LABELS[s],
            "train_rows": len(train),
            "oot_rows": len(rep),
            "reported_gini": round(gini(y_oot, p_rep), 4),
            "realised_gini": round(gini(y_oot, p_real), 4),
            "gini_gap_ci95": [round(lo, 4), round(hi, 4)],
            "promised_bad_rate_at_80pct_approval": round(_approved_bad_rate(y_oot, p_rep, cutoff), 4),
            "realised_bad_rate_at_same_cutoff": round(_approved_bad_rate(y_oot, p_real, cutoff), 4),
            "realised_approval_rate_at_same_cutoff": round(float((p_real <= cutoff).mean()), 4),
            "mean_predicted_pd_reported": round(float(p_rep.mean()), 4),
            "mean_predicted_pd_realised": round(float(p_real.mean()), 4),
            "oot_bad_rate": round(float(y_oot.mean()), 4),
            "audit_status": fr.manifest["audit"]["status"],
        }
        results[s]["gini_gap"] = round(results[s]["reported_gini"] - results[s]["realised_gini"], 4)
        audits[s] = fr.audit()
        drifts[s] = parity(rep, prod_vals, features)
        pi = permutation_importance(model, rep[features], y_oot, scoring="roc_auc", n_repeats=5,
                                    random_state=0)
        importances[s] = (pd.Series(pi.importances_mean, index=features).sort_values(ascending=False)
                          .head(5).round(4))
        r = results[s]
        log(f"{s:<10} reported Gini {r['reported_gini']:.3f} | realised {r['realised_gini']:.3f} | "
            f"gap {r['gini_gap']:+.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "spine_rows": len(spine), "excluded_in_default": int(excluded.sum()),
        "train_rows": int(is_train.sum()), "oot_rows": int(is_oot.sum()),
        "knowledge_epochs_replayed": prod.manifest["knowledge_epochs"],
        "parity_bitemporal_vs_replay_mismatches": int(bt_parity["mismatched_rows"].sum()),
        "features": features, "results": results,
        "top_features": {s: v.to_dict() for s, v in importances.items()},
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    _chart_gini(results, out_dir / "gini_reported_vs_realised.png")
    _chart_contamination(audits["event_time"], out_dir / "event_time_contamination.png")
    _write_report(summary, audits, drifts, importances, out_dir / "leakage_report.md")
    log(f"Wrote {out_dir / 'leakage_report.md'}")
    return summary


# --------------------------------------------------------------------- charts
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _chart_gini(results: dict, path: Path) -> None:
    order = ["bitemporal", "event_time", "naive"]
    fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=200, facecolor=SURFACE)
    _style(ax)
    for i, s in enumerate(order):
        r = results[s]
        rep, real = 100 * r["reported_gini"], 100 * r["realised_gini"]
        ax.plot([real, rep], [i, i], color=GRID, linewidth=6, solid_capstyle="round", zorder=1)
        # Reported is a ring, realised a dot: when they coincide both stay visible.
        ax.scatter([rep], [i], s=210, facecolor=SURFACE, edgecolor=REPORTED, linewidth=2.5, zorder=2,
                   label="Reported (out-of-time validation)" if i == 0 else None)
        ax.scatter([real], [i], s=80, color=REALISED, zorder=3,
                   label="Realised (production replay)" if i == 0 else None)
        gap = rep - real
        note = "reported = realised" if abs(gap) < 0.05 else f"{gap:.1f} pts overstated"
        ax.annotate(f"{rep:.1f} reported, {real:.1f} realised:  {note}", (max(rep, real), i),
                    xytext=(14, 0), textcoords="offset points", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(order)), [STRATEGY_LABELS[s] for s in order], color=INK, fontsize=10.5)
    lo = min(100 * results[s]["realised_gini"] for s in order)
    hi = max(100 * results[s]["reported_gini"] for s in order)
    ax.set_xlim(np.floor(lo / 5) * 5 - 2, hi + 30)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Gini on 2024 applications (%)", color=INK_2, fontsize=10)
    fig.suptitle("Leaky joins overstate model power. The bitemporal join reports what production delivers.",
                 x=0.012, ha="left", color=INK, fontsize=11.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9.5,
              labelcolor=INK_2, handletextpad=0.3, columnspacing=1.5, borderaxespad=0.3)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _chart_contamination(audit: pd.DataFrame, path: Path) -> None:
    a = audit[audit["contaminated_pct"] > 0].sort_values("contaminated_pct")
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(a) + 1.3), dpi=200, facecolor=SURFACE)
    _style(ax)
    ax.barh(a["feature"], a["contaminated_pct"], color=REPORTED, height=0.62)
    for y, (v, d) in enumerate(zip(a["contaminated_pct"], a["median_days_late"])):
        ax.annotate(f"{v:.1f}%  · {d:.1f} days late", (v, y), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, a["contaminated_pct"].max() * 1.35)
    ax.tick_params(axis="y", labelcolor=INK, labelsize=9.5)
    ax.set_xlabel("Rows that used data not yet landed at the decision (%).  "
                  "Label: median days from decision to arrival.", color=INK_2, fontsize=9.5)
    fig.suptitle("Event-time PIT join: share of rows built from data that had not landed yet",
                 x=0.012, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------- report
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _write_report(summary, audits, drifts, importances, path: Path) -> None:
    res = summary["results"]
    headline = pd.DataFrame([{
        "Strategy": r["label"],
        "Reported Gini": f"{r['reported_gini']:.3f}",
        "Realised Gini": f"{r['realised_gini']:.3f}",
        "Gap (95% CI)": f"{r['gini_gap']:+.3f} ({r['gini_gap_ci95'][0]:+.3f} to {r['gini_gap_ci95'][1]:+.3f})",
        "Promised bad rate @80% approval": f"{100 * r['promised_bad_rate_at_80pct_approval']:.1f}%",
        "Realised bad rate, same cut-off": f"{100 * r['realised_bad_rate_at_same_cutoff']:.1f}%",
        "Audit": r["audit_status"],
    } for r in res.values()])
    parts = [
        "# Leakage report",
        "",
        "Generated by `timefence experiment`. Synthetic data; numbers change with the seed.",
        "",
        f"- Applications scored: **{summary['spine_rows']:,}** "
        f"({summary['excluded_in_default']:,} excluded as already in default, using point-in-time data)",
        f"- Train: decisions in 2022-2023 ({summary['train_rows']:,}). "
        f"Out-of-time test: decisions in 2024 ({summary['oot_rows']:,}).",
        f"- Production replay: {summary['knowledge_epochs_replayed']} knowledge epochs, each a physical "
        "truncation of every source table to the rows that had landed.",
        f"- Bitemporal vs replay mismatches: **{summary['parity_bitemporal_vs_replay_mismatches']}** "
        f"(across {summary['oot_rows']:,} rows x {len(summary['features'])} features)",
        "",
        "## Reported vs realised",
        "",
        _md_table(headline),
        "",
        "![Reported vs realised Gini](gini_reported_vs_realised.png)",
        "",
        "## Leakage audit: event-time PIT join (all applications)",
        "",
        _md_table(audits["event_time"][["feature", "late_knowledge_rows", "contaminated_pct",
                                         "median_days_late", "status"]]),
        "",
        "![Event-time contamination](event_time_contamination.png)",
        "",
        "## Leakage audit: naive join (all applications)",
        "",
        _md_table(audits["naive"][["feature", "future_event_rows", "contaminated_pct", "status"]]),
        "",
        "## Leakage audit: bitemporal join (all applications)",
        "",
        _md_table(audits["bitemporal"][["feature", "future_event_rows", "late_knowledge_rows", "status"]]),
        "",
        "## Where the served value differed from the validated value (2024 applications)",
        "",
        _md_table(pd.DataFrame({
            "feature": drifts["bitemporal"]["feature"],
            "naive": drifts["naive"]["mismatch_pct"].map(lambda v: f"{v:.1f}%"),
            "event_time": drifts["event_time"]["mismatch_pct"].map(lambda v: f"{v:.1f}%"),
            "bitemporal": drifts["bitemporal"]["mismatch_pct"].map(lambda v: f"{v:.1f}%"),
        })),
        "",
        "## What each model leaned on (permutation importance, top 5, out-of-time)",
        "",
    ]
    for s, imp in importances.items():
        parts.append(f"**{STRATEGY_LABELS[s]}:** " + ", ".join(f"`{k}` ({v:.3f})" for k, v in imp.items()))
        parts.append("")
    path.write_text("\n".join(parts))
