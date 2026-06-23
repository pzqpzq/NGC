#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ngc_v1.common import RANDOM_SEED, V0_ROOT, V1_ROOT, ensure_dir, paired_bootstrap, read_observables, setup_matplotlib, write_json


def select_topk(g: pd.DataFrame, k: int) -> pd.DataFrame:
    return g.sort_values("law_score", ascending=False, kind="mergesort").head(min(k, len(g)))


def filter_for_knockout(g: pd.DataFrame, knockout: str) -> pd.DataFrame:
    if knockout == "full_law":
        return g

    if knockout == "drop_evidence_broker":
        threshold = g["early_kv_mass"].quantile(0.60)
        sub = g[g["early_kv_mass"] < threshold]
    elif knockout == "drop_mediator":
        threshold = g["mediator_score"].quantile(0.60)
        sub = g[g["mediator_score"] < threshold]
    elif knockout == "drop_settlement":
        threshold = g["settlement_score"].quantile(0.60)
        sub = g[g["settlement_score"] < threshold]
    elif knockout == "drop_veto":
        threshold = g["stability_guard"].quantile(0.40)
        sub = g[g["stability_guard"] <= threshold]
    elif knockout == "drop_reconstruction_guard":
        threshold = g["reconstruction_error_log10"].quantile(0.60)
        sub = g[g["reconstruction_error_log10"] >= threshold]
    else:
        raise ValueError(f"Unknown knockout: {knockout}")

    if len(sub) == 0:
        return g
    return sub


def evaluate(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    knockouts = [
        "full_law",
        "drop_evidence_broker",
        "drop_mediator",
        "drop_settlement",
        "drop_veto",
        "drop_reconstruction_guard",
    ]
    rows: List[Dict[str, object]] = []
    selected_rows = []
    for query_key, g in df.groupby("query_key", sort=False):
        for knockout in knockouts:
            pool = filter_for_knockout(g, knockout)
            top = select_topk(pool, k)
            first = top.iloc[0]
            rows.append(
                {
                    "query_key": query_key,
                    "model": first["model"],
                    "dataset": first["dataset"],
                    "knockout": knockout,
                    "pass1": float(first["label"]),
                    "passk": float(top["label"].max()),
                    "mean_at_k": float(top["label"].mean()),
                    "pool_size": int(len(pool)),
                    "topk_size": int(len(top)),
                    "early_kv_mass": float(top["early_kv_mass"].median()),
                    "mediator_score": float(top["mediator_score"].median()),
                    "settlement_score": float(top["settlement_score"].median()),
                    "stability_guard": float(top["stability_guard"].median()),
                    "reconstruction_error_log10": float(top["reconstruction_error_log10"].median()),
                    "departure_index": float(top["departure_index"].median()),
                }
            )
            selected_rows.append(top.assign(query_key=query_key, knockout=knockout))
    return pd.DataFrame(rows), pd.concat(selected_rows, ignore_index=True)


def summarize(per_query: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        per_query.groupby("knockout", as_index=False)
        .agg(
            n_queries=("query_key", "nunique"),
            pass1=("pass1", "mean"),
            pass5=("passk", "mean"),
            mean_at_5=("mean_at_k", "mean"),
            pool_size_median=("pool_size", "median"),
            early_kv_mass=("early_kv_mass", "median"),
            mediator_score=("mediator_score", "median"),
            settlement_score=("settlement_score", "median"),
            stability_guard=("stability_guard", "median"),
        )
        .sort_values("pass5", ascending=False)
    )
    full = per_query[per_query["knockout"] == "full_law"].set_index("query_key")
    boot_rows = []
    for knockout, g in per_query.groupby("knockout", sort=False):
        if knockout == "full_law":
            continue
        aligned = g.set_index("query_key").reindex(full.index)
        for metric in ["pass1", "passk", "mean_at_k"]:
            stat = paired_bootstrap(full[metric], aligned[metric], n_boot=n_boot, seed=seed)
            stat.update({"contrast": f"full_law_minus_{knockout}", "metric": metric})
            boot_rows.append(stat)
    return summary, pd.DataFrame(boot_rows)


def make_figure(summary: pd.DataFrame, boot: pd.DataFrame, out_dir: Path) -> None:
    plt = setup_matplotlib()
    plot = boot[boot["metric"] == "passk"].copy()
    plot["knockout"] = plot["contrast"].str.replace("full_law_minus_", "", regex=False)
    plot = plot.sort_values("diff")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y = np.arange(len(plot))
    ax.barh(y, plot["diff"], color="#59A14F")
    ax.errorbar(
        plot["diff"],
        y,
        xerr=[plot["diff"] - plot["ci_low"], plot["ci_high"] - plot["diff"]],
        fmt="none",
        ecolor="black",
        capsize=3,
        lw=1.0,
    )
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["knockout"])
    ax.set_xlabel("Full C316 pass@5 minus role-knockout pass@5")
    ax.set_title("Cached role knockout effects")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_role_knockout_effects.pdf")
    fig.savefig(out_dir / "fig_role_knockout_effects.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", type=Path, default=V0_ROOT)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    out_dir = ensure_dir(args.v1_root / "outputs/06_role_knockout")
    df = read_observables(args.v1_root, args.v0_root)
    per_query, selected = evaluate(df, args.top_k)
    summary, boot = summarize(per_query, args.n_bootstrap, args.seed)

    per_query.to_csv(out_dir / "role_knockout_per_query.csv", index=False)
    selected.to_csv(out_dir / "role_knockout_selected_candidates.csv.gz", index=False, compression="gzip")
    summary.to_csv(out_dir / "role_knockout_summary.csv", index=False)
    boot.to_csv(out_dir / "role_knockout_bootstrap.csv", index=False)
    make_figure(summary, boot, out_dir)
    write_json(
        out_dir / "manifest.json",
        {
            "rows_in": int(len(df)),
            "queries": int(df["query_key"].nunique()),
            "top_k": args.top_k,
            "n_bootstrap": args.n_bootstrap,
        },
    )

    print("Role knockout cached analysis complete")
    print(summary.to_string(index=False))
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
