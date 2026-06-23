#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ngc_v1.common import RANDOM_SEED, V0_ROOT, V1_ROOT, ensure_dir, paired_bootstrap, read_observables, setup_matplotlib, write_json


MATCH_FEATURES = ["tcr", "reconstruction_error_log10"]
ROLE_FEATURES = ["early_kv_mass", "late_q_mass", "late_mlp_mass", "role_entropy", "settlement_score"]


def scale_distance(pool: pd.DataFrame, selected: pd.Series) -> pd.Series:
    dist = pd.Series(np.zeros(len(pool)), index=pool.index, dtype=float)
    for col in MATCH_FEATURES:
        scale = pool[col].quantile(0.75) - pool[col].quantile(0.25)
        if not np.isfinite(scale) or scale < 1e-9:
            scale = pool[col].std()
        if not np.isfinite(scale) or scale < 1e-9:
            scale = 1.0
        dist += (pool[col] - selected[col]).abs() / scale
    return dist


def choose_nearest(pool: pd.DataFrame, selected: pd.Series, seed_value: int) -> Optional[pd.Series]:
    if pool.empty:
        return None
    tmp = pool.copy()
    tmp["_match_distance"] = scale_distance(tmp, selected)
    tmp["_tie"] = np.sin(np.arange(len(tmp)) + seed_value)
    return tmp.sort_values(["_match_distance", "_tie"], kind="mergesort").iloc[0]


def choose_control(g: pd.DataFrame, selected: pd.Series, control_type: str, seed_value: int) -> Optional[pd.Series]:
    pool = g[g["candidate_index"] != selected["candidate_index"]].copy()
    if pool.empty:
        return None

    q25 = g["departure_index"].quantile(0.25)
    q50 = g["departure_index"].quantile(0.50)
    q75 = g["departure_index"].quantile(0.75)

    if control_type == "reconstruction_tcr_matched":
        return choose_nearest(pool, selected, seed_value)

    if control_type == "too_conservative":
        sub = pool[pool["departure_index"] <= min(selected["departure_index"], q50)]
        if sub.empty:
            sub = pool[pool["departure_index"] <= q25]
        if sub.empty:
            sub = pool.sort_values("departure_index", ascending=True).head(max(1, min(8, len(pool))))
        return choose_nearest(sub, selected, seed_value)

    if control_type == "too_disruptive":
        sub = pool[pool["departure_index"] >= max(selected["departure_index"], q50)]
        if sub.empty:
            sub = pool[pool["departure_index"] >= q75]
        if sub.empty:
            sub = pool.sort_values("departure_index", ascending=False).head(max(1, min(8, len(pool))))
        return choose_nearest(sub, selected, seed_value)

    if control_type == "role_shuffled":
        tmp = pool.copy()
        tmp["_match_distance"] = scale_distance(tmp, selected)
        role_contrast = pd.Series(np.zeros(len(tmp)), index=tmp.index, dtype=float)
        for col in ROLE_FEATURES:
            scale = tmp[col].quantile(0.75) - tmp[col].quantile(0.25)
            if not np.isfinite(scale) or scale < 1e-9:
                scale = tmp[col].std()
            if not np.isfinite(scale) or scale < 1e-9:
                scale = 1.0
            role_contrast += (tmp[col] - selected[col]).abs() / scale
        tmp["_role_contrast"] = role_contrast
        match_cut = tmp["_match_distance"].quantile(0.35)
        sub = tmp[tmp["_match_distance"] <= match_cut].copy()
        if sub.empty:
            sub = tmp
        return sub.sort_values(["_role_contrast", "_match_distance"], ascending=[False, True], kind="mergesort").iloc[0]

    raise ValueError(f"Unknown control type: {control_type}")


def build_pairs(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    control_types = [
        "reconstruction_tcr_matched",
        "too_conservative",
        "too_disruptive",
        "role_shuffled",
    ]
    for i, (query_key, g) in enumerate(df.groupby("query_key", sort=False)):
        selected = g.sort_values("law_score", ascending=False, kind="mergesort").iloc[0]
        for control_type in control_types:
            control = choose_control(g, selected, control_type, seed + i)
            if control is None:
                continue
            row = {
                "query_key": query_key,
                "model": selected["model"],
                "dataset": selected["dataset"],
                "control_type": control_type,
                "selected_candidate_index": int(selected["candidate_index"]),
                "control_candidate_index": int(control["candidate_index"]),
                "selected_label": int(selected["label"]),
                "control_label": int(control["label"]),
                "delta_correct": int(selected["label"]) - int(control["label"]),
                "selected_law_score": float(selected["law_score"]),
                "control_law_score": float(control["law_score"]),
                "selected_tcr": float(selected["tcr"]),
                "control_tcr": float(control["tcr"]),
                "selected_reconstruction_error_log10": float(selected["reconstruction_error_log10"]),
                "control_reconstruction_error_log10": float(control["reconstruction_error_log10"]),
                "selected_departure_index": float(selected["departure_index"]),
                "control_departure_index": float(control["departure_index"]),
                "selected_stability_guard": float(selected["stability_guard"]),
                "control_stability_guard": float(control["stability_guard"]),
                "selected_early_kv_mass": float(selected["early_kv_mass"]),
                "control_early_kv_mass": float(control["early_kv_mass"]),
                "selected_settlement_score": float(selected["settlement_score"]),
                "control_settlement_score": float(control["settlement_score"]),
            }
            row["abs_tcr_delta"] = abs(row["selected_tcr"] - row["control_tcr"])
            row["abs_reconstruction_delta"] = abs(
                row["selected_reconstruction_error_log10"] - row["control_reconstruction_error_log10"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_pairs(pairs: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    boot_rows = []
    for control_type, g in pairs.groupby("control_type", sort=False):
        summaries.append(
            {
                "control_type": control_type,
                "n_pairs": int(len(g)),
                "selected_correct_rate": float(g["selected_label"].mean()),
                "control_correct_rate": float(g["control_label"].mean()),
                "delta_correct": float(g["delta_correct"].mean()),
                "abs_tcr_delta_median": float(g["abs_tcr_delta"].median()),
                "abs_reconstruction_delta_median": float(g["abs_reconstruction_delta"].median()),
                "selected_departure_median": float(g["selected_departure_index"].median()),
                "control_departure_median": float(g["control_departure_index"].median()),
                "selected_stability_guard_median": float(g["selected_stability_guard"].median()),
                "control_stability_guard_median": float(g["control_stability_guard"].median()),
            }
        )
        stat = paired_bootstrap(g["selected_label"], g["control_label"], n_boot=n_boot, seed=seed)
        stat.update({"control_type": control_type, "metric": "selected_correct_minus_control_correct"})
        boot_rows.append(stat)
    return pd.DataFrame(summaries), pd.DataFrame(boot_rows)


def make_figure(summary: pd.DataFrame, boot: pd.DataFrame, out_dir: Path) -> None:
    plt = setup_matplotlib()
    plot = summary.merge(boot[["control_type", "ci_low", "ci_high"]], on="control_type", how="left")
    order = plot.sort_values("delta_correct")["control_type"].tolist()
    plot = plot.set_index("control_type").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(plot))
    ax.barh(y, plot["delta_correct"], color="#4C78A8")
    ax.errorbar(
        plot["delta_correct"],
        y,
        xerr=[plot["delta_correct"] - plot["ci_low"], plot["ci_high"] - plot["delta_correct"]],
        fmt="none",
        ecolor="black",
        capsize=3,
        lw=1.0,
    )
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["control_type"])
    ax.set_xlabel("Selected C316 candidate minus matched control correctness")
    ax.set_title("Matched law-breaking controls")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_matched_law_breaking.pdf")
    fig.savefig(out_dir / "fig_matched_law_breaking.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", type=Path, default=V0_ROOT)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    out_dir = ensure_dir(args.v1_root / "outputs/04_matched_controls")
    df = read_observables(args.v1_root, args.v0_root)
    pairs = build_pairs(df, args.seed)
    summary, boot = summarize_pairs(pairs, args.n_bootstrap, args.seed)

    pairs.to_csv(out_dir / "matched_pairs.csv", index=False)
    summary.to_csv(out_dir / "matched_control_summary.csv", index=False)
    boot.to_csv(out_dir / "matched_control_bootstrap.csv", index=False)
    make_figure(summary, boot, out_dir)
    write_json(
        out_dir / "manifest.json",
        {
            "rows_in": int(len(df)),
            "queries": int(df["query_key"].nunique()),
            "matched_pairs": int(len(pairs)),
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "selector": "fixed_C316_top1_by_law_score",
        },
    )

    print("Matched law-breaking controls complete")
    print(summary.to_string(index=False))
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
