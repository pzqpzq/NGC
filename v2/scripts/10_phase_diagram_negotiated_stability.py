#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from jun15_utils import (
    DEFAULT_V2_ROOT,
    RANDOM_SEED,
    ensure_dir,
    load_router_predictions,
    paired_bootstrap,
    robust_fit,
    robust_transform,
    write_json,
)
from ngc_v2_utils import setup_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Jun 15 negotiated-stability phase diagram.")
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_V2_ROOT / "results/jun15_mechanistic_controls")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=3000)
    return parser.parse_args()


def add_phase_coordinates(df: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fits = {
        "departure_index": robust_fit(train["departure_index"]),
        "stability_guard": robust_fit(train["stability_guard"]),
        "reconstruction_score": robust_fit(train["reconstruction_score"]),
    }
    out = df.copy()
    out["phase_departure_z"] = robust_transform(out["departure_index"], fits["departure_index"]).clip(-8, 8)
    out["phase_stability_z"] = robust_transform(out["stability_guard"], fits["stability_guard"]).clip(-8, 8)
    out["phase_fidelity_z"] = robust_transform(out["reconstruction_score"], fits["reconstruction_score"]).clip(-8, 8)
    out["phase_stability_fidelity_z"] = 0.5 * out["phase_stability_z"] + 0.5 * out["phase_fidelity_z"]
    return out, fits


def select_rows(df: pd.DataFrame, selector: str, top_k: int, seed: int) -> pd.DataFrame:
    if selector == "random":
        rows = []
        for _, group in df.groupby("query_key", sort=False):
            rows.append(group.sample(n=min(top_k, len(group)), random_state=seed))
        return pd.concat(rows, ignore_index=False)
    score_cols = {
        "reconstruction": "reconstruction_score",
        "fixed_law": "score_fixed_transfer_law" if "score_fixed_transfer_law" in df else "law_fixed_transfer",
        "adaptive_router": "score_extra_trees_router" if "score_extra_trees_router" in df else "law_fixed_transfer",
    }
    score_col = score_cols[selector]
    return (
        df.sort_values(score_col, ascending=False, kind="mergesort")
        .groupby("query_key", group_keys=False)
        .head(top_k)
        .copy()
    )


def make_grid(df: pd.DataFrame, x_col: str, y_col: str, bins: int = 8) -> pd.DataFrame:
    x_edges = np.quantile(df[x_col].dropna(), np.linspace(0, 1, bins + 1))
    y_edges = np.quantile(df[y_col].dropna(), np.linspace(0, 1, bins + 1))
    x_edges = np.unique(x_edges)
    y_edges = np.unique(y_edges)
    work = df.copy()
    work["departure_bin"] = pd.cut(work[x_col], x_edges, include_lowest=True, duplicates="drop")
    work["stability_fidelity_bin"] = pd.cut(work[y_col], y_edges, include_lowest=True, duplicates="drop")
    grid = (
        work.groupby(["departure_bin", "stability_fidelity_bin"], observed=True)
        .agg(
            n=("label", "size"),
            correctness=("label", "mean"),
            departure_center=(x_col, "median"),
            stability_fidelity_center=(y_col, "median"),
            reconstruction_score=("reconstruction_score", "median"),
            stability_guard=("stability_guard", "median"),
        )
        .reset_index()
    )
    return grid


def best_region_from_train(train: pd.DataFrame, min_n: int = 80) -> dict:
    grid = make_grid(train, "phase_departure_z", "phase_stability_fidelity_z", bins=8)
    valid = grid[grid["n"].ge(min_n)].copy()
    if valid.empty:
        valid = grid.copy()
    row = valid.sort_values(["correctness", "n"], ascending=False).iloc[0]
    return {
        "departure_center": float(row["departure_center"]),
        "stability_fidelity_center": float(row["stability_fidelity_center"]),
        "radius": 0.90,
        "train_correctness": float(row["correctness"]),
        "train_n": int(row["n"]),
    }


def region_mask(df: pd.DataFrame, region: dict) -> pd.Series:
    dist = np.sqrt(
        (df["phase_departure_z"] - region["departure_center"]) ** 2
        + (df["phase_stability_fidelity_z"] - region["stability_fidelity_center"]) ** 2
    )
    return dist <= region["radius"]


def matched_reconstruction_contrast(test: pd.DataFrame, selected: pd.DataFrame, seed: int, n_boot: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    rng = np.random.default_rng(seed)
    selected_ids = set(selected["candidate_uid"].astype(str))
    for _, chosen in selected.groupby("query_key", sort=False).head(1).iterrows():
        group = test[test["query_key"].astype(str).eq(str(chosen["query_key"]))].copy()
        group = group[~group["candidate_uid"].astype(str).isin(selected_ids)]
        if group.empty:
            continue
        group["recon_gap"] = (pd.to_numeric(group["reconstruction_score"], errors="coerce") - float(chosen["reconstruction_score"])).abs()
        if "compression_bin" in group:
            matched = group[group["compression_bin"].eq(chosen["compression_bin"])]
            if not matched.empty:
                group = matched
        control = group.sort_values(["recon_gap"], kind="mergesort").iloc[0]
        rows.append(
            {
                "query_key": chosen["query_key"],
                "selected_label": int(chosen["label"]),
                "control_label": int(control["label"]),
                "selected_candidate_uid": chosen["candidate_uid"],
                "control_candidate_uid": control["candidate_uid"],
                "selected_reconstruction_score": float(chosen["reconstruction_score"]),
                "control_reconstruction_score": float(control["reconstruction_score"]),
                "selected_departure_z": float(chosen["phase_departure_z"]),
                "control_departure_z": float(control["phase_departure_z"]),
            }
        )
    match_df = pd.DataFrame(rows)
    if match_df.empty:
        return match_df, {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_boot": float("nan"), "n": 0}
    stat = paired_bootstrap(match_df["selected_label"], match_df["control_label"], n_boot=n_boot, seed=seed)
    return match_df, stat


def model_stats(test: pd.DataFrame) -> pd.DataFrame:
    features = test[["phase_departure_z", "phase_stability_fidelity_z"]].astype(float).copy()
    features["departure_sq"] = features["phase_departure_z"] ** 2
    features["stability_sq"] = features["phase_stability_fidelity_z"] ** 2
    features["interaction"] = features["phase_departure_z"] * features["phase_stability_fidelity_z"]
    y = test["label"].astype(int)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(features, y)
    prob = clf.predict_proba(features)[:, 1]
    rows = [{"term": name, "coefficient": float(coef)} for name, coef in zip(features.columns, clf.coef_[0])]
    rows.append({"term": "candidate_auc", "coefficient": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan")})
    return pd.DataFrame(rows)


def make_figure(test: pd.DataFrame, selected_sets: dict[str, pd.DataFrame], out_path: Path) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    rng = np.random.default_rng(RANDOM_SEED)
    sample = test.sample(n=min(9000, len(test)), random_state=RANDOM_SEED)
    colors = np.where(sample["label"].astype(int).to_numpy() == 1, "#2C9C95", "#A8B0B9")
    axes[0].scatter(sample["phase_departure_z"], sample["phase_stability_fidelity_z"], c=colors, s=7, alpha=0.35, linewidths=0)
    axes[0].set_xlabel("Departure from root (robust z)")
    axes[0].set_ylabel("Stability and fidelity (robust z)")
    axes[0].set_title("Candidate correctness in negotiated-stability phase space")

    contour_colors = {
        "random": "#8A8F98",
        "reconstruction": "#D08A2D",
        "fixed_law": "#3B6FB6",
        "adaptive_router": "#2C9C95",
    }
    for name, sdf in selected_sets.items():
        x = sdf["phase_departure_z"].to_numpy(dtype=float)
        y = sdf["phase_stability_fidelity_z"].to_numpy(dtype=float)
        if len(x) < 10:
            continue
        hist, xedges, yedges = np.histogram2d(x, y, bins=35, range=[[-5, 5], [-5, 5]], density=True)
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        levels = np.quantile(hist[hist > 0], [0.72, 0.88]) if np.any(hist > 0) else []
        if len(levels):
            axes[1].contour(xc, yc, hist.T, levels=np.unique(levels), colors=contour_colors[name], linewidths=1.7)
        axes[1].scatter([], [], color=contour_colors[name], label=name.replace("_", " "))
    axes[1].set_xlabel("Departure from root (robust z)")
    axes[1].set_ylabel("Stability and fidelity (robust z)")
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)
    axes[1].set_title("Selector density contours")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=320)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_root)
    fig_dir = ensure_dir(args.v2_root / "figures")
    df = load_router_predictions(args.v2_root)
    train_source = load_router_predictions(args.v2_root)
    if "split_random_70_15_15" in train_source and train_source["split_random_70_15_15"].eq("test").all():
        full = pd.read_parquet(args.v2_root / "results/router_candidate_bank.parquet")
        train_source = full[full["split_random_70_15_15"].eq("train")].copy()
    train_phase, fits = add_phase_coordinates(train_source, train_source)
    test, _ = add_phase_coordinates(df, train_source)
    test.to_csv(out_dir / "phase_diagram_source_data.csv.gz", index=False, compression="gzip")

    grid = make_grid(test, "phase_departure_z", "phase_stability_fidelity_z", bins=8)
    grid.to_csv(out_dir / "phase_diagram_binned_correctness.csv", index=False)

    selected_sets = {
        name: select_rows(test, name, args.top_k, args.seed)
        for name in ["random", "reconstruction", "fixed_law", "adaptive_router"]
    }
    for name, sdf in selected_sets.items():
        sdf.to_csv(out_dir / f"phase_selected_{name}.csv.gz", index=False, compression="gzip")

    region = best_region_from_train(train_phase)
    mask = region_mask(test, region)
    region_by_query = (
        test.assign(in_region=mask)
        .groupby(["query_key", "in_region"], as_index=False)["label"]
        .mean()
        .pivot(index="query_key", columns="in_region", values="label")
        .dropna()
    )
    if True in region_by_query.columns and False in region_by_query.columns:
        region_stat = paired_bootstrap(region_by_query[True], region_by_query[False], n_boot=args.n_bootstrap, seed=args.seed)
    else:
        region_stat = {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_boot": float("nan"), "n": 0}
    pd.DataFrame([{**region, **region_stat, "contrast": "best_train_region_minus_outside_on_test"}]).to_csv(
        out_dir / "phase_region_bootstrap.csv", index=False
    )

    matched, matched_stat = matched_reconstruction_contrast(test, selected_sets["fixed_law"], args.seed, args.n_bootstrap)
    matched.to_csv(out_dir / "phase_matched_reconstruction_pairs.csv", index=False)
    pd.DataFrame([{**matched_stat, "contrast": "fixed_law_top1_minus_reconstruction_matched_control"}]).to_csv(
        out_dir / "phase_matched_reconstruction_bootstrap.csv", index=False
    )

    stats = model_stats(test)
    stats.to_csv(out_dir / "phase_logistic_surface_stats.csv", index=False)
    make_figure(test, selected_sets, fig_dir / "jun15_phase_diagram_negotiated_stability")
    write_json(
        out_dir / "phase_diagram_manifest.json",
        {
            "rows": int(len(test)),
            "queries": int(test["query_key"].nunique()),
            "train_scalers": fits,
            "region": region,
            "outputs": {
                "figure_pdf": str(fig_dir / "jun15_phase_diagram_negotiated_stability.pdf"),
                "source_data": str(out_dir / "phase_diagram_source_data.csv.gz"),
            },
        },
    )
    print("Phase diagram complete")
    print(pd.DataFrame([{**region_stat, "contrast": "best_train_region_minus_outside_on_test"}]).to_string(index=False))
    print(pd.DataFrame([{**matched_stat, "contrast": "fixed_law_top1_minus_reconstruction_matched_control"}]).to_string(index=False))


if __name__ == "__main__":
    main()
