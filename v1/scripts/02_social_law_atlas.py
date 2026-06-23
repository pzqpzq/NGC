#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ngc_v1.common import V0_ROOT, V1_ROOT, ensure_dir, read_observables, setup_matplotlib, write_json


ROLE_FEATURES = [
    "early_kv_mass",
    "late_q_mass",
    "late_mlp_mass",
    "departure_index",
    "stability_guard",
    "role_entropy",
    "constitutional_balance",
    "mediator_score",
    "settlement_score",
]


def topk_by_query(df: pd.DataFrame, score_col: str, k: int = 5, ascending: bool = False) -> pd.DataFrame:
    return (
        df.sort_values(["query_key", score_col], ascending=[True, ascending], kind="mergesort")
        .groupby("query_key", group_keys=False)
        .head(k)
        .copy()
    )


def deterministic_random_by_query(df: pd.DataFrame, k: int = 5, seed: int = 20260528) -> pd.DataFrame:
    pieces = []
    rng = np.random.default_rng(seed)
    for _, g in df.groupby("query_key", sort=False):
        take = min(k, len(g))
        idx = rng.choice(g.index.to_numpy(), size=take, replace=False)
        pieces.append(g.loc[idx])
    return pd.concat(pieces, axis=0).copy()


def profile(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    prof = (
        df.groupby(["model", "dataset"], as_index=False)[ROLE_FEATURES + ["label", "raw_prob", "num_tokens", "tcr"]]
        .median(numeric_only=True)
        .rename(columns={"label": "candidate_correct_rate", "raw_prob": "raw_pass"})
    )
    prof["cohort"] = cohort
    prof["n_candidates"] = df.groupby(["model", "dataset"]).size().values
    prof["n_queries"] = df.groupby(["model", "dataset"])["query_key"].nunique().values
    return prof


def build_profiles(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    cohorts = [
        profile(topk_by_query(df, "law_score", top_k, ascending=False), "law_C316_topk"),
        profile(topk_by_query(df, "reconstruction_score", top_k, ascending=False), "reconstruction_topk"),
        profile(deterministic_random_by_query(df, top_k), "random_topk"),
        profile(df[df["label"] == 1], "successful_candidates"),
        profile(df[df["label"] == 0], "failed_candidates"),
    ]
    return pd.concat([c for c in cohorts if not c.empty], ignore_index=True)


def build_demand_profiles(signatures: pd.DataFrame) -> pd.DataFrame:
    # Use successful candidates as benchmark demand, falling back to law-selected candidates if needed.
    demand = signatures[signatures["cohort"] == "successful_candidates"].copy()
    if demand.empty:
        demand = signatures[signatures["cohort"] == "law_C316_topk"].copy()
    x = demand[ROLE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    xz = StandardScaler().fit_transform(x)
    n_clusters = min(4, len(demand))
    if n_clusters >= 2:
        demand["role_cluster"] = KMeans(n_clusters=n_clusters, random_state=20260528, n_init=20).fit_predict(xz)
    else:
        demand["role_cluster"] = 0
    if len(demand) >= 2:
        xy = PCA(n_components=2, random_state=20260528).fit_transform(xz)
    else:
        xy = np.zeros((len(demand), 2))
    demand["map_x"] = xy[:, 0]
    demand["map_y"] = xy[:, 1]
    demand["model_dataset"] = demand["model"].astype(str) + "::" + demand["dataset"].astype(str)
    return demand


def build_transfer_matrix(demand: pd.DataFrame) -> pd.DataFrame:
    if demand.empty:
        return pd.DataFrame()
    labels = demand["model_dataset"].tolist()
    x = demand[ROLE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    xz = StandardScaler().fit_transform(x)
    sim = cosine_similarity(xz)
    out = pd.DataFrame(sim, index=labels, columns=labels)
    out.index.name = "source"
    return out.reset_index()


def make_figures(signatures: pd.DataFrame, demand: pd.DataFrame, transfer: pd.DataFrame, out_dir: Path) -> None:
    plt = setup_matplotlib()
    import seaborn as sns

    law = signatures[signatures["cohort"] == "law_C316_topk"].copy()
    law["group"] = law["model"].astype(str) + "/" + law["dataset"].astype(str)
    heat = law.set_index("group")[ROLE_FEATURES].sort_index()
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    sns.heatmap(heat, cmap="vlag", center=0, linewidths=0.2, ax=ax)
    ax.set_title("C316 law-selected role signatures")
    ax.set_xlabel("Constitutional observable")
    ax.set_ylabel("Model / dataset")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_law_role_heatmap.pdf")
    fig.savefig(out_dir / "fig_law_role_heatmap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    scatter = ax.scatter(
        demand["map_x"],
        demand["map_y"],
        c=demand["role_cluster"],
        s=75,
        cmap="tab10",
        edgecolor="black",
        linewidth=0.4,
    )
    for _, row in demand.iterrows():
        ax.text(row["map_x"], row["map_y"], f"{row['model']}/{row['dataset']}", fontsize=7, ha="left", va="bottom")
    ax.set_title("Benchmark demand map from successful constitutions")
    ax.set_xlabel("Role signature PC1")
    ax.set_ylabel("Role signature PC2")
    fig.colorbar(scatter, ax=ax, label="Role cluster")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_benchmark_constitution_map.pdf")
    fig.savefig(out_dir / "fig_benchmark_constitution_map.png")
    plt.close(fig)

    if not transfer.empty:
        mat = transfer.set_index("source")
        fig, ax = plt.subplots(figsize=(9.0, 7.5))
        sns.heatmap(mat, cmap="mako", vmin=-1, vmax=1, linewidths=0.1, ax=ax)
        ax.set_title("Social-law transfer matrix")
        ax.set_xlabel("Target demand profile")
        ax.set_ylabel("Source demand profile")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_law_transfer_matrix.pdf")
        fig.savefig(out_dir / "fig_law_transfer_matrix.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", type=Path, default=V0_ROOT)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    out_dir = ensure_dir(args.v1_root / "outputs/02_law_atlas")
    df = read_observables(args.v1_root, args.v0_root)
    signatures = build_profiles(df, args.top_k)
    demand = build_demand_profiles(signatures)
    transfer = build_transfer_matrix(demand)

    signatures.to_csv(out_dir / "law_role_signatures.csv", index=False)
    demand.to_csv(out_dir / "benchmark_demand_profiles.csv", index=False)
    transfer.to_csv(out_dir / "law_transfer_matrix.csv", index=False)

    # Direct comparison between law-selected and reconstruction-only constitutions.
    law = signatures[signatures["cohort"] == "law_C316_topk"].set_index(["model", "dataset"])
    rec = signatures[signatures["cohort"] == "reconstruction_topk"].set_index(["model", "dataset"])
    delta = (law[ROLE_FEATURES] - rec[ROLE_FEATURES]).reset_index()
    delta.to_csv(out_dir / "law_vs_reconstruction_role_delta.csv", index=False)

    make_figures(signatures, demand, transfer, out_dir)
    write_json(
        out_dir / "manifest.json",
        {
            "rows_in": int(len(df)),
            "signature_rows": int(len(signatures)),
            "demand_rows": int(len(demand)),
            "top_k": int(args.top_k),
            "role_features": ROLE_FEATURES,
        },
    )

    print("Social-law atlas complete")
    print(f"Signature rows: {len(signatures)}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
