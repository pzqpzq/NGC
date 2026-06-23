#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ngc_v2_utils import RANDOM_SEED, ensure_dir, paired_bootstrap, query_metrics_for_score, setup_matplotlib, summarize_per_query


ROLE_EXCLUSIONS = {
    "full_method": [],
    "drop_reconstruction_guard_skeptic": [
        "U_ske",
        "U_veto",
        "stability_guard",
        "reconstruction_score",
        "reconstruction_error",
        "reconstruction_error_log10",
        "w_err_norm",
        "y_err_norm",
    ],
    "drop_settlement_stabilizer": ["U_stab", "U_set", "settlement_score", "late_q_mass", "late_mlp_mass", "late_block_frac"],
    "drop_evidence_broker_retrieval": ["U_ret", "evidence_broker_score", "early_kv_mass", "early_block_frac", "k_block_frac", "v_block_frac", "kv_frac"],
    "drop_mediator": ["U_med", "mediator_score"],
    "drop_veto_gate": ["U_veto"],
    "drop_role_depth_priors": [
        "early_block_frac",
        "mid_block_frac",
        "late_block_frac",
        "attention_block_frac",
        "mlp_block_frac",
        "mlp_block_frac_rich",
        "k_block_frac",
        "v_block_frac",
        "q_block_frac",
        "gate_block_frac",
        "up_block_frac",
        "down_block_frac",
        "attn_frac",
        "mlp_frac",
        "kv_frac",
        "q_frac",
    ],
    "reconstruction_only": ["*keep:reconstruction_score"],
    "stability_only": ["*keep:stability_guard", "*keep:stability_risk_log10", "*keep:U_ske", "*keep:U_veto"],
    "random_role_weights": ["*random_role_weights"],
}


def table_block(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    if df.empty:
        return "No rows."
    view = df[cols] if cols is not None else df
    return "```text\n" + view.to_string(index=False) + "\n```"


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_bank(v2_root: Path) -> pd.DataFrame:
    p = v2_root / "results/router_candidate_bank.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(v2_root / "results/router_candidate_bank.csv.gz")


def base_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat = [c for c in ["model", "dataset", "topology_family", "compression_bin"] if c in df.columns]
    blocked = {"label", "candidate_index", "cached_is_correct_rich", "compression_bin"}
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in blocked and c not in cat]
    return num, cat


def feature_subset(num: list[str], condition: str, seed: int) -> list[str]:
    rules = ROLE_EXCLUSIONS[condition]
    keep_rules = [r for r in rules if r.startswith("*keep:")]
    if keep_rules:
        requested = [r.split(":", 1)[1] for r in keep_rules]
        return [c for c in num if any(c == req or c.startswith(req + "_") for req in requested)]
    if "*random_role_weights" in rules:
        return num
    return [c for c in num if not any(token in c for token in rules)]


def make_model(num_cols: list[str], cat_cols: list[str], seed: int) -> Pipeline:
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("cat", one_hot_encoder(), cat_cols))
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers, remainder="drop")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=180,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=0.03,
                    random_state=seed,
                ),
            ),
        ]
    )


def add_random_role_weights(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    role_cols = [c for c in ["U_prop", "U_ske", "U_stab", "U_med", "U_ret", "U_set", "U_veto"] if c in train.columns]
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 1, size=len(role_cols))
    for frame in [train, test]:
        frame["random_role_score"] = frame[role_cols].to_numpy(dtype=float) @ weights if role_cols else 0.0
    return train, test


def train_condition(df: pd.DataFrame, condition: str, seed: int, top_k: int) -> tuple[dict, pd.DataFrame]:
    train = df[df["split_random_70_15_15"].eq("train")].copy()
    test = df[df["split_random_70_15_15"].eq("test")].copy()
    num, cat = base_features(df)
    if condition == "random_role_weights":
        train, test = add_random_role_weights(train, test, seed)
        num = num + ["random_role_score"]
    selected_num = feature_subset(num, condition, seed)
    model = make_model(selected_num, cat, seed)
    features = selected_num + cat
    model.fit(train[features], train["label"].astype(int))
    test = test.copy()
    test[f"score_{condition}"] = model.predict_proba(test[features])[:, 1]
    per_query = query_metrics_for_score(test, f"score_{condition}", top_k=top_k, ascending=False)
    summary = summarize_per_query(condition, per_query)
    summary["condition"] = condition
    summary["n_features"] = len(features)
    return summary, per_query.assign(condition=condition)


def load_fixed_ablation(v1_root: Path, v2_root: Path) -> pd.DataFrame:
    src = v1_root / "outputs/06_role_knockout/role_knockout_summary.csv"
    if not src.exists():
        return pd.DataFrame()
    fixed = pd.read_csv(src)
    fixed = fixed.rename(columns={"condition": "condition"})
    out = fixed.copy()
    out["source"] = "NGC-v1 cached fixed-law role ablation"
    out.to_csv(v2_root / "results/role_ablation_fixed.csv", index=False)
    return out


def make_figures(adaptive: pd.DataFrame, fixed: pd.DataFrame, fig_dir: Path) -> None:
    plt = setup_matplotlib()
    if not adaptive.empty:
        plot = adaptive.sort_values("pass5", ascending=True)
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        ax.barh(plot["condition"], 100 * plot["pass5"], color="#4778B3")
        ax.set_xlabel("Adaptive router pass@5 (%)")
        ax.set_ylabel("")
        ax.set_title("Adaptive router role ablations")
        fig.tight_layout()
        fig.savefig(fig_dir / "role_ablation_barplot.pdf")
        fig.savefig(fig_dir / "role_ablation_barplot.png", dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--v1-root", type=Path, default=Path(os.environ.get("NGC_V1_ROOT", repo_root / "v1")))
    parser.add_argument("--v2-root", type=Path, default=Path(os.environ.get("NGC_V2_ROOT", repo_root / "v2")))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    results_dir = ensure_dir(args.v2_root / "results")
    reports_dir = ensure_dir(args.v2_root / "reports")
    figures_dir = ensure_dir(args.v2_root / "figures")
    df = load_bank(args.v2_root)

    summaries = []
    per_query_frames = []
    for condition in ROLE_EXCLUSIONS:
        summary, per_query = train_condition(df, condition, args.seed, args.top_k)
        summaries.append(summary)
        per_query_frames.append(per_query)
    adaptive = pd.DataFrame(summaries).sort_values("pass5", ascending=False)
    adaptive.to_csv(results_dir / "role_ablation_adaptive.csv", index=False)
    per_query_all = pd.concat(per_query_frames, ignore_index=True)
    per_query_all.to_csv(results_dir / "role_ablation_adaptive_per_query.csv", index=False)

    fixed = load_fixed_ablation(args.v1_root, args.v2_root)
    full = per_query_all[per_query_all["condition"].eq("full_method")].set_index("query_key")
    contrasts = []
    for condition, group in per_query_all.groupby("condition"):
        if condition == "full_method":
            continue
        aligned = group.set_index("query_key").join(full[["pass5"]].rename(columns={"pass5": "full_pass5"}), how="inner")
        stat = paired_bootstrap(aligned["full_pass5"], aligned["pass5"], n_boot=args.n_bootstrap, seed=args.seed)
        stat.update({"condition": condition, "contrast": "full_method_minus_ablation"})
        contrasts.append(stat)
    pd.DataFrame(contrasts).to_csv(results_dir / "role_ablation_adaptive_bootstrap.csv", index=False)
    make_figures(adaptive, fixed, figures_dir)

    lines = [
        "# Role Ablation Summary",
        "",
        "Adaptive-router ablations retrain the same tree router after removing feature families.",
        "Fixed-law ablations reuse the exact cached NGC-v1 role-ablation output when available.",
        "",
        "## Adaptive Router",
        "",
        table_block(adaptive, ["condition", "n_queries", "top1_acc", "pass5", "mean_at_5", "n_features"]),
        "",
        "## Adaptive Bootstrap Contrasts",
        "",
        table_block(pd.DataFrame(contrasts)) if contrasts else "No contrasts.",
        "",
    ]
    if not fixed.empty:
        lines.extend(["## Fixed-Law Cached Ablation", "", table_block(fixed.head(20)), ""])
    (reports_dir / "role_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Role ablation complete")
    print(adaptive[["condition", "top1_acc", "pass5", "n_features"]].to_string(index=False))


if __name__ == "__main__":
    main()
