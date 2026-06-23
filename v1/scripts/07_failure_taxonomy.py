#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ngc_v1.common import NEGATIVE_GROUPS, V0_ROOT, V1_ROOT, ensure_dir, read_observables, setup_matplotlib, write_json


FEATURES = [
    "num_tokens",
    "reconstruction_error_log10",
    "stability_guard",
    "transport_shift_log10",
    "departure_index",
    "early_kv_mass",
    "late_q_mass",
    "late_mlp_mass",
    "settlement_score",
    "role_entropy",
    "tcr",
]


def load_fixed_gains(v0_root: Path) -> pd.DataFrame:
    path = v0_root / "nmi_may27_experiments/outputs/day1/law_selection_summary.csv"
    if not path.exists():
        rows = [{"model": m, "dataset": d, "gain": np.nan} for m, d in NEGATIVE_GROUPS]
        return pd.DataFrame(rows)
    df = pd.read_csv(path)
    fixed = df[df["protocol"] == "fixed"][["model", "dataset", "raw_pass", "ngc_pass", "gain"]].copy()
    fixed["is_negative_fixed_group"] = fixed["gain"] < 0
    return fixed


def selected_c316(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["query_key", "law_score"], ascending=[True, False], kind="mergesort")
        .groupby("query_key", as_index=False)
        .head(1)
        .copy()
    )


def classify_failures(sel: pd.DataFrame) -> pd.DataFrame:
    out = sel.copy()
    grouped = out.groupby(["model", "dataset"], group_keys=False)
    out["departure_q25"] = grouped["departure_index"].transform(lambda s: s.quantile(0.25))
    out["departure_q75"] = grouped["departure_index"].transform(lambda s: s.quantile(0.75))
    out["guard_q25"] = grouped["stability_guard"].transform(lambda s: s.quantile(0.25))
    out["early_kv_q35"] = grouped["early_kv_mass"].transform(lambda s: s.quantile(0.35))
    out["settlement_q75"] = grouped["settlement_score"].transform(lambda s: s.quantile(0.75))

    labels = []
    for _, row in out.iterrows():
        if int(row["label"]) == 1:
            labels.append("S0 selected_success")
        elif row["departure_index"] <= row["departure_q25"]:
            labels.append("F1 over_conservative")
        elif row["departure_index"] >= row["departure_q75"] and row["stability_guard"] <= row["guard_q25"]:
            labels.append("F2 over_disruptive")
        elif row["early_kv_mass"] <= row["early_kv_q35"] or row["settlement_score"] >= row["settlement_q75"]:
            labels.append("F3 wrong_community")
        elif (row["model"], row["dataset"]) in {("qwen3-4b", "aime"), ("qwen3-8b", "aime")}:
            labels.append("F4 brittle_small_model_math")
        elif row["dataset"] in {"aime", "gpqa", "math500"}:
            labels.append("F5 grading_sensitive_or_symbolic")
        else:
            labels.append("F6 residual_failure")
    out["failure_class"] = labels
    return out


def compare_groups(sel: pd.DataFrame, gains: pd.DataFrame) -> pd.DataFrame:
    merged = sel.merge(gains, on=["model", "dataset"], how="left")
    merged["fixed_group_sign"] = np.where(merged["gain"] < 0, "negative_fixed_gain", "nonnegative_fixed_gain")
    rows = []
    for feature in FEATURES:
        for sign, g in merged.groupby("fixed_group_sign"):
            rows.append(
                {
                    "feature": feature,
                    "group": sign,
                    "n": int(len(g)),
                    "mean": float(g[feature].mean()),
                    "median": float(g[feature].median()),
                    "std": float(g[feature].std()),
                }
            )
        neg = merged[merged["fixed_group_sign"] == "negative_fixed_gain"][feature]
        pos = merged[merged["fixed_group_sign"] == "nonnegative_fixed_gain"][feature]
        if len(neg) and len(pos):
            pooled = np.sqrt((neg.var() + pos.var()) / 2)
            d = float((neg.mean() - pos.mean()) / pooled) if pooled > 1e-12 else np.nan
            rows.append({"feature": feature, "group": "cohen_d_negative_minus_nonnegative", "n": len(merged), "mean": d, "median": d, "std": np.nan})
    return pd.DataFrame(rows), merged


def write_tex_table(gains: pd.DataFrame, out_path: Path) -> None:
    neg = gains[gains["gain"] < 0].sort_values("gain")
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Model & Dataset & Raw & NGC & Gain \\\\",
        "\\midrule",
    ]
    for _, row in neg.iterrows():
        lines.append(
            f"{row['model']} & {row['dataset']} & {row['raw_pass']:.3f} & {row['ngc_pass']:.3f} & {row['gain']:.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(merged: pd.DataFrame, out_dir: Path) -> None:
    plt = setup_matplotlib()
    import seaborn as sns

    counts = merged["failure_class"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.barh(counts.index, counts.values, color="#E15759")
    ax.set_xlabel("Selected C316 queries")
    ax.set_ylabel("")
    ax.set_title("Failure taxonomy for fixed C316 selected candidates")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_failure_taxonomy.pdf")
    fig.savefig(out_dir / "fig_failure_taxonomy.png")
    plt.close(fig)

    melt = merged.melt(
        id_vars=["fixed_group_sign"],
        value_vars=["departure_index", "stability_guard", "early_kv_mass", "settlement_score"],
        var_name="feature",
        value_name="value",
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    for ax, feature in zip(axes.ravel(), ["departure_index", "stability_guard", "early_kv_mass", "settlement_score"]):
        sub = melt[melt["feature"] == feature]
        sns.boxplot(data=sub, x="fixed_group_sign", y="value", ax=ax, color="#9C755F")
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("Positive versus negative fixed-law groups", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_failure_group_features.pdf")
    fig.savefig(out_dir / "fig_failure_group_features.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", type=Path, default=V0_ROOT)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    args = parser.parse_args()

    out_dir = ensure_dir(args.v1_root / "outputs/07_failure_taxonomy")
    df = read_observables(args.v1_root, args.v0_root)
    gains = load_fixed_gains(args.v0_root)
    selected = selected_c316(df)
    classified = classify_failures(selected)
    feature_summary, merged = compare_groups(classified, gains)

    class_summary = (
        merged.groupby(["model", "dataset", "failure_class"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values(["model", "dataset", "n"], ascending=[True, True, False])
    )
    examples = (
        merged[merged["failure_class"].str.startswith("F")]
        .sort_values(["gain", "failure_class", "law_score"], ascending=[True, True, False])
        .head(120)
    )

    gains.to_csv(out_dir / "fixed_law_group_gains.csv", index=False)
    class_summary.to_csv(out_dir / "failure_mode_summary.csv", index=False)
    feature_summary.to_csv(out_dir / "positive_vs_negative_feature_summary.csv", index=False)
    examples[
        [
            "query_key",
            "model",
            "dataset",
            "candidate_index",
            "label",
            "failure_class",
            "gain",
            "law_score",
            "departure_index",
            "stability_guard",
            "early_kv_mass",
            "settlement_score",
            "reconstruction_error_log10",
            "tcr",
            "num_tokens",
        ]
    ].to_csv(out_dir / "failure_mode_examples.csv", index=False)
    write_tex_table(gains, out_dir / "negative_case_table.tex")
    make_figures(merged, out_dir)
    write_json(
        out_dir / "manifest.json",
        {
            "selected_queries": int(len(selected)),
            "negative_fixed_gain_groups": gains[gains["gain"] < 0][["model", "dataset"]].to_dict(orient="records"),
            "outputs": {
                "summary": str(out_dir / "failure_mode_summary.csv"),
                "examples": str(out_dir / "failure_mode_examples.csv"),
            },
        },
    )

    print("Failure taxonomy complete")
    print(class_summary.head(20).to_string(index=False))
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
