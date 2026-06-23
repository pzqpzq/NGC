#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ngc_v2_utils import (
    RANDOM_SEED,
    ensure_dir,
    paired_bootstrap,
    query_metrics_for_score,
    safe_to_parquet,
    setup_matplotlib,
    summarize_per_query,
)


TARGETS = [80.0, 85.0, 90.0, 95.0]


def table_block(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "No rows."
    return "```text\n" + df[cols].to_string(index=False) + "\n```"


def load_bank(v2_root: Path) -> pd.DataFrame:
    parquet_path = v2_root / "results/router_candidate_bank.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(v2_root / "results/router_candidate_bank.csv.gz")


def target_pool(df: pd.DataFrame, target: float, family: str | None = None) -> pd.DataFrame:
    pool = df[pd.to_numeric(df["compression_bin"], errors="coerce").eq(float(target))].copy()
    if family is not None:
        pool = pool[pool["topology_family"].eq(family)].copy()
    return pool


def add_baseline_scores(pool: pd.DataFrame, target: float) -> pd.DataFrame:
    out = pool.copy()
    tcr_gap = (pd.to_numeric(out["tcr"], errors="coerce") - float(target)).abs().fillna(0.0)
    y_err = pd.to_numeric(out.get("y_err_norm_mean", np.nan), errors="coerce")
    w_err = pd.to_numeric(out.get("w_err_norm_mean", np.nan), errors="coerce")
    mlp_rich = pd.to_numeric(out.get("mlp_block_frac_rich", out.get("mlp_frac", 0.0)), errors="coerce").fillna(0.0)
    ffn_frac = (
        pd.to_numeric(out.get("gate_block_frac", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("up_block_frac", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("down_block_frac", 0.0), errors="coerce").fillna(0.0)
    )
    out["score_cached_reconstruction"] = pd.to_numeric(out["reconstruction_score"], errors="coerce").fillna(-1e9) - 0.01 * tcr_gap
    out["score_asvd_lite"] = -np.log10(y_err.clip(lower=1e-12).fillna(y_err.median() if y_err.notna().any() else 1.0)) - 0.10 * np.log10(
        w_err.clip(lower=1e-12).fillna(w_err.median() if w_err.notna().any() else 1.0)
    ) - 0.02 * tcr_gap
    out["score_dobi_svd_lite"] = (
        0.45 * pd.to_numeric(out.get("U_ske", 0.0), errors="coerce").fillna(0.0)
        + 0.25 * pd.to_numeric(out.get("U_cost", 0.0), errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out.get("U_stab", 0.0), errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(out.get("U_med", 0.0), errors="coerce").fillna(0.0)
        - 0.01 * tcr_gap
    )
    out["score_svd_llm_v2_lite"] = (
        pd.to_numeric(out["reconstruction_score"], errors="coerce").fillna(0.0)
        + 0.20 * pd.to_numeric(out.get("U_stab", 0.0), errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(out.get("U_cost", 0.0), errors="coerce").fillna(0.0)
        - 0.015 * tcr_gap
    )
    out["score_sola_lite"] = (
        0.42 * pd.to_numeric(out.get("U_set", 0.0), errors="coerce").fillna(0.0)
        + 0.28 * mlp_rich
        + 0.20 * ffn_frac
        + 0.10 * pd.to_numeric(out.get("U_ske", 0.0), errors="coerce").fillna(0.0)
        - 0.015 * tcr_gap
    )
    return out


def fixed_wtp_score(train: pd.DataFrame, test: pd.DataFrame, family: str, target: float) -> tuple[pd.DataFrame, str]:
    train_pool = target_pool(train, target, family)
    test_pool = target_pool(test, target, family)
    if train_pool.empty or test_pool.empty:
        return pd.DataFrame(), ""
    acc = (
        train_pool.groupby(["model", "dataset", "wtp_file_name"], dropna=False)["label"]
        .mean()
        .reset_index(name="train_acc")
        .sort_values(["model", "dataset", "train_acc"], ascending=[True, True, False], kind="mergesort")
    )
    winners = acc.groupby(["model", "dataset"], as_index=False).head(1)
    marked = test_pool.merge(winners[["model", "dataset", "wtp_file_name", "train_acc"]], on=["model", "dataset", "wtp_file_name"])
    if marked.empty:
        return pd.DataFrame(), ""
    marked["score_selected_fixed_wtp"] = 1.0
    return marked, "; ".join(f"{r.model}/{r.dataset}:{r.wtp_file_name}" for r in winners.itertuples(index=False))


def evaluate_score_method(
    test: pd.DataFrame,
    target: float,
    method: str,
    family: str | None,
    score_col: str,
    top_k: int,
    exactness: str,
    notes: str,
) -> tuple[dict, pd.DataFrame]:
    pool = target_pool(test, target, family)
    if pool.empty:
        return {}, pd.DataFrame()
    pool = add_baseline_scores(pool, target)
    per_query = query_metrics_for_score(pool, score_col, top_k=top_k, ascending=False)
    summary = summarize_per_query(method, per_query)
    summary.update(
        {
            "method": method,
            "family_filter": family or "all",
            "target_compression_percent": float(target),
            "exactness": exactness,
            "notes": notes,
            "n_candidates": int(len(pool)),
            "n_model_dataset_pairs": int(pool[["model", "dataset"]].drop_duplicates().shape[0]),
        }
    )
    return summary, per_query.assign(method=method, target_compression_percent=target)


def evaluate_grouped_rows(per_query: pd.DataFrame, method: str, target: float, exactness: str, notes: str) -> list[dict]:
    rows = []
    for (model, dataset), group in per_query.groupby(["model", "dataset"], dropna=False):
        row = summarize_per_query(method, group)
        row.update(
            {
                "method": method,
                "model": model,
                "dataset": dataset,
                "target_compression_percent": float(target),
                "exactness": exactness,
                "notes": notes,
            }
        )
        rows.append(row)
    return rows


def make_curve_figure(results: pd.DataFrame, fig_dir: Path) -> None:
    plt = setup_matplotlib()
    keep = results[results["model"].eq("ALL") & results["dataset"].eq("ALL")].copy()
    if keep.empty:
        keep = results.copy()
    methods = [
        "Raw dense model",
        "NGC fixed transfer law top-5",
        "Vanilla SVD cached fixed WTP",
        "Basis Sharing cached fixed WTP",
        "Dobi-SVD-lite activation selector",
        "SVD-LLM-V2-lite dynamic selector",
        "SoLA-lite FFN-sensitive selector",
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for method in methods:
        cur = keep[keep["method"].eq(method)].sort_values("target_compression_percent")
        if cur.empty:
            continue
        ax.plot(cur["target_compression_percent"], 100 * cur["pass5"], marker="o", linewidth=1.8, label=method)
    ax.set_xlabel("Target compression (%)")
    ax.set_ylabel("Held-out pass@5 or top-1 pass rate (%)")
    ax.set_title("Cached compression frontier by method family")
    ax.legend(frameon=False, fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(fig_dir / "baseline_accuracy_compression_curve.pdf")
    fig.savefig(fig_dir / "baseline_accuracy_compression_curve.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--v2-root", type=Path, default=Path(os.environ.get("NGC_V2_ROOT", repo_root / "v2")))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    results_dir = ensure_dir(args.v2_root / "results")
    reports_dir = ensure_dir(args.v2_root / "reports")
    figures_dir = ensure_dir(args.v2_root / "figures")
    df = load_bank(args.v2_root)
    train = df[df["split_random_70_15_15"].eq("train")].copy()
    test = df[df["split_random_70_15_15"].eq("test")].copy()

    rows = []
    per_query_frames = []
    for target in TARGETS:
        raw = (
            test[["query_key", "model", "dataset", "raw_prob"]]
            .drop_duplicates("query_key")
            .rename(columns={"raw_prob": "raw", "query_key": "query_key"})
        )
        raw["pass1"] = raw["raw"]
        raw["pass3"] = raw["raw"]
        raw["pass5"] = raw["raw"]
        raw["pass10"] = raw["raw"]
        raw["mean_at_5"] = raw["raw"]
        raw["selected_num_tokens"] = np.nan
        raw_all = summarize_per_query("Raw dense model", raw)
        raw_all.update(
            {
                "method": "Raw dense model",
                "model": "ALL",
                "dataset": "ALL",
                "target_compression_percent": float(target),
                "exactness": "exact cached raw predictions",
                "notes": "Raw dense rows are repeated for each compression target as the uncompressed reference.",
            }
        )
        rows.append(raw_all)
        rows.extend(evaluate_grouped_rows(raw, "Raw dense model", target, "exact cached raw predictions", "Uncompressed reference."))

        method_specs = [
            (
                "NGC fixed transfer law top-5",
                "NGC",
                "law_fixed_transfer",
                args.top_k,
                "exact cached NGC selector",
                "Uses the existing fixed transfer law score over cached NGC candidates.",
            ),
            (
                "NGC fixed transfer law top-1",
                "NGC",
                "law_fixed_transfer",
                1,
                "exact cached NGC selector",
                "Top-1 variant of the existing fixed transfer law.",
            ),
            (
                "ASVD-lite activation-aware selector",
                "Vanilla SVD",
                "score_asvd_lite",
                1,
                "simplified proxy",
                "Uses cached SVD candidates ranked by activation/output reconstruction statistics; not a full ASVD reproduction.",
            ),
            (
                "Dobi-SVD-lite activation selector",
                "Vanilla SVD",
                "score_dobi_svd_lite",
                1,
                "simplified proxy",
                "Dobi-style activation-truncation proxy over cached SVD candidates; full public reproduction was not run in this block.",
            ),
            (
                "SVD-LLM-V2-lite dynamic selector",
                "Vanilla SVD",
                "score_svd_llm_v2_lite",
                1,
                "simplified proxy",
                "Dynamic allocation proxy using reconstruction, stability, and cost features over cached SVD candidates.",
            ),
            (
                "SoLA-lite FFN-sensitive selector",
                "NGC",
                "score_sola_lite",
                1,
                "simplified proxy",
                "FFN-sensitive selector proxy emphasizing MLP/gate/up/down participation and settlement features.",
            ),
        ]
        for method, family, score_col, top_k, exactness, notes in method_specs:
            summary, per_query = evaluate_score_method(test, target, method, family, score_col, top_k, exactness, notes)
            if summary:
                summary.update({"model": "ALL", "dataset": "ALL"})
                rows.append(summary)
                rows.extend(evaluate_grouped_rows(per_query, method, target, exactness, notes))
                per_query_frames.append(per_query)

        for family, method in [
            ("Vanilla SVD", "Vanilla SVD cached fixed WTP"),
            ("Basis Sharing", "Basis Sharing cached fixed WTP"),
        ]:
            selected, winners = fixed_wtp_score(train, test, family, target)
            if selected.empty:
                continue
            per_query = query_metrics_for_score(selected, "score_selected_fixed_wtp", top_k=1, ascending=False)
            summary = summarize_per_query(method, per_query)
            summary.update(
                {
                    "method": method,
                    "model": "ALL",
                    "dataset": "ALL",
                    "target_compression_percent": float(target),
                    "exactness": "exact cached compressed WTP",
                    "notes": f"One cached {family} WTP is selected on the training split per model/dataset. Winners: {winners}",
                }
            )
            rows.append(summary)
            rows.extend(evaluate_grouped_rows(per_query, method, target, "exact cached compressed WTP", summary["notes"]))
            per_query_frames.append(per_query.assign(method=method, target_compression_percent=target))

    results = pd.DataFrame(rows)
    ordered_cols = [
        "method",
        "model",
        "dataset",
        "target_compression_percent",
        "exactness",
        "n_queries",
        "raw_acc",
        "top1_acc",
        "pass3",
        "pass5",
        "pass10",
        "mean_at_5",
        "median_selected_tokens",
        "notes",
    ]
    for col in ordered_cols:
        if col not in results:
            results[col] = np.nan
    results = results[ordered_cols + [c for c in results.columns if c not in ordered_cols]]
    results.to_csv(results_dir / "baseline_compression_results.csv", index=False)
    results.to_json(results_dir / "baseline_compression_results.jsonl", orient="records", lines=True)
    if per_query_frames:
        per_query_all = pd.concat(per_query_frames, ignore_index=True)
        safe_to_parquet(per_query_all, results_dir / "baseline_compression_per_query.parquet")

    contrasts = []
    if per_query_frames:
        per_query_all = pd.concat(per_query_frames, ignore_index=True)
        for target in TARGETS:
            fixed = per_query_all[
                per_query_all["method"].eq("NGC fixed transfer law top-5")
                & per_query_all["target_compression_percent"].eq(target)
            ].set_index("query_key")
            for method in sorted(per_query_all["method"].unique()):
                if method == "NGC fixed transfer law top-5":
                    continue
                cur = per_query_all[
                    per_query_all["method"].eq(method) & per_query_all["target_compression_percent"].eq(target)
                ].set_index("query_key")
                if cur.empty or fixed.empty:
                    continue
                aligned = cur.join(fixed[["pass5"]].rename(columns={"pass5": "fixed_pass5"}), how="inner")
                if aligned.empty:
                    continue
                stat = paired_bootstrap(aligned["pass5"], aligned["fixed_pass5"], n_boot=args.n_bootstrap, seed=args.seed)
                stat.update({"method": method, "reference": "NGC fixed transfer law top-5", "target_compression_percent": target})
                contrasts.append(stat)
    pd.DataFrame(contrasts).to_csv(results_dir / "baseline_vs_ngc_bootstrap.csv", index=False)

    make_curve_figure(results, figures_dir)

    all_rows = results[results["model"].eq("ALL") & results["dataset"].eq("ALL")].copy()
    best_rows = all_rows.sort_values(["target_compression_percent", "pass5"], ascending=[True, False])
    lines = [
        "# Expanded Compression Baseline Summary",
        "",
        "This report combines exact cached Raw/SVD/Basis-Sharing/NGC outputs with explicitly labeled lite proxy baselines.",
        "The proxy rows are useful diagnostics but must not be described as full public reproductions.",
        "",
        "Compression mapping: cached `tCR` values are treated as paper-facing compression percentages, so `tCR#89.57` is reported as approximately 90% compression.",
        "",
        "## Best Overall Rows By Target",
        "",
        table_block(
            best_rows.groupby("target_compression_percent", as_index=False).head(8),
            [
                "target_compression_percent",
                "method",
                "exactness",
                "n_queries",
                "raw_acc",
                "top1_acc",
                "pass5",
                "median_selected_tokens",
            ],
        ),
        "",
        "## Caveats",
        "",
        "- Dobi-SVD-lite, ASVD-lite, SVD-LLM-V2-lite, and SoLA-lite are simplified selector proxies over cached candidate banks.",
        "- Exact public baseline reproduction still requires executable model compression runs if the manuscript needs strict apples-to-apples claims.",
        "- Cached results are appropriate for screening and router design; final claims should rely on the live validation block.",
    ]
    (reports_dir / "baseline_compression_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote baseline compression results")
    print(all_rows.sort_values(["target_compression_percent", "pass5"], ascending=[True, False]).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
