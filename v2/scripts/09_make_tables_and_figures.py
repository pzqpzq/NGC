#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from ngc_v2_utils import ensure_dir, write_json


def table_block(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "No rows."
    return "```text\n" + df[cols].to_string(index=False) + "\n```"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--v1-root", type=Path, default=Path(os.environ.get("NGC_V1_ROOT", repo_root / "v1")))
    parser.add_argument("--v2-root", type=Path, default=Path(os.environ.get("NGC_V2_ROOT", repo_root / "v2")))
    args = parser.parse_args()

    reports = ensure_dir(args.v2_root / "reports")
    results = ensure_dir(args.v2_root / "results")
    figures = ensure_dir(args.v2_root / "figures")

    baseline = read_csv(results / "baseline_compression_results.csv")
    router = read_csv(results / "router_holdout_results.csv")
    oracle = read_csv(results / "oracle_headroom.csv")
    role_adapt = read_csv(results / "role_ablation_adaptive.csv")
    live_v1 = read_csv(args.v1_root / "outputs/12_adjudication/final_live_summary_adjudicated.csv")

    router_best_fixed = pd.DataFrame()
    router_best_adaptive = pd.DataFrame()
    if not router.empty:
        fixed_names = ["best_global_fixed_law", "fixed_transfer_law", "benchmark_specific_fixed_law", "model_specific_fixed_law"]
        router_best_fixed = router[router["selector"].isin(fixed_names)].sort_values(["pass5", "top1_acc"], ascending=False).head(1)
        router_best_adaptive = router[
            router["selector"].str.contains("router|cost_aware|query_tfidf", regex=True, na=False)
        ].sort_values(["pass5", "top1_acc"], ascending=False).head(1)

    baseline_insert = pd.DataFrame()
    if not baseline.empty:
        baseline_insert = (
            baseline[baseline["model"].eq("ALL") & baseline["dataset"].eq("ALL")]
            .sort_values(["target_compression_percent", "pass5"], ascending=[True, False])
            .groupby("target_compression_percent", as_index=False)
            .head(6)
        )
        baseline_insert.to_csv(results / "table_insert_baseline_top_rows.csv", index=False)

    router_insert = router.head(16) if not router.empty else pd.DataFrame()
    if not router_insert.empty:
        router_insert.to_csv(results / "table_insert_router_main.csv", index=False)
    if not oracle.empty:
        oracle.to_csv(results / "table_insert_oracle_headroom.csv", index=False)
    if not role_adapt.empty:
        role_adapt.to_csv(results / "table_insert_role_ablation_adaptive.csv", index=False)

    conclusion = "The v2 cached experiments are not complete enough to strengthen the manuscript beyond the live-validation claim."
    verdict = "No router verdict available."
    if not router_best_fixed.empty and not router_best_adaptive.empty:
        fixed = router_best_fixed.iloc[0]
        adaptive = router_best_adaptive.iloc[0]
        diff = float(adaptive["pass5"] - fixed["pass5"])
        if diff > 0.02:
            verdict = "Adaptive routing beats the best fixed law on cached random held-out pass@5."
        elif diff > 0.005:
            verdict = "Adaptive routing gives a small cached held-out gain over the best fixed law."
        else:
            verdict = "The fixed law remains the strongest cached Pareto point; adaptive routers do not yet show a robust pass@5 gain."
        conclusion = (
            f"Best fixed selector `{fixed['selector']}` reaches pass@5={fixed['pass5']:.4f}; "
            f"best adaptive selector `{adaptive['selector']}` reaches pass@5={adaptive['pass5']:.4f}. "
            f"{verdict}"
        )

    live_note = "No v2 executable live validation has been completed yet."
    if not live_v1.empty:
        live_note = "Existing NGC-v1 adjudicated live validation is available and should remain the executable evidence anchor until v2 live jobs finish."

    files_needed = [
        str(results / "baseline_compression_results.csv"),
        str(results / "router_holdout_results.csv"),
        str(results / "oracle_headroom.csv"),
        str(results / "router_cost_overhead.csv"),
        str(results / "role_ablation_adaptive.csv"),
        str(results / "role_ablation_fixed.csv"),
        str(figures / "baseline_accuracy_compression_curve.pdf"),
        str(figures / "router_fixed_vs_adaptive_by_benchmark.pdf"),
        str(figures / "router_oracle_gap.pdf"),
        str(figures / "router_cost_accuracy_pareto.pdf"),
        str(figures / "role_ablation_barplot.pdf"),
    ]

    lines = [
        "# Final Experiment Digest For Paper",
        "",
        "## Main Conclusion",
        "",
        conclusion,
        "",
        "## Router Verdict",
        "",
        verdict,
        "",
        "## Table Values To Insert",
        "",
        "### Baseline Compression",
        "",
        table_block(
            baseline_insert,
            [
                "target_compression_percent",
                "method",
                "exactness",
                "n_queries",
                "raw_acc",
                "top1_acc",
                "pass5",
            ],
        )
        if not baseline_insert.empty
        else "Baseline table not available.",
        "",
        "### Router Main",
        "",
        table_block(router_insert, ["selector", "n_queries", "raw_acc", "top1_acc", "pass3", "pass5", "pass10", "candidate_auc"])
        if not router_insert.empty
        else "Router table not available.",
        "",
        "## Figure Paths",
        "",
        "\n".join(f"- `{path}`" for path in files_needed if Path(path).exists()),
        "",
        "## Baseline Caveats",
        "",
        "- Raw dense, cached Vanilla SVD, cached Basis Sharing, and cached NGC rows are exact cached outputs.",
        "- ASVD-lite, Dobi-SVD-lite, SVD-LLM-V2-lite, and SoLA-lite are simplified proxy baselines over the cached candidate bank.",
        "- Full public reproduction of those baselines remains necessary for strict final baseline claims.",
        "",
        "## Live Validation Status",
        "",
        live_note,
        "",
        "## Recommended Wording",
        "",
        "Fixed social laws reveal a strong transferable selection prior over internal communication topologies. "
        "Cached router experiments quantify whether query-conditioned selection can exploit additional oracle headroom; "
        "the paper should phrase the adaptive-router claim according to the held-out and live-validation evidence rather than assuming a universal win.",
        "",
        "## Limitations",
        "",
        "- Cached labels are useful for method development but do not replace executable held-out inference.",
        "- Simplified baselines must be labeled as such.",
        "- Test labels were not used for router tuning in these scripts, but future symbolic law search must preserve nested validation.",
        "",
        "## Source Data Files Needed",
        "",
        "\n".join(f"- `{path}`" for path in files_needed),
    ]
    (reports / "final_experiment_digest_for_paper.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        reports / "final_digest_manifest.json",
        {
            "baseline_rows": int(len(baseline)),
            "router_rows": int(len(router)),
            "oracle_rows": int(len(oracle)),
            "role_adaptive_rows": int(len(role_adapt)),
            "figure_paths_existing": [path for path in files_needed if Path(path).exists()],
        },
    )
    print("Final digest written")
    print(reports / "final_experiment_digest_for_paper.md")


if __name__ == "__main__":
    main()
