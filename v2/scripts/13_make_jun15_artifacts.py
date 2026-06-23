#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jun15_utils import DEFAULT_V2_ROOT, ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Jun 15 mechanistic-control artifacts and write manuscript-facing summary.")
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_V2_ROOT / "results/jun15_mechanistic_controls")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def markdown_table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int = 20) -> str:
    if df.empty:
        return "No completed rows yet."
    view = df[cols] if cols else df
    return "```text\n" + view.head(max_rows).to_string(index=False) + "\n```"


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    ensure_dir(path.parent)
    if df.empty:
        path.write_text("% No rows available yet.\n", encoding="utf-8")
        return
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\footnotesize",
        df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}"),
        "\\end{table}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(args.out_root)
    report_dir = ensure_dir(args.v2_root / "reports")
    table_dir = ensure_dir(out_root / "tables")
    source_dir = ensure_dir(out_root / "source_data")

    phase_region = read_csv(out_root / "phase_region_bootstrap.csv")
    phase_matched = read_csv(out_root / "phase_matched_reconstruction_bootstrap.csv")
    phase_stats = read_csv(out_root / "phase_logistic_surface_stats.csv")
    causal_summary = read_csv(out_root / "causal_role_perturbation/causal_role_perturbation_summary.csv")
    causal_boot = read_csv(out_root / "causal_role_perturbation/causal_role_perturbation_bootstrap.csv")
    cost_summary = read_csv(out_root / "end_to_end_cost_audit/end_to_end_cost_audit_summary.csv")

    for name, df in [
        ("phase_region_bootstrap", phase_region),
        ("phase_matched_reconstruction_bootstrap", phase_matched),
        ("phase_logistic_surface_stats", phase_stats),
        ("causal_role_perturbation_summary", causal_summary),
        ("causal_role_perturbation_bootstrap", causal_boot),
        ("end_to_end_cost_audit_summary", cost_summary),
    ]:
        if not df.empty:
            df.to_csv(source_dir / f"{name}.csv", index=False)

    write_latex_table(phase_matched, table_dir / "table_jun15_phase_matched_reconstruction.tex", "Negotiated-stability matched reconstruction contrast.", "tab:jun15_phase_matched")
    write_latex_table(causal_boot, table_dir / "table_jun15_causal_role_perturbation.tex", "Causal perturbation of broker and settlement roles.", "tab:jun15_causal_role")
    write_latex_table(cost_summary, table_dir / "table_jun15_end_to_end_cost_audit.tex", "End-to-end wall-clock and memory audit.", "tab:jun15_cost_audit")

    figure_paths = [
        args.v2_root / "figures/jun15_phase_diagram_negotiated_stability.pdf",
        args.v2_root / "figures/jun15_causal_role_perturbation.pdf",
        args.v2_root / "figures/jun15_end_to_end_cost_audit.pdf",
    ]
    lines = [
        "# Jun 15 Mechanistic-Control Summary",
        "",
        "This report is generated from the Jun 15 mechanistic-control scripts. Claims below are descriptive and should be revised only after checking the completed source rows.",
        "",
        "## Phase Diagram",
        "",
        markdown_table(phase_region),
        "",
        "Matched reconstruction contrast:",
        "",
        markdown_table(phase_matched),
        "",
        "Logistic surface diagnostics:",
        "",
        markdown_table(phase_stats),
        "",
        "## Causal Role Perturbation",
        "",
        markdown_table(causal_summary, ["model", "dataset", "selector", "condition", "n_rows", "accuracy", "mean_f1", "median_seconds"] if not causal_summary.empty else None),
        "",
        "Paired contrasts:",
        "",
        markdown_table(causal_boot),
        "",
        "## End-To-End Cost Audit",
        "",
        markdown_table(
            cost_summary,
            [
                "model",
                "dataset",
                "condition",
                "n_rows",
                "accuracy",
                "median_generation_seconds",
                "median_tokens_per_second",
                "median_peak_memory_gb",
                "total_materialize_seconds",
                "ranking_median_seconds_per_query",
            ]
            if not cost_summary.empty and "ranking_median_seconds_per_query" in cost_summary
            else None,
        ),
        "",
        "## Figures",
        "",
        "\n".join(f"- `{path}`" for path in figure_paths if path.exists()) or "No figures generated yet.",
        "",
        "## Conservative Manuscript Guidance",
        "",
        "- Use the phase diagram as a visual diagnostic of where correct constitutions concentrate; do not force an inverted-U claim if the bootstrap/statistical rows do not support it.",
        "- Treat causal perturbation as the main executable mechanism check only after intact and perturbed runs cover the same query/rank pairs.",
        "- State deployment cost as measured ranking, materialization, and decoding overhead. Do not describe NGC top-k as zero-overhead because top-k requires multiple topology-conditioned generations.",
    ]
    summary_path = report_dir / "jun15_mechanistic_control_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        out_root / "jun15_artifact_manifest.json",
        {
            "summary": str(summary_path),
            "source_data_dir": str(source_dir),
            "table_dir": str(table_dir),
            "figures": [str(p) for p in figure_paths if p.exists()],
        },
    )
    print(summary_path)


if __name__ == "__main__":
    main()
