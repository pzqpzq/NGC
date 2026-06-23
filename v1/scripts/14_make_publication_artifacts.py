#!/usr/bin/env python3
"""Build Day 5 publication artifacts for the NGC social-law supplement.

The script reads completed Day 1--4 outputs and writes manuscript-facing
tables, figures, source data, a reproducibility manifest, an empirical report,
and an editable notebook under outputs/final. It intentionally does not mutate
NGC-v0 assets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--v1-root", type=Path, default=Path(os.environ.get("NGC_V1_ROOT", repo_root / "v1")))
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def pct(x: Any, digits: int = 1) -> str:
    return f"{100.0 * float(x):.{digits}f}"


def pp(x: Any, digits: int = 2) -> str:
    return f"{100.0 * float(x):+.{digits}f}"


def num(x: Any, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def sec(x: Any) -> str:
    value = float(x)
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3f}"


def tex_escape(text: Any) -> str:
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def tex_table(headers: list[str], rows: list[list[Any]], align: str | None = None, caption: str | None = None, label: str | None = None) -> str:
    align = align or ("l" + "r" * (len(headers) - 1))
    lines = []
    if caption:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
    lines.extend(
        [
            r"\begin{tabular}{" + align + "}",
            r"\toprule",
            " & ".join(tex_escape(h) for h in headers) + r" \\",
            r"\midrule",
        ]
    )
    for row in rows:
        lines.append(" & ".join(tex_escape(v) for v in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if caption:
        lines.append(r"\caption{" + tex_escape(caption) + "}")
        if label:
            lines.append(r"\label{" + tex_escape(label) + "}")
        lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    ensure(dst.parent)
    shutil.copy2(src, dst)
    return True


def save_df(df: pd.DataFrame, path: Path) -> None:
    ensure(path.parent)
    df.to_csv(path, index=False)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8dee9", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def make_table_selector(v1: Path, table_dir: Path) -> pd.DataFrame:
    df = read_csv(v1 / "outputs/03_selector/selector_metrics.csv")
    keep = [
        "random_expected",
        "reconstruction_only",
        "fixed_C316",
        "hist_gradient_boosting_negotiated_observables",
        "calibrated_ensemble",
        "logistic_scalar_law_only",
    ]
    sub = df[df["selector"].isin(keep)].copy()
    order = {name: idx for idx, name in enumerate(keep)}
    sub["order"] = sub["selector"].map(order)
    sub = sub.sort_values("order")
    label = {
        "random_expected": "Random expected",
        "reconstruction_only": "Reconstruction only",
        "fixed_C316": "Fixed C316 law",
        "hist_gradient_boosting_negotiated_observables": "Negotiated observables HGB",
        "calibrated_ensemble": "Calibrated ensemble",
        "logistic_scalar_law_only": "Scalar law only",
    }
    rows = []
    for _, r in sub.iterrows():
        rows.append(
            [
                label.get(r["selector"], r["selector"]),
                int(r["n_queries"]),
                pct(r["raw_pass"]),
                pct(r["pass1"]),
                pct(r["pass5"]),
                pp(r["gain_pass5_vs_raw"]),
                "" if pd.isna(r.get("candidate_auc")) else num(r["candidate_auc"], 3),
                "" if pd.isna(r.get("candidate_average_precision")) else num(r["candidate_average_precision"], 3),
            ]
        )
    write_text(
        table_dir / "table_selector_comparison.tex",
        tex_table(
            ["Selector", "Queries", "Raw", "Top-1", "Top-5", "Top-5 gain", "Candidate AUC", "AP"],
            rows,
            align="lrrrrrrr",
            caption="Held-out topology-selection comparison. Values are percentages except AUC/AP.",
            label="tab:ngc_selector_comparison",
        ),
    )
    return sub.drop(columns=["order"])


def make_table_live(v1: Path, table_dir: Path) -> pd.DataFrame:
    by_pair = read_csv(v1 / "outputs/08_live_grid/pooled_live_by_pair.csv")
    boot = read_csv(v1 / "outputs/12_adjudication/final_live_bootstrap_adjudicated.csv")
    final = read_csv(v1 / "outputs/12_adjudication/final_live_summary_adjudicated.csv")
    rows = []
    for _, r in by_pair.iterrows():
        rows.append(
            [
                r["pair"],
                int(r["n_queries"]),
                pct(r["live_raw"]),
                pct(r["live_top1"]),
                pct(r["live_topk"]),
                pp(float(r["live_topk"]) - float(r["live_raw"])),
                pct(r["live_topk_best_f1"]),
            ]
        )
    raw = final.loc[final["metric"] == "final_live_raw", "mean"].iloc[0]
    top1 = final.loc[final["metric"] == "final_live_top1", "mean"].iloc[0]
    topk = final.loc[final["metric"] == "final_live_topk", "mean"].iloc[0]
    n = int(final["n"].max())
    rows.append(["Pooled adjudication-ready", n, pct(raw), pct(top1), pct(topk), pp(float(topk) - float(raw)), ""])
    write_text(
        table_dir / "table_live_validation.tex",
        tex_table(
            ["Pair", "Queries", "Raw", "Top-1", "Top-5", "Top-5 - raw", "Best F1"],
            rows,
            align="lrrrrrr",
            caption="Expanded live validation. The pooled row includes the additional earlier qwen3-4b/Hotpot-QA run and is automatic-equivalent until manual adjudication fields are filled.",
            label="tab:ngc_live_validation",
        ),
    )
    return pd.concat([by_pair.assign(section="pair"), boot.assign(section="bootstrap")], ignore_index=True, sort=False)


def make_table_matched(v1: Path, table_dir: Path) -> pd.DataFrame:
    summary = read_csv(v1 / "outputs/04_matched_controls/matched_control_summary.csv")
    boot = read_csv(v1 / "outputs/04_matched_controls/matched_control_bootstrap.csv")
    merged = summary.merge(boot[["control_type", "ci_low", "ci_high", "p_boot"]], on="control_type", how="left")
    rows = []
    for _, r in merged.iterrows():
        rows.append(
            [
                r["control_type"].replace("_", " "),
                int(r["n_pairs"]),
                pct(r["selected_correct_rate"]),
                pct(r["control_correct_rate"]),
                pp(r["delta_correct"]),
                f"[{pp(r['ci_low'])}, {pp(r['ci_high'])}]",
                num(r["p_boot"], 3),
            ]
        )
    write_text(
        table_dir / "table_matched_law_breaking.tex",
        tex_table(
            ["Matched control", "Pairs", "Law", "Control", "Delta", "95% CI", "p_boot"],
            rows,
            align="lrrrrrr",
            caption="Matched law-breaking controls. Current confidence intervals overlap zero, so these results should be reported as mixed rather than decisive causal evidence.",
            label="tab:ngc_matched_controls",
        ),
    )
    return merged


def make_table_role_knockout(v1: Path, table_dir: Path) -> pd.DataFrame:
    summary = read_csv(v1 / "outputs/06_role_knockout/role_knockout_summary.csv")
    boot = read_csv(v1 / "outputs/06_role_knockout/role_knockout_bootstrap.csv")
    passk = boot[boot["metric"] == "passk"].copy()
    rows = []
    for _, r in summary.iterrows():
        label = str(r["knockout"]).replace("_", " ")
        contrast = f"full_law_minus_{r['knockout']}"
        b = passk[passk["contrast"] == contrast]
        if r["knockout"] == "full_law":
            delta, ci, p = "", "", ""
        elif len(b):
            br = b.iloc[0]
            delta = pp(br["diff"])
            ci = f"[{pp(br['ci_low'])}, {pp(br['ci_high'])}]"
            p = num(br["p_boot"], 3)
        else:
            delta, ci, p = "", "", ""
        rows.append([label, int(r["n_queries"]), pct(r["pass1"]), pct(r["pass5"]), pct(r["mean_at_5"]), delta, ci, p])
    write_text(
        table_dir / "table_role_knockout.tex",
        tex_table(
            ["Condition", "Queries", "Top-1", "Top-5", "Mean@5", "Full - condition", "95% CI", "p_boot"],
            rows,
            align="lrrrrrrr",
            caption="Role-knockout cached analysis. Positive full-minus-drop contrasts indicate roles whose removal degrades recovery.",
            label="tab:ngc_role_knockout",
        ),
    )
    return pd.concat([summary.assign(section="summary"), boot.assign(section="bootstrap")], ignore_index=True, sort=False)


def make_table_failure(v1: Path, table_dir: Path) -> pd.DataFrame:
    df = read_csv(v1 / "outputs/07_failure_taxonomy/failure_mode_summary.csv")
    agg = df.groupby("failure_class", as_index=False)["n"].sum().sort_values("n", ascending=False)
    rows = [[r["failure_class"], int(r["n"])] for _, r in agg.iterrows()]
    write_text(
        table_dir / "table_failure_taxonomy.tex",
        tex_table(
            ["Failure/success class", "Count"],
            rows,
            align="lr",
            caption="Failure taxonomy of fixed-law selected candidates across cached held-out queries.",
            label="tab:ngc_failure_taxonomy",
        ),
    )
    return agg


def make_table_overhead(v1: Path, table_dir: Path) -> pd.DataFrame:
    df = read_csv(v1 / "outputs/11_overhead/controlled_overhead_overall.csv")
    label = {
        "C316_ranking_only": "C316 ranking only",
        "reconstruction_ranking_only": "Reconstruction ranking only",
        "raw_generation": "Raw generation",
        "ngc_top1_generation": "NGC top-1 generation",
        "ngc_top5_generation": "NGC top-5 generation",
    }
    rows = []
    order = ["C316_ranking_only", "reconstruction_ranking_only", "raw_generation", "ngc_top1_generation", "ngc_top5_generation"]
    for condition in order:
        r = df[df["condition"] == condition].iloc[0]
        rows.append(
            [
                label.get(condition, condition),
                int(r["total_rows"]),
                sec(r["median_generation_seconds"]),
                num(r["median_peak_memory_gb"], 2),
                sec(r["total_model_load_seconds"]),
                sec(r["total_wtp_load_seconds"]),
                sec(r["total_materialize_seconds"]),
            ]
        )
    write_text(
        table_dir / "table_controlled_overhead.tex",
        tex_table(
            ["Condition", "Rows", "Median sec", "Peak GB", "Model-load sec", "WTP-load sec", "Materialize sec"],
            rows,
            align="lrrrrrr",
            caption="Controlled overhead benchmark over four Qwen3 model/dataset pairs.",
            label="tab:ngc_controlled_overhead",
        ),
    )
    return df


def make_table_stability(v1: Path, table_dir: Path) -> pd.DataFrame:
    quad = read_csv(v1 / "outputs/05_negotiated_stability/quadratic_tests.csv")
    mid = read_csv(v1 / "outputs/05_negotiated_stability/mid_departure_contrast.csv")
    keep = quad[quad["term"].isin(["departure_index_std", "departure_sq", "stability_guard_std", "reconstruction_error_log10_std"])].copy()
    rows = []
    for _, r in keep.iterrows():
        rows.append([r["term"], num(r["coefficient"], 4), f"[{num(r['ci_low'], 4)}, {num(r['ci_high'], 4)}]", num(r["p_boot_two_sided_zero"], 3)])
    mr = mid.iloc[0]
    rows.append([mr["contrast"], pp(mr["diff"]), f"[{pp(mr['ci_low'])}, {pp(mr['ci_high'])}]", num(mr["p_boot"], 3)])
    write_text(
        table_dir / "table_bounded_disagreement.tex",
        tex_table(
            ["Term/contrast", "Estimate", "95% CI", "p_boot"],
            rows,
            align="lrrr",
            caption="Bounded-disagreement diagnostics. Current results do not support the simple inverted-U hypothesis.",
            label="tab:ngc_bounded_disagreement",
        ),
    )
    return pd.concat([keep.assign(section="quadratic"), mid.assign(section="mid_contrast")], ignore_index=True, sort=False)


def make_evidence_stack(v1: Path, figure_dir: Path) -> None:
    selector = read_csv(v1 / "outputs/03_selector/selector_metrics.csv")
    role = read_csv(v1 / "outputs/06_role_knockout/role_knockout_bootstrap.csv")
    live = read_csv(v1 / "outputs/12_adjudication/final_live_bootstrap_adjudicated.csv")
    overhead = read_csv(v1 / "outputs/11_overhead/controlled_overhead_overall.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    ax = axes[0, 0]
    sel_order = ["fixed_C316", "reconstruction_only", "hist_gradient_boosting_negotiated_observables", "random_expected"]
    sel_labels = ["Fixed C316", "Reconstruction", "Negotiated HGB", "Random exp."]
    sel_vals = [float(selector.loc[selector["selector"] == s, "pass5"].iloc[0]) * 100 for s in sel_order]
    ax.bar(sel_labels, sel_vals, color=["#315c72", "#9aa6b2", "#6d597a", "#c4a484"], width=0.66)
    ax.set_ylabel("Held-out pass@5 (%)")
    ax.set_title("Cached topology selection")
    ax.tick_params(axis="x", rotation=18)
    style_axes(ax)

    ax = axes[0, 1]
    passk = role[(role["metric"] == "passk") & (role["contrast"].isin(["full_law_minus_drop_reconstruction_guard", "full_law_minus_drop_settlement", "full_law_minus_drop_evidence_broker", "full_law_minus_drop_mediator"]))]
    labels = [c.replace("full_law_minus_drop_", "").replace("_", " ") for c in passk["contrast"]]
    y = np.arange(len(passk))
    x = passk["diff"].astype(float).to_numpy() * 100
    lo = (passk["diff"].astype(float) - passk["ci_low"].astype(float)).to_numpy() * 100
    hi = (passk["ci_high"].astype(float) - passk["diff"].astype(float)).to_numpy() * 100
    ax.errorbar(x, y, xerr=[lo, hi], fmt="o", color="#315c72", ecolor="#7393a7", capsize=3)
    ax.axvline(0, color="#4c566a", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Full law - knockout (pp pass@5)")
    ax.set_title("Role-knockout effects")
    style_axes(ax)

    ax = axes[1, 0]
    live_labels = [str(c).replace("Adjudicated ", "").replace(" - ", "\n- ") for c in live["contrast"]]
    y = np.arange(len(live))
    x = live["diff"].astype(float).to_numpy() * 100
    lo = (live["diff"].astype(float) - live["ci_low"].astype(float)).to_numpy() * 100
    hi = (live["ci_high"].astype(float) - live["diff"].astype(float)).to_numpy() * 100
    colors = ["#b56576" if v < 0 else "#315c72" for v in x]
    ax.barh(y, x, color=colors, alpha=0.88)
    ax.errorbar(x, y, xerr=[lo, hi], fmt="none", ecolor="#1f2933", capsize=3, linewidth=1)
    ax.axvline(0, color="#4c566a", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(live_labels)
    ax.set_xlabel("Paired difference (pp)")
    ax.set_title("Live validation contrasts")
    style_axes(ax)

    ax = axes[1, 1]
    oh_order = ["C316_ranking_only", "raw_generation", "ngc_top1_generation", "ngc_top5_generation"]
    oh_labels = ["Rank", "Raw", "NGC top-1", "NGC top-5"]
    oh_vals = [float(overhead.loc[overhead["condition"] == s, "median_generation_seconds"].iloc[0]) for s in oh_order]
    ax.bar(oh_labels, oh_vals, color=["#88c0d0", "#9aa6b2", "#315c72", "#6d597a"], width=0.66)
    ax.set_yscale("log")
    ax.set_ylabel("Median seconds/query or answer")
    ax.set_title("Controlled overhead")
    ax.tick_params(axis="x", rotation=18)
    style_axes(ax)

    fig.suptitle("NGC social-law evidence stack", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figure_dir / "fig_social_law_evidence_stack.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_social_law_evidence_stack.png", bbox_inches="tight")
    plt.close(fig)


def copy_figures(v1: Path, figure_dir: Path) -> list[dict[str, Any]]:
    mapping = {
        "fig_bounded_disagreement": v1 / "outputs/05_negotiated_stability/fig_bounded_departure_curve",
        "fig_matched_law_breaking": v1 / "outputs/04_matched_controls/fig_matched_law_breaking",
        "fig_expanded_live_validation": v1 / "outputs/08_live_grid/fig_expanded_live_validation",
        "fig_failure_taxonomy": v1 / "outputs/07_failure_taxonomy/fig_failure_taxonomy",
        "fig_controlled_overhead": v1 / "outputs/11_overhead/fig_controlled_overhead",
        "fig_role_knockout_effects": v1 / "outputs/06_role_knockout/fig_role_knockout_effects",
        "fig_law_role_heatmap": v1 / "outputs/02_law_atlas/fig_law_role_heatmap",
    }
    copied = []
    for name, stem in mapping.items():
        for ext in [".pdf", ".png"]:
            src = stem.with_suffix(ext)
            dst = figure_dir / f"{name}{ext}"
            copied.append({"name": name + ext, "source": str(src), "dest": str(dst), "copied": copy_if_exists(src, dst)})
    return copied


def build_all_metrics(
    selector: pd.DataFrame,
    matched: pd.DataFrame,
    role: pd.DataFrame,
    stability: pd.DataFrame,
    failure: pd.DataFrame,
    overhead: pd.DataFrame,
    v1: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(group: str, comparison: str, estimate: Any, ci_low: Any = "", ci_high: Any = "", p_boot: Any = "", n: Any = "", interpretation: str = "") -> None:
        rows.append(
            {
                "group": group,
                "comparison": comparison,
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_boot": p_boot,
                "n": n,
                "interpretation": interpretation,
            }
        )

    fixed = selector[selector["selector"] == "fixed_C316"].iloc[0]
    hgb = selector[selector["selector"] == "hist_gradient_boosting_negotiated_observables"].iloc[0]
    rec = selector[selector["selector"] == "reconstruction_only"].iloc[0]
    add("selector", "fixed C316 pass@5", fixed["pass5"], n=fixed["n_queries"], interpretation="Best actionable cached selector by pass@5.")
    add("selector", "fixed C316 gain over raw", fixed["gain_pass5_vs_raw"], n=fixed["n_queries"], interpretation="Large cached top-k recovery relative to raw labels.")
    add("predictive", "HGB negotiated-observable candidate AUC", hgb["candidate_auc"], n=hgb["n_queries"], interpretation="Joint negotiated observables predict candidate correctness.")
    add("predictive", "reconstruction-only candidate AUC", rec["candidate_auc"], n=rec["n_queries"], interpretation="Weaker than joint negotiated observables.")

    for _, r in matched.iterrows():
        add("matched_control", r["control_type"], r["delta_correct"], r["ci_low"], r["ci_high"], r["p_boot"], r["n_pairs"], "Mixed; confidence interval overlaps zero.")

    role_boot = role[(role.get("section") == "bootstrap") if "section" in role.columns else pd.Series(False, index=role.index)]
    if not role_boot.empty:
        for _, r in role_boot[role_boot["metric"].isin(["passk", "mean_at_k"])].iterrows():
            add("role_knockout", f"{r['contrast']} ({r['metric']})", r["diff"], r["ci_low"], r["ci_high"], r["p_boot"], r["n"], "Positive values indicate degradation after role removal.")

    for _, r in stability.iterrows():
        if r.get("section") == "quadratic":
            add("bounded_disagreement", r["term"], r["coefficient"], r["ci_low"], r["ci_high"], r["p_boot_two_sided_zero"], r["n_boot_effective"], "Diagnostic regression term.")
        elif r.get("section") == "mid_contrast":
            add("bounded_disagreement", r["contrast"], r["diff"], r["ci_low"], r["ci_high"], r["p_boot"], r["n_queries"], "Negative result against simple mid-departure hypothesis.")

    live = read_csv(v1 / "outputs/12_adjudication/final_live_bootstrap_adjudicated.csv")
    for _, r in live.iterrows():
        add("live", r["contrast"], r["diff"], r["ci_low"], r["ci_high"], r["p_boot"], r["n"], "Automatic-equivalent until manual overrides are filled.")

    for _, r in overhead.iterrows():
        add("overhead", r["condition"], r["median_generation_seconds"], n=r["total_rows"], interpretation=f"Median peak memory {float(r['median_peak_memory_gb']):.3f} GB.")

    for _, r in failure.head(12).iterrows():
        add("failure_taxonomy", r["failure_class"], r["n"], interpretation="Aggregate count across cached held-out outputs.")

    return pd.DataFrame(rows)


def manuscript_artifact_mapping(v1: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "manuscript_area": "Central evidence figure",
                "recommended_artifact": "outputs/final/figures/fig_social_law_evidence_stack.pdf",
                "recommended_text": "Use as a compact summary of cached selection, role-knockout effects, live contrasts, and controlled overhead.",
                "claim_status": "supported with caveats",
            },
            {
                "manuscript_area": "Predictive social-law claim",
                "recommended_artifact": "outputs/final/tables/table_selector_comparison.tex",
                "recommended_text": "State that negotiated observables jointly predict candidate correctness better than reconstruction-only; do not claim the scalar law score alone is superior.",
                "claim_status": "supported for joint observables",
            },
            {
                "manuscript_area": "Causal law-breaking controls",
                "recommended_artifact": "outputs/final/tables/table_matched_law_breaking.tex",
                "recommended_text": "Report matched controls as mixed. Current CIs include zero and should not be used as decisive causal proof.",
                "claim_status": "not supported in strongest form",
            },
            {
                "manuscript_area": "Bounded-disagreement hypothesis",
                "recommended_artifact": "outputs/final/tables/table_bounded_disagreement.tex and outputs/final/figures/fig_bounded_disagreement.pdf",
                "recommended_text": "Report as diagnostic evidence: stability guard is beneficial, but the simple inverted-U/mid-departure prediction is not supported.",
                "claim_status": "mixed/negative",
            },
            {
                "manuscript_area": "Role-specific perturbations",
                "recommended_artifact": "outputs/final/tables/table_role_knockout.tex and outputs/final/figures/fig_role_knockout_effects.pdf",
                "recommended_text": "Use as the strongest mechanistic evidence. Dropping settlement and reconstruction guard significantly degrades pass@5.",
                "claim_status": "supported",
            },
            {
                "manuscript_area": "Expanded live validation",
                "recommended_artifact": "outputs/final/tables/table_live_validation.tex and outputs/final/figures/fig_expanded_live_validation.pdf",
                "recommended_text": "Claim robust top-5 recovery, not top-1 improvement. Across 375 consolidated live queries, top-5 improves by 9.87 percentage points.",
                "claim_status": "supported for top-k recovery",
            },
            {
                "manuscript_area": "Runtime and overhead",
                "recommended_artifact": "outputs/final/tables/table_controlled_overhead.tex and outputs/final/figures/fig_controlled_overhead.pdf",
                "recommended_text": "Ranking overhead is negligible compared with decoding. Avoid saying full top-5 generation is free, because it evaluates multiple answers.",
                "claim_status": "supported",
            },
            {
                "manuscript_area": "Manual adjudication",
                "recommended_artifact": "outputs/12_adjudication/manual_adjudication_sheet.csv",
                "recommended_text": "The sheet is ready but not filled. Current adjudicated metrics are automatic-equivalent.",
                "claim_status": "pending human review",
            },
        ]
    )


def write_report(v1: Path, report_dir: Path, metrics: pd.DataFrame) -> None:
    inv = read_json(v1 / "outputs/00_inventory/asset_manifest.json")
    obs = read_json(v1 / "outputs/01_observables/manifest.json")
    live_final = read_csv(v1 / "outputs/12_adjudication/final_live_summary_adjudicated.csv")
    live_boot = read_csv(v1 / "outputs/12_adjudication/final_live_bootstrap_adjudicated.csv")
    overhead = read_csv(v1 / "outputs/11_overhead/controlled_overhead_overall.csv")

    raw = live_final.loc[live_final["metric"] == "final_live_raw", "mean"].iloc[0]
    top1 = live_final.loc[live_final["metric"] == "final_live_top1", "mean"].iloc[0]
    topk = live_final.loc[live_final["metric"] == "final_live_topk", "mean"].iloc[0]
    topk_boot = live_boot[live_boot["contrast"] == "Adjudicated top-5 - raw"].iloc[0]
    rank_sec = overhead[overhead["condition"] == "C316_ranking_only"]["median_generation_seconds"].iloc[0]
    raw_sec = overhead[overhead["condition"] == "raw_generation"]["median_generation_seconds"].iloc[0]
    top5_sec = overhead[overhead["condition"] == "ngc_top5_generation"]["median_generation_seconds"].iloc[0]

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs,longtable,graphicx,hyperref}}
\usepackage{{caption}}
\title{{NGC Social-Law Supplemental Experiments: Empirical Report}}
\author{{Generated from the NGC public release workspace}}
\date{{{datetime.now().strftime('%Y-%m-%d %H:%M')}}}
\begin{{document}}
\maketitle

\section*{{Executive summary}}

The supplemental suite covers {obs.get('model_dataset_groups', 'NA')} model--dataset feature banks, {obs.get('rows', 'NA')} candidate rows, and {obs.get('queries', 'NA')} cached held-out queries across {', '.join(inv.get('models', []))} and {', '.join(inv.get('datasets', []))}. The cleanest manuscript claim is top-k recovery under a fixed social-law constitution plus mechanistic support from role knockouts. The current evidence does not justify a broad top-1 improvement claim, nor the strongest causal claim that matched law-breaking controls are consistently worse.

\paragraph{{Most defensible claim.}}
Across 375 consolidated live queries, automatic-equivalent adjudication gives raw accuracy {pct(raw)}\%, NGC top-1 accuracy {pct(top1)}\%, and NGC top-5 accuracy {pct(topk)}\%. The top-5 minus raw contrast is {pp(topk_boot['diff'])} percentage points with 95\% bootstrap CI [{pp(topk_boot['ci_low'])}, {pp(topk_boot['ci_high'])}], p\_boot={num(topk_boot['p_boot'], 3)}. Therefore the manuscript should emphasize recovery: NGC exposes correct reasoning trajectories among a small set of law-selected constitutions.

\paragraph{{Important cautions.}}
Matched controls are mixed and the bounded-disagreement curve does not support a simple inverted-U. The scalar social-law score alone should not be presented as a stronger correctness predictor than reconstruction. Human adjudication has not yet been entered; the current adjudication outputs equal automatic grading.

\section*{{Publication artifacts}}

Tables are in \texttt{{outputs/final/tables}} and figures are in \texttt{{outputs/final/figures}}. Source CSVs and the all-metrics summary are in \texttt{{outputs/final/source\_data}}.

\section*{{Key quantitative results}}

\begin{{itemize}}
\item Cached fixed C316 pass@5: 77.00\% over 300 internal split queries in the selector analysis; cached full-law role-knockout pass@5: 81.57\% over 1,503 held-out queries.
\item Joint negotiated observables HGB candidate AUC: 0.770; reconstruction-only candidate AUC: 0.520.
\item Full law vs dropping settlement: +1.86 pp pass@5, 95\% CI [+0.47, +3.33].
\item Full law vs dropping reconstruction guard: +2.00 pp pass@5, 95\% CI [+0.40, +3.59].
\item Controlled ranking overhead: {sec(rank_sec)} seconds/query; raw generation median: {sec(raw_sec)} seconds; NGC top-5 per-answer median: {sec(top5_sec)} seconds.
\end{{itemize}}

\section*{{Recommended manuscript wording}}

\begin{{quote}}
NGC reveals a recovery phenomenon rather than a solved top-1 routing mechanism. Across consolidated live validation, the correct answer is more often present among the top five law-selected constitutions than in the raw deterministic response, while the current top-1 selector remains below raw generation. Cached role-knockout experiments show that removing settlement and reconstruction-guard terms degrades pass@5, supporting the interpretation of the social law as an operational communication contract over internal communities.
\end{{quote}}

\section*{{Claims to avoid}}

\begin{{itemize}}
\item Avoid: NGC robustly improves top-1 generation.
\item Avoid: the scalar social-law score alone is better than reconstruction.
\item Avoid: matched law-breaking controls prove a causal social-law effect in the current suite.
\item Avoid: bounded disagreement follows a clean inverted-U in the current data.
\item Avoid: manual adjudication has been completed.
\end{{itemize}}

\section*{{Metric index}}

\begin{{longtable}}{{p{{0.17\linewidth}}p{{0.31\linewidth}}p{{0.11\linewidth}}p{{0.11\linewidth}}p{{0.11\linewidth}}p{{0.13\linewidth}}}}
\toprule
Group & Comparison & Estimate & CI low & CI high & p\_boot \\
\midrule
\endhead
"""
    for _, r in metrics.head(80).iterrows():
        tex += f"{tex_escape(r['group'])} & {tex_escape(r['comparison'])} & {tex_escape(r['estimate'])} & {tex_escape(r['ci_low'])} & {tex_escape(r['ci_high'])} & {tex_escape(r['p_boot'])} \\\\\n"
    tex += r"""\bottomrule
\end{longtable}

\end{document}
"""
    write_text(report_dir / "ngc_social_law_empirical_report.tex", tex.strip() + "\n")

    md = f"""# NGC Social-Law Supplemental Experiments: Day 5 Report

Workspace: the local NGC public release workspace or a user-provided `NGC_V1_ROOT`

## Bottom line

The strongest supported claim is top-k recovery with mechanistic role evidence. Across 375 consolidated live queries, raw accuracy is {pct(raw)}%, NGC top-1 is {pct(top1)}%, and NGC top-5 is {pct(topk)}%. The top-5 minus raw contrast is {pp(topk_boot['diff'])} percentage points with 95% CI [{pp(topk_boot['ci_low'])}, {pp(topk_boot['ci_high'])}].

## What to claim

- Claim that NGC exposes correct answers among a small set of law-selected constitutions.
- Claim that joint negotiated observables predict successful topologies better than reconstruction-only.
- Claim that settlement and reconstruction-guard roles matter in cached role-knockout tests.
- Claim that ranking overhead is negligible relative to decoding.

## What not to claim

- Do not claim robust top-1 improvement.
- Do not claim scalar law score alone beats reconstruction.
- Do not claim matched controls or bounded-disagreement are decisive in their strongest forms.
- Do not claim manual adjudication is complete; the current adjudicated metrics are automatic-equivalent.
"""
    write_text(report_dir / "ngc_social_law_empirical_report.md", md)


def write_notebook(out_dir: Path) -> None:
    nb_path = out_dir / "notebooks/day5_editable_figures.ipynb"
    ensure(nb_path.parent)

    def cell(kind: str, source: str) -> dict[str, Any]:
        base: dict[str, Any] = {"cell_type": kind, "metadata": {}, "source": source.strip("\n").splitlines(True)}
        if kind == "code":
            base.update({"execution_count": None, "outputs": []})
        return base

    code = r"""
from pathlib import Path
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get('NGC_V1_ROOT', Path.cwd()))
FINAL = ROOT / 'outputs/final'
FIG = FINAL / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 9,
    'savefig.dpi': 320,
})

def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#d8dee9', linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
"""
    fig_code = r"""
selector = pd.read_csv(ROOT / 'outputs/03_selector/selector_metrics.csv')
role = pd.read_csv(ROOT / 'outputs/06_role_knockout/role_knockout_bootstrap.csv')
live = pd.read_csv(ROOT / 'outputs/12_adjudication/final_live_bootstrap_adjudicated.csv')
overhead = pd.read_csv(ROOT / 'outputs/11_overhead/controlled_overhead_overall.csv')

fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))

ax = axes[0, 0]
sel_order = ['fixed_C316', 'reconstruction_only', 'hist_gradient_boosting_negotiated_observables', 'random_expected']
sel_labels = ['Fixed C316', 'Reconstruction', 'Negotiated HGB', 'Random exp.']
sel_vals = [float(selector.loc[selector.selector == s, 'pass5'].iloc[0]) * 100 for s in sel_order]
ax.bar(sel_labels, sel_vals, color=['#315c72', '#9aa6b2', '#6d597a', '#c4a484'])
ax.set_ylabel('Held-out pass@5 (%)')
ax.set_title('Cached topology selection')
ax.tick_params(axis='x', rotation=18)
style(ax)

ax = axes[0, 1]
passk = role[(role.metric == 'passk') & (role.contrast.isin([
    'full_law_minus_drop_reconstruction_guard',
    'full_law_minus_drop_settlement',
    'full_law_minus_drop_evidence_broker',
    'full_law_minus_drop_mediator',
]))]
labels = [c.replace('full_law_minus_drop_', '').replace('_', ' ') for c in passk.contrast]
y = np.arange(len(passk))
x = passk['diff'].astype(float).to_numpy() * 100
lo = (passk['diff'].astype(float) - passk['ci_low'].astype(float)).to_numpy() * 100
hi = (passk['ci_high'].astype(float) - passk['diff'].astype(float)).to_numpy() * 100
ax.errorbar(x, y, xerr=[lo, hi], fmt='o', color='#315c72', ecolor='#7393a7', capsize=3)
ax.axvline(0, color='#4c566a', linewidth=0.8)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel('Full law - knockout (pp pass@5)')
ax.set_title('Role-knockout effects')
style(ax)

ax = axes[1, 0]
labels = [c.replace('Adjudicated ', '').replace(' - ', '\n- ') for c in live.contrast]
y = np.arange(len(live))
x = live['diff'].astype(float).to_numpy() * 100
lo = (live['diff'].astype(float) - live['ci_low'].astype(float)).to_numpy() * 100
hi = (live['ci_high'].astype(float) - live['diff'].astype(float)).to_numpy() * 100
colors = ['#b56576' if v < 0 else '#315c72' for v in x]
ax.barh(y, x, color=colors, alpha=0.88)
ax.errorbar(x, y, xerr=[lo, hi], fmt='none', ecolor='#1f2933', capsize=3)
ax.axvline(0, color='#4c566a', linewidth=0.8)
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel('Paired difference (pp)')
ax.set_title('Live validation contrasts')
style(ax)

ax = axes[1, 1]
oh_order = ['C316_ranking_only', 'raw_generation', 'ngc_top1_generation', 'ngc_top5_generation']
oh_labels = ['Rank', 'Raw', 'NGC top-1', 'NGC top-5']
oh_vals = [float(overhead.loc[overhead.condition == s, 'median_generation_seconds'].iloc[0]) for s in oh_order]
ax.bar(oh_labels, oh_vals, color=['#88c0d0', '#9aa6b2', '#315c72', '#6d597a'])
ax.set_yscale('log')
ax.set_ylabel('Median seconds/query or answer')
ax.set_title('Controlled overhead')
ax.tick_params(axis='x', rotation=18)
style(ax)

fig.suptitle('NGC social-law evidence stack', fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG / 'fig_social_law_evidence_stack_edited.pdf', bbox_inches='tight')
fig.savefig(FIG / 'fig_social_law_evidence_stack_edited.png', bbox_inches='tight')
plt.show()
"""
    nb = {
        "cells": [
            cell("markdown", "# NGC Day 5 Editable Figures\n\nEdit labels, colors, legends, and panel composition here. The notebook reads source artifacts from `NGC_V1_ROOT` or the current working directory."),
            cell("code", code),
            cell("markdown", "## Evidence-stack figure"),
            cell("code", fig_code),
            cell("markdown", "## Directly copy existing publication figures\n\nRun this cell if you want a local editable staging directory containing the Day 2--4 figures."),
            cell(
                "code",
                r"""
copies = {
    'bounded_disagreement': ROOT / 'outputs/05_negotiated_stability/fig_bounded_departure_curve.pdf',
    'matched_law_breaking': ROOT / 'outputs/04_matched_controls/fig_matched_law_breaking.pdf',
    'expanded_live_validation': ROOT / 'outputs/08_live_grid/fig_expanded_live_validation.pdf',
    'failure_taxonomy': ROOT / 'outputs/07_failure_taxonomy/fig_failure_taxonomy.pdf',
    'controlled_overhead': ROOT / 'outputs/11_overhead/fig_controlled_overhead.pdf',
}
for name, src in copies.items():
    if src.exists():
        shutil.copy2(src, FIG / f'{name}_editable_source.pdf')
        print('copied', src)
""",
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def write_reproduce(v1: Path, out_dir: Path, files: list[dict[str, Any]]) -> None:
    text = f"""# Reproducing Day 5 Artifacts

Workspace:

```bash
cd /path/to/ngc-public-release/v1
```

Regenerate final tables, figures, source data, report, and notebook:

```bash
python -u scripts/14_make_publication_artifacts.py 2>&1 | tee logs/14_final_artifacts.log
```

The script reads only existing Day 1--4 outputs and writes to `outputs/final`.
It does not modify the external `NGC_V0_ROOT` artifact tree.

Important caveats:

- Manual adjudication fields are currently blank, so `outputs/12_adjudication/final_*_adjudicated.csv` is automatic-equivalent.
- The current evidence supports NGC top-k recovery, not robust top-1 improvement.
- Matched controls and bounded-disagreement diagnostics should be reported as mixed/negative in their strongest hypothesized forms.

Generated at: {datetime.now().isoformat(timespec='seconds')}
"""
    write_text(out_dir / "README_reproduce.md", text)


def copy_source_data(v1: Path, source_dir: Path) -> list[dict[str, Any]]:
    rels = [
        "outputs/00_inventory/asset_manifest.json",
        "outputs/01_observables/manifest.json",
        "outputs/02_law_atlas/law_role_signatures.csv",
        "outputs/02_law_atlas/law_transfer_matrix.csv",
        "outputs/03_selector/selector_metrics.csv",
        "outputs/03_selector/selector_paired_bootstrap.csv",
        "outputs/04_matched_controls/matched_control_summary.csv",
        "outputs/04_matched_controls/matched_control_bootstrap.csv",
        "outputs/05_negotiated_stability/mid_departure_contrast.csv",
        "outputs/05_negotiated_stability/quadratic_tests.csv",
        "outputs/06_role_knockout/role_knockout_summary.csv",
        "outputs/06_role_knockout/role_knockout_bootstrap.csv",
        "outputs/07_failure_taxonomy/failure_mode_summary.csv",
        "outputs/07_failure_taxonomy/failure_mode_examples.csv",
        "outputs/08_live_grid/pooled_live_summary.csv",
        "outputs/08_live_grid/pooled_live_bootstrap.csv",
        "outputs/08_live_grid/pooled_live_by_pair.csv",
        "outputs/11_overhead/controlled_overhead_overall.csv",
        "outputs/11_overhead/controlled_overhead_summary.csv",
        "outputs/12_adjudication/final_live_summary_adjudicated.csv",
        "outputs/12_adjudication/final_live_bootstrap_adjudicated.csv",
        "outputs/12_adjudication/manual_adjudication_flag_summary.csv",
        "outputs/12_adjudication/manual_vs_auto_summary.json",
    ]
    copied = []
    for rel in rels:
        src = v1 / rel
        dst = source_dir / rel.replace("outputs/", "").replace("/", "__")
        copied.append({"source": str(src), "dest": str(dst), "copied": copy_if_exists(src, dst)})
    return copied


def main() -> None:
    args = parse_args()
    v1 = args.v1_root.resolve()
    out_dir = (args.out_dir or (v1 / "outputs/final")).resolve()
    table_dir = ensure(out_dir / "tables")
    figure_dir = ensure(out_dir / "figures")
    source_dir = ensure(out_dir / "source_data")
    report_dir = ensure(out_dir / "report")
    ensure(v1 / "logs")

    selector = make_table_selector(v1, table_dir)
    live = make_table_live(v1, table_dir)
    matched = make_table_matched(v1, table_dir)
    role = make_table_role_knockout(v1, table_dir)
    failure = make_table_failure(v1, table_dir)
    overhead = make_table_overhead(v1, table_dir)
    stability = make_table_stability(v1, table_dir)

    make_evidence_stack(v1, figure_dir)
    copied_figs = copy_figures(v1, figure_dir)
    copied_source = copy_source_data(v1, source_dir)

    metrics = build_all_metrics(selector, matched, role, stability, failure, overhead, v1)
    save_df(metrics, out_dir / "all_metrics_summary.csv")
    save_df(metrics, source_dir / "all_metrics_summary.csv")
    mapping = manuscript_artifact_mapping(v1)
    save_df(mapping, out_dir / "day5_manuscript_artifact_mapping.csv")
    save_df(mapping, source_dir / "day5_manuscript_artifact_mapping.csv")
    write_report(v1, report_dir, metrics)
    write_notebook(out_dir)
    write_reproduce(v1, out_dir, copied_figs)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "v1_root": str(v1),
        "out_dir": str(out_dir),
        "tables": sorted(str(p.relative_to(out_dir)) for p in table_dir.glob("*.tex")),
        "figures": sorted(str(p.relative_to(out_dir)) for p in figure_dir.glob("*")),
        "source_data": sorted(str(p.relative_to(out_dir)) for p in source_dir.glob("*")),
        "reports": sorted(str(p.relative_to(out_dir)) for p in report_dir.glob("*")),
        "notebooks": sorted(str(p.relative_to(out_dir)) for p in (out_dir / "notebooks").glob("*.ipynb")),
        "copied_figures": copied_figs,
        "copied_source_data": copied_source,
        "caveats": [
            "Manual adjudication is prepared but not filled; final adjudicated metrics are automatic-equivalent.",
            "Top-k live recovery is supported; top-1 improvement is not supported.",
            "Matched controls and bounded-disagreement diagnostics are mixed or negative in their strongest hypothesized forms.",
        ],
    }
    write_text(out_dir / "reproducibility_manifest.json", json.dumps(manifest, indent=2))

    print(json.dumps({"out_dir": str(out_dir), "n_tables": len(manifest["tables"]), "n_figures": len(manifest["figures"]), "n_metrics": len(metrics)}, indent=2))


if __name__ == "__main__":
    main()
