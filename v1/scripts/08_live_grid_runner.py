#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ngc_v1.common import RANDOM_SEED, V0_ROOT, V1_ROOT, ensure_dir, paired_bootstrap, setup_matplotlib, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or aggregate expanded live NGC validation.")
    parser.add_argument("--project-dir", type=Path, default=V0_ROOT)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--out-root", type=Path, default=V1_ROOT / "outputs/08_live_grid")
    parser.add_argument("--model", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--mode", default="C316")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=75)
    parser.add_argument("--per-stratum", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--enable-thinking", default="false")
    parser.add_argument("--do-sample", default="false")
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def pair_name(model: str, dataset: str) -> str:
    return f"{model}_{dataset}".replace("/", "_").replace("-", "_")


def copy_expected_names(out_dir: Path) -> None:
    mapping = {
        "day4_live_outputs.jsonl": "live_outputs.jsonl",
        "day4_live_summary.csv": "live_summary.csv",
        "day4_live_paired_contrasts.csv": "live_paired_contrasts.csv",
        "day4_live_by_stratum.csv": "live_by_stratum.csv",
        "day4_live_per_query.csv": "live_per_query.csv",
        "day4_live_manifest.csv": "live_manifest.csv",
        "day4_live_bootstrap_ci.csv": "live_bootstrap_ci.csv",
        "fig_day4_live_audit.pdf": "fig_live_audit.pdf",
        "fig_day4_live_audit.png": "fig_live_audit.png",
    }
    for src, dst in mapping.items():
        s = out_dir / src
        if s.exists():
            shutil.copy2(s, out_dir / dst)


def run_one(args: argparse.Namespace) -> Path:
    if not args.model or not args.dataset:
        raise ValueError("--model and --dataset are required unless --aggregate-only is set")
    out_dir = ensure_dir(args.out_root / pair_name(args.model, args.dataset))
    script = args.project_dir / "nmi_may27_experiments/day4_live_validation.py"
    if not script.exists():
        raise FileNotFoundError(script)

    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--project-dir",
        str(args.project_dir),
        "--out-dir",
        str(out_dir),
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--mode",
        args.mode,
        "--top-k",
        str(args.top_k),
        "--max-queries",
        str(args.max_queries),
        "--per-stratum",
        str(args.per_stratum),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
        "--n-boot",
        str(args.n_boot),
    ]
    if args.skip_live:
        cmd.append("--skip-live")

    write_json(
        out_dir / "live_grid_run_config.json",
        {
            "command": cmd,
            "model": args.model,
            "dataset": args.dataset,
            "mode": args.mode,
            "top_k": args.top_k,
            "max_queries": args.max_queries,
            "per_stratum": args.per_stratum,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "skip_live": args.skip_live,
            "note": "Underlying Day 4 script hard-codes enable_thinking=False and do_sample=False.",
        },
    )
    print("[live-grid] running", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    expected = [out_dir / "day4_live_manifest.csv", out_dir / "day4_live_summary.csv"]
    if not args.skip_live:
        expected.extend([out_dir / "day4_live_outputs.jsonl", out_dir / "day4_live_per_query.csv"])
    if proc.returncode != 0 and not all(path.exists() for path in expected):
        raise SystemExit(proc.returncode)
    if proc.returncode != 0:
        print(
            "[live-grid] underlying Day 4 script returned nonzero after writing expected outputs; "
            "continuing because this is the known out-dir relative_to print issue.",
            flush=True,
        )
    copy_expected_names(out_dir)
    return out_dir


def metric_mean(rows: pd.DataFrame, metric: str) -> float:
    vals = pd.to_numeric(rows[metric], errors="coerce")
    return float(vals.mean())


def aggregate(out_root: Path, n_boot: int, seed: int) -> None:
    ensure_dir(out_root)
    all_per_query: List[pd.DataFrame] = []
    by_pair_rows: List[Dict[str, Any]] = []
    for per_query_path in sorted(out_root.glob("*/day4_live_per_query.csv")):
        pair_dir = per_query_path.parent
        config_path = pair_dir / "day4_config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        df = pd.read_csv(per_query_path)
        if "live_raw" not in df or df["live_raw"].replace("", np.nan).dropna().empty:
            continue
        df["model"] = config.get("model", pair_dir.name)
        df["dataset"] = config.get("dataset", pair_dir.name)
        df["pair"] = df["model"].astype(str) + "/" + df["dataset"].astype(str)
        all_per_query.append(df)
        by_pair_rows.append(
            {
                "pair": df["pair"].iloc[0],
                "model": df["model"].iloc[0],
                "dataset": df["dataset"].iloc[0],
                "n_queries": int(len(df)),
                "cached_raw": metric_mean(df, "cached_raw"),
                "cached_top1": metric_mean(df, "cached_top1"),
                "cached_topk": metric_mean(df, "cached_topk"),
                "live_raw": metric_mean(df, "live_raw"),
                "live_top1": metric_mean(df, "live_top1"),
                "live_topk": metric_mean(df, "live_topk"),
                "live_raw_f1": metric_mean(df, "live_raw_f1"),
                "live_top1_f1": metric_mean(df, "live_top1_f1"),
                "live_topk_best_f1": metric_mean(df, "live_topk_best_f1"),
            }
        )

    if not all_per_query:
        write_json(out_root / "pooled_live_manifest.json", {"status": "no_completed_live_runs"})
        print("[live-grid] no completed live runs found")
        return

    pooled = pd.concat(all_per_query, ignore_index=True)
    pooled.to_csv(out_root / "pooled_live_per_query.csv", index=False)
    by_pair = pd.DataFrame(by_pair_rows)
    by_pair.to_csv(out_root / "pooled_live_by_pair.csv", index=False)

    metrics = ["cached_raw", "cached_top1", "cached_topk", "live_raw", "live_top1", "live_topk"]
    summary_rows = []
    for metric in metrics + ["live_raw_f1", "live_top1_f1", "live_topk_best_f1"]:
        vals = pd.to_numeric(pooled[metric], errors="coerce").dropna()
        summary_rows.append({"metric": metric, "mean": float(vals.mean()), "n": int(len(vals))})
    pd.DataFrame(summary_rows).to_csv(out_root / "pooled_live_summary.csv", index=False)

    contrasts = []
    for left, right, label in [
        ("cached_top1", "cached_raw", "Cached top-1 - cached raw"),
        ("cached_topk", "cached_raw", "Cached top-5 - cached raw"),
        ("live_top1", "live_raw", "Live top-1 - live raw"),
        ("live_topk", "live_raw", "Live top-5 - live raw"),
        ("live_topk", "live_top1", "Live top-5 - live top-1"),
    ]:
        stat = paired_bootstrap(
            pd.to_numeric(pooled[left], errors="coerce"),
            pd.to_numeric(pooled[right], errors="coerce"),
            n_boot=n_boot,
            seed=seed,
        )
        stat.update({"contrast": label})
        contrasts.append(stat)
    pd.DataFrame(contrasts).to_csv(out_root / "pooled_live_bootstrap.csv", index=False)
    make_aggregate_figure(by_pair, out_root)
    write_json(
        out_root / "pooled_live_manifest.json",
        {
            "completed_pairs": by_pair["pair"].tolist(),
            "n_pairs": int(len(by_pair)),
            "n_queries": int(len(pooled)),
            "n_boot": int(n_boot),
            "seed": int(seed),
        },
    )
    print(f"[live-grid] aggregated {len(by_pair)} pairs and {len(pooled)} queries")


def make_aggregate_figure(by_pair: pd.DataFrame, out_root: Path) -> None:
    plt = setup_matplotlib()
    by_pair = by_pair.sort_values("pair")
    x = np.arange(len(by_pair))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - width, 100 * by_pair["live_raw"], width, label="Raw", color="#6E7781")
    ax.bar(x, 100 * by_pair["live_top1"], width, label="NGC top-1", color="#3B6FB6")
    ax.bar(x + width, 100 * by_pair["live_topk"], width, label="NGC top-5", color="#2C9C95")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n(n={n})" for p, n in zip(by_pair["pair"], by_pair["n_queries"])], rotation=0)
    ax.set_ylabel("Live pass rate (%)")
    ax.set_title("Expanded live validation by model-dataset pair")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_root / "fig_expanded_live_validation.pdf")
    fig.savefig(out_root / "fig_expanded_live_validation.png", dpi=320)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.project_dir = args.project_dir.resolve()
    args.v1_root = args.v1_root.resolve()
    args.out_root = args.out_root.resolve()
    if args.aggregate_only:
        aggregate(args.out_root, args.n_boot, args.seed)
    else:
        run_one(args)
        aggregate(args.out_root, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
