#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ngc_v1.common import RANDOM_SEED, V0_ROOT, V1_ROOT, ensure_dir, read_observables, run_capture, setup_matplotlib, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled runtime and memory benchmark for NGC live inference.")
    parser.add_argument("--project-dir", type=Path, default=V0_ROOT)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--out-root", type=Path, default=V1_ROOT / "outputs/11_overhead")
    parser.add_argument("--model", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--mode", default="C316")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-prompts", type=int, default=20)
    parser.add_argument("--per-stratum", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--ranking-repeats", type=int, default=200)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def pair_name(model: str, dataset: str) -> str:
    return f"{model}_{dataset}".replace("/", "_").replace("-", "_")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_live_module(project_dir: Path):
    sys.path.insert(0, str(project_dir))
    from nmi_may27_experiments import day4_live_validation as live

    return live


def build_live_manifest(args: argparse.Namespace, live) -> List[Dict[str, Any]]:
    ns = argparse.Namespace(
        project_dir=args.project_dir,
        out_dir=args.out_root,
        model=args.model,
        dataset=args.dataset,
        mode=args.mode,
        top_k=args.top_k,
        seed=args.seed,
        holdout_frac=0.5,
        max_queries=args.max_prompts,
        per_stratum=args.per_stratum,
        max_new_tokens=args.max_new_tokens,
        n_boot=1000,
        skip_live=True,
    )
    return live.build_manifest(ns)


def load_queries(live, dataset: str, manifest: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    test_set = live.load_clean_test_set(dataset)
    out = {}
    for row in manifest:
        item = test_set[int(row["query_key"])]
        out[row["query_key"]] = {"query": item["query"], "answer": item["label"] or item["cot_content"]}
    return out


def benchmark_rankings(args: argparse.Namespace, manifest: List[Dict[str, Any]], out_path: Path) -> List[Dict[str, Any]]:
    df = read_observables(args.v1_root, args.project_dir)
    keys = {f"{args.model}::{args.dataset}::{row['query_key']}" for row in manifest}
    sub = df[df["query_key"].isin(keys)].copy()
    rows = []
    for condition, score_col in [
        ("C316_ranking_only", "law_score"),
        ("reconstruction_ranking_only", "reconstruction_score"),
    ]:
        per_repeat_ms = []
        for repeat in range(args.ranking_repeats):
            t0 = time.perf_counter()
            for _, g in sub.groupby("query_key", sort=False):
                _ = g.sort_values(score_col, ascending=False, kind="mergesort").head(args.top_k)[["candidate_index", score_col]]
            elapsed_ms = 1000.0 * (time.perf_counter() - t0) / max(1, len(keys))
            per_repeat_ms.append(elapsed_ms)
        row = {
            "pair": f"{args.model}/{args.dataset}",
            "model": args.model,
            "dataset": args.dataset,
            "condition": condition,
            "run_type": "ranking_only",
            "n_prompts": len(keys),
            "n_units": args.ranking_repeats,
            "median_seconds": statistics.median(per_repeat_ms) / 1000.0,
            "mean_seconds": float(np.mean(per_repeat_ms) / 1000.0),
            "iqr_seconds": float((np.quantile(per_repeat_ms, 0.75) - np.quantile(per_repeat_ms, 0.25)) / 1000.0),
            "median_new_tokens": 0.0,
            "peak_memory_gb_median": 0.0,
        }
        rows.append(row)
        append_jsonl(
            out_path,
            {
                **row,
                "per_repeat_ms_per_query": per_repeat_ms,
            },
        )
    return rows


def peak_memory_gb(torch_mod: Any) -> float:
    if not torch_mod.cuda.is_available():
        return 0.0
    return float(torch_mod.cuda.max_memory_allocated() / 1024**3)


def reset_peak(torch_mod: Any) -> None:
    if torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()
        torch_mod.cuda.reset_peak_memory_stats()


def run_raw_generation(args: argparse.Namespace, live, manifest: List[Dict[str, Any]], queries: Dict[str, Dict[str, str]], trace_path: Path) -> None:
    t0 = time.perf_counter()
    torch_mod, model, tokenizer, _layers = live.load_model(args.model)
    load_seconds = time.perf_counter() - t0
    reset_peak(torch_mod)
    for idx, row in enumerate(manifest, start=1):
        q = queries[row["query_key"]]
        output, seconds, ntok = live.generate_answer(torch_mod, model, tokenizer, q["query"], args.max_new_tokens)
        stat = live.score_hotpot(output, q["answer"])
        append_jsonl(
            trace_path,
            {
                "pair": f"{args.model}/{args.dataset}",
                "model": args.model,
                "dataset": args.dataset,
                "condition": "raw_generation",
                "run_type": "generation",
                "query_id": row["query_id"],
                "query_key": row["query_key"],
                "rank": 0,
                "wtp_file": "",
                "generation_seconds": seconds,
                "num_new_tokens": ntok,
                "model_load_seconds": load_seconds if idx == 1 else 0.0,
                "wtp_load_seconds": 0.0,
                "materialize_seconds": 0.0,
                "peak_memory_gb": peak_memory_gb(torch_mod),
                **stat,
            },
        )
        print(f"[raw-overhead] {idx}/{len(manifest)} {row['query_id']} sec={seconds:.3f}", flush=True)
    del model
    live.cleanup_cuda(torch_mod)


def topology_jobs(manifest: List[Dict[str, Any]], max_rank: int) -> Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]]:
    jobs: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for row in manifest:
        for topo in row["topologies"]:
            if int(topo["rank"]) <= max_rank:
                jobs[topo["wtp_file"]].append((row, topo))
    return jobs


def run_ngc_generation(
    args: argparse.Namespace,
    live,
    manifest: List[Dict[str, Any]],
    queries: Dict[str, Dict[str, str]],
    max_rank: int,
    condition: str,
    trace_path: Path,
) -> None:
    import nsys_utils.nsys_config as nsys_config

    jobs_by_wtp = topology_jobs(manifest, max_rank=max_rank)
    for topo_idx, (wtp_file, jobs) in enumerate(sorted(jobs_by_wtp.items()), start=1):
        print(f"[{condition}] topology {topo_idx}/{len(jobs_by_wtp)} {wtp_file}; jobs={len(jobs)}", flush=True)
        t0 = time.perf_counter()
        torch_mod, model, tokenizer, _layers = live.load_model(args.model)
        model_load_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        nsys_records = live.load_wtp_records(args.project_dir, args.model, wtp_file)
        wtp_load_seconds = time.perf_counter() - t1
        t2 = time.perf_counter()
        nsys_config.replace_Linear_with_nSys_v2(model, nsys_records)
        model.eval()
        materialize_seconds = time.perf_counter() - t2
        reset_peak(torch_mod)
        for job_idx, (row, topo) in enumerate(jobs, start=1):
            q = queries[row["query_key"]]
            output, seconds, ntok = live.generate_answer(torch_mod, model, tokenizer, q["query"], args.max_new_tokens)
            stat = live.score_hotpot(output, q["answer"])
            append_jsonl(
                trace_path,
                {
                    "pair": f"{args.model}/{args.dataset}",
                    "model": args.model,
                    "dataset": args.dataset,
                    "condition": condition,
                    "run_type": "generation",
                    "query_id": row["query_id"],
                    "query_key": row["query_key"],
                    "rank": int(topo["rank"]),
                    "wtp_file": wtp_file,
                    "generation_seconds": seconds,
                    "num_new_tokens": ntok,
                    "model_load_seconds": model_load_seconds if job_idx == 1 else 0.0,
                    "wtp_load_seconds": wtp_load_seconds if job_idx == 1 else 0.0,
                    "materialize_seconds": materialize_seconds if job_idx == 1 else 0.0,
                    "peak_memory_gb": peak_memory_gb(torch_mod),
                    **stat,
                },
            )
            print(f"[{condition}] topo={topo_idx}/{len(jobs_by_wtp)} job={job_idx}/{len(jobs)} sec={seconds:.3f}", flush=True)
        del model
        del nsys_records
        live.cleanup_cuda(torch_mod)


def summarize_trace(trace_path: Path, out_dir: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "controlled_overhead_raw_traces.csv.gz", index=False, compression="gzip")
    summary_rows = []
    if df.empty:
        return pd.DataFrame()
    for condition, g in df.groupby("condition", sort=False):
        if "ranking_only" in set(g.get("run_type", [])):
            sec = g["median_seconds"].astype(float)
            summary_rows.append(
                {
                    "pair": g["pair"].iloc[0],
                    "model": g["model"].iloc[0],
                    "dataset": g["dataset"].iloc[0],
                    "condition": condition,
                    "n_rows": int(len(g)),
                    "median_generation_seconds": float(sec.median()),
                    "iqr_generation_seconds": float(g["iqr_seconds"].astype(float).median()),
                    "mean_generation_seconds": float(g["mean_seconds"].astype(float).mean()),
                    "median_new_tokens": 0.0,
                    "median_peak_memory_gb": 0.0,
                    "total_model_load_seconds": 0.0,
                    "total_wtp_load_seconds": 0.0,
                    "total_materialize_seconds": 0.0,
                }
            )
            continue
        gen = pd.to_numeric(g["generation_seconds"], errors="coerce")
        summary_rows.append(
            {
                "pair": g["pair"].iloc[0],
                "model": g["model"].iloc[0],
                "dataset": g["dataset"].iloc[0],
                "condition": condition,
                "n_rows": int(len(g)),
                "median_generation_seconds": float(gen.median()),
                "iqr_generation_seconds": float(gen.quantile(0.75) - gen.quantile(0.25)),
                "mean_generation_seconds": float(gen.mean()),
                "median_new_tokens": float(pd.to_numeric(g["num_new_tokens"], errors="coerce").median()),
                "median_peak_memory_gb": float(pd.to_numeric(g["peak_memory_gb"], errors="coerce").median()),
                "total_model_load_seconds": float(pd.to_numeric(g["model_load_seconds"], errors="coerce").sum()),
                "total_wtp_load_seconds": float(pd.to_numeric(g["wtp_load_seconds"], errors="coerce").sum()),
                "total_materialize_seconds": float(pd.to_numeric(g["materialize_seconds"], errors="coerce").sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "controlled_overhead_summary.csv", index=False)
    return summary


def run_one(args: argparse.Namespace) -> Path:
    if not args.model or not args.dataset:
        raise ValueError("--model and --dataset are required unless --aggregate-only is set")
    out_dir = ensure_dir(args.out_root / pair_name(args.model, args.dataset))
    trace_path = out_dir / "controlled_overhead_raw_traces.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    (out_dir / "gpustat_before.txt").write_text(run_capture(["gpustat", "-cp"]), encoding="utf-8")
    (out_dir / "nvidia_smi_before.txt").write_text(run_capture(["nvidia-smi"]), encoding="utf-8")

    live = load_live_module(args.project_dir)
    manifest = build_live_manifest(args, live)
    queries = load_queries(live, args.dataset, manifest)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config["num_manifest_queries"] = len(manifest)
    write_json(out_dir / "controlled_overhead_config.json", config)
    pd.DataFrame(
        [
            {
                "query_id": row["query_id"],
                "query_key": row["query_key"],
                "stratum": row["stratum"],
                "cached_raw": row["cached_raw"],
                "cached_top1": row["cached_top1"],
                "cached_topk": row["cached_topk"],
            }
            for row in manifest
        ]
    ).to_csv(out_dir / "controlled_overhead_manifest.csv", index=False)

    benchmark_rankings(args, manifest, trace_path)
    run_raw_generation(args, live, manifest, queries, trace_path)
    run_ngc_generation(args, live, manifest, queries, max_rank=1, condition="ngc_top1_generation", trace_path=trace_path)
    run_ngc_generation(args, live, manifest, queries, max_rank=args.top_k, condition="ngc_top5_generation", trace_path=trace_path)
    (out_dir / "gpustat_after.txt").write_text(run_capture(["gpustat", "-cp"]), encoding="utf-8")
    (out_dir / "nvidia_smi_after.txt").write_text(run_capture(["nvidia-smi"]), encoding="utf-8")
    summarize_trace(trace_path, out_dir)
    return out_dir


def aggregate(out_root: Path) -> None:
    summaries = []
    for path in sorted(out_root.glob("*/controlled_overhead_summary.csv")):
        summaries.append(pd.read_csv(path))
    if not summaries:
        write_json(out_root / "controlled_overhead_manifest.json", {"status": "no_pair_summaries"})
        return
    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(out_root / "controlled_overhead_summary.csv", index=False)
    overall = (
        all_summary.groupby("condition", as_index=False)
        .agg(
            n_pairs=("pair", "nunique"),
            total_rows=("n_rows", "sum"),
            median_generation_seconds=("median_generation_seconds", "median"),
            median_peak_memory_gb=("median_peak_memory_gb", "median"),
            total_model_load_seconds=("total_model_load_seconds", "sum"),
            total_wtp_load_seconds=("total_wtp_load_seconds", "sum"),
            total_materialize_seconds=("total_materialize_seconds", "sum"),
        )
        .sort_values("condition")
    )
    overall.to_csv(out_root / "controlled_overhead_overall.csv", index=False)
    write_latex_table(overall, out_root / "controlled_overhead_table.tex")
    make_figure(all_summary, out_root)
    write_json(
        out_root / "controlled_overhead_manifest.json",
        {
            "n_pairs": int(all_summary["pair"].nunique()),
            "conditions": sorted(all_summary["condition"].unique().tolist()),
            "outputs": {
                "summary": str(out_root / "controlled_overhead_summary.csv"),
                "overall": str(out_root / "controlled_overhead_overall.csv"),
                "figure": str(out_root / "fig_controlled_overhead.pdf"),
            },
        },
    )


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Condition & Rows & Median gen. sec & Median peak GB & Materialize sec \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['condition'].replace('_', ' ')} & {int(row['total_rows'])} & "
            f"{row['median_generation_seconds']:.4f} & {row['median_peak_memory_gb']:.2f} & "
            f"{row['total_materialize_seconds']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(summary: pd.DataFrame, out_root: Path) -> None:
    plt = setup_matplotlib()
    order = ["raw_generation", "ngc_top1_generation", "ngc_top5_generation", "C316_ranking_only", "reconstruction_ranking_only"]
    plot = summary.copy()
    plot["condition"] = pd.Categorical(plot["condition"], categories=order, ordered=True)
    plot = plot.sort_values(["condition", "pair"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, metric, title, ylabel in [
        (axes[0], "median_generation_seconds", "Generation and ranking latency", "Median seconds per unit"),
        (axes[1], "median_peak_memory_gb", "Peak GPU memory", "Median peak GB"),
    ]:
        pivot = plot.pivot(index="pair", columns="condition", values=metric)
        pivot = pivot[[c for c in order if c in pivot.columns]]
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=25)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_root / "fig_controlled_overhead.pdf")
    fig.savefig(out_root / "fig_controlled_overhead.png", dpi=320)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.project_dir = args.project_dir.resolve()
    args.v1_root = args.v1_root.resolve()
    args.out_root = args.out_root.resolve()
    ensure_dir(args.out_root)
    if args.aggregate_only:
        aggregate(args.out_root)
    else:
        run_one(args)
        aggregate(args.out_root)


if __name__ == "__main__":
    main()
