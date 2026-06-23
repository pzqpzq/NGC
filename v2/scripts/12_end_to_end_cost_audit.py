#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jun15_utils import (
    DEFAULT_V0_ROOT,
    DEFAULT_V2_ROOT,
    RANDOM_SEED,
    append_jsonl,
    build_query_manifest,
    cleanup_cuda,
    ensure_dir,
    generate_answer,
    load_model,
    load_queries,
    load_router_predictions,
    load_wtp_records,
    materialize_nsys,
    peak_memory_gb,
    reset_peak,
    score_prediction,
    selector_score_column,
    wtp_storage_stats,
    write_json,
)
from ngc_v2_utils import setup_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jun 15 end-to-end wall-clock and memory audit.")
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_V0_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_V2_ROOT / "results/jun15_mechanistic_controls/end_to_end_cost_audit")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--dataset", default="gpqa")
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--ranking-repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--conditions", default="raw_dense,vanilla_svd_top1,basis_sharing_top1,fixed_ngc_top1,fixed_ngc_top5,adaptive_ngc_top1,adaptive_ngc_top5")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def condition_manifest(df: pd.DataFrame, args: argparse.Namespace, condition: str) -> list[dict[str, Any]]:
    if condition == "vanilla_svd_top1":
        return build_query_manifest(df, args.model, args.dataset, selector="reconstruction", top_k=1, max_queries=args.max_queries, family="Vanilla SVD", seed=args.seed)
    if condition == "vanilla_svd_top5":
        return build_query_manifest(df, args.model, args.dataset, selector="reconstruction", top_k=args.top_k, max_queries=args.max_queries, family="Vanilla SVD", seed=args.seed)
    if condition == "basis_sharing_top1":
        return build_query_manifest(df, args.model, args.dataset, selector="reconstruction", top_k=1, max_queries=args.max_queries, family="Basis Sharing", seed=args.seed)
    if condition == "basis_sharing_top5":
        return build_query_manifest(df, args.model, args.dataset, selector="reconstruction", top_k=args.top_k, max_queries=args.max_queries, family="Basis Sharing", seed=args.seed)
    if condition == "fixed_ngc_top1":
        return build_query_manifest(df, args.model, args.dataset, selector="fixed", top_k=1, max_queries=args.max_queries, family="NGC", seed=args.seed)
    if condition == "fixed_ngc_top5":
        return build_query_manifest(df, args.model, args.dataset, selector="fixed", top_k=args.top_k, max_queries=args.max_queries, family="NGC", seed=args.seed)
    if condition == "adaptive_ngc_top1":
        return build_query_manifest(df, args.model, args.dataset, selector="adaptive", top_k=1, max_queries=args.max_queries, family="NGC", seed=args.seed)
    if condition == "adaptive_ngc_top5":
        return build_query_manifest(df, args.model, args.dataset, selector="adaptive", top_k=args.top_k, max_queries=args.max_queries, family="NGC", seed=args.seed)
    return build_query_manifest(df, args.model, args.dataset, selector="fixed", top_k=1, max_queries=args.max_queries, seed=args.seed)


def benchmark_ranking(df: pd.DataFrame, args: argparse.Namespace, manifest: list[dict[str, Any]], condition: str, trace_path: Path) -> None:
    query_keys = {row["query_key"] for row in manifest}
    sub = df[df["query_key"].astype(str).isin(query_keys)].copy()
    if condition.startswith("adaptive"):
        score_col = selector_score_column(sub, "adaptive")
    elif condition.startswith("fixed"):
        score_col = selector_score_column(sub, "fixed")
    else:
        score_col = selector_score_column(sub, "reconstruction")
    per_repeat = []
    for _ in range(args.ranking_repeats):
        t0 = time.perf_counter()
        for _, group in sub.groupby("query_key", sort=False):
            _ = group.sort_values(score_col, ascending=False, kind="mergesort").head(args.top_k)
        per_repeat.append((time.perf_counter() - t0) / max(1, len(query_keys)))
    append_jsonl(
        trace_path,
        {
            "condition": condition,
            "run_type": "ranking",
            "model": args.model,
            "dataset": args.dataset,
            "n_queries": len(query_keys),
            "ranking_repeats": args.ranking_repeats,
            "median_seconds": float(np.median(per_repeat)),
            "mean_seconds": float(np.mean(per_repeat)),
            "peak_memory_gb": 0.0,
        },
    )


def run_raw(args: argparse.Namespace, manifest: list[dict[str, Any]], queries: dict[str, dict[str, str]], trace_path: Path) -> None:
    t0 = time.perf_counter()
    torch_mod, model, tokenizer, _layers = load_model(args.project_dir, args.model)
    model_load_seconds = time.perf_counter() - t0
    total_params = sum(int(p.numel()) for p in model.parameters())
    reset_peak(torch_mod)
    for idx, row in enumerate(manifest, start=1):
        q = queries[row["query_key"]]
        output, seconds, ntok = generate_answer(torch_mod, model, tokenizer, q["query"], args.max_new_tokens)
        stat = score_prediction(args.dataset, output, q["answer"])
        append_jsonl(
            trace_path,
            {
                "condition": "raw_dense",
                "run_type": "generation",
                "model": args.model,
                "dataset": args.dataset,
                "query_id": row["query_id"],
                "query_key": row["query_key"],
                "rank": 0,
                "wtp_file": "",
                "generation_seconds": seconds,
                "num_new_tokens": ntok,
                "model_load_seconds": model_load_seconds if idx == 1 else 0.0,
                "wtp_load_seconds": 0.0,
                "materialize_seconds": 0.0,
                "peak_memory_gb": peak_memory_gb(torch_mod),
                "stored_bytes": 0,
                "stored_params": total_params,
                "output": output,
                **stat,
            },
        )
        print(f"[cost] raw {idx}/{len(manifest)} corr={stat['is_correct']} sec={seconds:.3f}", flush=True)
    del model
    cleanup_cuda(torch_mod)


def run_topology_condition(args: argparse.Namespace, condition: str, manifest: list[dict[str, Any]], queries: dict[str, dict[str, str]], trace_path: Path) -> None:
    jobs_by_wtp = defaultdict(list)
    for row in manifest:
        for topo in row["topologies"]:
            jobs_by_wtp[topo["wtp_file"]].append((row, topo))
    for topo_idx, (wtp_file, jobs) in enumerate(sorted(jobs_by_wtp.items()), start=1):
        print(f"[cost] {condition} topology={topo_idx}/{len(jobs_by_wtp)} jobs={len(jobs)} {wtp_file}", flush=True)
        cold_start = time.perf_counter()
        t0 = time.perf_counter()
        torch_mod, model, tokenizer, _layers = load_model(args.project_dir, args.model)
        model_load_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        nsys_records = load_wtp_records(args.project_dir, args.model, wtp_file)
        wtp_load_seconds = time.perf_counter() - t1
        stored_params = sum(int(p.numel()) for module in nsys_records.values() for p in module.parameters())
        storage = wtp_storage_stats(args.project_dir, args.model, wtp_file)
        t2 = time.perf_counter()
        materialize_nsys(model, nsys_records, args.project_dir)
        materialize_seconds = time.perf_counter() - t2
        reset_peak(torch_mod)
        for job_idx, (row, topo) in enumerate(jobs, start=1):
            q = queries[row["query_key"]]
            output, seconds, ntok = generate_answer(torch_mod, model, tokenizer, q["query"], args.max_new_tokens)
            stat = score_prediction(args.dataset, output, q["answer"])
            append_jsonl(
                trace_path,
                {
                    "condition": condition,
                    "run_type": "generation",
                    "model": args.model,
                    "dataset": args.dataset,
                    "query_id": row["query_id"],
                    "query_key": row["query_key"],
                    "rank": topo["rank"],
                    "wtp_file": wtp_file,
                    "generation_seconds": seconds,
                    "num_new_tokens": ntok,
                    "tokens_per_second": float(ntok / seconds) if seconds > 0 else float("nan"),
                    "model_load_seconds": model_load_seconds if job_idx == 1 else 0.0,
                    "wtp_load_seconds": wtp_load_seconds if job_idx == 1 else 0.0,
                    "materialize_seconds": materialize_seconds if job_idx == 1 else 0.0,
                    "cold_start_seconds": time.perf_counter() - cold_start if job_idx == 1 else 0.0,
                    "warm_materialized": int(job_idx > 1),
                    "peak_memory_gb": peak_memory_gb(torch_mod),
                    "stored_bytes": storage["total_bytes"],
                    "stored_params": stored_params,
                    "output": output,
                    **stat,
                },
            )
            print(f"[cost] {condition} {job_idx}/{len(jobs)} corr={stat['is_correct']} sec={seconds:.3f}", flush=True)
        del model
        del nsys_records
        cleanup_cuda(torch_mod)


def run_one(args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(args.out_root / f"{args.model}_{args.dataset}")
    trace_path = out_dir / "end_to_end_cost_audit_traces.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    df = load_router_predictions(args.v2_root)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    manifests = {condition: condition_manifest(df, args, condition) for condition in conditions if condition != "raw_dense"}
    base_manifest = next(iter(manifests.values())) if manifests else build_query_manifest(df, args.model, args.dataset, max_queries=args.max_queries)
    queries = load_queries(args.project_dir, args.dataset, base_manifest)
    write_json(out_dir / "end_to_end_cost_audit_manifest.json", {"conditions": conditions, "manifests": manifests, "base_manifest": base_manifest, "args": vars(args) | {"project_dir": str(args.project_dir), "v2_root": str(args.v2_root), "out_root": str(args.out_root)}})
    for condition, manifest in manifests.items():
        benchmark_ranking(df, args, manifest, condition, trace_path)
    if "raw_dense" in conditions:
        run_raw(args, base_manifest, queries, trace_path)
    for condition in conditions:
        if condition == "raw_dense":
            continue
        run_topology_condition(args, condition, manifests[condition], queries, trace_path)
    summarize(args.out_root)
    return out_dir


def summarize(out_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(out_root.glob("*/*traces.jsonl")):
        rows = [__import__("json").loads(line) for line in path.read_text().splitlines() if line.strip()]
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_root / "end_to_end_cost_audit_all_traces.csv.gz", index=False, compression="gzip")
    gen = df[df["run_type"].eq("generation")].copy()
    summary = (
        gen.groupby(["model", "dataset", "condition"], as_index=False)
        .agg(
            n_rows=("is_correct", "size"),
            accuracy=("is_correct", "mean"),
            median_generation_seconds=("generation_seconds", "median"),
            mean_generation_seconds=("generation_seconds", "mean"),
            median_tokens_per_second=("tokens_per_second", "median"),
            median_new_tokens=("num_new_tokens", "median"),
            median_peak_memory_gb=("peak_memory_gb", "median"),
            total_model_load_seconds=("model_load_seconds", "sum"),
            total_wtp_load_seconds=("wtp_load_seconds", "sum"),
            total_materialize_seconds=("materialize_seconds", "sum"),
            median_stored_bytes=("stored_bytes", "median"),
            median_stored_params=("stored_params", "median"),
        )
        .sort_values(["model", "dataset", "condition"])
    )
    ranking = df[df["run_type"].eq("ranking")].copy()
    if not ranking.empty:
        ranking_summary = ranking[["model", "dataset", "condition", "median_seconds", "mean_seconds", "n_queries"]].copy()
        ranking_summary = ranking_summary.rename(columns={"median_seconds": "ranking_median_seconds_per_query"})
        summary = summary.merge(ranking_summary, on=["model", "dataset", "condition"], how="left")
    summary.to_csv(out_root / "end_to_end_cost_audit_summary.csv", index=False)
    make_figure(summary, out_root)
    return summary


def make_figure(summary: pd.DataFrame, out_root: Path) -> None:
    if summary.empty:
        return
    plt = setup_matplotlib()
    plot = summary.copy()
    plot["pair"] = plot["model"] + "/" + plot["dataset"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))
    for ax, metric, title, ylabel in [
        (axes[0], "median_generation_seconds", "Generation latency", "Median seconds"),
        (axes[1], "median_peak_memory_gb", "Peak GPU memory", "Median GB"),
        (axes[2], "total_materialize_seconds", "Materialization cost", "Total seconds"),
    ]:
        pivot = plot.pivot_table(index="pair", columns="condition", values=metric, aggfunc="median")
        pivot.plot(kind="bar", ax=ax, width=0.84)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig_path = DEFAULT_V2_ROOT / "figures/jun15_end_to_end_cost_audit"
    ensure_dir(fig_path.parent)
    fig.savefig(fig_path.with_suffix(".pdf"))
    fig.savefig(fig_path.with_suffix(".png"), dpi=320)


def main() -> None:
    args = parse_args()
    args.project_dir = args.project_dir.resolve()
    args.v2_root = args.v2_root.resolve()
    args.out_root = args.out_root.resolve()
    if args.aggregate_only:
        summarize(args.out_root)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
