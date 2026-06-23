#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
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
    block_role,
    build_query_manifest,
    cleanup_cuda,
    collect_matched_non_role_blocks,
    collect_target_blocks,
    ensure_dir,
    generate_answer,
    get_submodule,
    iter_wtp_blocks,
    load_model,
    load_queries,
    load_router_predictions,
    load_wtp_records,
    materialize_nsys,
    paired_bootstrap,
    peak_memory_gb,
    read_wtp_json,
    reset_peak,
    run_scorer_unit_checks,
    score_prediction,
    set_submodule,
    write_csv,
    write_json,
)
from ngc_v2_utils import setup_matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executable causal perturbation of broker and settlement subspaces.")
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_V0_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_V2_ROOT / "results/jun15_mechanistic_controls/causal_role_perturbation")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--dataset", default="gpqa")
    parser.add_argument("--selector", default="fixed", choices=["fixed", "adaptive"])
    parser.add_argument("--family", default="NGC", help="Topology family for causal perturbation; use ALL to disable filtering.")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-target-blocks", type=int, default=2)
    parser.add_argument("--svd-rank-cap", type=int, default=128)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--conditions", default="intact,broker_random,broker_svd,settlement_random,settlement_svd,matched_nonrole_random")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--unit-checks-only", action="store_true")
    return parser.parse_args()


class FixedWeightLinear:
    def __init__(self, torch_mod: Any, weight: Any, bias: Any = None):
        import torch.nn as nn

        class _Fixed(nn.Module):
            def __init__(self, weight, bias):
                super().__init__()
                self.register_buffer("weight", weight.detach().clone())
                if bias is not None:
                    self.register_buffer("bias", bias.detach().clone())
                else:
                    self.bias = None

            def forward(self, x):
                out = x @ self.weight.T
                if self.bias is not None:
                    out = out + self.bias
                return out

        self.module = _Fixed(weight, bias)


class LowRankWeightLinear:
    def __init__(self, left: Any, right: Any, bias: Any = None):
        import torch.nn as nn

        class _LowRank(nn.Module):
            def __init__(self, left, right, bias):
                super().__init__()
                self.register_buffer("left", left.detach().clone())
                self.register_buffer("right", right.detach().clone())
                if bias is not None:
                    self.register_buffer("bias", bias.detach().clone())
                else:
                    self.bias = None

            def forward(self, x):
                in_dtype = x.dtype
                h = x.to(self.right.dtype) @ self.right
                out = h @ self.left.T
                if self.bias is not None:
                    out = out + self.bias
                return out.to(in_dtype)

        self.module = _LowRank(left, right, bias)


def perturb_output_neurons_random(nsys_records: dict[str, Any], target_blocks: set[str], seed: int) -> list[dict[str, Any]]:
    import torch

    rows = []
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    gen.manual_seed(seed)
    for nt_name, module in nsys_records.items():
        names = str(nt_name).split("-")
        for slice_id, block in enumerate(names):
            if block not in target_blocks or not hasattr(module, "slice_id2pos") or not hasattr(module, "output_neurons"):
                continue
            start, end = module.slice_id2pos[slice_id]
            original = module.output_neurons.data[start:end].detach().clone()
            noise = torch.randn(original.shape, generator=gen, device=original.device, dtype=torch.float32).to(original.dtype)
            orig_norm = original.float().norm()
            noise_norm = noise.float().norm().clamp_min(1e-8)
            module.output_neurons.data[start:end] = (noise.float() * (orig_norm / noise_norm)).to(original.dtype)
            rows.append(
                {
                    "nt_name": nt_name,
                    "block_name": block,
                    "slice_id": int(slice_id),
                    "operation": "output_neurons_norm_matched_random",
                    "original_norm": float(orig_norm),
                    "new_norm": float(module.output_neurons.data[start:end].float().norm()),
                }
            )
    return rows


def lowrank_factors(torch_mod: Any, weight: Any, rank: int, seed: int) -> tuple[Any, Any, int]:
    torch_mod.manual_seed(seed)
    q = min(int(rank), min(weight.shape) - 1)
    q = max(q, 1)
    w = weight.detach().float().cpu()
    # Keep the low-rank reconstruction factored. Materializing the full dense
    # out x in matrix on GPU causes avoidable OOM on 8B checkpoints.
    u, s, v = torch_mod.svd_lowrank(w, q=q, niter=2)
    left = (u[:, :q] * s[:q]).to(weight.dtype)
    right = v[:, :q].to(weight.dtype)
    return left, right, q


def replace_targets_with_svd(
    torch_mod: Any,
    model: Any,
    root_weights: dict[str, tuple[Any, Any]],
    target_blocks: set[str],
    rank_by_block: dict[str, int],
    svd_rank_cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for block in sorted(target_blocks):
        if block not in root_weights:
            continue
        weight, bias = root_weights[block]
        rank = int(rank_by_block.get(block, min(weight.shape) // 4))
        if svd_rank_cap and svd_rank_cap > 0:
            rank = min(rank, svd_rank_cap)
        t0 = time.perf_counter()
        left, right, rank = lowrank_factors(torch_mod, weight, rank=rank, seed=seed + len(rows))
        seconds = time.perf_counter() - t0
        fixed = LowRankWeightLinear(
            left.to(weight.device),
            right.to(weight.device),
            bias.to(weight.device) if bias is not None else None,
        ).module
        set_submodule(model, block, fixed)
        rows.append(
            {
                "block_name": block,
                "operation": "root_weight_lowrank_svd_factored_linear",
                "rank": int(rank),
                "seconds": float(seconds),
                "weight_shape": "x".join(map(str, weight.shape)),
                "left_shape": "x".join(map(str, left.shape)),
                "right_shape": "x".join(map(str, right.shape)),
            }
        )
    return rows


def target_blocks_for_condition(wtp_rows: list[list[Any]], model: str, condition: str, max_target_blocks: int, seed: int) -> list[str]:
    if condition.startswith("broker"):
        return collect_target_blocks(wtp_rows, model, "early_broker", max_target_blocks)
    if condition.startswith("settlement"):
        return collect_target_blocks(wtp_rows, model, "late_settlement", max_target_blocks)
    if condition.startswith("matched_nonrole"):
        return collect_matched_non_role_blocks(wtp_rows, model, max_target_blocks, seed)
    return []


def rank_by_block_from_records(nsys_records: dict[str, Any]) -> dict[str, int]:
    out = {}
    for nt_name, module in nsys_records.items():
        rank = int(getattr(module, "hidden_dim", 0) or 0)
        for block in str(nt_name).split("-"):
            out[block] = rank
    return out


def run_unit_checks(args: argparse.Namespace) -> pd.DataFrame:
    import torch

    df = load_router_predictions(args.v2_root)
    manifest = build_query_manifest(df, args.model, args.dataset, selector=args.selector, top_k=1, max_queries=1, seed=args.seed)
    rows = []
    for scorer_row in run_scorer_unit_checks():
        passed = int(scorer_row["expected"] == scorer_row["observed"])
        rows.append(
            {
                "check": f"scorer_{scorer_row['dataset']}_{len(rows)}",
                "passed": passed,
                "detail": f"expected={scorer_row['expected']} observed={scorer_row['observed']}",
            }
        )
    weight = torch.randn(16, 10, generator=torch.Generator().manual_seed(args.seed))
    left_a, right_a, rank_a = lowrank_factors(torch, weight, rank=4, seed=args.seed)
    left_b, right_b, rank_b = lowrank_factors(torch, weight, rank=4, seed=args.seed)
    rows.append(
        {
            "check": "svd_perturbation_deterministic",
            "passed": int(rank_a == rank_b and torch.allclose(left_a, left_b) and torch.allclose(right_a, right_b)),
            "detail": f"rank={rank_a}",
        }
    )
    bias = torch.randn(16, generator=torch.Generator().manual_seed(args.seed + 1))
    lowrank = LowRankWeightLinear(left_a, right_a, bias).module
    x = torch.randn(3, 10, generator=torch.Generator().manual_seed(args.seed + 2))
    expected = x @ right_a @ left_a.T + bias
    rows.append(
        {
            "check": "svd_lowrank_module_matches_factored_reconstruction",
            "passed": int(torch.allclose(lowrank(x), expected, atol=1e-5, rtol=1e-5)),
            "detail": "synthetic_linear",
        }
    )

    class _FakeNsys:
        def __init__(self):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.slice_id2pos = [(0, 2), (2, 4)]
            self.output_neurons = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)

    fake_a = {"model.layers.1.self_attn.k_proj-model.layers.20.self_attn.o_proj": _FakeNsys()}
    fake_b = {"model.layers.1.self_attn.k_proj-model.layers.20.self_attn.o_proj": _FakeNsys()}
    perturb_output_neurons_random(fake_a, {"model.layers.1.self_attn.k_proj"}, args.seed)
    perturb_output_neurons_random(fake_b, {"model.layers.1.self_attn.k_proj"}, args.seed)
    rows.append(
        {
            "check": "random_perturbation_deterministic",
            "passed": int(torch.allclose(next(iter(fake_a.values())).output_neurons, next(iter(fake_b.values())).output_neurons)),
            "detail": "synthetic_output_neurons",
        }
    )
    if not manifest:
        return pd.DataFrame([{"check": "manifest_non_empty", "passed": 0, "detail": "no manifest rows"}])
    wtp_file = manifest[0]["topologies"][0]["wtp_file"]
    wtp_rows = read_wtp_json(args.project_dir, args.model, wtp_file)
    blocks = list(iter_wtp_blocks(wtp_rows))
    rows.append({"check": "wtp_parses_blocks", "passed": int(len(blocks) > 0), "detail": str(len(blocks))})
    nsys_records = load_wtp_records(args.project_dir, args.model, wtp_file)
    targets = set(target_blocks_for_condition(wtp_rows, args.model, "broker_random", 1, args.seed))
    before = {}
    for nt_name, module in nsys_records.items():
        if hasattr(module, "output_neurons"):
            before[nt_name] = module.output_neurons.detach().clone()
    perturbed = perturb_output_neurons_random(nsys_records, targets, args.seed)
    untouched_ok = True
    changed_ok = bool(perturbed) or not targets
    for nt_name, module in nsys_records.items():
        if nt_name not in before:
            continue
        names = nt_name.split("-")
        for slice_id, block in enumerate(names):
            start, end = module.slice_id2pos[slice_id]
            same = bool((before[nt_name][start:end] == module.output_neurons[start:end]).all().item())
            if block in targets:
                changed_ok = changed_ok and not same
            else:
                untouched_ok = untouched_ok and same
    rows.append({"check": "perturb_changes_targeted_slice", "passed": int(changed_ok), "detail": ",".join(sorted(targets))})
    rows.append({"check": "perturb_preserves_untargeted_slices", "passed": int(untouched_ok), "detail": str(len(before))})
    return pd.DataFrame(rows)


def run_one(args: argparse.Namespace) -> Path:
    out_dir = ensure_dir(args.out_root / f"{args.model}_{args.dataset}_{args.selector}")
    trace_path = out_dir / "causal_role_perturbation_outputs.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    df = load_router_predictions(args.v2_root)
    manifest = build_query_manifest(
        df,
        args.model,
        args.dataset,
        selector=args.selector,
        top_k=args.top_k,
        max_queries=args.max_queries,
        family=None if args.family == "ALL" else args.family,
        seed=args.seed,
    )
    queries = load_queries(args.project_dir, args.dataset, manifest)
    write_json(out_dir / "causal_role_perturbation_manifest.json", {"manifest": manifest, "args": vars(args) | {"project_dir": str(args.project_dir), "v2_root": str(args.v2_root), "out_root": str(args.out_root)}})
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    for condition in conditions:
        jobs_by_wtp = defaultdict(list)
        for row in manifest:
            for topo in row["topologies"]:
                jobs_by_wtp[topo["wtp_file"]].append((row, topo))
        for wtp_idx, (wtp_file, jobs) in enumerate(sorted(jobs_by_wtp.items()), start=1):
            print(f"[causal] condition={condition} topology={wtp_idx}/{len(jobs_by_wtp)} {wtp_file} jobs={len(jobs)}", flush=True)
            torch_mod, model, tokenizer, _layers = load_model(args.project_dir, args.model)
            reset_peak(torch_mod)
            wtp_rows = read_wtp_json(args.project_dir, args.model, wtp_file)
            target_blocks = set(target_blocks_for_condition(wtp_rows, args.model, condition, args.max_target_blocks, args.seed))
            root_weights = {}
            for block in target_blocks:
                try:
                    layer = get_submodule(model, block)
                    root_weights[block] = (layer.weight.detach().clone(), layer.bias.detach().clone() if getattr(layer, "bias", None) is not None else None)
                except Exception:
                    pass
            t0 = time.perf_counter()
            nsys_records = load_wtp_records(args.project_dir, args.model, wtp_file)
            wtp_load_seconds = time.perf_counter() - t0
            perturb_rows = []
            if condition.endswith("random"):
                perturb_rows = perturb_output_neurons_random(nsys_records, target_blocks, args.seed)
            rank_by_block = rank_by_block_from_records(nsys_records)
            t1 = time.perf_counter()
            materialize_nsys(model, nsys_records, args.project_dir)
            materialize_seconds = time.perf_counter() - t1
            if condition.endswith("svd"):
                perturb_rows = replace_targets_with_svd(
                    torch_mod,
                    model,
                    root_weights,
                    target_blocks,
                    rank_by_block,
                    args.svd_rank_cap,
                    args.seed,
                )
            reset_peak(torch_mod)
            for job_idx, (row, topo) in enumerate(jobs, start=1):
                q = queries[row["query_key"]]
                output, seconds, ntok = generate_answer(torch_mod, model, tokenizer, q["query"], args.max_new_tokens)
                stat = score_prediction(args.dataset, output, q["answer"])
                append_jsonl(
                    trace_path,
                    {
                        "condition": condition,
                        "model": args.model,
                        "dataset": args.dataset,
                        "selector": args.selector,
                        "query_id": row["query_id"],
                        "query_key": row["query_key"],
                        "rank": topo["rank"],
                        "wtp_file": wtp_file,
                        "target_blocks": sorted(target_blocks),
                        "perturbation_rows": perturb_rows,
                        "generation_seconds": seconds,
                        "num_new_tokens": ntok,
                        "wtp_load_seconds": wtp_load_seconds if job_idx == 1 else 0.0,
                        "materialize_seconds": materialize_seconds if job_idx == 1 else 0.0,
                        "peak_memory_gb": peak_memory_gb(torch_mod),
                        "answer": q["answer"],
                        "output": output,
                        **stat,
                    },
                )
                print(f"[causal] {condition} {job_idx}/{len(jobs)} {row['query_id']} corr={stat['is_correct']} sec={seconds:.3f}", flush=True)
            del model
            del nsys_records
            cleanup_cuda(torch_mod)
    summarize(args.out_root, args.n_bootstrap, args.seed)
    return out_dir


def summarize(out_root: Path, n_boot: int, seed: int) -> pd.DataFrame:
    frames = []
    for path in sorted(out_root.glob("*/*outputs.jsonl")):
        rows = [pd.json_normalize(__import__("json").loads(line)).iloc[0].to_dict() for line in path.read_text().splitlines() if line.strip()]
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    ensure_dir(out_root)
    df.to_csv(out_root / "causal_role_perturbation_all_outputs.csv.gz", index=False, compression="gzip")
    summary = (
        df.groupby(["model", "dataset", "selector", "condition"], as_index=False)
        .agg(
            n_rows=("is_correct", "size"),
            accuracy=("is_correct", "mean"),
            mean_f1=("f1", "mean"),
            median_seconds=("generation_seconds", "median"),
            median_peak_memory_gb=("peak_memory_gb", "median"),
        )
        .sort_values(["model", "dataset", "condition"])
    )
    summary.to_csv(out_root / "causal_role_perturbation_summary.csv", index=False)
    contrasts = []
    for keys, group in df.groupby(["model", "dataset", "selector"]):
        pivot = group.pivot_table(index=["query_key", "rank"], columns="condition", values="is_correct", aggfunc="max")
        if "intact" not in pivot:
            continue
        for condition in pivot.columns:
            if condition == "intact":
                continue
            aligned = pivot[["intact", condition]].dropna()
            if aligned.empty:
                continue
            stat = paired_bootstrap(aligned["intact"], aligned[condition], n_boot=n_boot, seed=seed)
            contrasts.append(
                {
                    "model": keys[0],
                    "dataset": keys[1],
                    "selector": keys[2],
                    "contrast": f"intact_minus_{condition}",
                    **stat,
                }
            )
    contrast_df = pd.DataFrame(contrasts)
    contrast_df.to_csv(out_root / "causal_role_perturbation_bootstrap.csv", index=False)
    make_figure(summary, out_root)
    return summary


def make_figure(summary: pd.DataFrame, out_root: Path) -> None:
    if summary.empty:
        return
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    plot = summary.copy()
    plot["pair"] = plot["model"] + "/" + plot["dataset"]
    conditions = ["intact", "broker_random", "broker_svd", "settlement_random", "settlement_svd", "matched_nonrole_random"]
    pairs = sorted(plot["pair"].unique())
    x = np.arange(len(pairs))
    width = 0.13
    colors = ["#2C9C95", "#D95F5F", "#F0A33A", "#7666B3", "#4F77AA", "#9097A1"]
    for i, condition in enumerate(conditions):
        vals = []
        for pair in pairs:
            sub = plot[plot["pair"].eq(pair) & plot["condition"].eq(condition)]
            vals.append(float(sub["accuracy"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (i - 2.5) * width, np.array(vals) * 100.0, width, label=condition.replace("_", " "), color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=20, ha="right")
    ax.set_ylabel("Executable accuracy (%)")
    ax.set_title("Causal perturbation of role-specific subspaces")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig_path = DEFAULT_V2_ROOT / "figures/jun15_causal_role_perturbation"
    ensure_dir(fig_path.parent)
    fig.savefig(fig_path.with_suffix(".pdf"))
    fig.savefig(fig_path.with_suffix(".png"), dpi=320)


def main() -> None:
    args = parse_args()
    args.project_dir = args.project_dir.resolve()
    args.v2_root = args.v2_root.resolve()
    args.out_root = args.out_root.resolve()
    if args.unit_checks_only:
        out = ensure_dir(args.out_root)
        checks = run_unit_checks(args)
        checks.to_csv(out / "causal_role_perturbation_unit_checks.csv", index=False)
        print(checks.to_string(index=False))
        if not bool(checks["passed"].all()):
            raise SystemExit(1)
        return
    if args.aggregate_only:
        summarize(args.out_root, args.n_bootstrap, args.seed)
    else:
        run_one(args)


if __name__ == "__main__":
    main()
