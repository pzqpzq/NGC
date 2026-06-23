#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ngc_v2_utils import (
    RANDOM_SEED,
    compression_bin,
    ensure_dir,
    finite_or_nan,
    grouped_query_split,
    parse_tcr,
    safe_to_parquet,
    topology_family,
    write_json,
)


STAT_FIELDS = [
    "cur_inN_forceNorm",
    "root_inN_forceNorm",
    "trace-LT",
    "trace-nonLT",
    "norm-LT",
    "norm-nonLT",
    "w_err_norm",
    "y_err_norm",
]


def parse_topology_id(wtp_file: str) -> str:
    name = Path(str(wtp_file)).name
    match = re.search(r"wtpID#([^_.]+)", name)
    if match:
        return match.group(1)
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]


def layer_id(block_name: str) -> int:
    match = re.search(r"layers\.(\d+)\.", block_name)
    return int(match.group(1)) if match else -1


def block_kind(block_name: str) -> str:
    if ".self_attn.k_proj" in block_name:
        return "k"
    if ".self_attn.v_proj" in block_name:
        return "v"
    if ".self_attn.q_proj" in block_name:
        return "q"
    if ".self_attn.o_proj" in block_name:
        return "o"
    if ".mlp.gate_proj" in block_name:
        return "gate"
    if ".mlp.up_proj" in block_name:
        return "up"
    if ".mlp.down_proj" in block_name:
        return "down"
    if ".mlp." in block_name:
        return "mlp_other"
    return "other"


def summarize_weights(weights_stat: dict) -> dict:
    names = list(weights_stat.keys())
    out = {"n_modified_blocks_rich": len(names)}
    if not names:
        return out

    layers = np.array([layer_id(name) for name in names if layer_id(name) >= 0], dtype=float)
    max_layer = max(float(np.nanmax(layers)), 1.0) if len(layers) else 1.0
    if len(layers):
        out.update(
            {
                "mean_layer_rich": float(np.nanmean(layers)),
                "early_block_frac": float(np.mean(layers <= max_layer / 3.0)),
                "mid_block_frac": float(np.mean((layers > max_layer / 3.0) & (layers <= 2.0 * max_layer / 3.0))),
                "late_block_frac": float(np.mean(layers > 2.0 * max_layer / 3.0)),
            }
        )

    kinds = [block_kind(name) for name in names]
    for kind in ["k", "v", "q", "o", "gate", "up", "down"]:
        out[f"{kind}_block_frac"] = float(np.mean([item == kind for item in kinds]))
    out["attention_block_frac"] = float(np.mean([kind in {"k", "v", "q", "o"} for kind in kinds]))
    out["mlp_block_frac_rich"] = float(np.mean([kind in {"gate", "up", "down", "mlp_other"} for kind in kinds]))

    for field in STAT_FIELDS:
        vals = np.array([finite_or_nan(weights_stat[name].get(field)) for name in names], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        key = field.replace("-", "_")
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
        out[f"{key}_min"] = float(np.min(vals))
        out[f"{key}_max"] = float(np.max(vals))

    top_svd = []
    for name in names:
        top_svd.extend([finite_or_nan(x) for x in weights_stat[name].get("top3-svdvals-LT", [])])
    top_svd = np.array([x for x in top_svd if np.isfinite(x)], dtype=float)
    if len(top_svd):
        out["top3_svdvals_LT_mean"] = float(np.mean(top_svd))
        out["top3_svdvals_LT_max"] = float(np.max(top_svd))
    return out


def load_rich_candidate_rows(v0_root: Path) -> pd.DataFrame:
    rows = []
    feature_root = v0_root / "ngcLLM-features-record-v2"
    for feature_file in sorted(feature_root.glob("*/*/*.json")):
        model = feature_file.parent.parent.name
        dataset = feature_file.parent.name
        data = json.loads(feature_file.read_text(encoding="utf-8"))
        raw_probs = data.get("raw_probs", [])
        raw_numts = data.get("raw_numTs", [])
        for qid, candidates in data.get("content", {}).items():
            query_id = f"{model}/{dataset}/{qid}"
            for candidate_index, candidate in enumerate(candidates):
                wtp_file = candidate.get("wtp_fileName", "")
                row = {
                    "query_id": query_id,
                    "model": model,
                    "dataset": dataset,
                    "candidate_index": int(candidate_index),
                    "wtp_file": wtp_file,
                    "wtp_file_name": Path(str(wtp_file)).name,
                    "topology_id": parse_topology_id(str(wtp_file)),
                    "topology_family": topology_family(str(wtp_file)),
                    "tcr_from_wtp": parse_tcr(str(wtp_file)),
                    "compression_bin": compression_bin(parse_tcr(str(wtp_file))),
                    "cached_is_correct_rich": int(candidate.get("isCorr", 0)),
                    "num_tokens_rich": finite_or_nan(candidate.get("num_tokens")),
                }
                try:
                    idx = int(qid)
                    row["raw_numTs"] = finite_or_nan(raw_numts[idx]) if idx < len(raw_numts) else np.nan
                    row["raw_prob_from_feature"] = finite_or_nan(raw_probs[idx]) if idx < len(raw_probs) else np.nan
                except Exception:
                    row["raw_numTs"] = np.nan
                    row["raw_prob_from_feature"] = np.nan
                row.update(summarize_weights(candidate.get("weights_stat", {})))
                rows.append(row)
    return pd.DataFrame(rows)


def robust_z(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce")
    med = arr.median()
    mad = (arr - med).abs().median()
    if not np.isfinite(mad) or mad < 1e-12:
        std = arr.std()
        if not np.isfinite(std) or std < 1e-12:
            return pd.Series(np.zeros(len(arr)), index=values.index)
        return (arr - arr.mean()) / std
    return 0.67448975 * (arr - med) / mad


def sigmoid(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(-30, 30)
    return 1.0 / (1.0 + np.exp(-arr))


def add_role_law_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby(["model", "dataset"], group_keys=False)
    for col in [
        "departure_index",
        "stability_guard",
        "mediator_score",
        "settlement_score",
        "evidence_broker_score",
        "reconstruction_score",
        "raw_numTs",
        "num_tokens",
    ]:
        if col in out:
            out[f"{col}_rz"] = grouped[col].transform(robust_z)

    out["U_prop"] = sigmoid(out.get("departure_index_rz", 0.0))
    out["U_ske"] = sigmoid(out.get("stability_guard_rz", 0.0))
    out["U_stab"] = sigmoid(out.get("settlement_score_rz", 0.0))
    out["U_med"] = sigmoid(out.get("mediator_score_rz", 0.0))
    out["U_ret"] = sigmoid(out.get("evidence_broker_score_rz", 0.0))
    out["U_set"] = sigmoid(out.get("settlement_score_rz", 0.0))
    out["U_veto"] = sigmoid(out.get("stability_guard_rz", 0.0) - 0.20 * out.get("departure_index_rz", 0.0).abs())
    out["U_cost"] = sigmoid(-out.get("num_tokens_rz", 0.0))

    eps = 1e-8
    utilities = out[["U_prop", "U_ske", "U_stab"]].clip(lower=eps)
    nash = (utilities["U_prop"] * utilities["U_ske"] * utilities["U_stab"]) ** (1.0 / 3.0)
    mean_u = utilities.mean(axis=1)
    var_u = ((utilities.sub(mean_u, axis=0)) ** 2).mean(axis=1)
    agree = np.exp(-var_u / (mean_u * mean_u + 1e-6))

    out["law_fixed_transfer"] = out["law_score"]
    out["law_bargaining"] = nash * agree * (0.75 + 0.25 * out["role_entropy"].fillna(0.0))
    out["law_coalition"] = (0.35 * out["U_prop"] + 0.30 * out["U_ske"] + 0.20 * out["U_stab"] + 0.15 * out["U_med"]) * out["U_veto"]
    out["law_mediated_criticality"] = out["U_med"] * sigmoid(-out.get("stability_risk_z", 0.0).abs()) * (0.5 + 0.5 * out["role_entropy"].fillna(0.0))
    out["law_retrieve_then_settle"] = (0.55 * out["U_ret"] + 0.45 * out["U_set"]) * out["U_ske"]
    out["law_reconstruction_guarded_exploration"] = out["U_prop"] * sigmoid(out.get("reconstruction_score_rz", 0.0)) * out["U_veto"]
    out["law_settlement_only"] = out["U_set"] * sigmoid(out.get("late_mlp_mass", 0.0) + out.get("late_q_mass", 0.0))
    out["law_cost_aware_adaptive"] = out["law_bargaining"] - 0.08 * sigmoid(out.get("num_tokens_rz", 0.0))
    return out


def add_query_text(df: pd.DataFrame, v0_root: Path) -> pd.DataFrame:
    out = df.copy()
    sys.path.insert(0, str(v0_root))
    try:
        import llm_utils.load_data as load_ds
    except Exception as exc:
        out["query_text_load_error"] = str(exc)
        return out

    cache = {}
    query_text = {}
    answer_text = {}
    for dataset in sorted(out["dataset"].dropna().unique()):
        try:
            _train, test = load_ds.load_cleanDS(_dataCard=dataset)
            cache[dataset] = test
        except Exception as exc:
            cache[dataset] = exc
    for query_id, dataset in out[["query_id", "dataset"]].drop_duplicates().itertuples(index=False):
        try:
            idx = int(str(query_id).split("/")[-1])
            test = cache.get(dataset)
            if isinstance(test, Exception):
                raise test
            item = test[idx]
            query_text[query_id] = str(item.get("query", ""))
            answer_text[query_id] = str(item.get("label") or item.get("cot_content") or "")
        except Exception:
            query_text[query_id] = ""
            answer_text[query_id] = ""
    out["query_text"] = out["query_id"].map(query_text).fillna("")
    out["answer_text"] = out["query_id"].map(answer_text).fillna("")
    out["prompt_len_chars"] = out["query_text"].str.len().astype(float)
    out["answer_len_chars"] = out["answer_text"].str.len().astype(float)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--v0-root", type=Path, default=Path(os.environ.get("NGC_V0_ROOT", repo_root / "external" / "NGC-v0")))
    parser.add_argument("--v1-root", type=Path, default=Path(os.environ.get("NGC_V1_ROOT", repo_root / "v1")))
    parser.add_argument("--v2-root", type=Path, default=Path(os.environ.get("NGC_V2_ROOT", repo_root / "v2")))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-query-text", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.v2_root / "results")
    report_dir = ensure_dir(args.v2_root / "reports")

    base_path = args.v1_root / "outputs/01_observables/constitutional_candidates.parquet"
    if base_path.exists():
        base = pd.read_parquet(base_path)
    else:
        base = pd.read_csv(args.v1_root / "outputs/01_observables/constitutional_candidates.csv.gz")

    rich = load_rich_candidate_rows(args.v0_root)
    merged = base.merge(
        rich,
        on=["query_id", "model", "dataset", "candidate_index"],
        how="left",
        validate="one_to_one",
    )
    merged["tcr"] = pd.to_numeric(merged["tcr"], errors="coerce").fillna(merged["tcr_from_wtp"])
    merged["num_tokens"] = pd.to_numeric(merged["num_tokens"], errors="coerce").fillna(merged["num_tokens_rich"])
    merged["compression_bin"] = merged["tcr"].map(compression_bin)
    merged["candidate_uid"] = (
        merged["model"].astype(str)
        + "::"
        + merged["dataset"].astype(str)
        + "::"
        + merged["query_id"].astype(str)
        + "::cand"
        + merged["candidate_index"].astype(str)
    )
    split_map = grouped_query_split(merged["query_key"], seed=args.seed)
    merged["split_random_70_15_15"] = merged["query_key"].map(split_map)
    merged = add_role_law_scores(merged)
    if not args.skip_query_text:
        merged = add_query_text(merged, args.v0_root)

    parquet_ok = safe_to_parquet(merged, out_dir / "router_candidate_bank.parquet")
    merged.to_csv(out_dir / "router_candidate_bank.csv.gz", index=False, compression="gzip")

    schema_lines = ["# Router Candidate Bank Schema", ""]
    schema_lines.append(f"Rows: {len(merged)}")
    schema_lines.append(f"Queries: {merged['query_key'].nunique()}")
    schema_lines.append(f"Parquet written: {parquet_ok}")
    schema_lines.append("")
    schema_lines.append("Key columns:")
    for col in [
        "query_key",
        "candidate_uid",
        "model",
        "dataset",
        "candidate_index",
        "label",
        "raw_prob",
        "topology_family",
        "topology_id",
        "wtp_file_name",
        "tcr",
        "compression_bin",
        "law_fixed_transfer",
        "law_bargaining",
        "law_coalition",
        "law_mediated_criticality",
        "law_retrieve_then_settle",
        "law_reconstruction_guarded_exploration",
        "law_settlement_only",
        "law_cost_aware_adaptive",
    ]:
        if col in merged:
            schema_lines.append(f"- `{col}`")
    (out_dir / "router_candidate_bank_schema.md").write_text("\n".join(schema_lines) + "\n", encoding="utf-8")

    coverage = (
        merged.groupby(["model", "dataset", "topology_family"], dropna=False)
        .agg(n_queries=("query_key", "nunique"), n_candidates=("candidate_uid", "count"), acc=("label", "mean"))
        .reset_index()
    )
    coverage.to_csv(report_dir / "candidate_bank_coverage.csv", index=False)
    write_json(
        report_dir / "candidate_bank_manifest.json",
        {
            "rows": int(len(merged)),
            "queries": int(merged["query_key"].nunique()),
            "models": sorted(merged["model"].dropna().unique().tolist()),
            "datasets": sorted(merged["dataset"].dropna().unique().tolist()),
            "families": sorted(merged["topology_family"].dropna().unique().tolist()),
            "parquet_ok": bool(parquet_ok),
            "source_v1": str(base_path),
        },
    )
    print("Built router candidate bank")
    print(f"Rows: {len(merged)}")
    print(f"Queries: {merged['query_key'].nunique()}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
