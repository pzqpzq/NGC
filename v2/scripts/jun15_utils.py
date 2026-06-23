from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import re
import string
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


RANDOM_SEED = 20260615
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V0_ROOT = Path(os.environ.get("NGC_V0_ROOT", REPO_ROOT / "external" / "NGC-v0"))
DEFAULT_V1_ROOT = Path(os.environ.get("NGC_V1_ROOT", REPO_ROOT / "v1"))
DEFAULT_V2_ROOT = Path(os.environ.get("NGC_V2_ROOT", REPO_ROOT / "v2"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_candidate_bank(v2_root: Path = DEFAULT_V2_ROOT) -> pd.DataFrame:
    parquet_path = v2_root / "results/router_candidate_bank.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(v2_root / "results/router_candidate_bank.csv.gz")


def load_router_predictions(v2_root: Path = DEFAULT_V2_ROOT) -> pd.DataFrame:
    parquet_path = v2_root / "results/router_predictions_test.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = v2_root / "results/router_predictions_test.csv.gz"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return load_candidate_bank(v2_root).query("split_random_70_15_15 == 'test'").copy()


def robust_fit(values: pd.Series) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if arr.empty:
        return {"center": 0.0, "scale": 1.0}
    center = float(arr.median())
    mad = float((arr - center).abs().median())
    if not math.isfinite(mad) or mad < 1e-12:
        std = float(arr.std())
        mad = std / 1.4826 if math.isfinite(std) and std > 1e-12 else 1.0
    return {"center": center, "scale": 1.4826 * mad}


def robust_transform(values: pd.Series, fit: dict[str, float]) -> pd.Series:
    scale = fit["scale"] if abs(fit["scale"]) > 1e-12 else 1.0
    return (pd.to_numeric(values, errors="coerce") - fit["center"]) / scale


def paired_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    n_boot: int = 2000,
    seed: int = RANDOM_SEED,
) -> dict[str, float | int]:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    mask = np.isfinite(left_arr) & np.isfinite(right_arr)
    left_arr = left_arr[mask]
    right_arr = right_arr[mask]
    if len(left_arr) == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_boot": float("nan"), "n": 0}
    delta = left_arr - right_arr
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(delta), size=len(delta))
        draws[i] = delta[idx].mean()
    p_low = float((draws <= 0).mean())
    p_high = float((draws >= 0).mean())
    return {
        "diff": float(delta.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "p_boot": min(1.0, 2.0 * min(p_low, p_high)),
        "n": int(len(delta)),
    }


def parse_layer(block_name: str) -> int:
    match = re.search(r"layers\.(\d+)\.", block_name)
    return int(match.group(1)) if match else -1


def parse_kind(block_name: str) -> str:
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
        return "mlp"
    return "other"


def model_num_layers(model: str) -> int:
    if model.startswith("llama3"):
        return 32
    if model.startswith("qwen3"):
        return 36
    return 36


def block_role(block_name: str, model: str) -> str:
    layer = parse_layer(block_name)
    kind = parse_kind(block_name)
    n_layers = model_num_layers(model)
    if layer >= 0 and layer <= n_layers // 3 and kind in {"k", "v"}:
        return "early_broker"
    if layer >= (2 * n_layers) // 3 and kind in {"q", "gate", "up", "down", "mlp"}:
        return "late_settlement"
    return "non_role"


def iter_wtp_blocks(wtp_records: list[list[Any]]) -> Iterable[dict[str, Any]]:
    for record_index, row in enumerate(wtp_records):
        nsys_name, nt_name, y_errs = row
        names = str(nt_name).split("-")
        for slice_id, block in enumerate(names):
            yield {
                "record_index": record_index,
                "nsys_name": nsys_name,
                "nt_name": nt_name,
                "y_errs": y_errs,
                "block_name": block,
                "slice_id": slice_id,
            }


def read_wtp_json(project_dir: Path, model: str, wtp_file: str) -> list[list[Any]]:
    path = project_dir / "nsys-wtp-record" / model / Path(str(wtp_file)).name
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def wtp_storage_stats(project_dir: Path, model: str, wtp_file: str) -> dict[str, float | int | str]:
    wtp_path = project_dir / "nsys-wtp-record" / model / Path(str(wtp_file)).name
    rows = json.loads(wtp_path.read_text(encoding="utf-8"))
    ckpt_bytes = 0
    n_ckpts = 0
    n_blocks = 0
    for nsys_name, nt_name, y_errs in rows:
        n_blocks += len(str(nt_name).split("-"))
        ckpt = project_dir / "nsys-checkpoints-record" / model / nsys_name / f"{nt_name}#{';'.join(str(v) for v in y_errs)}.pt"
        if ckpt.exists():
            ckpt_bytes += ckpt.stat().st_size
            n_ckpts += 1
    return {
        "wtp_file": Path(str(wtp_file)).name,
        "wtp_json_bytes": int(wtp_path.stat().st_size) if wtp_path.exists() else 0,
        "checkpoint_bytes": int(ckpt_bytes),
        "total_bytes": int(ckpt_bytes + (wtp_path.stat().st_size if wtp_path.exists() else 0)),
        "n_checkpoints": int(n_ckpts),
        "n_blocks": int(n_blocks),
    }


def selector_score_column(df: pd.DataFrame, selector: str) -> str:
    mapping = {
        "fixed": ["score_fixed_transfer_law", "law_fixed_transfer", "law_score"],
        "adaptive": ["score_extra_trees_router"],
        "hgb": ["score_tree_hgb_router"],
        "reconstruction": ["score_reconstruction_only_selector", "reconstruction_score"],
        "raw_dense": ["score_raw_dense"],
    }
    for col in mapping.get(selector, [selector]):
        if col in df:
            return col
    raise KeyError(f"No score column found for selector={selector}")


def build_query_manifest(
    df: pd.DataFrame,
    model: str,
    dataset: str,
    selector: str = "fixed",
    top_k: int = 5,
    max_queries: int = 0,
    family: str | None = None,
    seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    work = df[df["model"].astype(str).eq(model) & df["dataset"].astype(str).eq(dataset)].copy()
    if "split_random_70_15_15" in work:
        work = work[work["split_random_70_15_15"].astype(str).eq("test")].copy()
    if family:
        work = work[work["topology_family"].astype(str).eq(family)].copy()
    if work.empty:
        return []
    score_col = selector_score_column(work, selector)
    rng = random.Random(seed)
    query_keys = sorted(work["query_key"].astype(str).unique())
    if max_queries and max_queries > 0:
        rng.shuffle(query_keys)
        query_keys = sorted(query_keys[:max_queries])
    rows = []
    for query_key in query_keys:
        group = work[work["query_key"].astype(str).eq(query_key)].copy()
        if selector == "random":
            group = group.sample(frac=1.0, random_state=seed).head(top_k)
        else:
            group = group.sort_values(score_col, ascending=False, kind="mergesort").head(top_k)
        if group.empty:
            continue
        first = group.iloc[0]
        q_short = str(query_key).split("::")[-1].split("/")[-1]
        topologies = []
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            topologies.append(
                {
                    "rank": int(rank),
                    "candidate_uid": str(row.get("candidate_uid", "")),
                    "candidate_index": int(row.get("candidate_index", rank - 1)),
                    "cached_label": int(row.get("label", 0)),
                    "score": float(row.get(score_col, float("nan"))),
                    "wtp_file": Path(str(row.get("wtp_file_name", row.get("wtp_file", "")))).name,
                    "topology_family": str(row.get("topology_family", "")),
                    "tcr": float(row.get("tcr", float("nan"))),
                    "num_tokens": float(row.get("num_tokens", float("nan"))),
                    "early_kv_mass": float(row.get("early_kv_mass", float("nan"))),
                    "settlement_score": float(row.get("settlement_score", float("nan"))),
                }
            )
        rows.append(
            {
                "query_key": query_key,
                "query_key_short": q_short,
                "query_id": f"{model}/{dataset}/{q_short}",
                "model": model,
                "dataset": dataset,
                "selector": selector,
                "family": family or "ALL",
                "raw_prob": float(first.get("raw_prob", float("nan"))),
                "cached_raw": int(float(first.get("raw_prob", 0)) > 0),
                "cached_top1": int(topologies[0]["cached_label"]) if topologies else 0,
                "cached_topk": int(any(t["cached_label"] for t in topologies)),
                "topologies": topologies,
            }
        )
    return rows


def load_live_dependencies(project_dir: Path):
    sys.path.insert(0, str(project_dir))
    from nmi_may27_experiments import day4_live_validation as live

    return live


def load_clean_test_set(project_dir: Path, dataset: str) -> list[dict[str, Any]]:
    sys.path.insert(0, str(project_dir))
    import llm_utils.load_data as load_ds

    _train, test = load_ds.load_cleanDS(_dataCard=dataset)
    return test


def load_queries(project_dir: Path, dataset: str, manifest: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    test = load_clean_test_set(project_dir, dataset)
    out = {}
    for row in manifest:
        idx = int(row["query_key_short"])
        item = test[idx]
        out[row["query_key"]] = {
            "query": str(item.get("query", "")),
            "answer": str(item.get("label") or item.get("cot_content") or ""),
        }
    return out


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def answer_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def strip_boxed(text: str) -> str:
    text = str(text)
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1]
    return text


def final_answer_fragment(text: str) -> str:
    text = strip_boxed(str(text)).strip()
    patterns = [
        r"final answer(?: is|:)?\s*([^\n\.]+)",
        r"answer(?: is|:)?\s*([^\n\.]+)",
        r"therefore[:,\s]+([^\n\.]+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def normalize_math_answer(text: str) -> str:
    frag = final_answer_fragment(text)
    frag = frag.replace("$", "").replace(",", "").strip()
    frag = re.sub(r"\\(?:left|right)", "", frag)
    frag = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", frag)
    frag = re.sub(r"\\[a-zA-Z]+\{([^{}]+)\}", r"\1", frag)
    frag = frag.strip().strip(".。,:;")
    frag = re.sub(r"\s+", "", frag)
    return frag.lower()


def numeric_value(text: str) -> Fraction | None:
    value = normalize_math_answer(text)
    value = value.replace("\\frac", "frac")
    frac_match = re.fullmatch(r"frac\{?(-?\d+)\}?\{?(-?\d+)\}?", value)
    if frac_match:
        den = int(frac_match.group(2))
        return Fraction(int(frac_match.group(1)), den) if den != 0 else None
    simple = re.fullmatch(r"-?\d+(?:/\d+)?", value)
    if simple:
        try:
            return Fraction(value)
        except Exception:
            return None
    dec = re.fullmatch(r"-?\d+(?:\.\d+)?", value)
    if dec:
        try:
            return Fraction(value)
        except Exception:
            return None
    return None


def extract_choice(text: str) -> str:
    text = str(text).strip()
    boxed = normalize_math_answer(text).upper()
    if boxed in {"A", "B", "C", "D", "E"}:
        return boxed
    matches = re.findall(r"(?:answer|option|choice)(?:\s+is|:)?\s*\(?([A-E])\)?", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"\b([A-E])\b", text.upper())
    return matches[-1].upper() if matches else ""


def score_prediction(dataset: str, prediction: str, answer: str) -> dict[str, Any]:
    dataset = dataset.lower()
    if dataset in {"aime", "math500", "math"}:
        pred_norm = normalize_math_answer(prediction)
        ans_norm = normalize_math_answer(answer)
        pred_val = numeric_value(prediction)
        ans_val = numeric_value(answer)
        exact = int(pred_norm == ans_norm or (pred_val is not None and ans_val is not None and pred_val == ans_val))
        return {
            "clean_pred": final_answer_fragment(prediction)[:500],
            "normalized_pred": pred_norm,
            "normalized_answer": ans_norm,
            "exact": float(exact),
            "contains_answer": float(ans_norm in pred_norm if ans_norm else False),
            "f1": float(exact),
            "is_correct": int(exact),
        }
    if dataset == "gpqa":
        pred_choice = extract_choice(prediction)
        ans_choice = extract_choice(answer)
        exact = int(bool(pred_choice) and bool(ans_choice) and pred_choice == ans_choice)
        f1 = answer_f1(prediction, answer)
        return {
            "clean_pred": str(prediction).strip()[:500],
            "normalized_pred": pred_choice,
            "normalized_answer": ans_choice,
            "exact": float(exact),
            "contains_answer": float(exact),
            "f1": f1,
            "is_correct": int(exact or f1 >= 0.8),
        }
    norm_pred = normalize_text(prediction)
    norm_answer = normalize_text(answer)
    exact = float(norm_pred == norm_answer)
    contains = float(bool(norm_answer) and norm_answer in norm_pred)
    f1 = answer_f1(prediction, answer)
    return {
        "clean_pred": str(prediction).strip()[:500],
        "normalized_pred": norm_pred[:500],
        "normalized_answer": norm_answer[:500],
        "exact": exact,
        "contains_answer": contains,
        "f1": f1,
        "is_correct": int(exact == 1.0 or contains == 1.0 or f1 >= 0.8),
    }


def run_scorer_unit_checks() -> list[dict[str, Any]]:
    cases = [
        ("aime", r"The final answer is \boxed{42}.", "42", 1),
        ("aime", "Answer: 1/2", r"\frac{1}{2}", 1),
        ("math500", "final answer: -7.", "-7", 1),
        ("math500", "The result is 7", "8", 0),
        ("gpqa", "I choose option C.", "C", 1),
        ("hotpot-qa", "Albert Einstein", "Einstein", 1),
    ]
    rows = []
    for dataset, pred, ans, expected in cases:
        stat = score_prediction(dataset, pred, ans)
        rows.append({"dataset": dataset, "prediction": pred, "answer": ans, "expected": expected, "observed": stat["is_correct"]})
    return rows


def get_submodule(root: Any, dotted_name: str) -> Any:
    cur = root
    for part in dotted_name.split("."):
        cur = getattr(cur, part)
    return cur


def set_submodule(root: Any, dotted_name: str, module: Any) -> None:
    parent_name, child_name = dotted_name.rsplit(".", 1)
    parent = get_submodule(root, parent_name)
    setattr(parent, child_name, module)


def cleanup_cuda(torch_mod: Any) -> None:
    gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()
        torch_mod.cuda.empty_cache()
        torch_mod.cuda.synchronize()


def reset_peak(torch_mod: Any) -> None:
    if torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()
        torch_mod.cuda.reset_peak_memory_stats()


def peak_memory_gb(torch_mod: Any) -> float:
    if not torch_mod.cuda.is_available():
        return 0.0
    torch_mod.cuda.synchronize()
    return float(torch_mod.cuda.max_memory_allocated() / 1024**3)


def generate_answer(torch_mod: Any, model: Any, tokenizer: Any, query: str, max_new_tokens: int) -> tuple[str, float, int]:
    prompt = query + "\n\nAnswer with only the final answer. Do not include reasoning."
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch_mod.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()
    seconds = time.perf_counter() - start
    new_ids = output_ids[0][len(inputs.input_ids[0]) :]
    output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return output, seconds, int(new_ids.numel())


def load_model(project_dir: Path, model_name: str):
    sys.path.insert(0, str(project_dir))
    import torch
    import llm_utils.load_llm as load_llm

    model, tokenizer, layers = load_llm.get_llm(model_name)
    model.eval()
    return torch, model, tokenizer, layers


def load_wtp_records(project_dir: Path, model_name: str, wtp_file: str):
    sys.path.insert(0, str(project_dir))
    import torch

    wtp_rows = read_wtp_json(project_dir, model_name, wtp_file)
    checkpoint_root = project_dir / "nsys-checkpoints-record" / model_name

    def map_func(storage: Any, loc: Any) -> Any:
        return storage.cuda()

    records = {}
    for nsys_name, nt_name, y_errs in wtp_rows:
        ckpt = checkpoint_root / nsys_name / f"{nt_name}#{';'.join(str(v) for v in y_errs)}.pt"
        records[nt_name] = torch.load(ckpt, map_location=map_func, weights_only=False)
    return records


def materialize_nsys(model: Any, nsys_records: dict[str, Any], project_dir: Path) -> None:
    sys.path.insert(0, str(project_dir))
    import nsys_utils.nsys_config as nsys_config

    nsys_config.replace_Linear_with_nSys_v2(model, nsys_records)
    model.eval()


def collect_target_blocks(wtp_rows: list[list[Any]], model: str, role: str, max_blocks: int) -> list[str]:
    candidates = []
    for item in iter_wtp_blocks(wtp_rows):
        block = item["block_name"]
        if block_role(block, model) == role:
            candidates.append(block)
    seen = []
    for block in candidates:
        if block not in seen:
            seen.append(block)
    return seen[:max_blocks]


def collect_matched_non_role_blocks(wtp_rows: list[list[Any]], model: str, max_blocks: int, seed: int) -> list[str]:
    blocks = []
    for item in iter_wtp_blocks(wtp_rows):
        block = item["block_name"]
        if block_role(block, model) == "non_role":
            blocks.append(block)
    blocks = sorted(set(blocks))
    rng = random.Random(seed)
    rng.shuffle(blocks)
    return sorted(blocks[:max_blocks])
