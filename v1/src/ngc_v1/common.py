from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
V1_ROOT = Path(os.environ.get("NGC_V1_ROOT", REPO_ROOT / "v1"))
V0_ROOT = Path(os.environ.get("NGC_V0_ROOT", REPO_ROOT / "external" / "NGC-v0"))
RANDOM_SEED = 20260528

NEGATIVE_GROUPS = {
    ("qwen3-4b", "aime"),
    ("qwen3-8b", "aime"),
    ("qwen3-8b", "mmlu-pro"),
    ("qwen3-8b", "sci-qa"),
    ("llama3-8b", "mmlu-pro"),
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def run_capture(cmd: Sequence[str], timeout: int = 30) -> str:
    try:
        proc = subprocess.run(
            list(cmd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return proc.stdout
    except Exception as exc:  # pragma: no cover - inventory must not fail hard here
        return f"FAILED: {cmd}: {exc}\n"


def python_env_snapshot() -> Dict[str, object]:
    mods = ["pandas", "numpy", "sklearn", "matplotlib", "yaml", "tqdm", "joblib", "pyarrow", "seaborn"]
    found = {}
    for mod in mods:
        try:
            spec = __import__("importlib").util.find_spec(mod)
            found[mod] = spec is not None
        except Exception:
            found[mod] = False
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "module_available": found,
        "cwd": str(Path.cwd()),
    }


def robust_z(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce")
    med = arr.median(skipna=True)
    mad = (arr - med).abs().median(skipna=True)
    if not np.isfinite(mad) or mad < 1e-12:
        std = arr.std(skipna=True)
        if not np.isfinite(std) or std < 1e-12:
            return pd.Series(np.zeros(len(arr)), index=values.index, dtype=float)
        return (arr - arr.mean(skipna=True)) / std
    return 0.67448975 * (arr - med) / mad


def minmax01(values: pd.Series) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce")
    lo = arr.min(skipna=True)
    hi = arr.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.full(len(arr), 0.5), index=values.index, dtype=float)
    return (arr - lo) / (hi - lo)


def role_entropy_frame(df: pd.DataFrame) -> pd.Series:
    cols = ["attn_frac", "mlp_frac", "kv_frac", "q_frac"]
    vals = df[cols].clip(lower=0).to_numpy(dtype=float)
    sums = vals.sum(axis=1, keepdims=True)
    sums[sums <= 1e-12] = 1.0
    probs = vals / sums
    ent = -(probs * np.log(probs + 1e-12)).sum(axis=1) / math.log(len(cols))
    return pd.Series(ent, index=df.index)


def add_constitutional_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["query_key"] = out["model"].astype(str) + "::" + out["dataset"].astype(str) + "::" + out["query_id"].astype(str)
    out["model_dataset"] = out["model"].astype(str) + "::" + out["dataset"].astype(str)
    out["candidate_uid"] = out["query_key"] + "::cand" + out["candidate_index"].astype(str)
    out["is_negative_group"] = [
        (m, d) in NEGATIVE_GROUPS for m, d in zip(out["model"].astype(str), out["dataset"].astype(str))
    ]

    grouped = out.groupby(["model", "dataset"], group_keys=False)
    out["force_ratio_z"] = grouped["force_ratio"].transform(robust_z)
    out["transport_shift_z"] = grouped["transport_shift_log10"].transform(robust_z)
    out["stability_risk_z"] = grouped["stability_risk_log10"].transform(robust_z)
    out["reconstruction_error_z"] = grouped["reconstruction_error_log10"].transform(robust_z)
    out["mean_layer_norm"] = grouped["mean_layer"].transform(minmax01)

    out["departure_index"] = out["force_ratio_z"] + out["transport_shift_z"]
    out["stability_guard"] = -out["stability_risk_z"] - out["reconstruction_error_z"]
    out["evidence_broker_score"] = out["kv_frac"].fillna(0.0) * (1.0 - out["mean_layer_norm"].fillna(0.5))
    out["mediator_score"] = out["spectral_flatness"].fillna(0.0) - out["transport_shift_z"].abs().fillna(0.0)
    out["settlement_score"] = (out["mlp_frac"].fillna(0.0) + out["q_frac"].fillna(0.0)) * out[
        "mean_layer_norm"
    ].fillna(0.5)
    out["constitutional_balance"] = (
        out["departure_index"].fillna(0.0) + out["stability_guard"].fillna(0.0) + out["mediator_score"].fillna(0.0)
    )
    out["role_entropy"] = role_entropy_frame(out)
    out["early_kv_mass"] = out["evidence_broker_score"]
    out["late_q_mass"] = out["q_frac"].fillna(0.0) * out["mean_layer_norm"].fillna(0.5)
    out["late_mlp_mass"] = out["mlp_frac"].fillna(0.0) * out["mean_layer_norm"].fillna(0.5)
    return out


def read_candidate_metrics(v0_root: Path = V0_ROOT) -> pd.DataFrame:
    path = v0_root / "nmi_may27_experiments/outputs/day2/candidate_metrics_holdout.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate metrics: {path}")
    return pd.read_csv(path)


def read_observables(v1_root: Path = V1_ROOT, v0_root: Path = V0_ROOT) -> pd.DataFrame:
    parquet_path = v1_root / "outputs/01_observables/constitutional_candidates.parquet"
    csv_path = v1_root / "outputs/01_observables/constitutional_candidates.csv.gz"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return add_constitutional_features(read_candidate_metrics(v0_root))


def combinatorial_pass_at_k(labels: Sequence[int], k: int) -> float:
    labels = list(labels)
    n = len(labels)
    c = int(np.sum(labels))
    if n == 0 or c == 0:
        return 0.0
    if k >= n:
        return 1.0
    bad = n - c
    if bad < k:
        return 1.0
    # Product form avoids large integer combinations.
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (bad - i) / (n - i)
    return 1.0 - prob_all_wrong


def query_metrics_for_score(
    df: pd.DataFrame,
    score_col: str,
    top_k: int = 5,
    ascending: bool = False,
    query_col: str = "query_key",
) -> pd.DataFrame:
    records = []
    for q, g in df.groupby(query_col, sort=False):
        sg = g.sort_values(score_col, ascending=ascending, kind="mergesort")
        labels = sg["label"].to_numpy(dtype=float)
        head = labels[: min(top_k, len(labels))]
        first = sg.iloc[0]
        records.append(
            {
                "query_key": q,
                "model": first["model"],
                "dataset": first["dataset"],
                "raw": float(first.get("raw_prob", np.nan)),
                "pass1": float(labels[0]) if len(labels) else np.nan,
                "passk": float(np.nanmax(head)) if len(head) else np.nan,
                "mean_at_k": float(np.nanmean(head)) if len(head) else np.nan,
                "n_candidates": int(len(labels)),
                "n_correct_candidates": int(np.nansum(labels)),
            }
        )
    return pd.DataFrame.from_records(records)


def random_expected_query_metrics(df: pd.DataFrame, top_k: int = 5, query_col: str = "query_key") -> pd.DataFrame:
    records = []
    for q, g in df.groupby(query_col, sort=False):
        first = g.iloc[0]
        labels = g["label"].to_numpy(dtype=int)
        records.append(
            {
                "query_key": q,
                "model": first["model"],
                "dataset": first["dataset"],
                "raw": float(first.get("raw_prob", np.nan)),
                "pass1": float(labels.mean()) if len(labels) else np.nan,
                "passk": combinatorial_pass_at_k(labels, top_k),
                "mean_at_k": float(labels.mean()) if len(labels) else np.nan,
                "n_candidates": int(len(labels)),
                "n_correct_candidates": int(labels.sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_query_metrics(name: str, per_query: pd.DataFrame) -> Dict[str, float]:
    return {
        "selector": name,
        "n_queries": int(len(per_query)),
        "raw_pass": float(per_query["raw"].mean()),
        "pass1": float(per_query["pass1"].mean()),
        "pass5": float(per_query["passk"].mean()),
        "mean_at_5": float(per_query["mean_at_k"].mean()),
        "gain_pass5_vs_raw": float((per_query["passk"] - per_query["raw"]).mean()),
    }


def paired_bootstrap(
    left: pd.Series,
    right: pd.Series,
    n_boot: int = 10000,
    seed: int = RANDOM_SEED,
) -> Dict[str, float]:
    left_arr = left.to_numpy(dtype=float)
    right_arr = right.to_numpy(dtype=float)
    mask = np.isfinite(left_arr) & np.isfinite(right_arr)
    left_arr = left_arr[mask]
    right_arr = right_arr[mask]
    n = len(left_arr)
    if n == 0:
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_boot": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    delta = left_arr - right_arr
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = delta[idx].mean()
    obs = float(delta.mean())
    p_low = float((diffs <= 0).mean())
    p_high = float((diffs >= 0).mean())
    p_boot = min(1.0, float(2 * min(p_low, p_high)))
    return {
        "diff": obs,
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_boot": p_boot,
        "n": int(n),
    }


def rank_auc_by_query(df: pd.DataFrame, score_col: str, query_col: str = "query_key") -> float:
    from sklearn.metrics import roc_auc_score

    aucs = []
    weights = []
    for _, g in df.groupby(query_col, sort=False):
        y = g["label"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y, g[score_col].to_numpy(dtype=float))))
            weights.append(len(g))
        except ValueError:
            continue
    if not aucs:
        return float("nan")
    return float(np.average(aucs, weights=weights))


def split_queries(query_keys: Sequence[str], seed: int = RANDOM_SEED) -> Tuple[set, set, set]:
    keys = np.array(sorted(set(map(str, query_keys))))
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(round(0.60 * n))
    n_val = int(round(0.20 * n))
    train = set(keys[:n_train])
    val = set(keys[n_train : n_train + n_val])
    test = set(keys[n_train + n_val :])
    return train, val, test


def safe_to_parquet(df: pd.DataFrame, path: Path) -> bool:
    try:
        ensure_dir(path.parent)
        df.to_parquet(path, index=False)
        return True
    except Exception as exc:
        (path.with_suffix(path.suffix + ".error.txt")).write_text(str(exc), encoding="utf-8")
        return False


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt
