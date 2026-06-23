from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


RANDOM_SEED = 20260601


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def safe_to_parquet(df: pd.DataFrame, path: Path) -> bool:
    try:
        ensure_dir(path.parent)
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def parse_tcr(value: str | float | int | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"tCR#([0-9.]+)", str(value))
    return float(match.group(1)) if match else float("nan")


def topology_family(wtp_file: str) -> str:
    name = Path(str(wtp_file)).name
    if "WTP#SVD#" in name:
        return "Vanilla SVD"
    if "WTP#BS#" in name:
        return "Basis Sharing"
    return "NGC"


def compression_bin(tcr: float, targets: Sequence[float] = (80.0, 85.0, 90.0, 95.0)) -> float:
    if not np.isfinite(tcr):
        return float("nan")
    return min(targets, key=lambda x: abs(float(tcr) - x))


def grouped_query_split(
    query_keys: Iterable[str],
    seed: int = RANDOM_SEED,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, str]:
    keys = np.array(sorted(set(map(str, query_keys))), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    split = {}
    for key in keys[:n_train]:
        split[str(key)] = "train"
    for key in keys[n_train : n_train + n_val]:
        split[str(key)] = "val"
    for key in keys[n_train + n_val :]:
        split[str(key)] = "test"
    return split


def paired_bootstrap(
    left: pd.Series,
    right: pd.Series,
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    left_arr = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_arr = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(left_arr) & np.isfinite(right_arr)
    left_arr = left_arr[mask]
    right_arr = right_arr[mask]
    n = len(left_arr)
    if n == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_boot": float("nan"), "n": 0}
    delta = left_arr - right_arr
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = delta[idx].mean()
    p_low = float((diffs <= 0).mean())
    p_high = float((diffs >= 0).mean())
    return {
        "diff": float(delta.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_boot": min(1.0, 2.0 * min(p_low, p_high)),
        "n": int(n),
    }


def query_metrics_for_score(
    df: pd.DataFrame,
    score_col: str,
    top_k: int = 5,
    ascending: bool = False,
) -> pd.DataFrame:
    rows = []
    for query_key, group in df.groupby("query_key", sort=False):
        ranked = group.sort_values(score_col, ascending=ascending, kind="mergesort")
        labels = ranked["label"].to_numpy(dtype=float)
        head = labels[: min(top_k, len(labels))]
        first = ranked.iloc[0]
        rows.append(
            {
                "query_key": query_key,
                "model": first.get("model", ""),
                "dataset": first.get("dataset", ""),
                "raw": float(first.get("raw_prob", np.nan)),
                "pass1": float(labels[0]) if len(labels) else np.nan,
                "pass3": float(np.nanmax(labels[: min(3, len(labels))])) if len(labels) else np.nan,
                "pass5": float(np.nanmax(head)) if len(head) else np.nan,
                "pass10": float(np.nanmax(labels[: min(10, len(labels))])) if len(labels) else np.nan,
                "mean_at_5": float(np.nanmean(head)) if len(head) else np.nan,
                "selected_candidate_uid": str(first.get("candidate_uid", "")),
                "selected_tcr": float(first.get("tcr", np.nan)),
                "selected_num_tokens": float(first.get("num_tokens", np.nan)),
                "n_candidates": int(len(labels)),
                "n_correct_candidates": int(np.nansum(labels)),
            }
        )
    return pd.DataFrame(rows)


def random_query_metrics(df: pd.DataFrame, top_k: int = 5, seed: int = RANDOM_SEED, n_seeds: int = 100) -> pd.DataFrame:
    rows = []
    for repeat in range(n_seeds):
        rng = np.random.default_rng(seed + repeat)
        for query_key, group in df.groupby("query_key", sort=False):
            first = group.iloc[0]
            take = min(top_k, len(group))
            idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
            selected = group.loc[idx]
            labels = selected["label"].to_numpy(dtype=float)
            rows.append(
                {
                    "query_key": query_key,
                    "repeat": repeat,
                    "model": first["model"],
                    "dataset": first["dataset"],
                    "raw": float(first.get("raw_prob", np.nan)),
                    "pass1": float(labels[0]) if len(labels) else np.nan,
                    "pass5": float(np.nanmax(labels)) if len(labels) else np.nan,
                    "mean_at_5": float(np.nanmean(labels)) if len(labels) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_per_query(name: str, per_query: pd.DataFrame) -> dict[str, float | str | int]:
    def mean_col(col: str) -> float:
        if col not in per_query:
            return float("nan")
        return float(pd.to_numeric(per_query[col], errors="coerce").mean())

    def median_col(col: str) -> float:
        if col not in per_query:
            return float("nan")
        return float(pd.to_numeric(per_query[col], errors="coerce").median())

    return {
        "selector": name,
        "n_queries": int(len(per_query)),
        "raw_acc": mean_col("raw"),
        "top1_acc": mean_col("pass1"),
        "pass3": mean_col("pass3"),
        "pass5": mean_col("pass5"),
        "pass10": mean_col("pass10"),
        "mean_at_5": mean_col("mean_at_5"),
        "median_selected_tokens": median_col("selected_num_tokens"),
    }


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
        }
    )
    return plt


def finite_or_nan(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")
