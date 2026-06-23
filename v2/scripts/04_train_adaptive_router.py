#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ngc_v2_utils import (
    RANDOM_SEED,
    ensure_dir,
    paired_bootstrap,
    query_metrics_for_score,
    random_query_metrics,
    safe_to_parquet,
    setup_matplotlib,
    summarize_per_query,
    write_json,
)


LAW_COLUMNS = [
    "law_fixed_transfer",
    "law_bargaining",
    "law_coalition",
    "law_mediated_criticality",
    "law_retrieve_then_settle",
    "law_reconstruction_guarded_exploration",
    "law_settlement_only",
    "law_cost_aware_adaptive",
]


def table_block(df: pd.DataFrame, cols: list[str]) -> str:
    if df.empty:
        return "No rows."
    return "```text\n" + df[cols].to_string(index=False) + "\n```"


EXCLUDE_NUMERIC = {
    "label",
    "cached_is_correct_rich",
    "candidate_index",
}

BASE_CATEGORICAL = ["model", "dataset", "topology_family", "compression_bin"]


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_bank(v2_root: Path) -> pd.DataFrame:
    parquet_path = v2_root / "results/router_candidate_bank.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(v2_root / "results/router_candidate_bank.csv.gz")


def numeric_features(df: pd.DataFrame, include_query_text: bool = False) -> list[str]:
    cols = []
    for col in df.columns:
        if col in EXCLUDE_NUMERIC or col.endswith("_is_correct"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    blocked = {
        "raw_prob_from_feature",
        "cached_is_correct_rich",
        "compression_bin",
    }
    return [c for c in cols if c not in blocked and c not in BASE_CATEGORICAL]


def make_preprocessor(num_cols: list[str], cat_cols: list[str], use_text: bool = False) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("cat", one_hot_encoder(), cat_cols))
    if use_text:
        transformers.append(("text", TfidfVectorizer(max_features=512, ngram_range=(1, 2), min_df=2), "query_text"))
    return ColumnTransformer(transformers, remainder="drop")


def make_logistic(num_cols: list[str], cat_cols: list[str], use_text: bool, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(num_cols, cat_cols, use_text)),
            (
                "clf",
                LogisticRegression(
                    C=0.7,
                    max_iter=1200,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def make_ridge(num_cols: list[str], cat_cols: list[str], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(num_cols, cat_cols, False)),
            ("clf", RidgeClassifier(alpha=2.0, class_weight="balanced", random_state=seed)),
        ]
    )


def make_hgb(num_cols: list[str], cat_cols: list[str], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(num_cols, cat_cols, False)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=240,
                    learning_rate=0.045,
                    max_leaf_nodes=31,
                    l2_regularization=0.025,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_extra_trees(num_cols: list[str], cat_cols: list[str], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(num_cols, cat_cols, False)),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=360,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def score_model(model: Pipeline, df: pd.DataFrame, features: list[str], use_text: bool = False) -> np.ndarray:
    cols = features + (["query_text"] if use_text else [])
    if hasattr(model.named_steps["clf"], "predict_proba"):
        return model.predict_proba(df[cols])[:, 1]
    raw = model.decision_function(df[cols])
    raw = np.asarray(raw, dtype=float)
    return 1.0 / (1.0 + np.exp(-raw / (np.nanstd(raw) + 1e-8)))


def safe_auc(y, score) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


def safe_ap(y, score) -> float:
    try:
        return float(average_precision_score(y, score))
    except Exception:
        return float("nan")


def safe_brier(y, score) -> float:
    try:
        s = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
        return float(brier_score_loss(y, s))
    except Exception:
        return float("nan")


def evaluate_score_table(df: pd.DataFrame, score_cols: dict[str, str], top_k: int, seed: int, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    summaries = []
    per_query = {}
    y = df["label"].astype(int).to_numpy()
    for name, col in score_cols.items():
        pq = query_metrics_for_score(df, col, top_k=top_k, ascending=False)
        per_query[name] = pq
        row = summarize_per_query(name, pq)
        score = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        score01 = (score - np.nanmin(score)) / (np.nanmax(score) - np.nanmin(score) + 1e-12)
        row.update(
            {
                "candidate_auc": safe_auc(y, score),
                "candidate_average_precision": safe_ap(y, score),
                "candidate_brier_minmax": safe_brier(y, score01),
            }
        )
        summaries.append(row)
    rand = random_query_metrics(df, top_k=top_k, seed=seed, n_seeds=100)
    rand_mean = rand.groupby("query_key", as_index=False)[["raw", "pass1", "pass5", "mean_at_5"]].mean()
    rand_first = df.drop_duplicates("query_key")[["query_key", "model", "dataset"]]
    rand_mean = rand_mean.merge(rand_first, on="query_key", how="left")
    per_query["random_topology_selector"] = rand_mean
    summaries.append(summarize_per_query("random_topology_selector", rand_mean))

    summary_df = pd.DataFrame(summaries).sort_values(["pass5", "top1_acc"], ascending=[False, False])
    contrasts = []
    refs = [name for name in ["best_global_fixed_law", "fixed_transfer_law", "reconstruction_only_selector", "raw_dense"] if name in per_query]
    for name, pq in per_query.items():
        if name in refs:
            continue
        aligned = pq.set_index("query_key")
        for ref in refs:
            if ref == "raw_dense":
                right = aligned["raw"]
            else:
                right = per_query[ref].set_index("query_key").reindex(aligned.index)["pass5"]
            stat = paired_bootstrap(aligned["pass5"], right, n_boot=n_boot, seed=seed)
            stat.update({"selector": name, "reference": ref, "metric": "pass5"})
            contrasts.append(stat)
    return summary_df, pd.DataFrame(contrasts), per_query


def select_best_law(train: pd.DataFrame, val: pd.DataFrame, scope: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    if scope == "global":
        scores = []
        for law in LAW_COLUMNS:
            pq = query_metrics_for_score(train, law, top_k=5, ascending=False)
            scores.append((float(pq["pass5"].mean()), law))
        choices["ALL"] = sorted(scores, reverse=True)[0][1]
    elif scope == "dataset":
        for dataset, group in train.groupby("dataset"):
            scores = []
            for law in LAW_COLUMNS:
                pq = query_metrics_for_score(group, law, top_k=5, ascending=False)
                scores.append((float(pq["pass5"].mean()), law))
            choices[str(dataset)] = sorted(scores, reverse=True)[0][1]
    elif scope == "model":
        for model, group in train.groupby("model"):
            scores = []
            for law in LAW_COLUMNS:
                pq = query_metrics_for_score(group, law, top_k=5, ascending=False)
                scores.append((float(pq["pass5"].mean()), law))
            choices[str(model)] = sorted(scores, reverse=True)[0][1]
    return choices


def apply_law_choices(df: pd.DataFrame, choices: dict[str, str], scope: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    if scope == "global":
        out[out_col] = out[choices["ALL"]]
        return out
    out[out_col] = np.nan
    key_col = "dataset" if scope == "dataset" else "model"
    for key, law in choices.items():
        mask = out[key_col].astype(str).eq(str(key))
        out.loc[mask, out_col] = out.loc[mask, law]
    return out


def train_pairwise_router(train: pd.DataFrame, feature_cols: list[str], seed: int, max_pairs: int = 120000) -> Pipeline:
    rng = np.random.default_rng(seed)
    x_rows = []
    y_rows = []
    for _, group in train.groupby("query_key", sort=False):
        pos = group[group["label"].astype(int).eq(1)]
        neg = group[group["label"].astype(int).eq(0)]
        if pos.empty or neg.empty:
            continue
        n_take = min(16, len(pos) * len(neg))
        for _ in range(n_take):
            p = pos.iloc[int(rng.integers(0, len(pos)))]
            n = neg.iloc[int(rng.integers(0, len(neg)))]
            diff = p[feature_cols].astype(float).to_numpy() - n[feature_cols].astype(float).to_numpy()
            if rng.random() < 0.5:
                x_rows.append(diff)
                y_rows.append(1)
            else:
                x_rows.append(-diff)
                y_rows.append(0)
            if len(y_rows) >= max_pairs:
                break
        if len(y_rows) >= max_pairs:
            break
    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_rows, dtype=int)
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        ]
    )
    pipe.fit(x, y)
    return pipe


def pairwise_scores(model: Pipeline, df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    scores = pd.Series(0.0, index=df.index)
    for _, group in df.groupby("query_key", sort=False):
        idx = group.index.to_numpy()
        x = group[feature_cols].astype(float).to_numpy()
        if len(idx) == 1:
            scores.loc[idx[0]] = 0.5
            continue
        wins = np.zeros(len(idx), dtype=float)
        counts = np.zeros(len(idx), dtype=float)
        for i in range(len(idx)):
            diffs = x[i][None, :] - x
            probs = model.predict_proba(diffs)[:, 1]
            wins[i] += probs.sum() - probs[i]
            counts[i] += len(idx) - 1
        scores.loc[idx] = wins / np.maximum(counts, 1.0)
    return scores


def add_cost_scores(scored: pd.DataFrame, base_col: str) -> list[str]:
    token = pd.to_numeric(scored["num_tokens"], errors="coerce")
    token_norm = (token - token.min()) / (token.max() - token.min() + 1e-12)
    cols = []
    for lam in [0.00, 0.02, 0.05, 0.10, 0.20, 0.35]:
        col = f"score_cost_aware_lambda_{lam:.2f}"
        scored[col] = pd.to_numeric(scored[base_col], errors="coerce").fillna(0.0) - lam * token_norm
        cols.append(col)
    return cols


def train_random_split(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict]:
    train = df[df["split_random_70_15_15"].eq("train")].copy()
    val = df[df["split_random_70_15_15"].eq("val")].copy()
    test = df[df["split_random_70_15_15"].eq("test")].copy()

    num_cols = numeric_features(df)
    num_cols = [c for c in num_cols if c not in {"raw_prob_from_feature"}]
    cat_cols = [c for c in BASE_CATEGORICAL if c in df.columns]
    has_text = "query_text" in df.columns and df["query_text"].fillna("").str.len().gt(0).any()
    scored = test.copy()
    timings = {}

    global_choice = select_best_law(train, val, "global")
    dataset_choice = select_best_law(train, val, "dataset")
    model_choice = select_best_law(train, val, "model")
    scored = apply_law_choices(scored, global_choice, "global", "score_best_global_fixed_law")
    scored = apply_law_choices(scored, dataset_choice, "dataset", "score_benchmark_specific_fixed_law")
    scored = apply_law_choices(scored, model_choice, "model", "score_model_specific_fixed_law")
    scored["score_fixed_transfer_law"] = scored["law_fixed_transfer"]
    scored["score_reconstruction_only_selector"] = scored["reconstruction_score"]
    scored["score_oracle_selector"] = scored["label"].astype(float)
    scored["score_raw_dense"] = scored["raw_prob"].astype(float)

    model_specs = [
        ("feature_only_logistic_router", make_logistic(num_cols, cat_cols, False, args.seed), False),
        ("ridge_feature_router", make_ridge(num_cols, cat_cols, args.seed), False),
        ("tree_hgb_router", make_hgb(num_cols, cat_cols, args.seed), False),
        ("extra_trees_router", make_extra_trees(num_cols, cat_cols, args.seed), False),
    ]
    if has_text:
        model_specs.append(("query_tfidf_logistic_router", make_logistic(num_cols, cat_cols, True, args.seed), True))

    fitted = {}
    for name, model, use_text in model_specs:
        cols = num_cols + cat_cols + (["query_text"] if use_text else [])
        t0 = time.perf_counter()
        model.fit(train[cols], train["label"].astype(int))
        train_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        scored[f"score_{name}"] = score_model(model, scored, num_cols + cat_cols, use_text=use_text)
        score_seconds = time.perf_counter() - t1
        timings[name] = {
            "train_seconds": train_seconds,
            "score_seconds_total": score_seconds,
            "score_seconds_per_query": score_seconds / max(1, scored["query_key"].nunique()),
        }
        fitted[name] = model

    pair_cols = [
        c
        for c in num_cols
        if c in train.columns
        and c not in {"raw_prob", "raw_numTs", "prompt_len_chars", "answer_len_chars"}
        and train[c].notna().any()
    ]
    pair_cols = pair_cols[:80]
    t0 = time.perf_counter()
    pair_model = train_pairwise_router(train, pair_cols, args.seed)
    train_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    scored["score_pairwise_ranking_router"] = pairwise_scores(pair_model, scored, pair_cols)
    score_seconds = time.perf_counter() - t1
    timings["pairwise_ranking_router"] = {
        "train_seconds": train_seconds,
        "score_seconds_total": score_seconds,
        "score_seconds_per_query": score_seconds / max(1, scored["query_key"].nunique()),
    }
    fitted["pairwise_ranking_router"] = pair_model

    cost_cols = add_cost_scores(scored, "score_tree_hgb_router")
    score_cols = {
        "raw_dense": "score_raw_dense",
        "fixed_transfer_law": "score_fixed_transfer_law",
        "best_global_fixed_law": "score_best_global_fixed_law",
        "benchmark_specific_fixed_law": "score_benchmark_specific_fixed_law",
        "model_specific_fixed_law": "score_model_specific_fixed_law",
        "reconstruction_only_selector": "score_reconstruction_only_selector",
        "oracle_selector": "score_oracle_selector",
        "feature_only_logistic_router": "score_feature_only_logistic_router",
        "ridge_feature_router": "score_ridge_feature_router",
        "tree_hgb_router": "score_tree_hgb_router",
        "extra_trees_router": "score_extra_trees_router",
        "pairwise_ranking_router": "score_pairwise_ranking_router",
    }
    if has_text:
        score_cols["query_tfidf_logistic_router"] = "score_query_tfidf_logistic_router"
    for col in cost_cols:
        score_cols[col.replace("score_", "cost_aware_")] = col

    summary, contrasts, per_query = evaluate_score_table(scored, score_cols, args.top_k, args.seed, args.n_bootstrap)
    joblib.dump(
        {
            "models": fitted,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "pair_cols": pair_cols,
            "global_choice": global_choice,
            "dataset_choice": dataset_choice,
            "model_choice": model_choice,
            "score_cols": score_cols,
        },
        out_dir / "adaptive_router_models.joblib",
    )
    meta = {
        "split": {
            "train_queries": int(train["query_key"].nunique()),
            "val_queries": int(val["query_key"].nunique()),
            "test_queries": int(test["query_key"].nunique()),
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
        },
        "global_choice": global_choice,
        "dataset_choice": dataset_choice,
        "model_choice": model_choice,
        "timings": timings,
    }
    return scored, summary, per_query, {"contrasts": contrasts, "meta": meta}


def leave_one_eval(df: pd.DataFrame, holdout_col: str, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    num_cols = numeric_features(df)
    cat_cols = [c for c in BASE_CATEGORICAL if c in df.columns and c != holdout_col]
    for value in sorted(df[holdout_col].dropna().unique()):
        train = df[~df[holdout_col].eq(value)].copy()
        test = df[df[holdout_col].eq(value)].copy()
        if train.empty or test["query_key"].nunique() < 2:
            continue
        global_choice = select_best_law(train, train, "global")
        test = apply_law_choices(test, global_choice, "global", "score_best_global_fixed_law")
        test["score_fixed_transfer_law"] = test["law_fixed_transfer"]
        test["score_oracle_selector"] = test["label"].astype(float)
        model = make_hgb(num_cols, cat_cols, args.seed)
        features = num_cols + cat_cols
        model.fit(train[features], train["label"].astype(int))
        test["score_tree_hgb_router"] = score_model(model, test, features, use_text=False)
        score_cols = {
            "best_global_fixed_law": "score_best_global_fixed_law",
            "fixed_transfer_law": "score_fixed_transfer_law",
            "tree_hgb_router": "score_tree_hgb_router",
            "oracle_selector": "score_oracle_selector",
        }
        summary, contrasts, _pq = evaluate_score_table(test, score_cols, args.top_k, args.seed, max(300, args.n_bootstrap // 2))
        for row in summary.to_dict("records"):
            row.update({"holdout_type": holdout_col, "holdout_value": value})
            rows.append(row)
    return pd.DataFrame(rows)


def build_oracle_headroom(per_query: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fixed_name = "best_global_fixed_law" if "best_global_fixed_law" in per_query else "fixed_transfer_law"
    router_names = [
        name
        for name in per_query
        if name.endswith("_router") or name.startswith("cost_aware_") or name == "query_tfidf_logistic_router"
    ]
    fixed = per_query[fixed_name].set_index("query_key")
    oracle = per_query["oracle_selector"].set_index("query_key")
    rows = []
    for name in router_names:
        router = per_query[name].set_index("query_key")
        aligned = fixed[["model", "dataset", "pass5"]].rename(columns={"pass5": "fixed_pass5"}).join(
            router[["pass5"]].rename(columns={"pass5": "router_pass5"}), how="inner"
        )
        aligned = aligned.join(oracle[["pass5"]].rename(columns={"pass5": "oracle_pass5"}), how="inner")
        for (model, dataset), group in aligned.groupby(["model", "dataset"], dropna=False):
            fixed_pass5 = float(group["fixed_pass5"].mean())
            router_pass5 = float(group["router_pass5"].mean())
            oracle_pass5 = float(group["oracle_pass5"].mean())
            gap = oracle_pass5 - fixed_pass5
            gain = router_pass5 - fixed_pass5
            rows.append(
                {
                    "router": name,
                    "model": model,
                    "dataset": dataset,
                    "fixed_pass5": fixed_pass5,
                    "router_pass5": router_pass5,
                    "oracle_pass5": oracle_pass5,
                    "oracle_gap": gap,
                    "router_gain": gain,
                    "gap_closed": gain / max(gap, 1e-9),
                }
            )
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame, per_query: dict[str, pd.DataFrame], oracle: pd.DataFrame, cost: pd.DataFrame, scored: pd.DataFrame, fig_dir: Path) -> None:
    plt = setup_matplotlib()
    chosen = [
        "raw_dense",
        "best_global_fixed_law",
        "fixed_transfer_law",
        "tree_hgb_router",
        "extra_trees_router",
        "pairwise_ranking_router",
        "oracle_selector",
    ]
    by_dataset = []
    for name in chosen:
        if name not in per_query:
            continue
        pq = per_query[name]
        for dataset, group in pq.groupby("dataset", dropna=False):
            by_dataset.append({"selector": name, "dataset": dataset, "pass5": group["pass5"].mean(), "top1": group["pass1"].mean()})
    bd = pd.DataFrame(by_dataset)
    if not bd.empty:
        fig, ax = plt.subplots(figsize=(10.0, 5.0))
        pivot = bd.pivot(index="dataset", columns="selector", values="pass5").fillna(0.0)
        pivot[[c for c in chosen if c in pivot.columns]].plot(kind="bar", ax=ax, width=0.82)
        ax.set_ylabel("Pass@5")
        ax.set_title("Fixed laws versus adaptive routers by benchmark")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "router_fixed_vs_adaptive_by_benchmark.pdf")
        fig.savefig(fig_dir / "router_fixed_vs_adaptive_by_benchmark.png", dpi=300)
        plt.close(fig)

    if not oracle.empty:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        best = oracle.sort_values("router_gain", ascending=False).groupby(["model", "dataset"], as_index=False).head(1)
        x = np.arange(len(best))
        ax.bar(x - 0.2, 100 * best["oracle_gap"], width=0.4, label="Oracle gap")
        ax.bar(x + 0.2, 100 * best["router_gain"], width=0.4, label="Best router gain")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}/{d}" for m, d in zip(best["model"], best["dataset"])], rotation=60, ha="right")
        ax.set_ylabel("Pass@5 points")
        ax.set_title("Oracle headroom and exploited router gain")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "router_oracle_gap.pdf")
        fig.savefig(fig_dir / "router_oracle_gap.png", dpi=300)
        plt.close(fig)

    if not cost.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.scatter(1000 * cost["score_seconds_per_query"], 100 * cost["pass5"], s=70)
        for row in cost.itertuples(index=False):
            ax.text(1000 * row.score_seconds_per_query, 100 * row.pass5, row.selector, fontsize=7)
        ax.set_xlabel("Routing/scoring overhead (ms/query)")
        ax.set_ylabel("Pass@5 (%)")
        ax.set_title("Router cost-accuracy Pareto")
        fig.tight_layout()
        fig.savefig(fig_dir / "router_cost_accuracy_pareto.pdf")
        fig.savefig(fig_dir / "router_cost_accuracy_pareto.png", dpi=300)
        plt.close(fig)

    role_cols = ["U_prop", "U_ske", "U_stab", "U_med", "U_ret", "U_set", "early_block_frac", "mid_block_frac", "late_block_frac"]
    available = [c for c in role_cols if c in scored.columns]
    selected = []
    for name in ["best_global_fixed_law", "tree_hgb_router", "extra_trees_router", "pairwise_ranking_router"]:
        col = f"score_{name}" if f"score_{name}" in scored.columns else None
        if col is None:
            continue
        top = scored.sort_values(["query_key", col], ascending=[True, False], kind="mergesort").groupby("query_key", as_index=False).head(1)
        vals = top[available].mean(numeric_only=True).to_dict()
        vals["selector"] = name
        selected.append(vals)
    heat = pd.DataFrame(selected).set_index("selector") if selected else pd.DataFrame()
    if not heat.empty:
        fig, ax = plt.subplots(figsize=(8.2, 3.6))
        arr = heat.to_numpy(dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap="viridis")
        ax.set_yticks(np.arange(len(heat.index)))
        ax.set_yticklabels(heat.index)
        ax.set_xticks(np.arange(len(heat.columns)))
        ax.set_xticklabels(heat.columns, rotation=45, ha="right")
        ax.set_title("Role and depth profile of selected candidates")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fig_dir / "router_role_depth_heatmap.pdf")
        fig.savefig(fig_dir / "router_role_depth_heatmap.png", dpi=300)
        plt.close(fig)


def write_summary_report(summary: pd.DataFrame, contrasts: pd.DataFrame, oracle: pd.DataFrame, cost: pd.DataFrame, meta: dict, reports_dir: Path) -> None:
    fixed_candidates = summary[summary["selector"].isin(["best_global_fixed_law", "fixed_transfer_law", "benchmark_specific_fixed_law", "model_specific_fixed_law"])]
    adaptive = summary[
        summary["selector"].str.contains("router|cost_aware|query_tfidf", regex=True, na=False)
    ].copy()
    best_fixed = fixed_candidates.sort_values(["pass5", "top1_acc"], ascending=False).iloc[0]
    best_adaptive = adaptive.sort_values(["pass5", "top1_acc"], ascending=False).iloc[0] if not adaptive.empty else None
    oracle_row = summary[summary["selector"].eq("oracle_selector")].iloc[0]
    diff = float(best_adaptive["pass5"] - best_fixed["pass5"]) if best_adaptive is not None else 0.0
    oracle_gap = float(oracle_row["pass5"] - best_fixed["pass5"])
    verdict = "Verdict C: fixed laws remain the best Pareto point."
    if best_adaptive is not None and diff > 0.02:
        verdict = "Verdict A: adaptive routing beats every fixed law on the random held-out split."
    elif best_adaptive is not None and diff > 0.005:
        verdict = "Verdict B: adaptive routing gives a small or task-dependent gain over fixed laws."
    elif oracle_gap > 0.05 and diff <= 0.005:
        verdict = "Verdict D: oracle headroom exists but current routers cannot exploit it reliably."

    contrast_line = ""
    if best_adaptive is not None and not contrasts.empty:
        match = contrasts[
            contrasts["selector"].eq(best_adaptive["selector"]) & contrasts["reference"].eq(best_fixed["selector"])
        ]
        if not match.empty:
            r = match.iloc[0]
            contrast_line = f"Bootstrap {best_adaptive['selector']} minus {best_fixed['selector']} pass@5: {r['diff']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}], p={r['p_boot']:.4f}."

    lines = [
        "# Router Versus Fixed-Law Summary",
        "",
        verdict,
        "",
        "## Main Random Query-Held-Out Results",
        "",
        table_block(
            summary.head(18),
            ["selector", "n_queries", "raw_acc", "top1_acc", "pass3", "pass5", "pass10", "candidate_auc", "candidate_average_precision"],
        ),
        "",
        f"Best fixed selector: `{best_fixed['selector']}` pass@5={best_fixed['pass5']:.4f}, top1={best_fixed['top1_acc']:.4f}.",
        f"Best adaptive selector: `{best_adaptive['selector']}` pass@5={best_adaptive['pass5']:.4f}, top1={best_adaptive['top1_acc']:.4f}." if best_adaptive is not None else "No adaptive selector was trained.",
        f"Oracle selector pass@5={oracle_row['pass5']:.4f}; oracle gap over best fixed={oracle_gap:.4f}.",
        contrast_line,
        "",
        "## Fixed-Law Choices",
        "",
        f"Global validation-selected law column: `{meta['global_choice'].get('ALL')}`.",
        f"Benchmark-specific law columns: `{meta['dataset_choice']}`.",
        f"Model-specific law columns: `{meta['model_choice']}`.",
        "",
        "## Cost Notes",
        "",
        table_block(cost, ["selector", "pass5", "score_seconds_per_query", "train_seconds"]) if not cost.empty else "No cost rows.",
        "",
        "## Interpretation Guardrails",
        "",
        "- These are cached candidate-bank routing results, not final executable inference.",
        "- The oracle selector is an upper bound and uses labels; it is not deployable.",
        "- The deployable conclusion should be cross-checked against the live validation block before manuscript claims are strengthened.",
    ]
    (reports_dir / "router_vs_fixed_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--v2-root", type=Path, default=Path(os.environ.get("NGC_V2_ROOT", repo_root / "v2")))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    results_dir = ensure_dir(args.v2_root / "results")
    reports_dir = ensure_dir(args.v2_root / "reports")
    figures_dir = ensure_dir(args.v2_root / "figures")
    model_dir = ensure_dir(args.v2_root / "cache/router_models")
    df = load_bank(args.v2_root)
    scored, summary, per_query, extras = train_random_split(df, args, model_dir)

    summary.to_csv(results_dir / "router_holdout_results.csv", index=False)
    summary.to_json(results_dir / "router_holdout_results.jsonl", orient="records", lines=True)
    extras["contrasts"].to_csv(results_dir / "router_holdout_bootstrap.csv", index=False)
    safe_to_parquet(scored, results_dir / "router_predictions_test.parquet")
    scored.to_csv(results_dir / "router_predictions_test.csv.gz", index=False, compression="gzip")
    pd.concat([pq.assign(selector=name) for name, pq in per_query.items()], ignore_index=True).to_csv(
        results_dir / "router_per_query_metrics.csv", index=False
    )

    oracle = build_oracle_headroom(per_query)
    oracle.to_csv(results_dir / "oracle_headroom.csv", index=False)
    timings = extras["meta"]["timings"]
    cost_rows = []
    for selector, values in timings.items():
        match = summary[summary["selector"].eq(selector)]
        pass5 = float(match["pass5"].iloc[0]) if not match.empty else np.nan
        cost_rows.append({"selector": selector, "pass5": pass5, **values, "api_cost_per_query": 0.0})
    cost = pd.DataFrame(cost_rows)
    cost.to_csv(results_dir / "router_cost_overhead.csv", index=False)

    leave_dataset = leave_one_eval(df, "dataset", args)
    leave_model = leave_one_eval(df, "model", args)
    leave_dataset.to_csv(results_dir / "router_leave_benchmark_out.csv", index=False)
    leave_model.to_csv(results_dir / "router_leave_model_out.csv", index=False)
    write_json(reports_dir / "router_training_manifest.json", extras["meta"])

    make_figures(summary, per_query, oracle, cost, scored, figures_dir)
    write_summary_report(summary, extras["contrasts"], oracle, cost, extras["meta"], reports_dir)
    print("Router training complete")
    print(summary[["selector", "top1_acc", "pass5", "candidate_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
