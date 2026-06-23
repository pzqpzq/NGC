#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ngc_v1.common import (
    RANDOM_SEED,
    V0_ROOT,
    V1_ROOT,
    ensure_dir,
    paired_bootstrap,
    query_metrics_for_score,
    random_expected_query_metrics,
    rank_auc_by_query,
    read_observables,
    safe_to_parquet,
    setup_matplotlib,
    split_queries,
    summarize_query_metrics,
    write_json,
)


NUMERIC_FEATURES = [
    "reconstruction_score",
    "reconstruction_error_log10",
    "law_score",
    "force_log_abs",
    "force_ratio",
    "transport_shift_log10",
    "stability_risk_log10",
    "spectral_flatness",
    "mean_layer",
    "attn_frac",
    "mlp_frac",
    "kv_frac",
    "q_frac",
    "tcr",
    "num_tokens",
    "departure_index",
    "stability_guard",
    "evidence_broker_score",
    "mediator_score",
    "settlement_score",
    "role_entropy",
    "constitutional_balance",
]

CATEGORICAL_FEATURES = ["model", "dataset"]

FEATURE_SETS = {
    "reconstruction_only": ["reconstruction_score"],
    "scalar_law_only": ["law_score"],
    "law_plus_reconstruction": ["law_score", "reconstruction_score"],
    "negotiated_observables": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
}


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(features: List[str]) -> ColumnTransformer:
    num = [c for c in features if c not in CATEGORICAL_FEATURES]
    cat = [c for c in features if c in CATEGORICAL_FEATURES]
    transformers = []
    if num:
        transformers.append(
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                num,
            )
        )
    if cat:
        transformers.append(("cat", one_hot_encoder(), cat))
    return ColumnTransformer(transformers, remainder="drop")


def make_logistic(features: List[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(features)),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def make_hgb(features: List[str], seed: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(features)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=180,
                    learning_rate=0.055,
                    max_leaf_nodes=31,
                    l2_regularization=0.02,
                    random_state=seed,
                ),
            ),
        ]
    )


def score_pipeline(model: Pipeline, df: pd.DataFrame, features: List[str]) -> np.ndarray:
    return model.predict_proba(df[features])[:, 1]


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(average_precision_score(y, p))
    except ValueError:
        return float("nan")


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        lr = LogisticRegression(solver="lbfgs").fit(logits, y)
        return float(lr.coef_[0, 0])
    except Exception:
        return float("nan")


def within_query_percent_rank(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("query_key")[col].rank(method="average", pct=True)


def evaluate_selector_scores(
    test_df: pd.DataFrame,
    score_cols: Dict[str, str],
    top_k: int,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    per_query = {}
    summaries = []
    for name, col in score_cols.items():
        pq = query_metrics_for_score(test_df, col, top_k=top_k, ascending=False)
        per_query[name] = pq
        row = summarize_query_metrics(name, pq)
        y = test_df["label"].to_numpy(dtype=int)
        s = test_df[col].to_numpy(dtype=float)
        s01 = np.clip((s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12), 1e-6, 1 - 1e-6)
        row.update(
            {
                "candidate_auc": safe_auc(y, s),
                "candidate_average_precision": safe_ap(y, s),
                "candidate_brier_minmax": float(brier_score_loss(y, s01)),
                "query_rank_auc": rank_auc_by_query(test_df, col),
            }
        )
        summaries.append(row)

    rand = random_expected_query_metrics(test_df, top_k=top_k)
    per_query["random_expected"] = rand
    summaries.append(summarize_query_metrics("random_expected", rand))

    summary_df = pd.DataFrame(summaries).sort_values("pass5", ascending=False)

    contrasts = []
    refs = ["fixed_C316", "reconstruction_only", "raw"]
    for name, pq in per_query.items():
        if name in refs:
            continue
        aligned = pq.set_index("query_key")
        for ref in refs:
            if ref == "raw":
                left = aligned["passk"]
                right = aligned["raw"]
            else:
                if ref not in per_query:
                    continue
                right_pq = per_query[ref].set_index("query_key").reindex(aligned.index)
                left = aligned["passk"]
                right = right_pq["passk"]
            stat = paired_bootstrap(left, right, n_boot=n_boot, seed=seed)
            stat.update({"selector": name, "reference": ref, "metric": "pass5"})
            contrasts.append(stat)
    contrasts_df = pd.DataFrame(contrasts)
    return summary_df, contrasts_df, per_query


def train_and_score(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, Dict[str, Pipeline], Dict[str, List[str]]]:
    train_keys, val_keys, test_keys = split_queries(df["query_key"], seed=seed)
    train_df = df[df["query_key"].isin(train_keys)].copy()
    val_df = df[df["query_key"].isin(val_keys)].copy()
    test_df = df[df["query_key"].isin(test_keys)].copy()

    y_train = train_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    models: Dict[str, Pipeline] = {}
    features_by_model: Dict[str, List[str]] = {}
    scored = test_df.copy()
    scored["score_fixed_C316"] = scored["law_score"].astype(float)
    scored["score_reconstruction_only"] = scored["reconstruction_score"].astype(float)

    point_rows = []
    for name, features in FEATURE_SETS.items():
        model_name = f"logistic_{name}"
        pipe = make_logistic(features)
        pipe.fit(train_df[features], y_train)
        scored[f"score_{model_name}"] = score_pipeline(pipe, test_df, features)
        models[model_name] = pipe
        features_by_model[model_name] = features
        point_rows.append(
            {
                "selector": model_name,
                "feature_set": name,
                "candidate_auc": safe_auc(y_test, scored[f"score_{model_name}"].to_numpy()),
                "candidate_average_precision": safe_ap(y_test, scored[f"score_{model_name}"].to_numpy()),
                "candidate_brier": float(brier_score_loss(y_test, scored[f"score_{model_name}"].to_numpy())),
                "calibration_slope": calibration_slope(y_test, scored[f"score_{model_name}"].to_numpy()),
            }
        )

    hgb_name = "hist_gradient_boosting_negotiated_observables"
    hgb_features = FEATURE_SETS["negotiated_observables"]
    hgb = make_hgb(hgb_features, seed)
    hgb.fit(train_df[hgb_features], y_train)
    scored[f"score_{hgb_name}"] = score_pipeline(hgb, test_df, hgb_features)
    models[hgb_name] = hgb
    features_by_model[hgb_name] = hgb_features
    point_rows.append(
        {
            "selector": hgb_name,
            "feature_set": "negotiated_observables",
            "candidate_auc": safe_auc(y_test, scored[f"score_{hgb_name}"].to_numpy()),
            "candidate_average_precision": safe_ap(y_test, scored[f"score_{hgb_name}"].to_numpy()),
            "candidate_brier": float(brier_score_loss(y_test, scored[f"score_{hgb_name}"].to_numpy())),
            "calibration_slope": calibration_slope(y_test, scored[f"score_{hgb_name}"].to_numpy()),
        }
    )

    scored["c316_query_rank"] = within_query_percent_rank(scored, "law_score")
    scored["reconstruction_query_rank"] = within_query_percent_rank(scored, "reconstruction_score")
    scored["score_calibrated_ensemble"] = (
        0.5 * scored[f"score_{hgb_name}"] + 0.3 * scored["c316_query_rank"] + 0.2 * scored["reconstruction_query_rank"]
    )

    split_manifest = {
        "n_train_queries": len(train_keys),
        "n_val_queries": len(val_keys),
        "n_test_queries": len(test_keys),
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
    }
    return scored, models, features_by_model, pd.DataFrame(point_rows), split_manifest


def leave_one_robustness(df: pd.DataFrame, holdout_col: str, seed: int, top_k: int) -> pd.DataFrame:
    rows = []
    features = FEATURE_SETS["negotiated_observables"]
    for value in sorted(df[holdout_col].unique()):
        train = df[df[holdout_col] != value].copy()
        test = df[df[holdout_col] == value].copy()
        if test["query_key"].nunique() < 2 or len(train) == 0:
            continue
        pipe = make_hgb(features, seed)
        pipe.fit(train[features], train["label"].astype(int))
        test = test.copy()
        test["score_hgb_leave_one"] = score_pipeline(pipe, test, features)
        test["score_fixed_C316"] = test["law_score"].astype(float)
        hgb_pq = query_metrics_for_score(test, "score_hgb_leave_one", top_k=top_k)
        fixed_pq = query_metrics_for_score(test, "score_fixed_C316", top_k=top_k)
        aligned = hgb_pq.set_index("query_key")
        fixed = fixed_pq.set_index("query_key").reindex(aligned.index)
        stat = paired_bootstrap(aligned["passk"], fixed["passk"], n_boot=2000, seed=seed)
        rows.append(
            {
                "holdout_type": holdout_col,
                "holdout_value": value,
                "n_queries": int(len(aligned)),
                "hgb_pass5": float(aligned["passk"].mean()),
                "fixed_C316_pass5": float(fixed["passk"].mean()),
                "raw_pass": float(aligned["raw"].mean()),
                "hgb_minus_fixed_pass5": stat["diff"],
                "ci_low": stat["ci_low"],
                "ci_high": stat["ci_high"],
                "p_boot": stat["p_boot"],
            }
        )
    return pd.DataFrame(rows)


def make_figures(summary: pd.DataFrame, scored: pd.DataFrame, out_dir: Path) -> None:
    plt = setup_matplotlib()
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    plot_df = summary.sort_values("pass5", ascending=True)
    ax.barh(plot_df["selector"], plot_df["pass5"], color="#4C78A8")
    ax.axvline(plot_df.loc[plot_df["selector"] == "fixed_C316", "pass5"].iloc[0], color="#C44E52", lw=1.5, label="fixed C316")
    ax.set_xlabel("Held-out query pass@5")
    ax.set_ylabel("")
    ax.set_title("Query-adaptive constitutional selector comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_selector_comparison.pdf")
    fig.savefig(out_dir / "fig_selector_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    y = scored["label"].astype(int).to_numpy()
    for name, col in [
        ("HGB negotiated", "score_hist_gradient_boosting_negotiated_observables"),
        ("Logistic negotiated", "score_logistic_negotiated_observables"),
        ("Ensemble", "score_calibrated_ensemble"),
    ]:
        if col not in scored:
            continue
        prob_true, prob_pred = calibration_curve(y, np.clip(scored[col], 1e-6, 1 - 1e-6), n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", lw=1.6, label=name)
    ax.plot([0, 1], [0, 1], color="black", lw=1.0, ls="--")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed correctness")
    ax.set_title("Candidate correctness calibration")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_selector_calibration.pdf")
    fig.savefig(out_dir / "fig_selector_calibration.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", type=Path, default=V0_ROOT)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    out_dir = ensure_dir(args.v1_root / "outputs/03_selector")
    df = read_observables(args.v1_root, args.v0_root)
    scored, models, features_by_model, point_metrics, split_manifest = train_and_score(df, args.seed)

    score_cols = {
        "fixed_C316": "score_fixed_C316",
        "reconstruction_only": "score_reconstruction_only",
        "logistic_reconstruction_only": "score_logistic_reconstruction_only",
        "logistic_scalar_law_only": "score_logistic_scalar_law_only",
        "logistic_law_plus_reconstruction": "score_logistic_law_plus_reconstruction",
        "logistic_negotiated_observables": "score_logistic_negotiated_observables",
        "hist_gradient_boosting_negotiated_observables": "score_hist_gradient_boosting_negotiated_observables",
        "calibrated_ensemble": "score_calibrated_ensemble",
    }
    summary, contrasts, per_query = evaluate_selector_scores(
        scored, score_cols, top_k=args.top_k, n_boot=args.n_bootstrap, seed=args.seed
    )

    point_metrics.to_csv(out_dir / "selector_candidate_metrics.csv", index=False)
    summary.to_csv(out_dir / "selector_metrics.csv", index=False)
    contrasts.to_csv(out_dir / "selector_paired_bootstrap.csv", index=False)
    safe_to_parquet(scored, out_dir / "per_query_rankings.parquet")
    scored.to_csv(out_dir / "per_query_rankings.csv.gz", index=False, compression="gzip")

    pd.concat(
        [pq.assign(selector=name) for name, pq in per_query.items()],
        ignore_index=True,
    ).to_csv(out_dir / "selector_per_query_metrics.csv", index=False)

    lod = leave_one_robustness(df, "dataset", args.seed, args.top_k)
    lom = leave_one_robustness(df, "model", args.seed, args.top_k)
    lod.to_csv(out_dir / "selector_leave_one_dataset.csv", index=False)
    lom.to_csv(out_dir / "selector_leave_one_model.csv", index=False)

    make_figures(summary, scored, out_dir)

    actionable = summary[~summary["selector"].isin(["random_expected"])].copy()
    best_name = actionable.iloc[0]["selector"]
    save_obj = {
        "best_selector": best_name,
        "models": models,
        "features_by_model": features_by_model,
        "score_columns": score_cols,
        "split_manifest": split_manifest,
        "note": "If best_selector is a baseline or ensemble, use score_columns and weights in the script logic.",
    }
    joblib.dump(save_obj, out_dir / "best_selector.joblib")

    write_json(
        out_dir / "manifest.json",
        {
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "top_k": args.top_k,
            "split": split_manifest,
            "best_actionable_selector": str(best_name),
            "note": "random_expected is a non-actionable diagnostic baseline and is excluded from best_actionable_selector.",
            "outputs": {
                "selector_metrics": str(out_dir / "selector_metrics.csv"),
                "paired_bootstrap": str(out_dir / "selector_paired_bootstrap.csv"),
                "per_query_rankings": str(out_dir / "per_query_rankings.parquet"),
            },
        },
    )

    print("Constitutional selector training complete")
    print(summary[["selector", "pass1", "pass5", "gain_pass5_vs_raw"]].to_string(index=False))
    print(f"Best actionable selector by pass@5: {best_name}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
