#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ngc_v1.common import (
    V0_ROOT,
    V1_ROOT,
    add_constitutional_features,
    ensure_dir,
    read_candidate_metrics,
    safe_to_parquet,
    write_json,
)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = {
        "query_key": "nunique",
        "candidate_uid": "count",
        "label": "mean",
        "raw_prob": "mean",
        "departure_index": "median",
        "stability_guard": "median",
        "evidence_broker_score": "median",
        "mediator_score": "median",
        "settlement_score": "median",
        "constitutional_balance": "median",
        "role_entropy": "median",
        "kv_frac": "median",
        "q_frac": "median",
        "mlp_frac": "median",
        "tcr": "median",
        "num_tokens": "median",
    }
    summary = df.groupby(["model", "dataset"], as_index=False).agg(agg_cols)
    summary = summary.rename(
        columns={
            "query_key": "n_queries",
            "candidate_uid": "n_candidates",
            "label": "candidate_correct_rate",
            "raw_prob": "raw_pass",
        }
    )
    summary["fixed_group_is_negative"] = [
        bool(v) for v in df.groupby(["model", "dataset"])["is_negative_group"].first().values
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", type=Path, default=V0_ROOT)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    args = parser.parse_args()

    out_dir = ensure_dir(args.v1_root / "outputs/01_observables")
    raw = read_candidate_metrics(args.v0_root)
    df = add_constitutional_features(raw)

    # Make the downstream scripts stable even when sklearn/matplotlib serializes dtypes.
    for col in ["label", "candidate_index"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    parquet_path = out_dir / "constitutional_candidates.parquet"
    wrote_parquet = safe_to_parquet(df, parquet_path)
    df.to_csv(out_dir / "constitutional_candidates.csv.gz", index=False, compression="gzip")

    summary = build_summary(df)
    summary.to_csv(out_dir / "summary_by_model_dataset.csv", index=False)

    dictionary = {
        "query_key": "model::dataset::query_id; grouping key for leakage-safe splits.",
        "departure_index": "robust_z(force_ratio) + robust_z(transport_shift_log10) within model-dataset.",
        "stability_guard": "-robust_z(stability_risk_log10) - robust_z(reconstruction_error_log10) within model-dataset.",
        "evidence_broker_score": "kv_frac times early-depth mass, operationalizing early K/V evidence brokerage.",
        "mediator_score": "spectral_flatness minus absolute transport-shift z-score.",
        "settlement_score": "(mlp_frac + q_frac) times normalized depth, operationalizing late answer settlement.",
        "constitutional_balance": "departure_index + stability_guard + mediator_score.",
        "role_entropy": "normalized entropy over attention, MLP, K/V and Q role fractions.",
        "is_negative_group": "Known cached fixed-law negative-gain model-dataset group.",
    }
    write_json(out_dir / "role_feature_dictionary.json", dictionary)
    write_json(
        out_dir / "manifest.json",
        {
            "source": str(args.v0_root / "nmi_may27_experiments/outputs/day2/candidate_metrics_holdout.csv"),
            "rows": int(len(df)),
            "queries": int(df["query_key"].nunique()),
            "model_dataset_groups": int(df[["model", "dataset"]].drop_duplicates().shape[0]),
            "wrote_parquet": bool(wrote_parquet),
            "outputs": {
                "parquet": str(parquet_path),
                "csv_gz": str(out_dir / "constitutional_candidates.csv.gz"),
                "summary": str(out_dir / "summary_by_model_dataset.csv"),
            },
        },
    )

    print("Constitutional observables complete")
    print(f"Rows: {len(df)}")
    print(f"Queries: {df['query_key'].nunique()}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
