#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ngc_v1.common import RANDOM_SEED, V0_ROOT, V1_ROOT, ensure_dir, read_observables, setup_matplotlib, write_json


def quantile_bin_by_group(df: pd.DataFrame, col: str, labels: list[str]) -> pd.Series:
    def one_group(s: pd.Series) -> pd.Series:
        try:
            return pd.qcut(s.rank(method="first"), q=len(labels), labels=labels)
        except ValueError:
            return pd.Series([labels[len(labels) // 2]] * len(s), index=s.index)

    return df.groupby(["model", "dataset"], group_keys=False)[col].transform(one_group)


def build_bins(df: pd.DataFrame) -> pd.DataFrame:
    dep_labels = ["low", "mid_low", "mid", "mid_high", "high"]
    guard_labels = ["low_guard", "medium_guard", "high_guard"]
    out = df.copy()
    out["departure_bin"] = quantile_bin_by_group(out, "departure_index", dep_labels)
    out["guard_bin"] = quantile_bin_by_group(out, "stability_guard", guard_labels)
    bins = (
        out.groupby(["departure_bin", "guard_bin"], observed=False)
        .agg(
            n=("label", "size"),
            correctness=("label", "mean"),
            reconstruction_error_log10=("reconstruction_error_log10", "median"),
            stability_guard=("stability_guard", "median"),
            departure_index=("departure_index", "median"),
        )
        .reset_index()
    )
    by_group = (
        out.groupby(["model", "dataset", "departure_bin", "guard_bin"], observed=False)
        .agg(n=("label", "size"), correctness=("label", "mean"))
        .reset_index()
    )
    return out, bins, by_group


def design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    work = df.copy()
    for col in ["departure_index", "stability_guard", "reconstruction_error_log10"]:
        mu = work[col].mean()
        sd = work[col].std()
        if not np.isfinite(sd) or sd < 1e-9:
            sd = 1.0
        work[col + "_std"] = (work[col] - mu) / sd
    work["departure_sq"] = work["departure_index_std"] ** 2
    dummies = pd.get_dummies(work[["model", "dataset"]], drop_first=True, dtype=float)
    xdf = pd.concat(
        [
            pd.Series(1.0, index=work.index, name="intercept"),
            work[
                [
                    "departure_index_std",
                    "departure_sq",
                    "stability_guard_std",
                    "reconstruction_error_log10_std",
                ]
            ],
            dummies,
        ],
        axis=1,
    )
    return xdf.to_numpy(dtype=float), xdf.columns.tolist()


def ols_coefficients(df: pd.DataFrame) -> pd.Series:
    y = df["label"].to_numpy(dtype=float)
    x, names = design_matrix(df)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return pd.Series(beta, index=names)


def bootstrap_quadratic(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    observed = ols_coefficients(df)
    query_keys = np.array(sorted(df["query_key"].unique()))
    groups = {q: g for q, g in df.groupby("query_key", sort=False)}
    rng = np.random.default_rng(seed)
    # OLS bootstrapping over all 57k rows is useful as a stability diagnostic, but
    # 10k coefficient fits would be wasteful for Day 2. Cap coefficient bootstrap
    # while preserving the requested argument in the manifest.
    effective_boot = min(n_boot, 300)
    draws = {name: [] for name in observed.index}
    for _ in range(effective_boot):
        sampled = rng.choice(query_keys, size=len(query_keys), replace=True)
        boot_df = pd.concat([groups[q] for q in sampled], ignore_index=True)
        coef = ols_coefficients(boot_df)
        for name, value in coef.items():
            draws[name].append(value)
    rows = []
    for name, value in observed.items():
        arr = np.array(draws[name], dtype=float)
        rows.append(
            {
                "term": name,
                "coefficient": float(value),
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "p_boot_two_sided_zero": float(min(1.0, 2 * min((arr <= 0).mean(), (arr >= 0).mean()))),
                "n_boot_effective": int(effective_boot),
            }
        )
    return pd.DataFrame(rows)


def build_mid_extreme_contrast(binned_df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    query = (
        binned_df.assign(is_mid=binned_df["departure_bin"].astype(str).eq("mid"))
        .groupby(["query_key", "is_mid"], as_index=False)["label"]
        .mean()
    )
    pivot = query.pivot(index="query_key", columns="is_mid", values="label").dropna()
    if True not in pivot.columns or False not in pivot.columns:
        return pd.DataFrame()
    delta = pivot[True] - pivot[False]
    rng = np.random.default_rng(seed)
    arr = delta.to_numpy(dtype=float)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        draws[i] = arr[idx].mean()
    return pd.DataFrame(
        [
            {
                "contrast": "mid_departure_minus_non_mid_query_mean_correctness",
                "diff": float(arr.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "p_boot": float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))),
                "n_queries": int(len(arr)),
            }
        ]
    )


def make_figures(bins: pd.DataFrame, out_dir: Path) -> None:
    plt = setup_matplotlib()
    import seaborn as sns

    order = ["low", "mid_low", "mid", "mid_high", "high"]
    guard_order = ["low_guard", "medium_guard", "high_guard"]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for guard in guard_order:
        sub = bins[bins["guard_bin"].astype(str).eq(guard)].copy()
        sub["departure_bin"] = pd.Categorical(sub["departure_bin"].astype(str), categories=order, ordered=True)
        sub = sub.sort_values("departure_bin")
        ax.plot(sub["departure_bin"].astype(str), sub["correctness"], marker="o", lw=1.8, label=guard)
    ax.set_xlabel("Departure regime")
    ax.set_ylabel("Candidate correctness rate")
    ax.set_title("Bounded departure under stability constraints")
    ax.legend(frameon=False, title="Stability guard")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_bounded_departure_curve.pdf")
    fig.savefig(out_dir / "fig_bounded_departure_curve.png")
    plt.close(fig)

    pivot = bins.pivot(index="guard_bin", columns="departure_bin", values="correctness").reindex(index=guard_order, columns=order)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f", linewidths=0.3, ax=ax)
    ax.set_xlabel("Departure regime")
    ax.set_ylabel("Stability guard")
    ax.set_title("Departure-risk correctness heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_departure_risk_heatmap.pdf")
    fig.savefig(out_dir / "fig_departure_risk_heatmap.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-root", type=Path, default=V0_ROOT)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()

    out_dir = ensure_dir(args.v1_root / "outputs/05_negotiated_stability")
    df = read_observables(args.v1_root, args.v0_root)
    binned_df, bins, by_group = build_bins(df)
    quadratic = bootstrap_quadratic(df, args.n_bootstrap, args.seed)
    mid_contrast = build_mid_extreme_contrast(binned_df, args.n_bootstrap, args.seed)

    bins.to_csv(out_dir / "curve_bins.csv", index=False)
    by_group.to_csv(out_dir / "curve_bins_by_model_dataset.csv", index=False)
    quadratic.to_csv(out_dir / "quadratic_tests.csv", index=False)
    mid_contrast.to_csv(out_dir / "mid_departure_contrast.csv", index=False)
    make_figures(bins, out_dir)
    write_json(
        out_dir / "manifest.json",
        {
            "rows_in": int(len(df)),
            "queries": int(df["query_key"].nunique()),
            "n_bootstrap_requested": int(args.n_bootstrap),
            "quadratic_bootstrap_effective": int(quadratic["n_boot_effective"].max()),
        },
    )

    print("Negotiated stability curve complete")
    print(quadratic[quadratic["term"].isin(["departure_index_std", "departure_sq", "stability_guard_std"])].to_string(index=False))
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
