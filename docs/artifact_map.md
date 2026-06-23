# Artifact Map

## Manuscript-Facing Social-Law Evidence

- `v1/scripts/01_build_constitutional_observables.py`: builds candidate observables and social-role features from cached candidate metrics.
- `v1/scripts/02_social_law_atlas.py`: summarizes benchmark demand and role signatures across social laws.
- `v1/scripts/03_train_constitutional_selector.py`: evaluates fixed and learned selectors on grouped held-out queries.
- `v1/scripts/04_matched_law_breaking_controls.py`: matched controls for law-breaking versus reconstruction-only explanations.
- `v1/scripts/05_negotiated_stability_curve.py`: bounded-disagreement / negotiated-stability diagnostics.
- `v1/scripts/06_role_knockout_cached.py`: role-knockout analyses for evidence broker, mediator, settlement, and reconstruction guard.
- `v1/scripts/07_failure_taxonomy.py`: failure-mode summary for cases where fixed social laws do not improve.
- `v1/scripts/08_live_grid_runner.py`: wrapper/aggregator for executable live validation.
- `v1/scripts/11_controlled_overhead.py`: controlled overhead summary.
- `v1/scripts/14_make_publication_artifacts.py`: regenerates final tables, selected figures, source data, and report artifacts from v1 outputs.

Representative outputs live in:

- `examples/source_data/v1_social_laws/`
- `examples/figures/v1_social_laws/`
- `examples/tables/v1_social_laws/`

## Router And Baseline Evidence

- `v2/scripts/01_build_feature_bank.py`: merges v1 observables with richer candidate-bank features.
- `v2/scripts/02_run_baselines.py`: cached raw, SVD, basis-sharing, NGC, and lite/proxy baseline comparison.
- `v2/scripts/04_train_adaptive_router.py`: adaptive router training and grouped held-out evaluation.
- `v2/scripts/07_role_ablation.py`: adaptive-router feature-family ablation.
- `v2/scripts/09_make_tables_and_figures.py`: manuscript-facing router/baseline tables and figures.

Representative outputs live in:

- `examples/source_data/v2_router_baselines/`
- `examples/figures/v2_router_baselines/`

## Mechanistic Controls

- `v2/scripts/10_phase_diagram_negotiated_stability.py`: phase-space geometry for negotiated stability.
- `v2/scripts/11_causal_role_perturbation.py`: executable perturbation of broker and settlement subspaces.
- `v2/scripts/12_end_to_end_cost_audit.py`: wall-clock and memory audit.
- `v2/scripts/13_make_jun15_artifacts.py`: table/figure assembly for Jun15 controls.

Representative outputs live in:

- `examples/source_data/jun15_mechanistic_controls/`
- `examples/figures/jun15_mechanistic_controls/`
- `examples/tables/jun15_mechanistic_controls/`

## Core NGC Utilities

- `ngc_core/nsys_utils/generate_ONT.py`: topology-family generation.
- `ngc_core/nsys_utils/nsys_config.py`: shared-neuron / low-rank module definitions.
- `ngc_core/nsys_utils/capture_Acts.py`: activation-capture helpers.
- `ngc_core/nsys_utils/train_new.py`: neural-system fitting helper.
