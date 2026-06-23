# Reproducibility Scope

## Included

- Core implementation fragments for NGC topology construction and low-rank block materialization.
- Cached analysis scripts for the social-law and router/baseline claims.
- A query-complete reduced candidate bank for running smoke tests.
- Representative source data, tables, and figures that directly support the manuscript.
- Portable defaults via `NGC_V0_ROOT`, `NGC_V1_ROOT`, and `NGC_V2_ROOT`.

## Excluded

- Full raw benchmark predictions and complete candidate banks.
- Model checkpoints, compressed weight checkpoints, and CUDA cache directories.
- Local model/dataset loader files that contained private machine paths.
- Remote host inventories, shell logs, monitoring logs, and personal environment dumps.
- Internal exploratory experiments that do not map directly to the manuscript's main claims.

## Reproduction Levels

1. Smoke test: run the v2 cached baseline/router scripts on `examples/sample_input`.
2. Cached paper-scale rerun: provide local equivalents of the v0/v1/v2 artifact roots and rerun the v1/v2 scripts.
3. Live inference rerun: additionally provide public model checkpoints, dataset paths, and NGC checkpoint/topology artifacts.

This package is intended to make levels 1 and 2 clear and auditable, while documenting the extra assets required for level 3.
