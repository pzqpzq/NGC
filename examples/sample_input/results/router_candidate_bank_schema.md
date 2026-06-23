# Router Candidate Bank Schema

Rows: 16748
Queries: 450
Parquet written: True

Included model/dataset pairs:
- llama3-8b / gpqa: 75 queries
- llama3-8b / hotpot-qa: 75 queries
- llama3-8b / math500: 75 queries
- qwen3-8b / gpqa: 75 queries
- qwen3-8b / hotpot-qa: 75 queries
- qwen3-8b / math500: 75 queries

Key columns:
- `query_key`
- `candidate_uid`
- `model`
- `dataset`
- `candidate_index`
- `label`
- `raw_prob`
- `topology_family`
- `topology_id`
- `wtp_file_name`
- `tcr`
- `compression_bin`
- `law_fixed_transfer`
- `law_bargaining`
- `law_coalition`
- `law_mediated_criticality`
- `law_retrieve_then_settle`
- `law_reconstruction_guarded_exploration`
- `law_settlement_only`
- `law_cost_aware_adaptive`
