# scripts

Command-line tools around the framework. Run them from the repository root; every script prints
its options with `--help`, and its module docstring describes inputs, outputs and an example.

## Datasets and splits

`make data` (or `make data-openea | data-dbp1m | data-2m`) runs `download_data.sh`, which downloads
the releases and calls the preparation and split scripts below.

| Script | Purpose |
|---|---|
| `download_data.sh` | Download, unpack, convert and split the four datasets of the thesis. |
| `extract_zip.py` | Unzip helper used by `download_data.sh` (skips macOS metadata, handles nested archives). |
| `prepare_dbp1m.py` | Reshape the LargeEA DBP1M release into the OpenEA layout; drop alignment links whose entities are missing from the triples. |
| `prepare_fb_dbp2m.py` | Reshape FB-DBP2M into the OpenEA layout; prefix the per-KG numeric ids so they do not collide (the alignment file lists the Freebase id first). |
| `split_openea.py` | Seeded train/valid/test splits from `ent_links` (`--train 0.3 --valid 0.1 --folds 1` gives the `316_1fold/1` splits of the thesis). |
| `make_30pct_fold.py` | A seeded 30/10/60 `custom_30pct_fold/1` split for other OpenEA datasets. |

## Partitions

`src/partition.py` builds the base partition; `make partition` (`launch/partition.sh`) calls it and
the refinement below.

| Script | Purpose |
|---|---|
| `make_refined_kg2_partition.py` | Copy a partition and refine its KG2 assignment by label propagation (`..._kg2localityref`, used for every multi-node result). |
| `eval_kg2_assignment.py` | Offline check of a KG2 assignment: share of pairs split across ranks, load, KG2 edge cut. |
| `make_joint_metis_partition.py` | Alternative partition: KG1 and KG2 in one METIS call (load-balancing study). |
| `make_divea_aligned_partition.py` | Alternative partition derived from an ELsEA/DivEA run's own subtask division (ablation). |

## Results and evaluation

| Script | Purpose |
|---|---|
| `show_results.py` | Print a summary of the runs under `outputs/results/` (`make results`). |
| `summarize_rank_split.py` | Same-rank / cross-rank breakdown of a run made with `--eval_rank_split_dir`. |
| `unified_eval.py` | **The unified evaluator.** Scores an export bundle (`embeddings.npy`, `ids.npy`, `kg_labels.npy`, `test_links.json`) with cosine or L1 similarity, optional CSLS, and test-side or all-entity candidates. Every comparison number of the thesis comes from it. |
| `tie_rank_check.py` | How much of a score comes from exact ties between candidates (relevant for LightEA). |
| `rescore_embeddings.py` | Re-score a DistEA export bundle (`--export_dir`) under other inference protocols. |
| `subtask_unified_eval.py` | Score ELsEA and DistEA under ELsEA's per-subtask protocol. |

## Comparison methods

The comparison methods are not part of this repository; clone them from their authors and run
the launchers from inside their checkouts. Each launcher patches only what is needed to run the
published code on current libraries and on our splits, without editing the upstream files.

| Method | Prepare input | Launch | Export to `unified_eval.py` |
|---|---|---|---|
| ELsEA / DivEA | `prepare_dbp1m_for_elsea.py` (URI datasets), `export_to_divea.py` | `run_elsea_divea.py` | `subtask_unified_eval.py` (reads ELsEA's subtask results) |
| ClusterEA | fold swap via `CLUSTEREA_FOLD_DIR` | `run_clusterea.py` | `export_clusterea_to_unified.py` |
| LightEA | `prepare_for_lightea.py` | `run_lightea.py` | written by the launcher |
| LargeEA | — | `run_largeea.py` | own evaluation |
| LargeGNN | — | `run_largegnn.py` | own evaluation |
| RREA (original, TensorFlow) | `prepare_rrea_tf_data.py` | `run_rrea_tf_reference.py` | `export_rrea_tf_to_unified.py` |

`replay_elsea_multigpu.py` estimates ELsEA's wall-clock time on 2, 4 or 8 GPUs from the subtask
times in its `running.log`; the thesis uses it for the 8-GPU equal-GPU ratio.

The ClusterEA and LargeEA launchers read DBP1M in LargeEA's `mkdata` layout from
`data/clusterea_mkdata/` and the OpenEA datasets from `data/OpenEA_dataset_v2.0/`; set
`DISTEA_MKDATA_DIR` / `DISTEA_OPENEA_DIR` to use other locations.

## Other

| Script | Purpose |
|---|---|
| `build_grpc_server.sh` | Build the Go cache server (`make grpc-server`). |
