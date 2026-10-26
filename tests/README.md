# tests

Every `test_*.py` file is a standalone script that runs on CPU, needs neither a GPU nor DGL, and
exits non-zero on failure. A few start a small `gloo` process group on the loopback interface.

```bash
make test                                         # every CPU test (about 1.5 minutes)
.venv/bin/python tests/test_table_consensus.py    # a single file
```

Tests that need `nxmetis` (`make nxmetis`) or the Go server (`make grpc-server`) skip the
affected cases when these are missing.

## GPU checks (run by hand on a GPU node)

| File | Checks | Needs |
|---|---|---|
| `gpu_parity_eval_device.py` | Chunked evaluation on the GPU equals the CPU result (`DISTEA_EVAL_DEVICE=cuda`). | 1 GPU |
| `gpu_nccl_eval_shard.py` | Sharded evaluation and pseudo-seed mining give the unsharded result under NCCL. | `torchrun --nproc_per_node=2` |
| `gpu_nccl_allreduce_wait.py` | `DistTrainer.allreduce_gpu_wait_s` measures injected rank skew under NCCL. | `torchrun --nproc_per_node=2` |

## Core components (`test_stage_*.py`)

| File | Tests |
|---|---|
| `test_stage_a.py` | Data loading, KG2 assignment, deterministic entity features, GCN forward, alignment loss; RREA kernels identical on every rank. |
| `test_stage_b.py` | GCN model and the alignment loss at the KG boundary (empty pairs, no KG2 nodes, identical embeddings). |
| `test_stage_c.py` | Embedding write / halo-row refresh, Adam momentum reset of patched rows, pseudo-seed merge. |
| `test_stage_d.py` | KG2 counterpart candidates (`dist_ctx_kg2.py`) and their use as negatives. |
| `test_stage_e.py` | Evaluation metrics (L1, cosine, CSLS, Hits@k, MRR) and the export bundle. |
| `test_stage_f.py` | Owner-compute dispatch: requester and owner build matching send/receive sequences. |
| `test_stage_g.py` | KG2 halo nodes keep cross-rank KG2–KG2 edges; the no-halo ablation has none. |
| `test_stage_h.py` | Cross-rank consistency regulariser (`--consistency_lambda`). |
| `test_stage_i.py` | `--shared_h_init`: gradients confined to local rows; touch-count weighted averaging. |
| `test_stage_j.py` | Locality-based KG2 assignment, edge weights, fold-aware partition paths. |
| `test_stage_k.py` | VBIP scoring on the RREA graph. |
| `test_stage_l.py` | DBP1M / FB-DBP2M preparation scripts (symlinks, dangling links, id prefixes). |
| `test_stage_m.py` | Final-embedding push through the cache backends, including a real Go-server round trip. |
| `test_stage_n.py` | Mini-batch building blocks: `SparseEntityTable`, fan-out sampling, GCN blocks. |
| `test_stage_o.py` | `scripts/export_to_divea.py` keeps the (h, r, t) column order. |
| `test_stage_p.py` | RREA mini-batch blocks; `RREA.forward_blocks` equals `forward`. |
| `test_stage_q.py` | Mini-batch cross-rank fetch: ownership split and owner map. |
| `test_stage_r.py` | Mini-batch evaluation fast paths. |
| `test_stage_s.py` | Halo-sync backends (`none` / `file` / `grpc`) and their factory. |
| `test_stage_t.py` | NHSM loss equals the ClusterEA reference. |
| `test_stage_u.py` | Per-epoch halo sampling (`--halo_sample_p`). |
| `test_stage_v.py` | Vectorised and DGL mini-batch samplers equal the Python sampler. |
| `test_stage_w.py` | Cluster-wise (GAS) RREA forward equals the full forward with exact history; end-to-end GAS epoch. |

## Exactness of optimisations and options

Most of these tests check that an optimisation reproduces the reference computation exactly (or
up to floating-point summation order), or pin down the behaviour of one option.

| File | Tests |
|---|---|
| `test_attn_softmax_axis.py` | Attention-softmax grouping (`--attn_softmax`) and `--rrea_input_relu`. |
| `test_chunked_eval.py` | Chunked evaluation equals the dense pipeline (Hits@k, MRR, CSLS argmax). |
| `test_data_cache.py` | The parsed-triples cache reproduces a plain parse; invalidation and fallback. |
| `test_defer_test_eval.py` | `--defer_test_eval` evaluates the selected iteration's state. |
| `test_edge_attention_core.py` | Fused attention core equals the original concatenation in values and gradients. |
| `test_epoch_selector.py` | Best-epoch and best-iteration selection logic. |
| `test_eval_chunk_fit.py` | Evaluation block sizing to free GPU memory; results independent of block size. |
| `test_eval_owned_rows.py` | Ownership invariant: every entity is scored once, with its owner's row. |
| `test_eval_pseudo_split.py` | Test-pair grouping by overlap with pseudo-seeds (`--eval_pseudo_split`). |
| `test_eval_rank_split.py` | Same-rank / cross-rank metrics equal the subset metrics. |
| `test_eval_shard.py` | Sharded evaluation and pseudo-seed mining over 3 ranks equal the unsharded result. |
| `test_feature_procs.py` | Parallel feature initialisation is bit-identical to the serial loop. |
| `test_full_graph_ablation.py` | `--ablate_full_graph`: every rank holds the whole graph. |
| `test_gas_assign_cache.py` | Cached METIS cluster assignments: keys, invalidation, GPU-built incidence matrices. |
| `test_gas_bfs_chunk_scale.py` | The BFS-chunk clustering fallback at DBP1M-like scale. |
| `test_gas_checkpoint.py` | Gradient checkpointing leaves outputs, gradients and history unchanged. |
| `test_gas_closure_cap.py` | `--gas_closure_cap` and `--gas_history_on_device`. |
| `test_gas_compact_loss.py` | Loss on a compact row table equals the full-view loss. |
| `test_gas_csr_gather.py` | Incidence-matrix gathers and scatters (`--gas_csr_gather`). |
| `test_gas_csr_spmm.py` | CSR sparse products with fixed transposes (`--gas_csr_spmm`). |
| `test_gas_hard_neg.py` | Hard negatives inside the GAS epoch. |
| `test_gas_layer0_history.py` | Layer-0 history (`--gas_layer0_history`). |
| `test_gas_metis_csr.py` | `--gas_cluster_method metis_csr`: balanced assignment, lower edge cut, fallback. |
| `test_gas_metis_fallback.py` | BFS-chunk fallback on any METIS failure. |
| `test_gas_row_slice_index.py` | Row-slice index equals the mask-based reference. |
| `test_gas_softmax_history.py` | Softmax-denominator history (`--gas_softmax_history`) reproduces `RREA.forward`. |
| `test_gas_src_sorted.py` | Source-sorted clusters and segment softmax give the same outputs and gradients. |
| `test_halo_halo_drop.py` | `--drop_halo_halo_edges`. |
| `test_halo_refresh_warm_start.py` | The halo refresh survives the warm-start table restore. |
| `test_hard_neg.py` | Hard-negative mining, sampling and merging. |
| `test_init_scale.py` | `--init_scale`; 1.0 is bit-identical. |
| `test_joint_metis_partition.py` | Joint KG1 + KG2 METIS partition keeps training pairs together. |
| `test_kg2_refine.py` | Label-propagation refinement of a KG2 assignment. |
| `test_lr_schedule.py` | Learning-rate schedules (none / cosine / linear / step, warm-up). |
| `test_minibatch_loss.py` | Loss selection on the mini-batch path mirrors the full-graph path. |
| `test_neg_pool_frac.py` | Uniform negatives from a fixed subset (`--neg_pool_frac`). |
| `test_nhsm_chunked.py` | Row-chunked NHSM equals the dense form. |
| `test_nhsm_compact.py` | NHSM on a compact table equals the full-table NHSM. |
| `test_pseudo_seed_policy.py` | Pseudo-seed policies: merge rules, candidate sets, mutual nearest neighbours. |
| `test_rel_vec_per_relation.py` | Per-relation normalisation of relation vectors equals the per-edge path. |
| `test_rescore_variants.py` | Offline re-scoring (`src/rescore.py`): CSLS, top-K, Sinkhorn, parity with `eval.py`. |
| `test_rrea_dropout.py` | `--rrea_dropout`. |
| `test_rrea_edge_chunk.py` | Edge-chunked RREA forward is exact in outputs and gradients. |
| `test_seed.py` | `--seed`: per-rank streams; no effect on id-seeded features. |
| `test_segment_softmax.py` | Segment softmax equals the scatter-based softmax. |
| `test_table_consensus.py` | Table consensus (`mean` / `delta` / `delta_dc`) over 3 ranks. |
| `test_train_opt.py` | Optimiser construction and weight-decay groups. |
| `test_vector_graph.py` | Vectorised local-graph construction equals the Python loops exactly. |
