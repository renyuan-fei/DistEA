# src — the DistEA framework

`main.py` (in the repository root) parses the options, initialises `torch.distributed`, loads the
dataset and hands control to `DistEAFramework.run()` in `framework.py`. Every rank runs the same
code on its own partition.

## Data flow of one run

```
dataload.py          OpenEAData: triples and train/valid/test links, global entity ids
        |            (KG1 ids first, KG2 ids after them)
partition.py         offline, once: METIS partition of KG1 + KG2 assignment
counterpart_disc.py  -> outputs/partitions/.../{rank}.npy, kg2_assignments.json, meta.json
kg2_refine.py        (refined assignment, written by scripts/make_refined_kg2_partition.py)
        |
dist_graph.py        per rank: local graph [local KG1 | halo KG1 | local KG2 | halo KG2]
        |
framework.py         outer iterations: train -> evaluate -> mine pseudo-seeds -> train
  gas.py               cluster-wise training of the RREA encoder with historical embeddings
  models/rrea.py       the RREA encoder (full-graph forward, used for evaluation)
  table_consensus.py   periodic merge of entity rows replicated on several ranks
  dist_trainer.py      parameter all-reduce; full-graph training epoch
  eval.py              distributed evaluation (ownership-filtered gather, CSLS, Hits@k, MRR)
        |
outputs/results/<dataset>/<timestamp>/metrics.json, alignment.csv
```

## Modules

**Data and partitioning**

| Module | Purpose |
|---|---|
| `dataload.py` | `OpenEAData`: reads an OpenEA-layout dataset and assigns integer ids (optional `.npz` cache). |
| `graph_metrics.py` | Relation functionality (r2f / r2if) edge weights in the style of ELsEA. |
| `partition.py` | Offline METIS partitioning of KG1 and assignment of KG2 entities to ranks. |
| `counterpart_disc.py` | KG2 assignment strategies: round-robin and locality-based majority vote. |
| `kg2_refine.py` | Label-propagation refinement of a KG2 assignment (fewer pairs split across ranks). |
| `dist_graph.py` | `DistGraphLoader`: the per-rank local graph with two-sided halo nodes; RREA graph construction. |

**Model and training**

| Module | Purpose |
|---|---|
| `models/rrea.py` | PyTorch port of the RREA encoder (relational reflection attention). |
| `models/gcn.py` | PyTorch port of GCN-Align (secondary encoder), deterministic entity features, and the alignment losses (margin, InfoNCE, NHSM). |
| `gas.py` | Cluster-wise training with historical embeddings (GNNAutoScale) for RREA: cluster construction, forward/backward per cluster, optimised sparse kernels. |
| `framework.py` | `DistEAFramework`: the training loop over iterations and epochs, pseudo-seed mining, model selection, result writing. |
| `dist_trainer.py` | `DistTrainer`: one full-graph training epoch and the all-reduce of the shared parameters. |
| `table_consensus.py` | Periodic consensus of entity rows held by several ranks (`mean`, `delta`, `delta_dc`). |
| `train_opt.py` | Optimiser construction (weight-decay groups) and learning-rate schedules. |
| `epoch_select.py` | Best-epoch / best-iteration selection on the validation split. |
| `hard_neg.py` | Optional hard-negative mining for the margin loss. |
| `seeding.py` | `--seed`: seeds Python, NumPy and torch per rank. |

**Evaluation**

| Module | Purpose |
|---|---|
| `eval.py` | Distributed and chunked evaluation, CSLS, pseudo-seed mining, export bundles. |
| `rescore.py` | Offline re-scoring of exported embeddings under other inference protocols. |

**Optional communication mechanisms** (off in the final protocol; studied in the thesis)

| Module | Purpose |
|---|---|
| `dist_halo_sync.py` | Halo-row refresh between iterations through a shared file system or the gRPC server. |
| `leapgnn_cache.py`, `leapgnn/` | File and gRPC embedding caches; `leapgnn/` is the Python client of `cache_server/` (adapted from LeapGNN). |
| `dist_ocd.py` | Owner-compute dispatch: point-to-point halo refresh between the owning ranks. |
| `dist_vbip.py` | VBIP pair weighting of the alignment loss (needs DGL). |
| `dist_ctx_kg2.py` | Extra negative candidates from KG2 counterparts held by other ranks. |
| `comm_stats.py` | Timing and byte counters for communication, reported in `metrics.json`. |
| `dist_utils.py` | Small `torch.distributed` portability helpers. |

**Mini-batch path** (`--minibatch`; an alternative to cluster-wise training, compared in the thesis)

| Module | Purpose |
|---|---|
| `sparse_table.py` | `SparseEntityTable`: entity rows materialised on demand, with their own Adam state. |
| `minibatch_sampler.py`, `minibatch_csr.py`, `rrea_minibatch.py` | Multi-hop neighbour sampling and RREA blocks (Python, vectorised and DGL backends). |
| `minibatch_ocd.py` | Cross-rank fetch of rows needed by a batch. |
| `minibatch_eval.py`, `minibatch_loss.py` | Batched inference and loss selection for the mini-batch path. |

## Environment switches

Some exact, result-preserving speed-ups are switched on through environment variables rather
than options, so that they can be enabled for whole jobs; `launch/protocol.sh` sets all of them.

| Variable | Effect |
|---|---|
| `DISTEA_EVAL_DEVICE=cuda` | Chunked evaluation on the GPU. |
| `DISTEA_EVAL_SHARD=1` | Evaluation and pseudo-seed mining split across ranks. |
| `DISTEA_VECTOR_GRAPH=1` | Vectorised construction of the local graph. |
| `DISTEA_DATA_CACHE=1` | Cache of the parsed triples (`DISTEA_DATA_CACHE_DIR`). |
| `DISTEA_FEAT_PROCS=P` | Deterministic feature initialisation with *P* worker processes. |
| `DISTEA_GAS_FAST_BUILD=1` | Cluster construction on the GPU and cached METIS assignments. |
| `DISTEA_GAS_SEGMENT_SOFTMAX=1` | Segment softmax over source-sorted edges. |
| `DISTEA_GAS_COMPILE_HEAD=1` | `torch.compile` of the attention head. |
| `DISTEA_GAS_TF32=1` | TF32 matrix products in the training step (evaluation stays fp32). |
| `DISTEA_GAS_CHECKPOINT=1` | Gradient checkpointing of the attention heads (saves memory, costs time). |
