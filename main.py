"""
DistEA entry point — compatible with torchrun / torch.distributed.launch.

Single-machine 4-GPU example:
    torchrun --nproc_per_node=4 main.py \\
        --data_dir  data/OpenEA_dataset_v2.0/EN_DE_15K_V1 \\
        --fold_dir  data/OpenEA_dataset_v2.0/EN_DE_15K_V1/721_5fold/1 \\
        --partition_dir outputs/partitions/EN_DE_15K_V1/721_5fold_1/dist_True/4_metis_kg2locality

Multi-machine 2-node × 4-GPU example (run on each node):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \\
        --master_addr=<master-ip> --master_port=29500 main.py [args]
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.dataload import OpenEAData
from src.framework import DistEAFramework
from src.seeding import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="DistEA — Distributed Entity Alignment")

    # ---- data ----
    p.add_argument("--data_dir", required=True,
                   help="Root of the OpenEA dataset folder (e.g. EN_DE_15K_V1)")
    p.add_argument("--fold_dir", required=True,
                   help="Fold directory containing train/valid/test_links")
    p.add_argument("--partition_dir", required=True,
                   help="Directory with METIS partition files (0.npy … N.npy, "
                        "ent2id.json, kg2_assignments.json)")

    # ---- model ----
    p.add_argument("--hidden_dim", type=int, default=256,
                   help="Deprecated and ignored: the encoders have no per-layer "
                        "learnable transform, so the entity table h_init operates "
                        "at --output_dim directly. Accepted so that older command "
                        "lines still parse; omit it (or set it equal to "
                        "--output_dim) to avoid the startup warning.")
    p.add_argument("--output_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2,
                   help="RREA attention depth (RREA/runner.py default: 2)")
    p.add_argument("--attn_heads", type=int, default=1,
                   help="RREA attention heads (RREA/runner.py's training run "
                        "uses 1, not create_model()'s own default of 2)")
    p.add_argument("--dropout", type=float, default=0.2)         # ELsEA default

    # ---- training (defaults aligned with ELsEA GCN_Align_run.py) ----
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_epochs", type=int, default=200,
                   help="Training epochs per outer iteration (default 200). Caution: "
                        "much larger budgets (e.g. 1200, the RREA/ClusterEA recipe) "
                        "combined with --use_infonce on a single rank were observed "
                        "to make Hit@1 fall across outer iterations, whereas "
                        "--no_use_infonce under the same budget kept improving; "
                        "re-check before raising this well above 200 for a "
                        "multi-iteration InfoNCE run.")
    p.add_argument("--max_iterations", type=int, default=5)
    p.add_argument("--later_iter_epochs", type=int, default=None,
                   help="If set, iterations after the first train for "
                        "min(this, --max_epochs) epochs instead of the full "
                        "--max_epochs, as RREA/runner.py's iterative_training() "
                        "does (full budget for turn 0, min(100, max_train_epoch) "
                        "afterwards). Default: unset, every iteration uses "
                        "--max_epochs.")
    p.add_argument("--neg_k", type=int, default=1)               # ELsEA default
    p.add_argument("--margin", type=float, default=6.0)          # ELsEA default
    p.add_argument("--eval_interval", type=int, default=1,
                   help="Evaluate every N iterations")
    p.add_argument("--use_csls", action="store_true", default=True,
                   help="Use CSLS-corrected similarity at eval time (default on), "
                        "as RREA/runner.py's CSLS_test() does with k=10; GCN-Align's "
                        "get_hits() has no CSLS. Pass --no_use_csls to disable.")
    p.add_argument("--no_use_csls", dest="use_csls", action="store_false")
    p.add_argument("--csls_k", type=int, default=10,
                   help="CSLS neighbourhood size (RREA/runner.py default: 10)")
    p.add_argument("--eval_metric", choices=["cosine", "l1"], default="cosine",
                   help="Similarity metric at eval time (default: cosine, the RREA "
                        "protocol: L2-normalised embeddings ranked by dot product). "
                        "'l1' ranks by L1/cityblock distance, as GCN-Align's "
                        "get_hits() does.")
    p.add_argument("--seed_threshold", type=float, default=0.85,
                   help="Deprecated and ignored: pseudo-seeds are mutual nearest "
                        "neighbours with no similarity threshold, as in RREA (see "
                        "eval.py's generate_pseudo_seeds). Accepted so that older "
                        "command lines still parse.")
    p.add_argument("--pseudo_seed_policy", choices=["legacy", "unaligned", "testside"], default="legacy",
                   help="How pseudo-seeds are mined. legacy: mutual-NN over every "
                        "entity, and candidates equal to a test pair are rejected (this "
                        "uses test labels and discards correct predictions; not "
                        "recommended). unaligned: RREA's protocol -- mutual-NN among "
                        "entities outside the train+valid links, no label filter, "
                        "search on DISTEA_EVAL_DEVICE; the logged precision against "
                        "the test links is for logging only. testside: same, but the "
                        "candidate pool is the entity sets of the test links (the "
                        "candidates eval ranks against; their pairing is never used). "
                        "Use testside for pseudo-seed runs.")
    p.add_argument("--pseudo_seed_refresh", choices=["accumulate", "replace"], default="accumulate",
                   help="With --pseudo_seed_policy unaligned|testside: accumulate "
                        "(default, RREA) keeps every round's pseudo-seeds and removes "
                        "their entities from the pool; replace re-mines the whole pool "
                        "each round on top of the original train links and drops the "
                        "previous round's set.")
    p.add_argument("--pseudo_seed_top_frac", type=float, default=1.0,
                   help="With --pseudo_seed_policy unaligned|testside: keep only this "
                        "fraction of each round's mutual-NN pairs, the most confident "
                        "by CSLS-adjusted similarity (1.0 = all, as in RREA).")
    p.add_argument("--use_pseudo_seeds", action="store_true", default=False,
                   help="Augment train_links with mutual-nearest-neighbour pseudo-seeds "
                        "after each iteration (iterative EA, as in RREA/runner.py: no "
                        "confidence threshold). The pseudo-seed similarity uses CSLS "
                        "(--use_csls/--csls_k), and once an entity has been used by an "
                        "accepted pseudo-seed it cannot receive a second, conflicting "
                        "pair in a later iteration. Default off.")

    # ---- ELsEA-style loss / optimizer ----
    p.add_argument("--loss", choices=["margin", "nhsm"], default="margin",
                   help="Training loss. 'margin' (default): the ELsEA-style margin "
                        "loss (align_loss, optionally mixed with InfoNCE). 'nhsm': "
                        "Dual-AMN Normalized Hard Sample Mining (whole-table "
                        "negatives, per-row z-scored logits, logsumexp(30x+10)), "
                        "ported from ClusterEA's PyTorch code; under nhsm, "
                        "--neg_k/--neg_resample_every/--use_infonce and MM "
                        "pair-weights have no effect. With --minibatch, nhsm gives "
                        "a strong first iteration but degrades over later outer "
                        "iterations, so use it with --max_iterations 1 there.")
    p.add_argument("--nhsm_gamma", type=float, default=1.0,
                   help="NHSM margin gamma (Dual-AMN default 1.0; distinct from "
                        "--margin, which belongs to the margin-loss path).")
    p.add_argument("--amp", choices=["none", "bf16"], default="none",
                   help="'bf16' runs the model forward under "
                        "torch.autocast(bfloat16); loss math, backward, "
                        "optimizer and allreduce stay fp32 (no GradScaler "
                        "needed -- bf16 shares fp32's exponent range). "
                        "Full-graph training path only. Default 'none'.")
    p.add_argument("--compile", action="store_true", default=False,
                   help="Wrap the full-graph training forward in torch.compile "
                        "(fixed shapes, so it compiles once). The variable-shape "
                        "minibatch path is never compiled; state_dict, warm start "
                        "and eval use the uncompiled module, so checkpoints and "
                        "metrics are unchanged. Default off.")
    p.add_argument("--gas_closure_cap", type=int, default=0,
                   help="With --gas_clusters: keep at most this many non-batch "
                        "in-edges per destination node in each cluster's softmax "
                        "closure (uniform sample, weighted n_other/n_kept so the "
                        "softmax denominator stays an unbiased estimate; every "
                        "batch-src edge is always kept). The exact closure is "
                        "dominated by hub entities whose full in-edge set lands in "
                        "every cluster; a cap of 128-512 bounds it. Approximate, "
                        "sample fixed per outer iteration. 0 (default) = exact "
                        "closure. See gas.py build_gas_clusters.")
    p.add_argument("--gas_cluster_method", choices=["metis", "bfs", "metis_csr", "rcm"], default="metis",
                   help="How --gas_clusters partitions the local graph. metis "
                        "(default): nxmetis through a networkx graph, with a "
                        "BFS-chunk fallback above the size guard (always taken at "
                        "DBP1M scale). metis_csr: METIS on a scipy-built CSR via "
                        "the C-level entry point (pymetis if installed, else the "
                        "vendored extension), after dropping hubs of degree > "
                        "--gas_hub_degree (assigned round-robin); takes seconds at "
                        "DBP1M scale and gives a real edge cut, which bounds how "
                        "often a non-hub edge is replicated across cluster "
                        "closures. rcm: scipy reverse-Cuthill-McKee order (hub "
                        "edges removed) cut into equal chunks. bfs: the chunk "
                        "heuristic directly.")
    p.add_argument("--gas_hub_degree", type=int, default=1000,
                   help="metis_csr only: nodes whose undirected degree exceeds "
                        "this are left out of the METIS graph (at DBP1M, a few "
                        "hundred such nodes carry about 17%% of a rank's edges). "
                        "0 = keep every node.")
    p.add_argument("--eval_every_epochs", type=int, default=0,
                   help="Also run the eval every this many epochs inside an "
                        "iteration (logged, and stored under the iteration's "
                        "'epoch_evals' in metrics.json together with the training "
                        "seconds elapsed so far), giving a time-vs-accuracy curve "
                        "from one run. Each eval costs about one eval's wall time. "
                        "0 (default) = off.")
    # ---- seed / per-epoch split evals / epoch selection ----
    p.add_argument("--seed", type=int, default=None,
                   help="Seed random/numpy/torch with seed+rank (negative "
                        "sampler, GAS cluster order, dropout). h_init, rel_emb "
                        "and attention kernels stay ID-seeded regardless. "
                        "None (default) = unseeded.")
    p.add_argument("--eval_every_splits", type=str, default="test",
                   help="Comma-separated splits evaluated at each "
                        "--eval_every_epochs point: test (unprefixed keys), "
                        "valid (valid_*), train (train_*; the current seed "
                        "set). Default 'test'. Each epoch_evals entry also "
                        "records lr and h_init_norm_mean.")
    p.add_argument("--keep_best_epoch", action="store_true", default=False,
                   help="Snapshot model + h_init (on CPU) whenever the "
                        "--select_by split's Hit@1 improves at an "
                        "--eval_every_epochs point (the final epoch is "
                        "scored too), restore that snapshot before the "
                        "iteration's test eval, and report test numbers for "
                        "the selected epoch. Needs --eval_every_epochs > 0; "
                        "with --select_by valid the valid split is added to "
                        "--eval_every_splits automatically. Default off.")
    p.add_argument("--epoch_patience", type=int, default=0,
                   help="Stop the iteration's epoch loop once the "
                        "--select_by Hit@1 has not improved for this many "
                        "consecutive --eval_every_epochs evaluations "
                        "(rank-0 decision, broadcast). Implies the same "
                        "snapshot/restore as --keep_best_epoch. 0 (default) "
                        "= off.")
    p.add_argument("--select_by", choices=["test", "valid"], default="test",
                   help="Split whose Hit@1 picks the epoch (with "
                        "--keep_best_epoch/--epoch_patience) and the 'best' "
                        "iteration written to metrics.json. Default 'test'; "
                        "use 'valid' for reported numbers so that model "
                        "selection does not see the test split.")
    # ---- hard negatives from the GAS final history ----
    p.add_argument("--hard_neg_pool", type=int, default=0,
                   help="Mine this many nearest wrong entities per train pair "
                        "and side from the GAS final-history table "
                        "(src/hard_neg.py; no extra forward) and append "
                        "--hard_neg_k of them, sampled per epoch, to the "
                        "uniform negatives of the margin loss. GAS path only. "
                        "0 (default) = off.")
    p.add_argument("--hard_neg_k", type=int, default=1,
                   help="Hard negatives per pair and side drawn from the "
                        "pool each epoch (on top of --neg_k uniform ones).")
    p.add_argument("--hard_neg_every", type=int, default=25,
                   help="Re-mine the pools every this many epochs "
                        "(each mining takes ~8-12 s at DBP1M; values below 10 warn).")
    p.add_argument("--hard_neg_warmup", type=int, default=10,
                   help="First epoch (1-based) that trains with hard "
                        "negatives; earlier epochs let the table leave its "
                        "random init so the pools are meaningful.")
    p.add_argument("--hard_neg_side", choices=["other", "any"], default="other",
                   help="other (default): KG1 candidates for corrupting e1 and KG2 "
                        "candidates for e2 (what the loss corrupts); any: whole table.")
    p.add_argument("--hard_neg_dtype", choices=["bf16", "fp32"], default="bf16",
                   help="Matmul dtype for mining on CUDA (bf16 halves the "
                        "normalised copy: 1.3 GB at DBP1M).")
    # ---- regularisation: decoupled weight decay + lr schedule ----
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="Decoupled (AdamW) weight decay on the encoder "
                        "parameters (rel_emb, attention kernels) and, unless "
                        "--weight_decay_h_init says otherwise, on h_init. "
                        "The loss-mix weight is never decayed. adam only. "
                        "0 (default) = plain Adam.")
    p.add_argument("--weight_decay_h_init", type=float, default=None,
                   help="Separate decay for the entity table h_init "
                        "(default: same as --weight_decay).")
    p.add_argument("--lr_schedule", choices=["none", "cosine", "linear", "step"], default="none",
                   help="Per-epoch lr schedule (stepped once after each "
                        "epoch, rebuilt per iteration): cosine / linear decay "
                        "to --lr_min_ratio x lr over the iteration, or step "
                        "decay by --lr_gamma every --lr_step_size epochs. "
                        "none (default) = constant lr.")
    p.add_argument("--lr_warmup_epochs", type=int, default=0,
                   help="Linear lr warmup over this many epochs before "
                        "the schedule (0 = none).")
    p.add_argument("--lr_min_ratio", type=float, default=0.0,
                   help="Final lr as a fraction of --lr for cosine/linear; "
                        "floor for step.")
    p.add_argument("--lr_step_size", type=int, default=50,
                   help="step schedule: epochs per decay.")
    p.add_argument("--lr_gamma", type=float, default=0.5,
                   help="step schedule: multiplier per decay.")
    p.add_argument("--nhsm_negs", choices=["full", "compact"], default="full",
                   help="With --loss nhsm on the GAS path: full (default) = "
                        "negatives are the whole table (~15 s/epoch and "
                        "OOM-prone at DBP1M); compact = a uniform pool of "
                        "--nhsm_pool rows plus the pair rows per cluster "
                        "(ClusterEA's in-batch shape), on the compact table.")
    p.add_argument("--nhsm_pool", type=int, default=10000,
                   help="Pool size for --nhsm_negs compact.")
    p.add_argument("--neg_pool_frac", type=float, default=1.0,
                   help="Ablation (GAS path only): draw the uniform negatives from "
                        "a fixed random subset holding this fraction of the local "
                        "table (seeded 12345+rank) -- a single-GPU stand-in for "
                        "the per-rank negative pool of a world_size=1/frac "
                        "partition, graph untouched. 1.0 (default) = off.")
    p.add_argument("--attn_softmax", choices=["dst", "src"], default="dst",
                   help="Edge endpoint that groups the RREA attention softmax. "
                        "src matches the original Keras layer, which normalises "
                        "over the edges of the node messages are summed into; "
                        "dst (default) groups by the neighbour instead, which "
                        "gives a degree-dependent output scale. Use src for the "
                        "faithful RREA behaviour.")
    p.add_argument("--rrea_input_relu", action="store_true", default=False,
                   help="Apply relu to the input features before the depth-0 "
                        "slice / layer 1, as the original RREA layer.py does. "
                        "Default off.")
    p.add_argument("--init_scale", type=float, default=1.0,
                   help="Multiply the ID-seeded unit-norm initial entity rows "
                        "(h_init) and relation rows (rel_emb) by this factor. The "
                        "original RREA starts both tables at U(-0.05, 0.05), i.e. "
                        "row norm ~0.33 at dim 128, and the L1 margin is an "
                        "absolute quantity. Default 1.0.")
    p.add_argument("--rrea_dropout", type=float, default=0.0,
                   help="The original RREA dropout (runner.py: 0.2 on the "
                        "final concatenated embedding; the layers have none). "
                        "--dropout does not act on the RREA path. GAS applies it "
                        "to the loss rows only, never to the history tables. "
                        "0 (default) = off.")
    p.add_argument("--rrea_edge_chunk", type=int, default=0,
                   help="Process the RREA attention forward in dst-grouped edge "
                        "chunks of about this many edges (exact: chunk boundaries "
                        "never split a softmax group). Bounds the per-edge "
                        "intermediates of the full-graph forward -- which eval, "
                        "pseudo-seeds and the end-of-iteration dump use even with "
                        "--gas_clusters -- to one chunk; needed at DBP1M scale, "
                        "where the whole-graph eval forward does not fit in 80 GB. "
                        "2000000 is a good value there. 0 (default) = whole graph "
                        "in one pass.")
    p.add_argument("--gas_softmax_history", action="store_true", default=False,
                   help="With --gas_clusters: keep a per-edge history of "
                        "exp(attention logit) (one fp64 value per edge per "
                        "channel/layer/head) so each cluster processes only its "
                        "own batch-src edges -- a dst's softmax denominator is "
                        "the fresh sum over those edges plus its stale total "
                        "minus their stale values. Every edge is recomputed once "
                        "per epoch and no denominator is more than one epoch "
                        "stale (the same trade GAS makes for features), so the "
                        "per-epoch edge work is 1x the local edge count instead "
                        "of several times that for the exact closure. Makes "
                        "--gas_closure_cap moot. Default off.")
    p.add_argument("--gas_layer0_history", action="store_true", default=False,
                   help="With --gas_clusters: take the layer-0 features of "
                        "out-of-batch needed nodes from a history table (stale "
                        "<= 1 epoch, like layers >= 1) instead of recomputing "
                        "them exactly per cluster. The exact layer-0 spmm "
                        "re-reads every hub destination's full in-edge list in "
                        "every cluster that touches it; with this flag the "
                        "layer-0 work is 1x the local edges and the cluster "
                        "build stops slicing hub rows. Tables are warmed once "
                        "with the exact whole-graph rows. Approximate (exact at "
                        "the fixed point); default off.")
    p.add_argument("--no_save_model", action="store_true", default=False,
                   help="Skip writing model.pt (model weights + the whole "
                        "h_init table, 0.9 GB at DBP1M/N=1) at the end of the "
                        "run; metrics.json / alignment outputs are unaffected. "
                        "Default off.")
    p.add_argument("--fused_adam", action="store_true", default=False,
                   help="Use torch.optim.Adam(fused=True) on CUDA for the main "
                        "training optimizer (one fused kernel per step instead "
                        "of foreach; same update up to fp32 rounding). CPU runs "
                        "ignore it. Default off.")
    p.add_argument("--gas_csr_gather", action="store_true", default=False,
                   help="With --gas_clusters: do the per-edge dst/src gathers and "
                        "the message scatter of each attention head as spmm's "
                        "with per-cluster 0/1 incidence matrices (CSR + "
                        "precomputed transposes), so their backward is a "
                        "segmented reduction instead of the sort-based "
                        "embedding/index_put_ backward (over half of an epoch's "
                        "time at DBP1M). Exact; falls back to the index path if "
                        "CSR is unavailable. Default off.")
    p.add_argument("--gas_csr_spmm", action="store_true", default=False,
                   help="With --gas_clusters: store each cluster's layer-0 "
                        "slices as CSR with precomputed CSR transposes, so the "
                        "layer-0 backward is one spmm instead of "
                        "torch.sparse.mm's per-call transpose + coalesce. "
                        "Exact; falls back to COO if the torch build lacks CSR "
                        "support. Default off.")
    p.add_argument("--gas_clusters_on_device", action="store_true", default=False,
                   help="With --gas_clusters: keep every cluster's index tensors "
                        "and row-sliced layer-0 matrices on the compute device "
                        "after the build, avoiding per-step host-to-device "
                        "copies. Exact; default off.")
    p.add_argument("--gas_history_on_device", action="store_true", default=False,
                   help="With --gas_clusters: keep the GAS historical-embedding "
                        "tables on the compute device instead of CPU (2 channels "
                        "x (layers-1) x N x dim fp32, ~660 MB at DBP1M/N=4), "
                        "removing a CPU gather + host-to-device copy of every "
                        "cluster's needed rows per channel per layer per step. "
                        "Exact; default off.")
    p.add_argument("--gas_clusters", type=int, default=0,
                   help="GNNAutoScale-style (GAS) cluster-wise training: "
                        "partition the local graph into this many METIS "
                        "clusters and train one cluster per step, with "
                        "out-of-batch neighbors read from historical "
                        "embeddings (exact 1-hop aggregation, no sampling). "
                        "Mutually exclusive with --minibatch. 0 (default) = "
                        "off (full-graph epochs).")
    p.add_argument("--halo_sample_p", type=float, default=None,
                   help="BNS-GCN-style halo sampling: keep this fraction of "
                        "halo nodes each training epoch (fresh draw per epoch; "
                        "edges to dropped halo nodes removed, adjacency "
                        "renormalised; eval always uses the full graph). Sets "
                        "both --halo_sample_p_kg1 and --halo_sample_p_kg2 "
                        "unless they are given explicitly. Default: no "
                        "sampling (p=1).")
    p.add_argument("--halo_sample_p_kg1", type=float, default=None,
                   help="Per-epoch keep fraction for halo_kg1 only.")
    p.add_argument("--halo_sample_p_kg2", type=float, default=None,
                   help="Per-epoch keep fraction for halo_kg2 only.")
    p.add_argument("--use_infonce", action="store_true", default=True,
                   help="Use ELsEA-style combined loss = w0·InfoNCE + w1·align (default ON). "
                        "Pass --no_use_infonce to fall back to pure align_loss.")
    p.add_argument("--no_use_infonce", dest="use_infonce", action="store_false")
    p.add_argument("--infonce_temp", type=float, default=0.1,
                   help="Temperature for InfoNCE (ELsEA infoNce1 default)")
    p.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam",
                   help="adam (DistEA default) / sgd (GCN-Align original; pair "
                        "with --lr 40) / rmsprop (RREA original; pair with "
                        "--lr 0.005)")
    p.add_argument("--neg_resample_every", type=int, default=1,
                   help="Re-sample negatives every N epochs (default 1, as "
                        "RREA/runner.py's get_train_set() does every epoch). "
                        "GCN-Align's reference script resamples every 10 epochs; "
                        "set 10 to match it with --optimizer sgd.")

    # ---- iterative EA controls ----
    p.add_argument("--warm_start", action="store_true", default=True,
                   help="Load previous iteration's model weights as warm-start (default on).")
    p.add_argument("--no_warm_start", dest="warm_start", action="store_false",
                   help="Disable warm-start; re-initialise model weights each iteration.")
    p.add_argument("--persist_optimizer", action="store_true", default=False,
                   help="Keep the same model/optimizer/h_init/loss_weight objects "
                        "across iterations instead of rebuilding and warm-start "
                        "loading them, so Adam's moment estimates carry over, as "
                        "in RREA/runner.py's iterative_training() (one continuous "
                        "model and optimizer across all turns). Avoids the "
                        "peak-then-decline over iterations seen without it. Halo "
                        "rows are patched onto the reused h_init in place. "
                        "Overrides --warm_start's per-iteration reload. Default off.")
    p.add_argument("--halo_refresh_after_warm_start", action="store_true", default=False,
                   help="Re-apply the per-iteration halo-row refresh (--cache_backend "
                        "file/grpc) on top of --warm_start's restored entity table. "
                        "Without it, on the warm_start path without "
                        "--persist_optimizer, the restore overwrites the freshly "
                        "patched halo rows with the previous iteration's local "
                        "copies, so the cross-rank refresh has no effect after "
                        "iteration 0 (a rank-0 warning fires when that happens). "
                        "Default off to keep the original semantics; set it for any "
                        "run meant to refresh halos between iterations.")
    p.add_argument("--eval_on_valid", action="store_true", default=False,
                   help="Evaluate on valid_links each iteration and use for early stopping.")
    p.add_argument("--patience", type=int, default=3,
                   help="Stop early if valid Hit@1 does not improve for this many iterations "
                        "(only active when --eval_on_valid is set).")
    p.add_argument("--min_iterations", type=int, default=2,
                   help="Minimum number of iterations to run before allowing early stop or "
                        "convergence detection. Prevents stopping in iteration 1 when "
                        "embeddings are still warming up.")

    # ---- distributed optimization mechanisms ----
    p.add_argument("--use_mm", action="store_true", default=False,
                   help="Weight the alignment loss's train pairs by VBIP "
                        "informativeness scores, with the global VBIP statistics "
                        "re-synced every --mm_sync_interval iterations. Default off.")
    p.add_argument("--mm_sync_interval", type=int, default=5,
                   help="Iterations between VBIP global-stat syncs when --use_mm.")
    p.add_argument("--vbip_impl", choices=["simplified", "elsea"], default="simplified",
                   help="VBIP scoring implementation used by --use_mm. 'simplified' "
                        "(default): a loosely ELsEA-inspired diffusion (unweighted "
                        "mean aggregation, gamma-damped residual mix, single final "
                        "sigmoid). 'elsea': a port of ELsEA's VBIP "
                        "(divea/eviPassing.py) -- weighted-sum aggregation over the "
                        "r2f/r2if edge weights, two passes with ratio normalisation "
                        "and 2*sigmoid(gamma*x)-1 squashing between them (ELsEA's "
                        "leave-one-out candidate-selection loop, a partition-time "
                        "mechanism, is not included).")
    p.add_argument("--use_ocd", action="store_true", default=False,
                   help="Owner-Compute Dispatch: refresh halo rows via batched "
                        "point-to-point send/recv instead of the shared-file cache.")
    p.add_argument("--minibatch", action="store_true", default=False,
                   help="Mini-batch neighbour-sampling training: replaces the "
                        "full-graph per-iteration loop with per-batch "
                        "sampled-neighbourhood training (RREA.forward_blocks) plus "
                        "per-batch cross-rank on-demand fetch (minibatch_ocd.py), so "
                        "at most one batch's sampled frontier is materialised at "
                        "once (memory-bounded at DBP1M scale). RREA only; honours "
                        "--loss/--use_infonce (src/minibatch_loss.py). "
                        "--use_ocd/--use_saca/--use_mm/--use_halo_kg2/"
                        "--shared_h_init/--persist_optimizer/--warm_start are not "
                        "used on this path (see DistEAFramework._run_minibatch). "
                        "Default off.")
    p.add_argument("--batch_size", type=int, default=512,
                   help="--minibatch: seed pairs per training/eval batch.")
    p.add_argument("--minibatch_sampler", choices=["python", "vectorized", "dgl"],
                   default="vectorized",
                   help="--minibatch: neighbour-sampling / block-construction "
                        "backend. 'python' = rrea_minibatch.py's reference "
                        "dict-of-lists loops (slow); 'vectorized' (default) = "
                        "minibatch_csr.py numpy/CSR implementation with identical "
                        "structural semantics (tests/test_stage_v.py parity "
                        "tests); 'dgl' = DGL's C++ sample_neighbors for the "
                        "sampling step + CSR blocks (falls back to 'vectorized' "
                        "with a warning when DGL is not importable; samples K arcs "
                        "among all arcs before dropping seen nodes, so per-hop "
                        "frontiers can be < fanout, as in GraphSAGE).")
    p.add_argument("--minibatch_reset_table", action="store_true", default=False,
                   help="--minibatch: rebuild the SparseEntityTable (deterministic "
                        "init) + optimizer + loss-mix weight at every outer iteration, "
                        "warm-starting only the model weights -- the full-graph path's "
                        "default per-iteration semantics (it rebuilds h_init unless "
                        "--persist_optimizer). Default off = table and Adam state carried "
                        "across iterations.")
    p.add_argument("--minibatch_neg_extra", type=int, default=0,
                   help="--minibatch: add this many random entities this rank owns to every "
                        "training batch's seed set (no pairs) so their embeddings exist and "
                        "serve as margin/NHSM negatives, like the full-graph path's "
                        "all-entity negative sampling. 0 (default) = seed rows only. "
                        "See minibatch_loss.extend_seeds_with_negatives.")
    p.add_argument("--minibatch_halo_snapshot", action="store_true", default=False,
                   help="--minibatch: reproduce the full-graph path's per-iteration halo "
                        "semantics -- at every outer iteration after the first, freeze the "
                        "previous iteration's entity table as what this rank serves to peers' "
                        "on-demand fetches, and re-learn a fresh local table (model weights "
                        "warm-started). Implies --minibatch_reset_table's reset. Default off.")
    p.add_argument("--eval_fanout", type=int, default=None,
                   help="--minibatch: fanout for eval/pseudo-seed inference. None = "
                        "same as --fanout (sampled, noisy inference); "
                        "-1 = full neighbourhood (deterministic, recommended at "
                        "15K-100K scale where a batch reaches most of the graph anyway).")
    p.add_argument("--eval_batch_size", type=int, default=None,
                   help="--minibatch: seed entities per eval/pseudo-seed inference batch "
                        "(None = 4 x --batch_size). Each eval batch costs nearly a "
                        "full-graph forward at 15K (nested active sets), so fewer, "
                        "larger batches cut eval time roughly proportionally.")
    p.add_argument("--fanout", type=int, default=None,
                   help="--minibatch: cap on neighbours sampled per node per hop. "
                        "None = full neighbourhood, which saturates a 30K-entity "
                        "graph immediately and is far slower per epoch; 30 works "
                        "well at 15K scale.")
    p.add_argument("--use_saca", action="store_true", default=False,
                   help="Staleness-Aware Communication Avoiding: skip rewriting an "
                        "entity's embedding when it changed less than --saca_threshold.")
    p.add_argument("--saca_threshold", type=float, default=0.01,
                   help="Relative-L2 change threshold below which --use_saca skips a write.")
    p.add_argument("--use_halo_kg2", action="store_true", default=False,
                   help="KG2 Counterpart Halo: give each rank extra, possibly cross-rank "
                        "KG2 negative-sampling candidates selected via ELsEA-style "
                        "multi-hop BFS from this rank's alignment anchors (see "
                        "dist_ctx_kg2.py), so alignment_loss's negatives are not "
                        "limited to this rank's ~1/world_size slice of KG2. "
                        "Requires --cache_dir; not supported with --use_ocd.")
    p.add_argument("--halo_kg2_ratio", type=float, default=0.5,
                   help="halo_kg2 candidate budget per rank, as a multiple of this rank's "
                        "local KG2 count (e.g. 0.5 = 50%% more KG2 visibility).")
    p.add_argument("--halo_kg2_max_hop", type=int, default=None,
                   help="Max BFS hops from anchors when selecting halo_kg2 candidates. "
                        "Default: 2 * --num_layers (matches ELsEA's CounterpartDiscovery "
                        "sizing, i.e. a GCN layer's receptive field).")
    p.add_argument("--no_halo_kg1", action="store_true", default=False,
                   help="Ablation: disable the graph-structural halo_kg1 mechanism "
                        "(dist_graph.py's DistGraphLoader) -- the local graph keeps "
                        "only this rank's own KG1 partition, with no 1-hop cross-rank "
                        "KG1 neighbours. Unrelated to --use_halo_kg2 (a KG2 "
                        "negative-sampling-candidate mechanism).")
    p.add_argument("--table_consensus_every", type=int, default=0,
                   help="Table consensus: every E epochs (and after the last) merge "
                        "each h_init row held by more than one rank (owner + halo "
                        "copies) across its copies, so the alignment signal landing "
                        "on halo copies reaches the owner row that eval reads "
                        "(table_consensus.py). 0 (default) = off.")
    p.add_argument("--table_consensus_mode", choices=["mean", "delta", "delta_dc"], default="mean",
                   help="mean: copies := their mean; delta: copies := base + sum of "
                        "each copy's change since the previous sync; delta_dc: delta "
                        "with AdamW's decoupled decay counted once instead of once per copy.")
    p.add_argument("--eval_rank_split_dir", type=str, default="",
                   help="Diagnostic: partition dir whose ownership labels every "
                        "evaluated link same_rank / cross_rank (KG1 owner from "
                        "{r}.npy, KG2 owner from kg2_assignments.json); adds "
                        "same_rank/Hit@k, cross_rank/Hit@k, .../MRR, .../n to every "
                        "eval. Point an N=1 run at an N>1 partition for a "
                        "like-for-like split. Empty (default) = off.")
    p.add_argument("--eval_pseudo_split", action="store_true", default=False,
                   help="Diagnostic: label every evaluated test pair ps_pair (the "
                        "pair itself was mined as a pseudo-seed), ps_entity (not "
                        "mined, but one of its entities is in some pseudo-seed) or "
                        "ps_clean, and report ps_*/Hit@k, .../MRR, .../n next to the "
                        "overall metrics. Candidates are unchanged; only the scored "
                        "rows are split. Combines with --eval_rank_split_dir as "
                        "'<rank>|<ps>'. Default off.")
    p.add_argument("--ablate_full_graph", action="store_true", default=False,
                   help="Ablation: every rank loads the whole graph (halo = "
                        "complement of its owned sets, all edges), while ownership "
                        "-- eval rows and train-pair sharding -- stays the "
                        "partition's. Separates graph truncation from pair "
                        "sharding / per-rank h_init. Memory per rank = world_size 1.")
    p.add_argument("--no_halo_kg2", action="store_true", default=False,
                   help="Ablation: disable the graph-structural halo_kg2 mechanism "
                        "(dist_graph.py's DistGraphLoader) -- the local graph keeps "
                        "only this rank's own KG2 assignment, with no 1-hop "
                        "cross-rank KG2 neighbours, so KG2-KG2 edges that cross a "
                        "rank boundary are dropped.")

    p.add_argument("--defer_test_eval", action="store_true", default=False,
                   help="With --select_by valid (+ --eval_on_valid): skip the "
                        "per-iteration test eval and run it once after the loop on "
                        "the iteration valid chose (its state is kept on every "
                        "rank). Same reported test metrics; the other iterations "
                        "carry no test numbers. Saves one eval per extra iteration "
                        "(about 30 s at DBP1M). Default off.")
    p.add_argument("--drop_halo_halo_edges", action="store_true", default=False,
                   help="Build each rank's RREA graph from the triples touching an "
                        "owned entity only, leaving out those whose two ends are "
                        "both halo rows (about 28%% of a rank's graph at DBP1M "
                        "N=8). Halo rows then aggregate only from owned neighbours; "
                        "this reduces load but changes accuracy. No-op at N=1. "
                        "Default off.")

    # ---- distributed ----
    p.add_argument("--backend", default="nccl",
                   help="torch.distributed backend (nccl / gloo)")
    p.add_argument("--task_id", default="",
                   help="Optional free-form label stamped into metrics.json (and "
                        "the export metadata) for grouping runs.")
    p.add_argument("--sim", action="store_true", default=False,
                   help="Simulation mode: run N ranks as processes on fewer "
                        "physical GPUs than ranks. Forces --backend gloo (NCCL "
                        "cannot share one GPU across ranks) and maps each rank to "
                        "cuda:(local_rank %% ngpus). The same collectives run "
                        "regardless of how ranks map to devices, so --sim changes "
                        "scheduling only, never numerics.")

    # ---- results ----
    p.add_argument("--results_dir", default="outputs/results",
                   help="Directory to write metrics.json and alignment.csv after training. "
                        "Set to empty string to disable.")
    p.add_argument("--export_dir", default="",
                   help="Directory to write the scripts/unified_eval.py export bundle "
                        "(embeddings.npy/ids.npy/kg_labels.npy/test_links.json) from the "
                        "final test eval, for cross-method comparison. Used as given "
                        "(not namespaced under --results_dir). Leave empty to disable.")
    p.add_argument("--export_dtype", choices=["fp32", "fp16"], default="fp32",
                   help="dtype of embeddings.npy in --export_dir (fp16 halves "
                        "the 2.45 GB DBP1M bundle; exact rescoring parity "
                        "needs fp32). meta.json is always written.")

    # ---- embedding cache (shared filesystem) ----
    p.add_argument("--cache_dir", default="",
                   help="Directory on the shared filesystem for per-rank embedding "
                        "files (ids_{rank}.npy / emb_{rank}.npy). "
                        "Enables halo refresh between iterations and the final "
                        "embedding export (Stage C). Leave empty to disable.")
    p.add_argument("--cache_backend", choices=["none", "file", "grpc"], default="none",
                   help="Backend for the between-iteration halo refresh and the "
                        "Stage C push of final embeddings to a post-training lookup "
                        "service (src/dist_halo_sync.py, src/leapgnn_cache.py). "
                        "'none' (default): disabled. 'file': shared-filesystem "
                        "files, no external infrastructure. 'grpc': LeapGNN's Go "
                        "cache_server (see cache_server/README.md), read-only, "
                        "restarted around each push; needs --cache_dir and "
                        "--grpc_addrs (falls back to 'file' if --grpc_addrs is "
                        "empty). Not a training-time speedup, and unrelated to "
                        "--use_mm.")
    p.add_argument("--grpc_addrs", default="",
                   help="Comma-separated host:port list for --cache_backend grpc's "
                        "Go server(s). Required (non-empty) when --cache_backend=grpc.")
    p.add_argument("--halo_refresh_interval", type=int, default=0,
                   help="Ablation (LeapGNN-inspired): also refresh halo_kg1 and "
                        "halo_kg2 rows every N epochs within an iteration's "
                        "training loop, not just once between iterations. With hard "
                        "overwrites (--halo_blend_alpha 1.0) this lowered Hit@1, "
                        "mostly through the stale Adam momentum that "
                        "--reset_halo_momentum addresses. Requires --cache_dir; "
                        "only acts when world_size > 1. 0 (default) = refresh only "
                        "between iterations.")
    p.add_argument("--halo_blend_alpha", type=float, default=1.0,
                   help="Only affects --halo_refresh_interval's mid-epoch refresh. "
                        "1.0 (default) hard-overwrites halo rows with the remote "
                        "value; <1.0 blends instead: h_init[halo] = "
                        "(1-alpha)*local + alpha*remote. Blending (e.g. 0.3) "
                        "recovers part of the hard-overwrite loss but did not beat "
                        "refreshing only between iterations.")
    p.add_argument("--reassign_kg2_every", type=int, default=0,
                   help="Re-run assign_kg2_to_ranks_locality every N iterations, "
                        "seeded from train_links after this run's pseudo-seeds "
                        "(--use_pseudo_seeds) have been merged in, instead of "
                        "keeping the assignment computed by partition.py. Rank 0 "
                        "rewrites kg2_assignments.json and all ranks barrier "
                        "before the next iteration's DistGraphLoader.load() "
                        "re-reads it. Requires a partition built with "
                        "--kg2_assignment locality. Not supported with "
                        "--persist_optimizer (the persisted h_init/optimizer "
                        "state assumes a fixed row layout). 0 (default) = "
                        "assignment fixed at partition time.")
    p.add_argument("--reset_halo_momentum", action="store_true", default=False,
                   help="Zero Adam's exp_avg/exp_avg_sq for exactly the h_init rows "
                        "a halo refresh just patched, at both patch sites "
                        "(--persist_optimizer's iteration-boundary refresh and "
                        "--halo_refresh_interval's mid-epoch one). The patch writes "
                        "into h_init.data, bypassing optimizer.step(), so the "
                        "carried-over moments would otherwise reflect gradients of "
                        "the pre-patch value. Improved Hit@1 by about 3%% at "
                        "world_size=2 and 10%% at world_size=4 (relative). Adam "
                        "only; no effect with --optimizer sgd/rmsprop. Default off.")
    p.add_argument("--consistency_lambda", type=float, default=0.0,
                   help="Weight of an h_init cross-rank consistency regulariser: "
                        "adds lambda * mean(||h_init[halo] - remote_target||^2) to "
                        "the training loss every epoch, where remote_target is the "
                        "halo snapshot used by the per-iteration refresh. Unlike "
                        "--halo_refresh_interval it leaves the overwrite itself "
                        "unchanged and adds a continuous pull toward the last-known "
                        "remote value. Best combined with --persist_optimizer "
                        "--reset_halo_momentum. 0.0 (default) = disabled.")
    p.add_argument("--shared_h_init", action="store_true", default=False,
                   help="Alternative to halo patching: every rank holds the full "
                        "(N_global, output_dim) entity table and all_reduce(AVG)s "
                        "it every epoch, like RREA's rel_emb/attn_kernels, instead "
                        "of a per-rank local+halo table. Makes "
                        "--reset_halo_momentum/--consistency_lambda/"
                        "--halo_refresh_interval/--cache_dir moot. A dense "
                        "all-reduce per epoch is only practical at the 15K-100K "
                        "scale. It narrowed the same-rank/cross-rank accuracy gap "
                        "but gave lower aggregate Hit@1 than --reset_halo_momentum, "
                        "because averaging dilutes rows only one rank touches (see "
                        "--shared_h_init_weighted_avg). Default off.")
    p.add_argument("--shared_h_init_weighted_avg", action="store_true", default=False,
                   help="Only affects --shared_h_init. Plain all_reduce(AVG) divides "
                        "every h_init row by world_size, so a row only one rank "
                        "touched is averaged with the other ranks' stale copies, "
                        "diluting its effective learning rate. This flag sums each "
                        "row over the ranks whose local subgraph included it this "
                        "epoch and divides by that count, at the cost of one extra "
                        "small all_reduce (a per-row touch-count vector). Default off.")

    return p.parse_args()


def main():
    args = parse_args()

    # --halo_sample_p is a convenience that fills whichever
    # per-side fraction wasn't given explicitly; unset means no sampling.
    _p_both = args.halo_sample_p
    args.halo_sample_p_kg1 = (
        args.halo_sample_p_kg1 if args.halo_sample_p_kg1 is not None
        else (_p_both if _p_both is not None else 1.0)
    )
    args.halo_sample_p_kg2 = (
        args.halo_sample_p_kg2 if args.halo_sample_p_kg2 is not None
        else (_p_both if _p_both is not None else 1.0)
    )
    for _nm in ("halo_sample_p_kg1", "halo_sample_p_kg2"):
        _v = getattr(args, _nm)
        if not (0.0 < _v <= 1.0):
            raise SystemExit(f"--{_nm} must be in (0, 1], got {_v}")
    if args.gas_clusters and getattr(args, "minibatch", False):
        raise SystemExit("--gas_clusters and --minibatch are mutually "
                         "exclusive memory-saving paths.")
    if args.gas_clusters < 0:
        raise SystemExit(f"--gas_clusters must be >= 0, got {args.gas_clusters}")

    # partition.py writes meta.json recording which fold_dir produced the
    # partition files; a partition built from a different fold would silently
    # mismatch the train/test links. Every rank re-checks it here (before
    # dist.init_process_group, so a mismatch fails all ranks identically
    # instead of hanging on a partial collective). Partitions without a
    # meta.json only trigger a warning, since their provenance is unknown.
    meta_path = os.path.join(args.partition_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            partition_meta = json.load(f)
        expected_fold = os.path.normpath(args.fold_dir)
        actual_fold = partition_meta.get("fold_dir")
        if actual_fold != expected_fold:
            raise SystemExit(
                f"--partition_dir {args.partition_dir!r} was built from "
                f"fold_dir {actual_fold!r}, but this run passed --fold_dir "
                f"{expected_fold!r}. Re-run src/partition.py with the "
                f"correct --fold_dir (or point --partition_dir at the "
                f"right output) before training — otherwise KG2 "
                f"assignments and train/test links won't correspond to "
                f"the actual partition, silently."
            )
    else:
        logging.warning(
            f"{meta_path} not found — cannot verify --partition_dir "
            f"{args.partition_dir!r} matches --fold_dir {args.fold_dir!r} "
            f"(partition predates this check). Re-partition if unsure."
        )

    # Run timestamp shared by all ranks (ranks start within the same second in
    # practice). The timestamp has only 1-second resolution, so two jobs that
    # start in the same second would share (and overwrite) one results
    # directory; it is therefore suffixed with SLURM_JOB_ID, which is identical
    # on every rank of a job and needs no cross-rank communication. Without
    # SLURM the bare timestamp is used.
    _job_id = os.environ.get("SLURM_JOB_ID", "")
    args.run_timestamp = time.strftime("%Y%m%d_%H%M%S") + (f"_{_job_id}" if _job_id else "")

    if args.sim and args.backend != "gloo":
        logging.warning(
            f"--sim forces gloo (was --backend {args.backend}): NCCL requires "
            f"one exclusive GPU per rank and cannot run the multi-rank-per-GPU "
            f"simulation this flag is for."
        )
        args.backend = "gloo"
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    # --seed: seed+rank so ranks draw different negatives; ID-seeded
    # tables are untouched (see src/seeding.py). Before any RNG consumer.
    args.seed_effective = seed_everything(args.seed, rank)

    level = logging.INFO if rank == 0 else logging.WARNING
    log_format = "%(asctime)s [rank %(process)d] %(levelname)s %(message)s"
    # force=True: any logging.warning() emitted before this point (e.g.
    # "--sim forces gloo" above, or minibatch_csr.import_dgl's stub notice)
    # makes Python install an implicit WARNING-level root handler, after
    # which a plain basicConfig() is a silent no-op and every INFO line
    # (epoch losses, eval metrics) would be lost from stderr and train.log.
    logging.basicConfig(level=level, format=log_format, force=True)

    # Rank 0 mirrors all log output to a per-run file inside results_dir
    if rank == 0 and args.results_dir:
        dataset_name = os.path.basename(os.path.normpath(args.data_dir))
        run_dir = os.path.join(args.results_dir, dataset_name, args.run_timestamp)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "train.log")
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        logging.info(f"Run timestamp: {args.run_timestamp}")
        logging.info(f"Log file: {log_path}")

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        # When simulating multiple ranks on fewer GPUs (e.g. 1 GPU, 2 ranks),
        # wrap local_rank so every process still gets a valid CUDA device.
        # Use gloo backend in this case — NCCL requires one device per rank.
        cuda_idx = local_rank % n_gpus
        device = torch.device(f"cuda:{cuda_idx}")
        torch.cuda.set_device(device)
        # n_gpus is per-node (torch.cuda.device_count() only sees this
        # node's devices), so it is compared against the per-node rank count,
        # not the global world_size (e.g. 2 nodes x 4 GPUs = world_size 8,
        # but 4 local GPUs for 4 local ranks is a 1:1 mapping). torchrun sets
        # LOCAL_WORLD_SIZE; fall back to world_size if launched some other
        # way (correct for single-node runs).
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
        if rank == 0 and n_gpus < local_world_size:
            logging.warning(
                f"GPUs ({n_gpus}) < local ranks ({local_world_size}): "
                f"sharing GPU(s) across ranks on this node. Use --backend gloo."
            )
    else:
        device = torch.device("cpu")
        if rank == 0:
            logging.warning("CUDA not available — running on CPU")

    if rank == 0:
        logging.info(f"Loading OpenEA data from {args.data_dir} / {args.fold_dir}")
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs",
        "partitions",
        os.path.basename(os.path.normpath(args.data_dir)),
    )

    data = OpenEAData(args.data_dir, args.fold_dir, output_dir)
    data.load_weights()   # loads r2f/r2if edge weights if KGs_edge_info.json exists

    if rank == 0:
        weight_status = "loaded" if data.edge_weights1 else "not found, using uniform 1.0"
        logging.info(
            f"KG1 entities: {data.ent_num_1}  "
            f"KG2 entities: {data.ent_num_2}  "
            f"Relations: {data.rel_num}  "
            f"Edge weights: {weight_status}"
        )

    framework = DistEAFramework(
        rank=rank,
        world_size=world_size,
        data=data,
        partition_dir=args.partition_dir,
        cfg=args,
        device=device,
    )
    framework.run()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
