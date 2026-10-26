#!/usr/bin/env bash
# The final DistEA training protocol, as used for every result in the thesis.
#
# Source this file, then call
#     distea_protocol <dataset> <N>
# to fill the bash arrays DISTEA_ARGS (arguments for main.py) and set
# DATA_DIR / FOLD_DIR / PARTITION_DIR for that dataset and node count N.
#
# Datasets: 15k | 100k | dbp1m | 2m (see the top-level README for how to
# prepare them). Paths are relative to the repository root and can be
# overridden through the environment (DATA_ROOT, PARTITION_ROOT).

DATA_ROOT="${DATA_ROOT:-data}"
PARTITION_ROOT="${PARTITION_ROOT:-outputs/partitions}"

# Exact, result-preserving speed switches (environment variables read by src/).
export DISTEA_EVAL_DEVICE="${DISTEA_EVAL_DEVICE:-cuda}"   # chunked evaluation on the GPU
export DISTEA_EVAL_SHARD="${DISTEA_EVAL_SHARD:-1}"        # split evaluation / pseudo-seed mining across ranks
export DISTEA_VECTOR_GRAPH="${DISTEA_VECTOR_GRAPH:-1}"    # vectorised local-graph construction
export DISTEA_DATA_CACHE="${DISTEA_DATA_CACHE:-1}"        # cache parsed triples as .npz
export DISTEA_FEAT_PROCS="${DISTEA_FEAT_PROCS:-4}"        # parallel deterministic feature init
export DISTEA_GAS_FAST_BUILD="${DISTEA_GAS_FAST_BUILD:-1}"            # GPU cluster build + cached METIS assignment
export DISTEA_GAS_SEGMENT_SOFTMAX="${DISTEA_GAS_SEGMENT_SOFTMAX:-1}"  # segment softmax over src-sorted edges
export DISTEA_GAS_COMPILE_HEAD="${DISTEA_GAS_COMPILE_HEAD:-1}"        # torch.compile the attention head
export DISTEA_GAS_CHECKPOINT="${DISTEA_GAS_CHECKPOINT:-0}"            # gradient checkpointing (memory only)
# TF32 for training-step matmuls only (evaluation stays fp32); verified not to
# change accuracy. Set to 0 for exact fp32 arithmetic.
export DISTEA_GAS_TF32="${DISTEA_GAS_TF32:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

distea_protocol() {
    local dataset="$1" n="$2"
    local name fold k wd extra=()
    case "$dataset" in
        15k)   name=EN_DE_15K_V1;  DATA_DIR="$DATA_ROOT/OpenEA_dataset_v2.0/EN_DE_15K_V1"
               fold=custom_30pct_fold/1; wd=1.0; extra=(--init_scale 0.3)
               case "$n" in 1) k=8;; 2) k=4;; 4) k=2;; 8) k=1;; *) k=1;; esac ;;
        100k)  name=EN_DE_100K_V1; DATA_DIR="$DATA_ROOT/OpenEA_dataset_v2.0/EN_DE_100K_V1"
               fold=721_5fold/1;        wd=1.0; extra=(--init_scale 0.3); k=8 ;;
        dbp1m) name=DBP1M_DE;      DATA_DIR="$DATA_ROOT/DBP1M_DE"
               fold=316_1fold/1;        wd=0.3; extra=(--rrea_input_relu)
               case "$n" in 1) k=32;; 2) k=16;; *) k=8;; esac ;;
        2m)    name=FB_DBP2M;      DATA_DIR="$DATA_ROOT/FB_DBP2M"
               fold=316_1fold/1;        wd=0.3; extra=(--rrea_input_relu)
               case "$n" in 1) k=80;; 2) k=48;; 4) k=28;; *) k=16;; esac ;;
        *) echo "unknown dataset '$dataset' (expected 15k | 100k | dbp1m | 2m)" >&2; return 1 ;;
    esac
    FOLD_DIR="$DATA_DIR/$fold"
    local fold_id="${fold//\//_}"
    # One node: the locality-based KG2 assignment. Several nodes: the refined
    # assignment written by scripts/make_refined_kg2_partition.py ("...ref").
    if [ "$n" -eq 1 ]; then
        PARTITION_DIR="$PARTITION_ROOT/$name/$fold_id/dist_True/1_metis_kg2locality"
    else
        PARTITION_DIR="$PARTITION_ROOT/$name/$fold_id/dist_True/${n}_metis_kg2localityref"
    fi

    DISTEA_ARGS=(
        --data_dir "$DATA_DIR" --fold_dir "$FOLD_DIR" --partition_dir "$PARTITION_DIR"
        # encoder and optimiser
        --output_dim 128 --num_layers 2 --attn_heads 1 --attn_softmax src
        --optimizer adam --lr 5e-4 --weight_decay "$wd" --dropout 0.2 --fused_adam
        --neg_k 1 --margin 6.0
        "${extra[@]}"
        # schedule: 150 epochs, mine pseudo-seeds, 50 more epochs
        --max_epochs 150 --max_iterations 2 --min_iterations 2 --later_iter_epochs 50
        --use_pseudo_seeds --pseudo_seed_policy testside
        # model selection on the validation split; test is evaluated once
        --eval_on_valid --eval_every_epochs 50 --eval_every_splits valid
        --keep_best_epoch --epoch_patience 1 --select_by valid --patience 3
        --defer_test_eval
        --eval_metric cosine --use_csls --csls_k 10
        # cluster-wise (GAS) training
        --gas_clusters "$k" --gas_cluster_method metis_csr --gas_hub_degree 1000
        --gas_softmax_history --gas_csr_spmm --gas_csr_gather
        --rrea_edge_chunk 2000000
        --reset_halo_momentum --seed 1 --no_save_model
    )
    # Cluster matrices and history tables on the GPU, except for 2M on a
    # single node, which keeps them in host memory to fit into 80 GB.
    if ! { [ "$dataset" = 2m ] && [ "$n" -eq 1 ]; }; then
        DISTEA_ARGS+=(--gas_clusters_on_device --gas_history_on_device)
    fi
    # Several nodes: periodic table consensus of replicated entity rows.
    if [ "$n" -gt 1 ]; then
        DISTEA_ARGS+=(--table_consensus_every 5 --table_consensus_mode delta_dc)
    fi
}
