#!/bin/bash
# Train with the final protocol on N Slurm nodes, one GPU per node (the
# multi-node deployment of the thesis).
#
#   sbatch --nodes=<N> launch/run_slurm.sbatch <dataset>
#
# Adapt the #SBATCH lines and the environment block below to your cluster.
# The repository must be on a filesystem shared by all nodes.
#SBATCH --job-name=distea
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?run with sbatch}"
dataset="${1:?usage: sbatch --nodes=<N> launch/run_slurm.sbatch <15k|100k|dbp1m|2m>}"
n="$SLURM_JOB_NUM_NODES"

# --- environment: activate the Python environment that has DistEA installed ---
# e.g. module load CUDA/12.4.1; source .venv/bin/activate
# (or set PYTHON=<path to the environment's python>; `make train-slurm` sets it to .venv/bin/python)
# Network interface used for NCCL / gloo traffic between nodes (cluster-specific).
# export NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0
# -------------------------------------------------------------------------------

source launch/protocol.sh
distea_protocol "$dataset" "$n"
[ -d "$PARTITION_DIR" ] || { echo "missing $PARTITION_DIR -- run: bash launch/partition.sh $dataset $n" >&2; exit 1; }

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}"
export DISTEA_ARGS_STR="$(printf '%q ' "${DISTEA_ARGS[@]}")"

srun --export=ALL bash -c '
    eval "args=($DISTEA_ARGS_STR)"
    "${PYTHON:-python}" -m torch.distributed.run --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node=1 --node_rank="$SLURM_NODEID" \
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        main.py "${args[@]}" --backend nccl \
        --task_id "'"${dataset}"'_N'"${n}"'" --results_dir outputs/results \
        --cache_dir "outputs/embedding_cache/$SLURM_JOB_ID" ${EXTRA_ARGS:-}
'
