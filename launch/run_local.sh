#!/usr/bin/env bash
# Train with the final protocol on one machine.
#
#   bash launch/run_local.sh <dataset> [N]
#
# N ranks are started with torchrun on this machine, one GPU each (N defaults
# to 1). This is the single-node multi-GPU deployment of the thesis; for N
# separate nodes use launch/run_slurm.sbatch. Extra main.py arguments can be
# appended through EXTRA_ARGS, e.g. EXTRA_ARGS="--max_epochs 10 --later_iter_epochs 10".
set -euo pipefail
cd "$(dirname "$0")/.."
source launch/protocol.sh
dataset="${1:?usage: run_local.sh <15k|100k|dbp1m|2m> [N]}"; n="${2:-1}"
distea_protocol "$dataset" "$n"
[ -d "$PARTITION_DIR" ] || { echo "missing $PARTITION_DIR -- run: bash launch/partition.sh $dataset $n" >&2; exit 1; }
# shellcheck disable=SC2086
"${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node="$n" main.py "${DISTEA_ARGS[@]}" \
    --backend "${BACKEND:-nccl}" --task_id "${dataset}_N${n}" \
    --results_dir outputs/results --cache_dir outputs/embedding_cache ${EXTRA_ARGS:-}
