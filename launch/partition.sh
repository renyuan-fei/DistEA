#!/usr/bin/env bash
# Build the offline partition for <dataset> and node count <N> (run once).
#
#   bash launch/partition.sh dbp1m 8
#
# N = 1 writes 1_metis_kg2locality; N > 1 writes N_metis_kg2locality and then
# its refined KG2 assignment N_metis_kg2localityref, which training uses.
set -euo pipefail
cd "$(dirname "$0")/.."
source launch/protocol.sh
dataset="${1:?usage: partition.sh <15k|100k|dbp1m|2m> <N>}"; n="${2:?missing N}"
distea_protocol "$dataset" "$n"
PYTHON="${PYTHON:-python}"
base="${PARTITION_DIR%ref}"
if [ ! -d "$base" ]; then
    "$PYTHON" src/partition.py --data_dir "$DATA_DIR" --fold_dir "$FOLD_DIR" \
        --partition "$n" --kg2_assignment locality --output_base "$PARTITION_ROOT"
fi
if [ "$n" -gt 1 ] && [ ! -d "$PARTITION_DIR" ]; then
    "$PYTHON" scripts/make_refined_kg2_partition.py --data_dir "$DATA_DIR" \
        --fold_dir "$FOLD_DIR" --partition_dir "$base" --output_base "$PARTITION_ROOT"
fi
echo "partition ready: $PARTITION_DIR"
