#!/usr/bin/env bash
# build_grpc_server.sh — compile the DistEA gRPC cache server binary.
#
# Usage:
#   bash scripts/build_grpc_server.sh            # output: bin/distea_server
#   bash scripts/build_grpc_server.sh --race     # with race detector
#   OUTPUT=/custom/path bash scripts/build_grpc_server.sh
#
# Requires: Go 1.23.x  (GOTOOLCHAIN=local prevents auto-download of newer toolchain)

set -euo pipefail

export GOTOOLCHAIN=local

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${REPO_ROOT}/bin"
OUTPUT="${OUTPUT:-${BIN_DIR}/distea_server}"

mkdir -p "${BIN_DIR}"

cd "${REPO_ROOT}/cache_server"

echo "[build] tidying modules…"
go mod tidy

echo "[build] compiling → ${OUTPUT}"
go build "${@}" -o "${OUTPUT}" .

echo "[build] done: ${OUTPUT}"
