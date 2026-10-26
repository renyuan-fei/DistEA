# cache_server — gRPC embedding cache server (optional)

A standalone Go gRPC server that caches entity embeddings in memory and serves
feature-lookup requests from the Python training processes. It is adapted from
the LeapGNN cache server and extended with DistEA's partition loading.

It is only needed for `--cache_backend grpc`, one of the halo-refresh backends
(`src/dist_halo_sync.py`). The final training protocol does not use it, and the
default backend (`none`) and the `file` backend need no server.

---

## Overview

During distributed training each rank computes embeddings for its local partition.
The cache server stores those embeddings in memory and serves feature-lookup
requests — including cross-partition requests forwarded to peer servers:

```
Python rank 0        Python rank 1  …  Python rank N-1
     │                    │                   │
  emb_0.npy            emb_1.npy         emb_{N-1}.npy   (shared filesystem)
     │                    │                   │
     └──────────── gRPC ──┴── gRPC ───────────┘
                         ↓
              cache_server instances
              (one per rank, talk to each other)
```

Each server instance owns one partition's embeddings. Feature requests that
target another partition are forwarded to the peer server via an internal
gRPC call and assembled before returning to the caller.

> **Note:** The Python training loop writes embeddings to the shared filesystem
> directly and uses `dist.barrier()` for synchronisation. The gRPC server is an
> optional higher-throughput alternative for multi-machine deployments.

---

## Directory Structure

```
cache_server/
├── server.go              Entry point: init → main (blocking on ready file)
├── go.mod                 Go module "distea"
│
├── common/
│   ├── configmodel.go     ServerConfig struct (all runtime fields)
│   ├── cmdparser.go       CLI flag parsing (overrides YAML)
│   └── yamlparser.go      YAML config loading (configor)
│
├── services/
│   ├── distcache.go       DistCache: partition routing, Nid→Pid map, registry
│   ├── static_cache.go    StaticCache: full embeddings in memory (DistEA default)
│   ├── p3_cache.go        P3Cache: column-striped feature sharding (LeapGNN mode)
│   └── services.go        Start(): initialises the global DCRuntime
│
├── rpc/
│   ├── rpc.go             gRPC server setup and listener
│   ├── distcache_rpc.go   Service registration, DCSubmit / DCSubmitFeatures handlers
│   ├── distcache_rpc_imple.go  Per-operation implementations (opGet*, opReset, …)
│   └── cache/             Protobuf-generated stubs (do not edit)
│       ├── distcache.pb.go
│       └── distcache_grpc.pb.go
│
├── peerclient/
│   └── peerrpc.go         gRPC client pool for peer-server feature forwarding
│
└── conf/
    ├── distea.yaml        Configuration used by DistEA
    └── static_cache.yaml  Default when the binary is started without --config
```

---

## Build

```bash
# From the repository root
make grpc-server                          # -> bin/distea_server
bash scripts/build_grpc_server.sh --race  # with the race detector

# Or directly
cd cache_server && go build -o ../bin/distea_server .
```

Requires **Go 1.23**. No CGO; the result is a self-contained static binary.
`GrpcHaloSync` starts and stops the server processes itself during training, so
the commands below are only needed to run the server by hand.

---

## Configuration

All settings can be provided via a YAML file and/or CLI flags.
**CLI flags take precedence over YAML values.**

### Key Fields (`conf/distea.yaml`)

| YAML key | CLI flag | Default | Description |
|----------|----------|---------|-------------|
| `rpcport` | `--rpcport` | `18112` | gRPC listen port |
| `cache_type` | `--cachetype` | — | `static` (DistEA) or `p3` (LeapGNN) |
| `cache_group` | `--cachegroup` | — | Comma-separated `host:port` for all ranks |
| `partition_dir` | `--partition_dir` | — | Dir containing `{rank}.npy` files |
| `emb_dir` | `--emb_dir` | — | Dir containing `emb_{rank}.npy` files |
| `distearank` | `--distea_rank` | `-1` | Explicit rank index (`-1` = auto from IP) |
| `bindaddr` | `--bind_addr` | `""` | gRPC bind address (`0.0.0.0` = all interfaces) |
| `readyfile` | `--ready_file` | — | File created when server is fully ready |
| `statistic` | `--statistic` | `false` | Enable hit-ratio counters (reduces throughput) |

### DistEA mode and LeapGNN mode

| Mode | Trigger | Node IDs | Embeddings |
|------|---------|----------|------------|
| **DistEA** | `emb_dir` is set | `{partition_dir}/{rank}.npy` | `{emb_dir}/emb_{rank}.npy` |
| **LeapGNN** | `emb_dir` empty | `{dataset}/dist_True/{N}_{type}/{rank}.npy` | `{dataset}/feat.npy` |

---

## Running

### Single host, 4 ranks

```bash
for rank in 0 1 2 3; do
  port=$((18112 + rank))
  bin/distea_server \
    --config      cache_server/conf/distea.yaml \
    --partition_dir outputs/partitions/EN_DE_15K_V1/721_5fold_1/dist_True/4_metis_kg2locality \
    --emb_dir     outputs/embedding_cache \
    --cachegroup  "127.0.0.1:18112,127.0.0.1:18113,127.0.0.1:18114,127.0.0.1:18115" \
    --rpcport     $port \
    --distea_rank $rank \
    --bind_addr   0.0.0.0 \
    --ready_file  outputs/embedding_cache/ready_${rank}.txt \
    > outputs/server_logs/server_${rank}.log 2>&1 &
done
```

### Several hosts

Use the hosts' IP addresses instead of `127.0.0.1`, omit `--distea_rank` (auto-detected from IP),
and omit `--bind_addr` (auto-detected):

```bash
bin/distea_server \
  --config      cache_server/conf/distea.yaml \
  --partition_dir /data/partitions/4_metis \
  --emb_dir     /data/embedding_cache \
  --cachegroup  "10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4" \
  --rpcport     18112
```

---

## Startup Sequence

```
server.go init()
  ├── common.Parser()      parse CLI + YAML → global Config
  ├── services.Start()     load Nid→Pid map + embeddings into StaticCache
  ├── rpc.Start()          start gRPC listener (background goroutine)
  └── peerclient.Build()   connect to all peer servers

server.go main()
  ├── delete stale ready file (if exists)
  ├── create ready file  ← callers poll this before issuing gRPC calls
  └── block on empty channel (server runs indefinitely)
```

The ready file is created **after** all peer connections are established,
giving a simple full-readiness signal without requiring a coordinator.

---

## gRPC Operations

Defined in `rpc/cache/distcache.proto` (generated stubs in `rpc/cache/`).

| `OpType` | Handler | Description |
|----------|---------|-------------|
| `get_features_by_client` | `opGetFeaturesByClient` | Client fetch (may forward to peers) |
| `get_features_by_peer_server` | `opPeerGet` | Peer-server fetch (local shard only) |
| `get_stream_features_by_client` | `opGetStreamFeatures` | Streaming fetch via shared memory |
| `get_feature_dim` | `opGetFeatDim` | Returns feature dimension |
| `get_cache_info` | `opGetCacheInfo` | Returns hit-ratio statistics |
| `get_statistic` | `opGetStatistic` | Returns whether statistic mode is on |
| `reset` | `opReset` | Resets hit-ratio counters |

---

## Design Notes

**Single-host rank identity** — When several ranks run on one host they share the same IP,
so IP-based rank detection fails. Set `--distea_rank` explicitly (or rely on
`GrpcHaloSync` which sets it automatically) to bypass IP matching.

**Port-per-rank** — Each rank listens on `BASE_PORT + rank`. The full
`--cachegroup` list (all `host:port` pairs) is passed to every instance so
each server knows how to reach its peers.

**Peer forwarding** — `StaticCache.Get()` partitions the requested node IDs
by `Nid2Pid`, serves local IDs from memory, and forwards remote IDs to the
owning peer server via `peerclient.GrpcClients`. Results are assembled before
returning to the Python caller.

**Ready-file protocol** — The server creates `ready_{rank}.txt` after all peer
connections are established. Any client that needs to issue gRPC calls should
poll for this file before proceeding, giving a simple readiness barrier.
