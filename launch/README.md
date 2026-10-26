# launch — the final training protocol

| File | Purpose |
|---|---|
| `protocol.sh` | The final protocol: every `main.py` option and environment switch, per dataset and node count. Sourced by the other scripts. |
| `partition.sh` | Builds the partition a run needs (once per dataset and node count). |
| `run_local.sh` | Runs *N* ranks on one machine with `torchrun`, one GPU each. |
| `run_slurm.sbatch` | Runs on *N* Slurm nodes with one GPU each. |

The usual entry points are the make targets, which call these scripts:

```bash
make partition DATASET=100k N=4      # bash launch/partition.sh 100k 4
make train DATASET=100k N=4          # bash launch/run_local.sh 100k 4  (single-node multi-GPU)
make train-slurm DATASET=100k N=4    # sbatch --nodes=4 launch/run_slurm.sbatch 100k  (multi-node)
```

Datasets: `15k`, `100k`, `dbp1m`, `2m`. Paths default to `data/` and `outputs/partitions/` and can
be moved with `DATA_ROOT` and `PARTITION_ROOT`. Additional `main.py` options can be passed with
`EXTRA_ARGS`, e.g. `EXTRA_ARGS="--max_epochs 10 --later_iter_epochs 10"` for a short smoke run
(this is what `make smoke` does).

## What the protocol sets

Common to all datasets:

- RREA encoder, 2 layers, dimension 128, one attention head, softmax grouped by source entity.
- Adam, learning rate 5e-4; the ELsEA-style combined loss of InfoNCE and a margin loss with one
  uniform negative per positive (margin 6), with learnable mixing weights.
- 150 epochs; then mutual-nearest-neighbour pseudo-seeds mined from the test-side candidates
  with CSLS, and 50 more epochs.
- Validation every 50 epochs; the best epoch and iteration on the validation split are kept, and
  the test split is evaluated once, on the selected state (cosine similarity with CSLS, *k* = 10).
- Cluster-wise training with METIS clusters (hubs of degree > 1000 excluded from the METIS call),
  softmax-denominator history and sparse incidence kernels.
- On several nodes: table consensus of replicated rows every 5 epochs (`delta_dc`), and the
  refined KG2 assignment.

Per dataset:

| Dataset | Weight decay | Other | Clusters at N = 1 / 2 / 4 / 8 |
|---|---|---|---|
| `15k` | 1.0 | `--init_scale 0.3` | 8 / 4 / 2 / 1 |
| `100k` | 1.0 | `--init_scale 0.3` | 8 / 8 / 8 / 8 |
| `dbp1m` | 0.3 | `--rrea_input_relu` | 32 / 16 / 8 / 8 |
| `2m` | 0.3 | `--rrea_input_relu` | 80 / 48 / 28 / 16 |

On `2m` with one node, the cluster matrices and history tables stay in host memory (the run
needs about 96 GB of host memory); in every other configuration they are kept on the GPU.

## Adapting `run_slurm.sbatch`

The script assumes one task and one GPU per node and a repository on a shared file system.
Before the first run, edit the `#SBATCH` lines (partition, time, memory) and the environment
block: load your CUDA module, activate the Python environment, and, if needed, set
`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` to the interface that connects the nodes. The first
node acts as the rendezvous host.
