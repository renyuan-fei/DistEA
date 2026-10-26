# DistEA

DistEA is a framework for **distributed entity alignment** (EA) on large knowledge graphs.
It partitions both knowledge graphs across *N* compute nodes, each holding about 1/*N* of the
entities, and trains a single [RREA](https://arxiv.org/abs/2008.07962) encoder over all
partitions in a data-parallel fashion with PyTorch and `torch.distributed`.

This repository contains the code of the master's thesis on DistEA: the training framework, the
scripts that prepare the datasets and partitions, the launch scripts of the final training
protocol, the unified evaluator used for all comparison methods, and the test suite.

## How it works

Training on *N* nodes would normally cut every edge that crosses a partition boundary and keep the
alignment signal of a training pair away from the node that evaluates it. DistEA keeps the
accuracy cost of partitioning small with four mechanisms:

| Mechanism | What it does | Code |
|---|---|---|
| Two-sided halo nodes | Each node also loads the 1-hop neighbours of its entities in **both** KGs, so cross-partition edges stay in the local subgraph. | `src/dist_graph.py` |
| Locality-aware KG2 assignment | KG2 entities are placed by a majority vote over the KG2 graph, seeded from the training pairs, and then refined by label propagation, so KG2 neighbourhoods are not scattered across nodes. | `src/counterpart_disc.py`, `src/kg2_refine.py` |
| Table consensus | Every few epochs the rows of entities replicated on several nodes are merged (sum of the per-node deltas), so cross-node alignment signal reaches the row used for evaluation. | `src/table_consensus.py` |
| Cluster-wise training | Each local subgraph is split into METIS clusters that are trained one at a time, with historical embeddings for out-of-cluster neighbours (GNNAutoScale-style). The cluster count is scaled with the local subgraph size. This also lets million-entity graphs train on one GPU. | `src/gas.py` |

An ownership invariant makes the evaluation independent of the number of partitions: every test
entity is ranked with exactly one representation, the one held by the node that owns it
(`src/eval.py`).

## Results

Final protocol, one NVIDIA A100 80 GB GPU per node. Each cell gives Hits@1 on the test split and
the job wall-clock time in minutes (start-up and evaluation included), as reported in the thesis.

| Dataset | Entities (KG1 / KG2) | 1 node | 2 nodes | 4 nodes | 8 nodes |
|---|---|---|---|---|---|
| EN-DE-15K (OpenEA) | 15 K / 15 K | 0.7542 / 1.4 | 0.7638 / 0.6 | 0.7634 / 1.1 | 0.7614 / 2.2 |
| EN-DE-100K (OpenEA) | 100 K / 100 K | 0.5731 / 1.5 | 0.5600 / 1.1 | 0.5525 / 1.0 | 0.5523 / 1.7 |
| DBP1M EN-DE | 1.12 M / 0.56 M | 0.2546 / 10.0 | 0.2494 / 6.3 | 0.2411 / 4.4 | 0.2386 / 3.0 |
| FB-DBP2M | 2.33 M / 2.34 M | 0.2560 / 153 | 0.2582 / 39.4 | 0.2540 / 23.5 | 0.2490 / 16.6 |

Multi-node training costs at most 2.1 percentage points of Hits@1 relative to one node. On
FB-DBP2M, the single-node run keeps its cluster matrices in host memory; from two nodes on, all
state fits on the GPUs.

## Repository layout

```
main.py         entry point, one process per GPU (started by torchrun)
src/            the framework: data loading, partitioned graphs, RREA encoder, cluster-wise
                training, table consensus, distributed evaluation
launch/         the final training protocol and its launchers (one machine / Slurm)
scripts/        dataset preparation, partition refinement, unified evaluation, and the
                launchers and exporters of the comparison methods
tests/          CPU test suite (no GPU needed)
cache_server/   optional Go gRPC server for the gRPC halo-refresh backend
```

Each folder has its own README.

## Installation

Requirements: Linux, Python 3.12 and a CUDA GPU for training (the tests run on CPU). Every step is
a `make` target; `make help` lists them all.

With [uv](https://docs.astral.sh/uv/):

```bash
make install    # .venv with PyTorch, NumPy, SciPy, NetworkX, pymetis, ... (from uv.lock)
make nxmetis    # build networkx-metis from source (used by partitioning and clustering)
make test       # CPU test suite, about 1.5 minutes
```

With conda, using the exact package versions of the thesis runs (`requirements.txt`, PyTorch
2.8.0 for CUDA 12.8; set `TORCH_INDEX` for another CUDA version):

```bash
make conda-env                  # creates the conda env "distea" and builds nxmetis
conda activate distea
make test
```

Without `.venv`, the make targets use the `python` of the active environment. If the PyTorch
build does not match your CUDA driver, reinstall it from the matching index, e.g.
`pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126`. Two components are
optional: DGL (`make dgl`; used only by the VBIP pair weighting and the DGL mini-batch sampler) and
the Go cache server (`make grpc-server`; used only with `--cache_backend grpc`).

## Data

All datasets are converted to the OpenEA layout: `rel_triples_1`, `rel_triples_2`, `ent_links`,
and a split directory with `train_links`, `valid_links` and `test_links`. Only relation triples
are used; entity names, attribute triples and descriptions are never loaded.

One command downloads everything into `data/`, converts DBP1M and FB-DBP2M, and creates the splits
(about 0.7 GB of downloads; DBP1M is on Google Drive and needs `gdown`, e.g. `pip install gdown`):

```bash
make data            # or one dataset: make data-openea | make data-dbp1m | make data-2m
```

| Dataset | Download | Result in `data/` | Split (train / valid / test) |
|---|---|---|---|
| EN-DE-15K, EN-DE-100K | [OpenEA v2.0](https://github.com/nju-websoft/OpenEA) ([zip on figshare](https://ndownloader.figshare.com/files/34234391)) | `OpenEA_dataset_v2.0/EN_DE_15K_V1`, `.../EN_DE_100K_V1` | 15K: `custom_30pct_fold/1`, 30/10/60; 100K: `721_5fold/1`, 20/10/70 (OpenEA's own) |
| DBP1M EN-DE | [LargeEA](https://github.com/ZJU-DAILY/LargeEA) release ([Google Drive](https://drive.google.com/file/d/15jeGD-6pVGlqI5jCn7KJfGIER6AeoQ-L)) | `DBP1M_DE` (from `mkdata/` via `scripts/prepare_dbp1m.py`) | `316_1fold/1`, 30/10/60 |
| FB-DBP2M | [DivEA](https://github.com/uqbingliu/DivEA) data ([Dropbox](https://www.dropbox.com/scl/fo/j95qpgb4cpxjcu1am3koy/ADnyA1olK-S5oGrbfV1Cb18?rlkey=cs89tuhnwl2hmvcy2sk5r86nm&dl=1)), originally released by LIME | `FB_DBP2M` (from `datasets/2m/fb_dbp/` via `scripts/prepare_fb_dbp2m.py`) | `316_1fold/1`, 30/10/60 |

The DBP1M and FB-DBP2M splits are made by `scripts/split_openea.py --train 0.3 --valid 0.1
--folds 1` (seed 42) and are byte-identical to the ones used in the thesis. The EN-DE-15K split is
made by `scripts/make_30pct_fold.py` with the same proportions; the exact 15K split of the thesis
was drawn by a different shuffle and is not distributed here.

## Training

The final protocol is defined in one place, [`launch/protocol.sh`](launch/protocol.sh): RREA with
two layers and dimension 128; 150 epochs, then test-side pseudo-seeds and 50 more epochs; model
selection on the validation split; CSLS evaluation; cluster-wise training; and, on several nodes,
table consensus every 5 epochs. The datasets are named `15k`, `100k`, `dbp1m` and `2m`.

```bash
make smoke DATASET=15k                 # 10 + 10 epochs on one GPU: checks the installation
make train DATASET=dbp1m N=1           # one machine, N GPUs (ranks started by torchrun)
make train-slurm DATASET=dbp1m N=8     # N Slurm nodes with one GPU each
make results                           # summary of all runs
```

`make train` and `make train-slurm` first build the partition the run needs (`make partition`:
METIS for KG1, locality-based KG2 assignment, refined for N > 1, stored under
`outputs/partitions/`); this is done once per dataset and node count. Before the first Slurm run,
adapt the `#SBATCH` lines and the environment block of
[`launch/run_slurm.sbatch`](launch/run_slurm.sbatch). Additional `main.py` options can be given
with `EXTRA_ARGS="..."`.

Each run writes `outputs/results/<dataset>/<timestamp>/metrics.json` (per-iteration and final
Hits@1/5/10/50 and MRR, timings, communication statistics and the full argument list) and
`alignment.csv`.

`main.py` can also be called directly. `python main.py --help` lists every option, including the
mechanisms that are off in the final protocol (mini-batch training, halo refresh through files or
gRPC, owner-compute dispatch, VBIP weighting, hard negatives, and ablation switches).

## Comparison methods

The thesis compares DistEA with ELsEA, ClusterEA, LightEA, LargeGNN, LargeEA and the original
TensorFlow RREA, all on the same splits and scored by one evaluator,
[`scripts/unified_eval.py`](scripts/unified_eval.py). The scripts that convert our splits into each
method's input format, launch the published implementations and export their embeddings to the
unified evaluator are listed in [`scripts/README.md`](scripts/README.md). The comparison methods
themselves are not included; obtain them from their authors' repositories.

## Tests

```bash
make test                                # every CPU test
python tests/test_table_consensus.py     # one file
make lint                                # code style (ruff)
```

The tests need neither a GPU nor DGL. The files `tests/gpu_*.py` are optional GPU/NCCL checks run
by hand on a GPU node. See [`tests/README.md`](tests/README.md).

## Acknowledgements

DistEA builds on the RREA encoder (Mao et al., 2020), on ELsEA/DivEA for partitioned entity
alignment, on LeapGNN for distributed GNN training and the gRPC cache server (`cache_server/`,
`src/leapgnn/`), and on GNNAutoScale for historical embeddings.

## License

MIT, see [LICENSE](LICENSE).
