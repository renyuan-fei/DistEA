.DEFAULT_GOAL := help
SHELL := /bin/bash

# Python of the project environment: .venv (from `make install`) if it exists, otherwise the
# python of the active environment (e.g. a conda env created by `make conda-env`).
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
export PYTHON

# Run parameters: DATASET = 15k | 100k | dbp1m | 2m, N = number of ranks / nodes.
DATASET ?= 15k
N       ?= 1
# Passed through to the launch scripts when set.
export EXTRA_ARGS BACKEND DATA_ROOT PARTITION_ROOT

# nxmetis (networkx-metis) has no wheel for Python 3.12; it is built from source.
NXMETIS_REPO   := https://github.com/networkx/networkx-metis
NXMETIS_DIR    := modules/networkx-metis
NXMETIS_DEPS   := "Cython>=0.29.36,<3.0" setuptools wheel
NXMETIS_CFLAGS := -Wno-implicit-function-declaration -Wno-int-conversion

.PHONY: help
help: ## Show this help
	@echo "Usage: make <target> [DATASET=15k|100k|dbp1m|2m] [N=<ranks>]"
	@awk 'BEGIN {FS = ":.*?## "} \
	  /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
	  /^[a-zA-Z0-9_-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Setup

.PHONY: install
install: ## Create .venv with uv and install the Python dependencies
	uv sync

# Installer used by `make nxmetis`: "uv pip" for the uv environment, "pip" inside conda.
PIP ?= uv pip

.PHONY: nxmetis
nxmetis: ## Build and install nxmetis from source (PIP=pip inside a conda env)
	@[ -f $(NXMETIS_DIR)/setup.py ] || git clone --depth 1 $(NXMETIS_REPO) $(NXMETIS_DIR)
	$(PIP) install $(NXMETIS_DEPS)
	CFLAGS="$(NXMETIS_CFLAGS)" $(PIP) install --no-build-isolation -e $(NXMETIS_DIR)

.PHONY: setup
setup: install nxmetis ## install + nxmetis

# conda alternative: the exact versions of requirements.txt in a conda environment.
CONDA_ENV ?= distea
TORCH_INDEX ?= https://download.pytorch.org/whl/cu128

.PHONY: conda-env
conda-env: ## Create conda env CONDA_ENV with requirements.txt and nxmetis (TORCH_INDEX for CUDA)
	conda create -n $(CONDA_ENV) python=3.12 -y
	conda run -n $(CONDA_ENV) pip install torch==2.8.0 --index-url $(TORCH_INDEX)
	conda run -n $(CONDA_ENV) pip install -r requirements.txt
	@[ -f $(NXMETIS_DIR)/setup.py ] || git clone --depth 1 $(NXMETIS_REPO) $(NXMETIS_DIR)
	conda run -n $(CONDA_ENV) pip install $(NXMETIS_DEPS)
	conda run -n $(CONDA_ENV) env CFLAGS="$(NXMETIS_CFLAGS)" pip install --no-build-isolation -e $(NXMETIS_DIR)
	@echo "Activate with: conda activate $(CONDA_ENV)"

.PHONY: dgl
dgl: ## Install the optional DGL accelerator (DGL_CUDA=cu121|cu124|repo for CPU)
	uv pip install dgl -f "https://data.dgl.ai/wheels/$(or $(DGL_CUDA),cu124)/repo.html"

.PHONY: grpc-server
grpc-server: ## Build the optional Go cache server -> bin/distea_server (Go 1.23)
	bash scripts/build_grpc_server.sh

##@ Data

.PHONY: data
data: ## Download and prepare all four datasets into data/
	bash scripts/download_data.sh all

.PHONY: data-openea
data-openea: ## Download EN-DE-15K and EN-DE-100K (OpenEA v2.0)
	bash scripts/download_data.sh openea

.PHONY: data-dbp1m
data-dbp1m: ## Download and prepare DBP1M EN-DE
	bash scripts/download_data.sh dbp1m

.PHONY: data-2m
data-2m: ## Download and prepare FB-DBP2M
	bash scripts/download_data.sh fb_dbp2m

##@ Training (final protocol, see launch/README.md)

.PHONY: partition
partition: ## Build the partition for DATASET and N (skipped if it exists)
	bash launch/partition.sh $(DATASET) $(N)

.PHONY: train
train: partition ## Train DATASET on this machine with N GPUs (EXTRA_ARGS adds main.py options)
	bash launch/run_local.sh $(DATASET) $(N)

.PHONY: train-slurm
train-slurm: partition ## Submit DATASET to N Slurm nodes with one GPU each
	sbatch --nodes=$(N) launch/run_slurm.sbatch $(DATASET)

.PHONY: smoke
smoke: ## Short end-to-end run of DATASET (10 + 10 epochs) to check an installation
	$(MAKE) --no-print-directory train EXTRA_ARGS="--max_epochs 10 --later_iter_epochs 10"

.PHONY: results
results: ## Print a summary of the runs in outputs/results
	$(PYTHON) scripts/show_results.py outputs/results

##@ Development

.PHONY: test
test: ## Run the CPU test suite (no GPU or DGL needed)
	@fail=0; for f in tests/test_*.py; do \
	  $(PYTHON) "$$f" > /dev/null 2>&1 && echo "ok    $$f" || { echo "FAIL  $$f"; fail=1; }; \
	done; exit $$fail

.PHONY: lint
lint: ## Check code style with ruff (pip install ruff in a conda env)
	$(PYTHON) -m ruff check .

.PHONY: format
format: ## Apply ruff's automatic fixes
	$(PYTHON) -m ruff check --fix .

.PHONY: clean
clean: ## Remove caches (keeps data/ and outputs/)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .ruff_cache .pytest_cache
