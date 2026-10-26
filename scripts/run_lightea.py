#!/usr/bin/env python
"""
run_lightea.py -- run LightEA's released pipeline (run_lightea_default.py,
converted verbatim from the official LightEA.ipynb) on a dataset prepared by
scripts/prepare_for_lightea.py, without editing the LightEA checkout, and
export its final features for unified scoring.

Reads:  --dataset_dir (prepare_for_lightea.py output) and the checkout's
        run_lightea_default.py / utils.py.
Writes: --out_dir (see "Outputs" below).

What changes relative to the released notebook, and nothing else:
  * dataset/path: point at --dataset_dir instead of the notebook's hard-coded
    list index.
  * load_aligned_pair: the released one shuffles ref_ent_ids and re-splits
    30/70 at random; replaced by a reader of our fixed fold (train_pairs /
    test_pairs written by prepare_for_lightea.py) so every method trains and
    tests on the same pairs.
  * CUDA_VISIBLE_DEVICES: the notebook forces "0"; dropped so the job
    scheduler's assignment stands.
  * optional overrides of the notebook's own hyper-parameters (--ent_dim,
    --epochs, --seed), applied as text substitutions of the same lines, and
    printed so every run's config is visible in its log. Defaults leave the
    notebook's values (ent_dim 1024, depth 2, top_k 500, 3 rounds, seed
    12306) untouched.
Name features stay off (using_name_features = False in the notebook) --
structure only, like every other method compared.

Outputs in --out_dir:
  lightea_own.json   LightEA's own last-round test(): Hit@1 = len(right_list)
                     / len(test_pair) (its Sinkhorn ranking, test-side pool)
  export bundle      embeddings.npy / ids.npy / kg_labels.npy / test_links.json
                     restricted to test-side entities (the only rows unified
                     scoring with --candidate_pool test_side reads -- full
                     DBP1M features would be ~55 GB at the default dims),
                     for scripts/unified_eval.py.

Run from the LightEA checkout root (its utils.py is imported by name):
    cd projects/LightEA && python <DistEA>/scripts/run_lightea.py \\
        --dataset_dir EA_datasets/EN_DE_15K_V1_custom30_1 \\
        --out_dir <DistEA>/export/lightea/EN_DE_15K_V1/custom_30pct_fold_1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np


def fixed_split_loader(dataset_dir: str):
    def load_aligned_pair(file_path, ratio=0.3):  # same signature as utils.load_aligned_pair
        def read(name):
            with open(os.path.join(dataset_dir, name)) as f:
                return np.array([line.split("\t") for line in f.read().split("\n") if line], dtype=np.int64)
        return read("train_pairs"), read("test_pairs")
    return load_aligned_pair


def patch_source(src: str, dataset_dir: str, args) -> str:
    def sub(pattern, repl, what):
        nonlocal src
        new, n = re.subn(pattern, repl, src, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"[run_lightea] could not patch {what}: pattern {pattern!r} not found once")
        src = new

    sub(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=.*$', "pass  # [run_lightea] keep Slurm's CUDA_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES")
    sub(r"^dataset\s*=\s*\[.*$", f"dataset = {os.path.basename(os.path.normpath(dataset_dir)) + '/'!r}", "dataset")
    sub(r'^path\s*=\s*"\./EA_datasets/"\s*\+\s*dataset\s*$',
        f"path = {os.path.normpath(dataset_dir) + '/'!r}", "path")
    if args.seed is not None:
        sub(r"^np\.random\.seed\(\d+\)", f"np.random.seed({args.seed})", "seed")
    if args.ent_dim is not None:
        sub(r"^ent_dim,\s*depth,\s*top_k\s*=\s*\d+", f"ent_dim, depth, top_k = {args.ent_dim}", "ent_dim")
    if args.epochs is not None:
        sub(r"^epochs\s*=\s*\d+", f"epochs = {args.epochs}", "epochs")
    # The notebook picks rel_dim by `if "EN" in dataset:` -- ent_dim//2 for
    # its cross-lingual sets (DBP_ZH_EN, ..., SRPRS_DE_EN), ent_dim//3 for the
    # monolingual DBP_WD / DBP_YG. Our directory names don't follow that
    # convention (DBP1M_DE_316_1 has no "EN" though it is DE-EN), so the
    # branch is decided by --cross_lingual instead of the name.
    sub(r'^if "EN" in dataset:', f"if {args.cross_lingual == 'yes'}:  # [run_lightea] --cross_lingual",
        "cross-lingual rel_dim branch")
    # TF's GPU SparseTensorDenseMatMul refuses output columns * nnz > 2^31
    # (2M: the ent-ent graph has > 2^31/128 nonzeros).
    # batch_sparse_matmul's column chunk (128) is only a memory knob -- each
    # output column depends on its own input column alone -- so shrink it
    # just when that limit would be hit; a no-op on 15K / 100K / DBP1M.
    sub(r"^(def batch_sparse_matmul\(sparse_tensor,dense_tensor,batch_size = 128,save_mem = False\):)$",
        r"\1\n    batch_size = max(1, min(batch_size, (2**31 - 1) // max(1, int(sparse_tensor.indices.shape[0]))))"
        "  # [run_lightea] TF 2^31 limit",
        "batch_sparse_matmul chunk cap")
    return src


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_dir", required=True, help="output of prepare_for_lightea.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--script", default="run_lightea_default.py", help="released notebook, converted")
    p.add_argument("--ent_dim", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cross_lingual", choices=["yes", "no"], default="yes",
                   help="notebook's 'EN in dataset' branch: yes -> rel_dim = ent_dim//2 (all our "
                        "datasets are EN-DE); no -> ent_dim//3 (its monolingual DBP_WD/DBP_YG)")
    p.add_argument("--faiss_cpu_fallback", action="store_true",
                   help="if faiss cannot move its index to the GPU (out of memory), search on the CPU "
                        "instead of failing (2M)")
    p.add_argument("--export_dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--keep_graph_cache", action="store_true",
                   help="reuse LightEA's graph_cache.pkl (debug only; timing then excludes graph build)")
    args = p.parse_args()

    sys.path.insert(0, os.getcwd())
    import utils as lightea_utils  # the checkout's own utils.py

    lightea_utils.load_aligned_pair = fixed_split_loader(args.dataset_dir)

    if args.faiss_cpu_fallback:
        # 2M: TF's allocator keeps the ~80 GB it grew to
        # while building the features, so utils.sparse_sinkhorn_sims'
        # faiss.index_cpu_to_gpu cannot get even its 1.6 GB temp buffer.
        # Fall back to the CPU index for that call -- the notebook's own
        # `if len(gpus)` CPU branch; with its exact 'Flat' index the
        # neighbours are the same, only slower. The TF Sinkhorn after it
        # stays on the GPU. Logged every time it triggers.
        import faiss
        _to_gpu = faiss.index_cpu_to_gpu

        def index_cpu_to_gpu_or_cpu(res, device, index, *a, **kw):
            try:
                return _to_gpu(res, device, index, *a, **kw)
            except RuntimeError as e:
                if "out of memory" not in str(e):
                    raise
                # The job environment may export OMP_NUM_THREADS=1, which
                # leaves faiss' CPU search on one of the allocated cores.
                # Give it the whole allocation; only this
                # fallback path changes, so 15K/100K/DBP1M timings are untouched.
                n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
                faiss.omp_set_num_threads(n_threads)
                print(f"[run_lightea] faiss GPU index out of memory, using the CPU index with "
                      f"{n_threads} threads (--faiss_cpu_fallback): {str(e).splitlines()[-1][:160]}",
                      flush=True)
                return index

        faiss.index_cpu_to_gpu = index_cpu_to_gpu_or_cpu

    # utils.load_graph pickles its built graph next to the data and reuses it
    # on later runs; drop it so every run's time includes graph construction
    # (complete end-to-end timing).
    cache = os.path.join(args.dataset_dir, "graph_cache.pkl")
    if os.path.exists(cache) and not args.keep_graph_cache:
        os.remove(cache)
        print(f"[run_lightea] removed stale {cache} (graph rebuilt and timed this run)", flush=True)

    with open(args.script) as f:
        src = patch_source(f.read(), args.dataset_dir, args)
    print(f"[run_lightea] dataset_dir={args.dataset_dir} ent_dim={args.ent_dim or 'default'} "
          f"epochs={args.epochs or 'default'} seed={args.seed or 'default'} "
          f"cross_lingual={args.cross_lingual} faiss_cpu_fallback={args.faiss_cpu_fallback}", flush=True)

    ns: dict = {"__name__": "__lightea__", "__file__": os.path.abspath(args.script)}
    t0 = time.time()
    exec(compile(src, args.script, "exec"), ns)
    t_run = time.time() - t0

    test_pair = ns["test_pair"]
    features = ns["features"]
    right_list = ns["right_list"]
    os.makedirs(args.out_dir, exist_ok=True)
    own = {
        "hit@1": len(right_list) / len(test_pair),
        "n_test": int(len(test_pair)),
        "n_train_final": int(len(ns["train_pair"])),
        "feature_dim": int(features.shape[1]),
        "ent_dim": int(ns["ent_dim"]), "rel_dim": int(ns["rel_dim"]), "epochs": int(ns["epochs"]),
        "run_seconds": t_run,
    }
    with open(os.path.join(args.out_dir, "lightea_own.json"), "w") as f:
        json.dump(own, f, indent=2)
    print(f"[run_lightea] own protocol: {own}", flush=True)

    with open(os.path.join(args.dataset_dir, "meta.json")) as f:
        meta = json.load(f)
    e1 = np.unique(test_pair[:, 0])
    e2 = np.unique(test_pair[:, 1])
    ids = np.concatenate([e1, e2]).astype(np.int64)
    if ids.max() >= features.shape[0]:
        raise SystemExit(f"test entity id {ids.max()} beyond feature rows {features.shape[0]}")
    np.save(os.path.join(args.out_dir, "embeddings.npy"), features[ids].astype(args.export_dtype))
    np.save(os.path.join(args.out_dir, "ids.npy"), ids)
    np.save(os.path.join(args.out_dir, "kg_labels.npy"),
            (ids >= meta["kg2_id_min"]).astype(np.int8))
    with open(os.path.join(args.out_dir, "test_links.json"), "w") as f:
        json.dump(test_pair.tolist(), f)
    print(f"[run_lightea] exported {len(e1)} KG1 + {len(e2)} KG2 test-side rows "
          f"(dim {features.shape[1]}, {args.export_dtype}) to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
