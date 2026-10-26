#!/usr/bin/env python
"""
export_clusterea_to_unified.py -- convert ClusterEA embeddings into a
unified_eval.py export bundle.

Reads:  ClusterEA's saved step-1 embeddings
        (`{save_folder}/embeddings.pkl_<suffix>`, a pickled
        `(ea, (kg1_emb, kg2_emb))`), DistEA's ent2id.json and the fold's
        test_links.
Writes: --out_dir/{embeddings.npy, ids.npy, kg_labels.npy, test_links.json},
        so ClusterEA is scored by the same code and protocol as DistEA and
        ELsEA.

Id spaces: ClusterEA's `ea.ent1`/`ea.ent2` are `{uri: local_id}` (row index
into kg1_emb / kg2_emb respectively, each 0-based per KG). DistEA's global
ids come from partition.py's ent2id.json (`{uri: global_id}`, KG1 then KG2).
Mapping goes through the URI, never by assuming the two first-seen orders
coincide. test_links are the FOLD's test_links (URI pairs) mapped the same
way -- only meaningful for a checkpoint trained with CLUSTEREA_FOLD_DIR set
to that same fold (run_clusterea.py::apply_fold_override); a checkpoint
trained on ClusterEA's own prefix split would leak that fold's test pairs
into training and must not be scored this way.

Must run inside the clusterea-gpu env (unpickling `ea` needs ClusterEA's own
dataset/dto classes + torch 1.12) with CUDA/11.7.0 loaded (ClusterEA's utils
import torch_sparse -> libcusparse), from ClusterEA/src as cwd or with
--clusterea_src pointing at it. The written bundle is plain npy/json and is
then scored in the distea env.

Usage:
    python scripts/export_clusterea_to_unified.py \
        --pkl tmp_fold/embeddings.pkl_small_ids_de_dual-amn_it5 \
        --ent2id_json outputs/partitions/EN_DE_15K_V1/custom_30pct_fold_1/dist_True/5_metis_kg2locality/ent2id.json \
        --fold_dir data/OpenEA_dataset_v2.0/EN_DE_15K_V1/custom_30pct_fold/1 \
        --out_dir export/clusterea/EN_DE_15K_V1/custom_30pct_fold_1/n1/dual-amn
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np


def to_numpy(x) -> np.ndarray:
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().float().numpy()
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float32)


def map_rows(ent_dict: dict, emb: np.ndarray, ent2id: dict, label: str):
    assert emb.shape[0] == len(ent_dict), f"{label}: {emb.shape[0]} rows vs {len(ent_dict)} entities"
    gids = np.full(emb.shape[0], -1, dtype=np.int64)
    miss = 0
    for uri, lid in ent_dict.items():
        g = ent2id.get(uri)
        if g is None:
            miss += 1
            continue
        gids[int(lid)] = int(g)
    keep = gids >= 0
    print(f"  [{label}] {int(keep.sum())}/{emb.shape[0]} entities mapped to global ids ({miss} URIs absent from ent2id)")
    return emb[keep], gids[keep]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl", required=True)
    p.add_argument("--ent2id_json", required=True)
    p.add_argument("--fold_dir", required=True, help="dir with test_links (URI pairs)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clusterea_src", default=os.getcwd(), help="ClusterEA/src (for unpickling ea)")
    args = p.parse_args()

    sys.path.insert(0, args.clusterea_src)
    with open(args.pkl, "rb") as f:
        ea, embs = pickle.load(f)
    kg1_emb, kg2_emb = (to_numpy(e) for e in embs)
    with open(args.ent2id_json) as f:
        ent2id = json.load(f)

    e1, i1 = map_rows(ea.ent1, kg1_emb, ent2id, "kg1")
    e2, i2 = map_rows(ea.ent2, kg2_emb, ent2id, "kg2")

    test_links, miss = [], 0
    with open(os.path.join(args.fold_dir, "test_links"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            u1, u2 = line.rstrip("\n").split("\t")
            if u1 in ent2id and u2 in ent2id:
                test_links.append([int(ent2id[u1]), int(ent2id[u2])])
            else:
                miss += 1
    print(f"  test_links: {len(test_links)} pairs ({miss} unmapped)")

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "embeddings.npy"), np.concatenate([e1, e2]).astype(np.float32))
    np.save(os.path.join(args.out_dir, "ids.npy"), np.concatenate([i1, i2]).astype(np.int64))
    np.save(os.path.join(args.out_dir, "kg_labels.npy"),
            np.array([0] * len(i1) + [1] * len(i2), dtype=np.int8))
    with open(os.path.join(args.out_dir, "test_links.json"), "w") as f:
        json.dump(test_links, f)
    print(f"[export_clusterea_to_unified] wrote {args.out_dir} "
          f"({len(i1)} KG1 + {len(i2)} KG2 rows, dim {e1.shape[1]}, train_cnt in pkl = {getattr(ea, 'train_cnt', '?')})")


if __name__ == "__main__":
    main()
