#!/usr/bin/env python
"""
export_rrea_tf_to_unified.py -- convert the original TensorFlow RREA
runner's embeddings into a unified_eval.py export bundle.

Reads:  the runner's `emb.npz` (embs = L2-normalised (N, D) table indexed by
        the ids in ent_ids_1/2, ent1_ids / ent2_ids) and the prepared data
        dir's ref_ent_ids_test.
Writes: --out_dir/{embeddings.npy, ids.npy, kg_labels.npy, test_links.json,
        meta.json}, restricted -- like DistEA's own export -- to the
        test-side KG1 and KG2 rows by default (--all_kg1 / --all_kg2 widen
        it), so the number is produced by the same evaluator and protocol
        (cosine + CSLS-10) as every other method.

Ids: prepare_rrea_tf_data.py wrote DistEA's own ids with an identity new->old
map, so emb row i IS DistEA global id i; test_links come from the same
prepared dir (ref_ent_ids_test), i.e. the fold's test_links.

Usage:
    python scripts/export_rrea_tf_to_unified.py \
        --emb_npz <RREA output_dir>/emb.npz \
        --data_dir projects/ELsEA/DivEA/data/rrea_tf/EN_DE_100K_V1_721_5fold_1 \
        --out_dir export/rrea_tf/EN_DE_100K_V1/721_5fold_1/ws1 \
        [--all_kg1]   # keep every KG1 row instead of the test-side ones
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def read_pairs(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                a, b = line.split()
                out.append((int(a), int(b)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emb_npz", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--all_kg1", action="store_true")
    p.add_argument("--all_kg2", action="store_true",
                   help="keep every KG2 row as a candidate; default restricts KG2 to the "
                        "test-side entities, which is what DistEA's selective_eval export "
                        "contains (same candidate set -> comparable Hit@k)")
    a = p.parse_args()

    z = np.load(a.emb_npz)
    embs = np.asarray(z["embs"], dtype=np.float32)
    ent1 = np.asarray(z["ent1_ids"], dtype=np.int64)
    ent2 = np.asarray(z["ent2_ids"], dtype=np.int64)
    test = read_pairs(os.path.join(a.data_dir, "ref_ent_ids_test"))
    test_e1 = sorted({e1 for e1, _ in test})
    kg1_rows = np.array(test_e1 if not a.all_kg1 else ent1.tolist(), dtype=np.int64)
    test_e2 = sorted({e2 for _, e2 in test})
    kg2_rows = np.array(test_e2 if not a.all_kg2 else ent2.tolist(), dtype=np.int64)
    assert embs.shape[0] > max(int(kg1_rows.max()), int(kg2_rows.max()))
    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, "embeddings.npy"), np.concatenate([embs[kg1_rows], embs[kg2_rows]]).astype(np.float32))
    np.save(os.path.join(a.out_dir, "ids.npy"), np.concatenate([kg1_rows, kg2_rows]).astype(np.int64))
    np.save(os.path.join(a.out_dir, "kg_labels.npy"),
            np.concatenate([np.zeros(len(kg1_rows), np.int8), np.ones(len(kg2_rows), np.int8)]))
    with open(os.path.join(a.out_dir, "test_links.json"), "w") as f:
        json.dump([[int(e1), int(e2)] for e1, e2 in test], f)
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump({"source": "rrea_tf", "emb_npz": os.path.abspath(a.emb_npz), "data_dir": os.path.abspath(a.data_dir),
                   "n_kg1": int(len(kg1_rows)), "n_kg2": int(len(kg2_rows)), "width": int(embs.shape[1]),
                   "kg2_candidates": "all" if a.all_kg2 else "test_side",
                   "n_test_links": len(test), "dtype": "fp32"}, f, indent=2)
    print(f"wrote {a.out_dir}: {len(kg1_rows)} KG1 rows x {len(kg2_rows)} KG2 rows, {len(test)} test links, width {embs.shape[1]}")


if __name__ == "__main__":
    main()
