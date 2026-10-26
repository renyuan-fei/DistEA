#!/usr/bin/env python
"""
make_refined_kg2_partition.py -- write a sibling partition directory whose
KG1 split ({r}.npy, ent2id.json) is copied verbatim from an existing
locality partition and whose kg2_assignments.json is refined by
src/kg2_refine.refine_kg2_assignment (KG1-side vote + label propagation).
A training comparison against the source directory therefore changes the
KG2 assignment only.

Reads:  the dataset/fold and an existing partition directory.
Writes: <partition_dir>ref (e.g. 4_metis_kg2locality -> 4_metis_kg2localityref);
        meta.json records the source.

Usage:
    python scripts/make_refined_kg2_partition.py \
        --data_dir data/DBP1M_DE --fold_dir data/DBP1M_DE/316_1fold/1 \
        --partition_dir outputs/partitions/DBP1M_DE/316_1fold_1/dist_True/4_metis_kg2locality [--rounds 3]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from counterpart_disc import _write_kg2_assignments  # noqa: E402
from dataload import OpenEAData  # noqa: E402
from kg2_refine import refine_kg2_assignment  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--fold_dir", required=True)
    p.add_argument("--partition_dir", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--no_kg1_vote", action="store_true")
    p.add_argument("--output_base", default="outputs/partitions")
    a = p.parse_args()
    src = os.path.normpath(a.partition_dir)
    dst = src + "ref"
    out_dir = os.path.join(a.output_base, os.path.basename(os.path.normpath(a.data_dir)))
    data = OpenEAData(a.data_dir, a.fold_dir, out_dir)
    files = sorted(glob.glob(os.path.join(src, "[0-9]*.npy")), key=lambda x: int(os.path.basename(x).split(".")[0]))
    w = len(files)
    kg1 = np.asarray(sorted(data.kg1_entities), dtype=np.int64)
    kg2 = np.asarray(sorted(data.kg2_entities), dtype=np.int64)
    n = int(max(kg1.max(), kg2.max())) + 1
    kg1_rank = np.full(n, -1, dtype=np.int64)
    for r, fn in enumerate(files):
        kg1_rank[np.load(fn).astype(np.int64)] = r
    with open(os.path.join(src, "kg2_assignments.json")) as f:
        stored = json.load(f)
    kg2_rank = np.full(n, -1, dtype=np.int64)
    for r, ids in stored.items():
        kg2_rank[np.asarray(ids, dtype=np.int64)] = int(r)
    assert (kg2_rank[kg2] >= 0).all(), "source assignment does not cover every KG2 entity"
    new = refine_kg2_assignment(data, kg1_rank, kg2_rank, w, rounds=a.rounds, kg1_vote=not a.no_kg1_vote)

    def share(links, asg):
        x = np.asarray(links, dtype=np.int64).reshape(-1, 2)
        return float(np.mean(kg1_rank[x[:, 0]] != asg[x[:, 1]]))

    valid, test = data.load_valid_alignment(), data.load_test_alignment()
    for name, asg in (("source", kg2_rank), ("refined", new)):
        load = np.bincount(asg[kg2], minlength=w)
        print(f"[refine] {name:8s} cross valid {share(valid, asg):.4f} test {share(test, asg):.4f} "
              f"KG2 load max/mean {load.max() / load.mean():.2f} min {load.min()}", flush=True)
    os.makedirs(dst, exist_ok=True)
    for fn in files + [os.path.join(src, "ent2id.json")]:
        if os.path.exists(fn):
            shutil.copy2(fn, os.path.join(dst, os.path.basename(fn)))
    result = {r: kg2[new[kg2] == r].tolist() for r in range(w)}
    _write_kg2_assignments(result, dst)
    meta = {}
    if os.path.exists(os.path.join(src, "meta.json")):
        with open(os.path.join(src, "meta.json")) as f:
            meta = json.load(f)
    meta.update({"kg2_assignment": "locality_refined", "refined_from": src,
                 "refine_rounds": a.rounds, "refine_kg1_vote": not a.no_kg1_vote})
    with open(os.path.join(dst, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[refine] wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
