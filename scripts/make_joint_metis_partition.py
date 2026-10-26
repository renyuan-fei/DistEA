#!/usr/bin/env python
"""
make_joint_metis_partition.py -- partition KG1 and KG2 together in one
METIS call (load balancing), so each rank's KG2 share is balanced by METIS
instead of following the KG1 split through the locality BFS.

Reads:  the dataset/fold and ent2id.json of an existing partition.
Writes: two partition directories under --out_dir (see below).

Graph: one vertex per KG1 entity and one per KG2 entity without a train
link; a train-linked KG2 entity is contracted into its KG1 partner's vertex,
so a train pair is never split. Edges: the KG1 and KG2 triples mapped onto
those vertices (undirected, de-duplicated, self-loops dropped). Vertex
weight: 1 + the KG1 + KG2 triple degree of what the vertex holds, i.e. the
vertex's share of a rank's training graph. pymetis, recursive bisection.
Uses train links only; valid / test links are read just to print the
cross-rank share.

Writes two sibling directories under --out_dir:
  {N}_metis_joint     KG1 split ({r}.npy) and kg2_assignments.json from METIS
  {N}_metis_jointref  the same KG1 split, KG2 refined by
                      kg2_refine.refine_kg2_assignment (like *_kg2localityref)
ent2id.json is copied from --ent2id_from (any partition of the same data).

Why (heaviest rank's RREA training graph relative to an even 1/N share,
with --drop_halo_halo_edges): DBP1M N=8 1.58x on the
refined locality partition, 1.41x degree-balanced, 1.23x jointref; N=4 1.37x
/ 1.24x / 1.15x. Cross-rank test pairs N=8 23.38% / 23.07% / 22.80%, N=4
17.00% / 17.01% / 18.09%. Every epoch waits for the heaviest rank.

Usage:
    python scripts/make_joint_metis_partition.py --data_dir data/DBP1M_DE \\
        --fold_dir data/DBP1M_DE/316_1fold/1 --n_parts 8 \\
        --ent2id_from outputs/partitions/DBP1M_DE/dist_True/8_metis_kg2localityref \\
        --out_dir outputs/partitions/DBP1M_DE/316_1fold_1/dist_True
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def joint_metis_assign(kg1: np.ndarray, kg2: np.ndarray, kg1_triples: np.ndarray,
                       kg2_triples: np.ndarray, train: np.ndarray, n_parts: int):
    """kg1 / kg2: entity id arrays; *_triples: (T, 3) id arrays; train:
    (P, 2) [kg1_id, kg2_id]. Returns (kg1_rank, kg2_rank, edge_cut) with the
    rank arrays indexed by global entity id (-1 where not applicable)."""
    import pymetis

    kg1 = np.asarray(kg1, dtype=np.int64)
    kg2 = np.asarray(kg2, dtype=np.int64)
    train = np.asarray(train, dtype=np.int64).reshape(-1, 2)
    n = int(max(kg1.max(), kg2.max())) + 1
    rep = np.full(n, -1, dtype=np.int64)
    rep[kg1] = np.arange(len(kg1))
    assert (rep[train[:, 0]] >= 0).all(), "a train link's KG1 side is not a KG1 entity"
    rep[train[:, 1]] = rep[train[:, 0]]
    free2 = kg2[rep[kg2] < 0]
    rep[free2] = len(kg1) + np.arange(len(free2))
    V = len(kg1) + len(free2)
    kg1_rank = np.full(n, -1, dtype=np.int64)
    kg2_rank = np.full(n, -1, dtype=np.int64)
    if n_parts <= 1:
        kg1_rank[kg1] = 0
        kg2_rank[kg2] = 0
        return kg1_rank, kg2_rank, 0
    tri = np.concatenate([np.asarray(kg1_triples, dtype=np.int64).reshape(-1, 3),
                          np.asarray(kg2_triples, dtype=np.int64).reshape(-1, 3)])
    deg = np.bincount(np.concatenate([tri[:, 0], tri[:, 2]]), minlength=n)
    ents = np.concatenate([kg1, kg2])
    vw = np.bincount(rep[ents], weights=deg[ents], minlength=V).astype(np.int64) + 1
    a, b = rep[tri[:, 0]], rep[tri[:, 2]]
    keep = a != b
    rows = np.concatenate([a[keep], b[keep]])
    cols = np.concatenate([b[keep], a[keep]])
    adj = sp.csr_matrix((np.ones(rows.shape[0], dtype=np.int32), (rows, cols)), shape=(V, V))
    adj.sum_duplicates()
    cut, part = pymetis.part_graph(
        int(n_parts), xadj=adj.indptr.astype(np.int64), adjncy=adj.indices.astype(np.int64),
        vweights=vw, recursive=True,
    )
    part = np.asarray(part, dtype=np.int64)
    kg1_rank[kg1] = part[rep[kg1]]
    kg2_rank[kg2] = part[rep[kg2]]
    return kg1_rank, kg2_rank, int(cut)


def main() -> None:
    from counterpart_disc import _write_kg2_assignments
    from dataload import OpenEAData
    from dist_graph import _triples_array
    from kg2_refine import refine_kg2_assignment

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--fold_dir", required=True)
    p.add_argument("--n_parts", type=int, required=True)
    p.add_argument("--ent2id_from", required=True, help="partition dir of the same data to copy ent2id.json from")
    p.add_argument("--out_dir", required=True, help="parent of the {N}_metis_joint[ref] directories")
    p.add_argument("--rounds", type=int, default=3, help="refine_kg2_assignment rounds (as make_refined_kg2_partition.py)")
    p.add_argument("--output_base", default="outputs/partitions")
    a = p.parse_args()
    ds = os.path.basename(os.path.normpath(a.data_dir))
    data = OpenEAData(a.data_dir, a.fold_dir, os.path.join(a.output_base, ds))
    kg1 = np.asarray(sorted(data.kg1_entities), dtype=np.int64)
    kg2 = np.asarray(sorted(data.kg2_entities), dtype=np.int64)
    train = np.asarray(data.load_train_alignment(), dtype=np.int64).reshape(-1, 2)
    kg1_rank, kg2_rank, cut = joint_metis_assign(
        kg1, kg2, _triples_array(data, "kg1_triples"), _triples_array(data, "kg2_triples"), train, a.n_parts)
    refined = refine_kg2_assignment(data, kg1_rank, kg2_rank, a.n_parts, rounds=a.rounds)
    valid = np.asarray(data.load_valid_alignment(), dtype=np.int64).reshape(-1, 2)
    test = np.asarray(data.load_test_alignment(), dtype=np.int64).reshape(-1, 2)
    print(f"[joint] {ds} N={a.n_parts}: edge-cut {cut}", flush=True)
    for tag, k2r in (("joint", kg2_rank), ("jointref", refined)):
        dst = os.path.join(a.out_dir, f"{a.n_parts}_metis_{tag}")
        os.makedirs(dst, exist_ok=True)
        for r in range(a.n_parts):
            np.save(os.path.join(dst, f"{r}.npy"), kg1[kg1_rank[kg1] == r])
        _write_kg2_assignments({r: kg2[k2r[kg2] == r].tolist() for r in range(a.n_parts)}, dst)
        shutil.copy2(os.path.join(a.ent2id_from, "ent2id.json"), os.path.join(dst, "ent2id.json"))
        with open(os.path.join(dst, "meta.json"), "w") as f:
            json.dump({"data_dir": os.path.normpath(a.data_dir), "fold_dir": os.path.normpath(a.fold_dir),
                       "n_parts": a.n_parts, "kg2_assignment": "joint_metis" + ("_refined" if tag == "jointref" else ""),
                       "kg1_node_weight": "degree_joint", "refine_rounds": a.rounds if tag == "jointref" else 0}, f)
        l1 = np.bincount(kg1_rank[kg1], minlength=a.n_parts)
        l2 = np.bincount(k2r[kg2], minlength=a.n_parts)
        cv = float(np.mean(kg1_rank[valid[:, 0]] != k2r[valid[:, 1]]))
        ct = float(np.mean(kg1_rank[test[:, 0]] != k2r[test[:, 1]]))
        print(f"[joint] {tag:8s} {dst}: cross valid {cv:.4f} test {ct:.4f}; KG1 per rank {l1.min()}-{l1.max()}, "
              f"KG2 per rank {l2.min()}-{l2.max()}", flush=True)


if __name__ == "__main__":
    main()
