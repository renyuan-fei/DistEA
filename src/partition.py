"""
partition.py — METIS graph partitioning + KG2 entity assignment.

Run once before training:
    uv run python src/partition.py \\
        --data_dir data/OpenEA_dataset_v2.0/EN_DE_15K_V1 \\
        --fold_dir data/OpenEA_dataset_v2.0/EN_DE_15K_V1/721_5fold/1 \\
        --partition 4                   # --kg2_assignment locality (default)

Output: outputs/partitions/EN_DE_15K_V1/721_5fold_1/dist_True/4_metis_kg2locality/
    0.npy … (N-1).npy          KG1 entity IDs assigned to each rank
    ent2id.json                 entity name → global integer ID
    kg2_assignments.json        {rank: [kg2_entity_ids]}
    meta.json                   {data_dir, fold_dir, n_parts} — checked by main.py

The fold identifier (here "721_5fold_1", derived from --fold_dir's path
relative to --data_dir) is its own directory level so two different
folds partitioned at the same --partition count never collide on the same
path.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import shutil

import networkx as nx
import nxmetis

from counterpart_disc import (
    assign_kg2_to_ranks,
    assign_kg2_to_ranks_locality,
    compute_kg2_locality_edge_weights,
)
from dataload import OpenEAData
from graph_metrics import graph_metrics_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def build_kg1_graph(data, edge_weights: dict | None = None) -> nx.Graph:
    """
    Build an undirected NetworkX graph from KG1 triples.

    edge_weights: optional dict keyed by tuple(sorted((h, t))) (see
    counterpart_disc.compute_kg2_locality_edge_weights) -- default None
    builds the plain unweighted graph (no `weight` edge attribute at all).

    When edge_weights is given, self-loop triples (h == t, ~261 of them in
    EN_DE_15K_V1's real KG1 data) are skipped rather than added as
    self-loop edges: METIS's own C-level `CheckGraph` validation tolerates
    them fine in the plain unweighted case (`adjwgt=None` when every weight
    is 1, per nxmetis's own `partition()`, bypasses whatever check trips on
    them), but asserts
    `libmetis__CheckGraph` once real per-edge weights are actually supplied.
    Dropping them changes nothing about the partition objective either way
    -- a self-loop's two endpoints are always in the same partition by
    definition, so it never contributes to any edge-cut, weighted or not.
    """
    G = nx.Graph()
    G.add_nodes_from(data.kg1_entities)
    for h, _r, t in data.kg1_triples:
        if edge_weights is None:
            G.add_edge(h, t)
        elif h != t:
            G.add_edge(h, t, weight=edge_weights.get(tuple(sorted((h, t))), 1))
    return G


def metis_partition(G: nx.Graph, n_parts: int):
    """
    Partition G into n_parts using METIS.

    Returns a list of n_parts sorted entity-ID lists.
    """
    # recursive=True (METIS_PartGraphRecursive) instead of the kway default:
    # on this KG1 graph (disconnected, 165 components) the kway path hits a
    # `libmetis__CheckBnd2: Assertion 'nbnd == graph->nbnd' failed` crash in
    # the vendored METIS C code at n_parts=8 (n_parts=4 happened to work).
    # Recursive bisection avoids that refinement codepath entirely and also
    # produces a more evenly balanced split (verified: exact 3750x4 / 1875x8
    # vs kway's uneven ~3689-3862 at n_parts=4).
    _, parts = nxmetis.partition(G, n_parts, recursive=True)
    return [sorted(p) for p in parts]


def metis_partition_degree_csr(data, n_parts: int, with_kg2: bool = False) -> list[list[int]]:
    """--kg1_node_weight degree: METIS on a scipy CSR of the KG1 graph with
    vertex weight 1 + the entity's KG1 triple count, through pymetis (a
    release build of METIS 5). The vendored nxmetis path (networkx graph +
    an assertion-enabled METIS) had not returned after 23 min on DBP1M's
    1.12M-entity KG1. Same objective as
    metis_partition: unweighted edge-cut of the undirected, de-duplicated
    KG1 graph, recursive bisection; only the vertex weights and the METIS
    build differ. Returns sorted entity-ID lists like metis_partition.

    with_kg2 (--kg1_node_weight degree_kg2): each train-linked KG1 entity
    also carries its KG2 counterpart's KG2 triple count -- the KG2 locality
    assignment puts that counterpart (and, through the BFS, much of its KG2
    neighbourhood) on the same rank, so the KG2 half of a rank's load
    follows the KG1 split. Train links only (no valid / test labels)."""
    import pymetis
    import scipy.sparse as sp

    n = int(data.ent_num_1)
    tri = np.asarray(data.kg1_triples, dtype=np.int64).reshape(-1, 3)
    h, t = tri[:, 0], tri[:, 2]
    inside = (h < n) & (t < n)
    if not inside.all():
        logging.warning(f"kg1_node_weight=degree: {int((~inside).sum())} KG1 triples touch ids >= "
                        f"ent_num_1 and are left out of the METIS graph")
    deg = np.bincount(h[inside], minlength=n) + np.bincount(t[inside], minlength=n)
    if with_kg2:
        tri2 = np.asarray(data.kg2_triples, dtype=np.int64).reshape(-1, 3)
        n_all = int(data.ent_num_1 + data.ent_num_2)
        deg2 = np.bincount(tri2[:, 0], minlength=n_all) + np.bincount(tri2[:, 2], minlength=n_all)
        links = np.asarray(data.load_train_alignment(), dtype=np.int64).reshape(-1, 2)
        ok = links[:, 0] < n
        np.add.at(deg, links[ok, 0], deg2[links[ok, 1]])
    keep = inside & (h != t)
    rows = np.concatenate([h[keep], t[keep]])
    cols = np.concatenate([t[keep], h[keep]])
    adj = sp.csr_matrix((np.ones(rows.shape[0], dtype=np.int8), (rows, cols)), shape=(n, n))
    adj.sum_duplicates()
    objval, part = pymetis.part_graph(
        int(n_parts), xadj=adj.indptr.astype(np.int64), adjncy=adj.indices.astype(np.int64),
        vweights=(deg + 1).astype(np.int64), recursive=True,
    )
    part = np.asarray(part, dtype=np.int64)
    w = np.bincount(part, weights=deg + 1, minlength=n_parts)
    logging.info(f"kg1_node_weight=degree{'_kg2' if with_kg2 else ''} (pymetis CSR): edge-cut {objval}, part weights "
                 f"{int(w.min())}-{int(w.max())} (mean {w.mean():.0f}), part sizes "
                 f"{int(np.bincount(part, minlength=n_parts).min())}-"
                 f"{int(np.bincount(part, minlength=n_parts).max())}")
    return [np.nonzero(part == r)[0].tolist() for r in range(n_parts)]


def run_partition(
    data_dir: str, fold_dir: str, output_base: str, n_parts: int,
    kg2_assignment: str = "locality",
    kg1_edge_weight: str = "none",
    kg1_node_weight: str = "none",
):
    output_dir = os.path.join(output_base, os.path.basename(os.path.normpath(data_dir)))
    # Fold-aware path: the fold identifier is its own directory level,
    # derived from --fold_dir's path relative to --data_dir (e.g.
    # "custom_30pct_fold/1" -> "custom_30pct_fold_1"), so two different
    # folds partitioned at the same n_parts never collide (meta.json's
    # mismatch check in main.py is a second safety net). Falls back to
    # fold_dir's own basename if it isn't nested under data_dir (unusual,
    # but keeps the function total rather than raising on an edge case).
    #
    # Inserted ABOVE "dist_True", not replacing it: leapgnn_cache.py's
    # GrpcEmbeddingCache._split_partition_dir() and the Go server's own
    # ConstructNid2Pid both hard-parse partition_dir's last two segments as
    # literal ".../dist_True/{N}_{type}/" (tests/test_stage_m.py). Keeping
    # "dist_True" as the immediate parent of the {n_parts}{suffix} leaf
    # means both the Python split logic and the Go server's path
    # reconstruction work unchanged -- "dataset_root" just resolves one
    # level deeper, which is all either side assumes about it.
    fold_rel = os.path.relpath(os.path.normpath(fold_dir), os.path.normpath(data_dir))
    fold_id = os.path.basename(os.path.normpath(fold_dir)) if fold_rel.startswith("..") \
        else fold_rel.replace(os.sep, "_")
    # --kg2_assignment locality and --kg1_edge_weight kg2_locality each write
    # to their own SEPARATE, suffixed directory within that fold's directory
    # rather than a shared one — other jobs may point --partition_dir at
    # this dataset+fold+n_parts combo on a shared filesystem, and two
    # default-off axes composing independently means neither silently
    # overwrites the other's output.
    suffix = "_metis"
    if kg1_edge_weight == "kg2_locality":
        suffix += "_kg2weighted"
    if kg1_node_weight == "degree":
        suffix += "_degw"
    elif kg1_node_weight == "degree_kg2":
        suffix += "_degw2"
    if kg2_assignment == "locality":
        suffix += "_kg2locality"
    partition_dir = os.path.join(output_dir, fold_id, "dist_True", f"{n_parts}{suffix}")
    os.makedirs(partition_dir, exist_ok=True)

    data = OpenEAData(data_dir, fold_dir, output_dir)

    # Compute graph metrics if needed
    edge_info_path = os.path.join(output_dir, "KGs_edge_info.json")
    if not os.path.exists(edge_info_path):
        logging.info("Computing graph metrics…")
        graph_metrics_main(data)
    data.load_weights()

    logging.info(f"KG1 entities: {data.ent_num_1}, KG2 entities: {data.ent_num_2}")

    # METIS partition of KG1. For --kg2_assignment locality (with the
    # default --kg1_edge_weight none), reuse the EXISTING
    # dist_True/{n_parts}_metis/ KG1 split verbatim if it's already been
    # computed (METIS itself has no bearing on KG2 assignment strategy, and
    # copying guarantees a bit-identical KG1 split for a fair
    # single-variable comparison against the round-robin baseline, rather
    # than risking two different METIS runs producing slightly different
    # partitions). --kg1_edge_weight kg2_locality changes the KG1 split
    # itself (that's the whole point), so it never reuses this shortcut --
    # METIS always re-runs with the weighted graph in that case.
    existing_metis_dir = os.path.join(output_dir, fold_id, "dist_True", f"{n_parts}_metis")
    reused_existing = False
    if (
        kg2_assignment == "locality"
        and kg1_edge_weight == "none"
        and kg1_node_weight == "none"
        and os.path.isdir(existing_metis_dir)
    ):
        part_files = sorted(
            f for f in os.listdir(existing_metis_dir) if f.endswith(".npy")
        )
        if len(part_files) == n_parts:
            logging.info(
                f"Reusing existing KG1 partition from {existing_metis_dir} "
                f"(bit-identical split, only KG2 assignment differs)"
            )
            partitions = []
            for rank in range(n_parts):
                src = os.path.join(existing_metis_dir, f"{rank}.npy")
                dst = os.path.join(partition_dir, f"{rank}.npy")
                shutil.copy(src, dst)
                partitions.append(sorted(np.load(src).tolist()))
            shutil.copy(
                os.path.join(existing_metis_dir, "ent2id.json"),
                os.path.join(partition_dir, "ent2id.json"),
            )
            reused_existing = True

    if not reused_existing:
        logging.info(f"Running METIS partition into {n_parts} parts…")
        edge_weights = None
        if kg1_edge_weight == "kg2_locality":
            edge_weights = compute_kg2_locality_edge_weights(data)
            logging.info(
                f"kg1_edge_weight=kg2_locality: {len(edge_weights)} KG1-KG1 "
                f"edges weighted (both endpoints train-linked with "
                f"same-component KG2 counterparts)"
            )
        if kg1_node_weight in ("degree", "degree_kg2") and edge_weights is None:
            # Balance the parts' triple load, not their entity count (vertex
            # weight = 1 + KG1 triple degree). Each rank's training graph is
            # the triples touching its rows, and with count-balanced parts
            # the heaviest DBP1M N=8 rank carried 2.17x the even share --
            # every epoch waits for it.
            partitions = metis_partition_degree_csr(data, n_parts, with_kg2=kg1_node_weight == "degree_kg2")
        else:
            G = build_kg1_graph(data, edge_weights=edge_weights)
            if kg1_node_weight == "degree":
                # Same vertex weights on the networkx path (nxmetis reads
                # the node attribute "weight"; edges keep their own).
                tri = np.asarray(data.kg1_triples, dtype=np.int64).reshape(-1, 3)
                deg = np.bincount(tri[:, 0], minlength=data.ent_num_1) \
                    + np.bincount(tri[:, 2], minlength=data.ent_num_1)
                nx.set_node_attributes(G, {v: int(deg[v]) + 1 for v in G.nodes}, "weight")
            partitions = metis_partition(G, n_parts)

        # Write partition files
        for rank, part in enumerate(partitions):
            out = os.path.join(partition_dir, f"{rank}.npy")
            np.save(out, np.array(part, dtype=np.int64))
            logging.info(f"  Partition {rank}: {len(part)} entities → {out}")

        # Save ent2id
        ent2id_path = os.path.join(partition_dir, "ent2id.json")
        with open(ent2id_path, "w") as f:
            json.dump(data.ent2id, f)
        logging.info(f"ent2id written to {ent2id_path}")

    # KG2 assignment
    if kg2_assignment == "locality":
        assign_kg2_to_ranks_locality(data, partitions, partition_dir)
    else:
        assign_kg2_to_ranks(data, partitions, partition_dir)

    # Record which fold produced this partition. main.py checks this file
    # against its own --fold_dir at startup so a stale/mismatched partition
    # fails fast instead of silently training on the wrong train/test split.
    meta_path = os.path.join(partition_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "data_dir": os.path.normpath(data_dir),
            "fold_dir": os.path.normpath(fold_dir),
            "n_parts": n_parts,
            "kg2_assignment": kg2_assignment,
            "kg1_edge_weight": kg1_edge_weight,
            "kg1_node_weight": kg1_node_weight,
        }, f)

    logging.info(f"Partition complete. Output: {partition_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="DistEA — METIS Partitioner")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--fold_dir", required=True)
    p.add_argument(
        "--partition",
        type=int,
        default=4,
        help="Number of partitions (= number of training ranks)",
    )
    p.add_argument(
        "--output_base",
        default="outputs/partitions",
        help="Base directory for partition output",
    )
    p.add_argument(
        "--kg2_assignment",
        choices=["round_robin", "locality"],
        default="locality",
        help="How to place KG2 entities not in train_links. locality "
             "(default; +55%%/+121%% relative Hit@1 at ws=2/4 over "
             "round_robin, holding through ws=128): "
             "BFS-layered majority vote over the KG2-KG2 graph from "
             "train-linked seeds -- see "
             "counterpart_disc.assign_kg2_to_ranks_locality. round_robin "
             "(ablation baseline): i %% world_size, ignoring KG2-KG2 graph "
             "structure entirely. Writes to its own suffixed directory (reusing the "
             "existing {n}_metis/ KG1 split verbatim if already computed "
             "for locality, for a fair single-variable comparison) rather "
             "than overwriting the other strategy's output other jobs may "
             "be pointing --partition_dir at.",
    )
    p.add_argument(
        "--kg1_edge_weight",
        choices=["none", "kg2_locality"],
        default="none",
        help="Whether METIS's own KG1 partitioning objective is nudged by "
             "KG2 structure. "
             "none (default): plain unweighted "
             "edge-cut minimization. kg2_locality: KG1-KG1 edges where BOTH "
             "endpoints are train-linked get a weight bonus if their KG2 "
             "counterparts fall in the same weakly-connected component of "
             "the KG2-KG2 graph -- see "
             "counterpart_disc.compute_kg2_locality_edge_weights. Every "
             "other edge (no train-linked signal available) keeps the "
             "unweighted default, avoiding a chicken-and-egg dependency on "
             "a partition result that doesn't exist yet. Complementary to, "
             "not a replacement for, --kg2_assignment locality -- the "
             "downstream KG2 assignment pass runs unchanged either way. "
             "Writes to its own suffixed directory (composes with "
             "--kg2_assignment locality's suffix when both are set) rather "
             "than overwriting the unweighted output other jobs may be "
             "pointing --partition_dir at.",
    )
    p.add_argument(
        "--kg1_node_weight",
        choices=["none", "degree", "degree_kg2"],
        default="none",
        help="METIS vertex weights for the KG1 split. none (default): "
             "every entity weighs 1, so parts have equal entity "
             "counts. degree: 1 + the entity's KG1 triple count, so parts carry "
             "similar triple loads -- each rank's training graph is the triples "
             "touching its rows, and the heaviest count-balanced DBP1M N=8 rank "
             "held 2.17x the even share. degree_kg2: each train-linked entity "
             "also carries its KG2 counterpart's KG2 triple count (the KG2 "
             "assignment follows the train links). Write to their own "
             "_degw / _degw2-suffixed directories and never reuse an existing "
             "{n}_metis split.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_partition(
        args.data_dir, args.fold_dir, args.output_base, args.partition,
        kg2_assignment=args.kg2_assignment,
        kg1_edge_weight=args.kg1_edge_weight,
        kg1_node_weight=args.kg1_node_weight,
    )
