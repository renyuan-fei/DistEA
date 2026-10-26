#!/usr/bin/env python
"""
make_divea_aligned_partition.py -- build a DistEA partition directory whose
KG1 split is DERIVED FROM an ELsEA/DivEA run's own subtask division.

Reads:  the dataset/fold and the DivEA data dir's partition_*/kg_partition.json.
Writes: a partition directory (see below).

With this partition the two systems train on the same split, and the only
remaining difference is DistEA's runtime halo completion + shared training
vs DivEA's static context copies + independent per-subtask models (an
ablation that isolates the partitioning).

Why "derived" and not "copied verbatim" -- DivEA's subtasks are NOT a
partition in DistEA's sense (example counts from EN_DE_15K, N=5):
  - kg1_partition.entities ("own"): the subtask's test KG1 entities.
    Disjoint across subtasks, union = all 9000 test entities.  -> copied
    exactly: each becomes local to that rank.
  - kg1_partition.ctx_entities: non-test KG1 entities DivEA copies in as
    static training context. OVERLAP across subtasks (6000 slots, 4325
    distinct) and 1675 non-test KG1 entities appear in NO subtask.
  - kg2_partition.entities: counterpart-discovery candidate sets. Heavily
    overlapping (27634 slots / 14400 distinct; 600 KG2 entities in none).
DistEA needs every entity owned by exactly one rank (its embedding row has
one owner), so the leftover choices are made by the SAME BFS-layered
majority vote DistEA's own KG2 assignment uses (counterpart_disc.
assign_kg2_to_ranks_locality), restricted to the ranks DivEA put the entity
in, so no choice ever contradicts DivEA where DivEA was unambiguous:

  KG1: own -> its subtask (exact). Other KG1 entities -> BFS-layered vote
       over KG1-KG1 adjacency among ranks whose ctx contains them (all ranks
       if uncovered); ties/unreached -> least-loaded allowed rank.
  KG2 (--kg2_mode):
    locality          DistEA's own assign_kg2_to_ranks_locality on the
                      derived KG1 split (isolates the KG1 split).
    divea_constrained locality, then every non-train-seeded KG2 entity whose
                      rank is not among the subtasks DivEA listed it in is
                      moved to the allowed rank holding most of its KG2-KG2
                      neighbours (tie -> least loaded). Train-seeded entities
                      are never moved (train-pair co-location is fundamental
                      to both systems) and are only counted.

Writes the standard partition_dir layout ({rank}.npy, ent2id.json,
kg2_assignments.json, meta.json) plus stats.json, under the same fold-aware
path scheme partition.py uses: {output_base}/{dataset}/{fold_id}/dist_True/
{N}_{tag}. Reads DistEA's src/ modules; modifies none of them.

Usage:
    python scripts/make_divea_aligned_partition.py \
        --data_dir data/OpenEA_dataset_v2.0/EN_DE_15K_V1 \
        --fold_dir data/OpenEA_dataset_v2.0/EN_DE_15K_V1/custom_30pct_fold/1 \
        --elsea_data_dir projects/ELsEA/DivEA/data/IDS15K_V1/en_de \
        --kg2_mode divea_constrained --tag divea_kg2divea
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from counterpart_disc import _write_kg2_assignments, assign_kg2_to_ranks_locality  # noqa: E402
from dataload import OpenEAData  # noqa: E402
from dist_ctx_kg2 import build_kg2_adjacency  # noqa: E402


def undirected_adj(triples) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for h, _, t in triples:
        if h != t:
            adj[h].add(t)
            adj[t].add(h)
    return adj


def load_subtasks(elsea_data_dir: str):
    dirs = sorted(glob.glob(os.path.join(elsea_data_dir, "partition_*")),
                  key=lambda p: int(p.rsplit("_", 1)[1]))
    own, ctx, kg2 = [], [], []
    for d in dirs:
        with open(os.path.join(d, "kg_partition.json")) as f:
            kp = json.load(f)
        own.append({int(x) for x in kp["kg1_partition"]["entities"]})
        ctx.append({int(x) for x in kp["kg1_partition"]["ctx_entities"]})
        kg2.append({int(x) for x in kp["kg2_partition"]["entities"]})
    return own, ctx, kg2


def layered_vote_assign(rank_of: dict[int, int], remaining: list[int], allowed: dict[int, set[int]],
                        adj: dict[int, set[int]], load: list[int], n: int) -> dict[str, int]:
    """Mirror of assign_kg2_to_ranks_locality's layering: decide every entity
    reached in a layer from the frontier's state at the START of the layer."""
    undecided = set(remaining)
    frontier = set(rank_of)
    stats = Counter()
    while frontier and undecided:
        layer = {nb for e in frontier for nb in adj.get(e, ()) if nb in undecided}
        if not layer:
            break
        snapshot = dict(rank_of)
        decided_now = {}
        for e in sorted(layer):
            ok = allowed.get(e) or set(range(n))
            votes = Counter(snapshot[nb] for nb in adj.get(e, ()) if nb in snapshot and snapshot[nb] in ok)
            if votes:
                top = max(votes.values())
                tied = [r for r, v in votes.items() if v == top]
                r = min(tied, key=lambda x: (load[x], x))
                stats["voted"] += 1
            else:
                r = min(ok, key=lambda x: (load[x], x))
                stats["no_vote_least_loaded"] += 1
            decided_now[e] = r
        for e, r in decided_now.items():
            rank_of[e] = r
            load[r] += 1
            undecided.discard(e)
        frontier = set(decided_now)
    for e in sorted(undecided):  # never reached from any decided entity
        ok = allowed.get(e) or set(range(n))
        r = min(ok, key=lambda x: (load[x], x))
        rank_of[e] = r
        load[r] += 1
        stats["unreached_least_loaded"] += 1
    return dict(stats)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--fold_dir", required=True)
    p.add_argument("--output_base", default="outputs/partitions")
    p.add_argument("--elsea_data_dir", required=True, help="DivEA data dir containing partition_*/kg_partition.json")
    p.add_argument("--kg2_mode", choices=["locality", "divea_constrained"], required=True)
    p.add_argument("--tag", required=True, help="partition dir suffix, e.g. divea_kg2locality / divea_kg2divea")
    args = p.parse_args()

    output_dir = os.path.join(args.output_base, os.path.basename(os.path.normpath(args.data_dir)))
    fold_rel = os.path.relpath(os.path.normpath(args.fold_dir), os.path.normpath(args.data_dir))
    fold_id = os.path.basename(os.path.normpath(args.fold_dir)) if fold_rel.startswith("..") \
        else fold_rel.replace(os.sep, "_")

    own, ctx, kg2sets = load_subtasks(args.elsea_data_dir)
    n = len(own)
    partition_dir = os.path.join(output_dir, fold_id, "dist_True", f"{n}_{args.tag}")
    os.makedirs(partition_dir, exist_ok=True)

    data = OpenEAData(args.data_dir, args.fold_dir, output_dir)
    kg1_all = set(data.kg1_entities)
    kg2_all = set(data.kg2_entities)

    # ---- KG1 ---------------------------------------------------------------
    rank_of: dict[int, int] = {}
    for r, s in enumerate(own):
        for e in s:
            assert e in kg1_all, f"DivEA own entity {e} not a KG1 id"
            assert e not in rank_of, f"DivEA own entity {e} in two subtasks"
            rank_of[e] = r
    load = [len(s) for s in own]
    ctx_members: dict[int, set[int]] = defaultdict(set)
    for r, s in enumerate(ctx):
        for e in s:
            if e not in rank_of:
                ctx_members[e].add(r)
    remaining = sorted(kg1_all - set(rank_of))
    n_ctx_single = sum(1 for e in remaining if len(ctx_members.get(e, ())) == 1)
    n_ctx_multi = sum(1 for e in remaining if len(ctx_members.get(e, ())) > 1)
    n_uncovered = sum(1 for e in remaining if e not in ctx_members)
    kg1_stats = layered_vote_assign(rank_of, remaining, ctx_members, undirected_adj(data.kg1_triples), load, n)
    partitions = [sorted(e for e, r in rank_of.items() if r == i) for i in range(n)]
    assert sum(len(x) for x in partitions) == len(kg1_all)
    for r, part in enumerate(partitions):
        np.save(os.path.join(partition_dir, f"{r}.npy"), np.array(part, dtype=np.int64))
    with open(os.path.join(partition_dir, "ent2id.json"), "w") as f:
        json.dump(data.ent2id, f)
    # how faithful is the derived split to DivEA's ctx placement?
    ctx_agree = sum(1 for e in remaining if e in ctx_members and rank_of[e] in ctx_members[e])
    print(f"[kg1] own exact={sum(len(s) for s in own)}  ctx single-membership={n_ctx_single} "
          f"multi={n_ctx_multi} uncovered-by-DivEA={n_uncovered}  vote stats={kg1_stats}  "
          f"placed within a DivEA ctx membership={ctx_agree}/{len(remaining) - n_uncovered}  "
          f"rank sizes={[len(x) for x in partitions]}")

    # ---- KG2 ---------------------------------------------------------------
    result = assign_kg2_to_ranks_locality(data, partitions, partition_dir)  # writes kg2_assignments.json
    kg2_stats: dict = {"mode": args.kg2_mode}
    if args.kg2_mode == "divea_constrained":
        allowed: dict[int, set[int]] = defaultdict(set)
        for r, s in enumerate(kg2sets):
            for e in s:
                allowed[e].add(r)
        seeded = {k2 for _, k2 in data.load_train_alignment()}
        kg2_rank = {e: r for r, lst in result.items() for e in lst}
        kg2_load = [len(result[r]) for r in range(n)]
        kg2_adj = build_kg2_adjacency(data.kg2_triples)
        moved = seeded_viol = 0
        for e in sorted(kg2_all):
            ok = allowed.get(e)
            if not ok or kg2_rank[e] in ok:
                continue
            if e in seeded:
                seeded_viol += 1  # keep train-pair co-location; count only
                continue
            votes = Counter(kg2_rank[nb] for nb in kg2_adj.get(e, ()) if nb in kg2_rank and kg2_rank[nb] in ok)
            if votes:
                top = max(votes.values())
                r = min([x for x, v in votes.items() if v == top], key=lambda x: (kg2_load[x], x))
            else:
                r = min(ok, key=lambda x: (kg2_load[x], x))
            kg2_load[kg2_rank[e]] -= 1
            kg2_rank[e] = r
            kg2_load[r] += 1
            moved += 1
        result = {r: sorted(e for e, rr in kg2_rank.items() if rr == r) for r in range(n)}
        _write_kg2_assignments(result, partition_dir)
        still = sum(1 for e in kg2_all if allowed.get(e) and kg2_rank[e] not in allowed[e])
        kg2_stats.update({
            "uncovered_by_divea": sum(1 for e in kg2_all if e not in allowed),
            "moved_to_satisfy_divea_membership": moved,
            "train_seeded_left_outside_divea_membership": seeded_viol,
            "remaining_violations": still,
        })
        assert still == seeded_viol, (still, seeded_viol)
    kg2_stats["rank_sizes"] = [len(result[r]) for r in range(n)]
    print(f"[kg2] {kg2_stats}")

    stats = {"kg1": {"own_exact": sum(len(s) for s in own), "ctx_single": n_ctx_single, "ctx_multi": n_ctx_multi,
                     "uncovered_by_divea": n_uncovered, "ctx_placement_agreement": ctx_agree,
                     "vote": kg1_stats, "rank_sizes": [len(x) for x in partitions]},
             "kg2": kg2_stats}
    with open(os.path.join(partition_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(partition_dir, "meta.json"), "w") as f:
        json.dump({
            "data_dir": os.path.normpath(args.data_dir),
            "fold_dir": os.path.normpath(args.fold_dir),
            "n_parts": n,
            "kg2_assignment": "locality" if args.kg2_mode == "locality" else "locality_divea_constrained",
            "kg1_edge_weight": "none",
            "kg1_partition_source": os.path.normpath(args.elsea_data_dir),
        }, f)
    print(f"[make_divea_aligned_partition] wrote {partition_dir}")


if __name__ == "__main__":
    main()
