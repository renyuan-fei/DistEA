#!/usr/bin/env python
"""
prepare_for_lightea.py -- convert an OpenEA-style dataset + one of our folds
into LightEA's input format, using DistEA's own
global entity ids so LightEA's exported features line up with every other
method's unified-eval bundle.

Input (URI based, as DistEA reads it):
    <data_dir>/rel_triples_1, rel_triples_2     h \\t r \\t t  (URIs)
    <fold_dir>/train_links, test_links           e1 \\t e2      (URIs)
    --ent2id_json  DistEA partition's ent2id.json {uri: global_id}, KG1 ids
                   first then KG2 (any partition count; ids don't depend on N)

Output (LightEA's EA_datasets/<name>/ layout, integer ids):
    triples_1, triples_2   h \\t r \\t t   entity ids = DistEA global ids;
                           KG1 relations 0..R1-1, KG2 relations R1..R1+R2-1
    ent_ids_1, ent_ids_2   id \\t uri
    train_pairs, test_pairs  e1 \\t e2    our fold, read by run_lightea.py
                             (LightEA's own load_aligned_pair would randomly
                             re-split 30/70 -- the launcher replaces it)
    ref_ent_ids            train + test, for format compatibility only
    meta.json              counts + provenance

valid_links is not used: LightEA has no validation-based selection, and the
ClusterEA runs on the same folds (run_clusterea.py fold override) also train
on train_links and test on test_links only.

Usage:
    python scripts/prepare_for_lightea.py \\
        --data_dir data/OpenEA_dataset_v2.0/EN_DE_15K_V1 \\
        --fold_dir data/OpenEA_dataset_v2.0/EN_DE_15K_V1/custom_30pct_fold/1 \\
        --ent2id_json outputs/partitions/EN_DE_15K_V1/custom_30pct_fold_1/dist_True/1_metis_kg2locality/ent2id.json \\
        --out_dir projects/LightEA/EA_datasets/EN_DE_15K_V1_custom30_1
"""

from __future__ import annotations

import argparse
import json
import os


def read_tsv(path: str, ncol: int):
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == ncol:
                yield parts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--fold_dir", required=True)
    p.add_argument("--ent2id_json", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    with open(args.ent2id_json) as f:
        ent2id = json.load(f)

    kg_ents: list[set[str]] = [set(), set()]
    triples: list[list[tuple[str, str, str]]] = [[], []]
    rels: list[dict[str, int]] = [{}, {}]
    for k in (0, 1):
        for h, r, t in read_tsv(os.path.join(args.data_dir, f"rel_triples_{k + 1}"), 3):
            triples[k].append((h, r, t))
            kg_ents[k].update((h, t))
            rels[k].setdefault(r, len(rels[k]))

    # KG1-only ids must all precede KG2-only ids (DistEA's ent2id convention);
    # LightEA only needs ids unique across both KGs, but the export side
    # labels test rows by kg2_id_min. A URI present in both KGs (DBP1M_DE: 16
    # website literals, in no train/test link) keeps its single DistEA id --
    # one node in both graphs, as in src/dataload.py and the ELsEA export.
    missing = [u for k in (0, 1) for u in kg_ents[k] if u not in ent2id]
    if missing:
        raise SystemExit(f"{len(missing)} triple entities absent from ent2id (e.g. {missing[:3]})")
    shared = kg_ents[0] & kg_ents[1]
    max_kg1 = max(ent2id[u] for u in kg_ents[0] - shared)
    min_kg2 = min(ent2id[u] for u in kg_ents[1] - shared)
    if max_kg1 >= min_kg2:
        raise SystemExit(f"KG1/KG2 id ranges overlap (max KG1 {max_kg1} >= min KG2 {min_kg2})")

    os.makedirs(args.out_dir, exist_ok=True)
    r_off = [0, len(rels[0])]
    for k in (0, 1):
        with open(os.path.join(args.out_dir, f"triples_{k + 1}"), "w") as f:
            for h, r, t in triples[k]:
                f.write(f"{ent2id[h]}\t{rels[k][r] + r_off[k]}\t{ent2id[t]}\n")
        with open(os.path.join(args.out_dir, f"ent_ids_{k + 1}"), "w", encoding="utf-8") as f:
            for u in sorted(kg_ents[k], key=lambda u: ent2id[u]):
                f.write(f"{ent2id[u]}\t{u}\n")

    counts = {}
    all_pairs = []
    for split in ("train", "test"):
        pairs, skipped = [], 0
        for e1, e2 in read_tsv(os.path.join(args.fold_dir, f"{split}_links"), 2):
            if (e1 in ent2id and e2 in ent2id and e1 in kg_ents[0] and e2 in kg_ents[1]
                    and e1 not in shared and e2 not in shared):
                pairs.append((ent2id[e1], ent2id[e2]))
            else:
                skipped += 1
        with open(os.path.join(args.out_dir, f"{split}_pairs"), "w") as f:
            for a, b in pairs:
                f.write(f"{a}\t{b}\n")
        counts[split] = len(pairs)
        counts[f"{split}_skipped"] = skipped
        all_pairs += pairs
    with open(os.path.join(args.out_dir, "ref_ent_ids"), "w") as f:
        for a, b in all_pairs:
            f.write(f"{a}\t{b}\n")

    meta = {
        "data_dir": os.path.abspath(args.data_dir),
        "fold_dir": os.path.abspath(args.fold_dir),
        "ent2id_json": os.path.abspath(args.ent2id_json),
        "n_ent_kg1": len(kg_ents[0]), "n_ent_kg2": len(kg_ents[1]), "n_shared": len(shared),
        "kg1_id_max": max_kg1, "kg2_id_min": min_kg2,
        "n_rel_kg1": len(rels[0]), "n_rel_kg2": len(rels[1]),
        "n_triples_kg1": len(triples[0]), "n_triples_kg2": len(triples[1]),
        **counts,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prepare_for_lightea] wrote {args.out_dir}: {json.dumps(meta)}")


if __name__ == "__main__":
    main()
