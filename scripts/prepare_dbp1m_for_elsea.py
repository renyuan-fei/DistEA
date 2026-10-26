"""
prepare_dbp1m_for_elsea.py -- convert a URI-based OpenEA-layout dataset
(e.g. DBP1M_DE) directly into ELsEA/DivEA's UniformData directory layout.

Reads:  <data_dir>/rel_triples_1/2 (URI triples) and <fold_dir>/train_links,
        test_links (URI pairs).
Writes: --dst_dir/{kgid}_entity_id2uri.txt, {kgid}_relation_id2uri.txt,
        {kgid}_triple_rel.txt, train_alignment.txt, test_alignment.txt,
        alignment_of_entity.txt.

Unlike scripts/export_to_divea.py (which starts from an already-integerized
OpenEA-raw export -- ent_ids_1/2 + triples_1/2 + ref_ent_ids_train/test),
DBP1M_DE ships URIs directly
(see data/DBP1M_DE/rel_triples_1: "http://dbpedia.org/resource/... \t
http://dbpedia.org/ontology/... \t http://dbpedia.org/resource/..."). This
script fuses two existing pieces of logic instead of re-deriving them:

1. src/dataload.py's OpenEAData._build_dict() id-assignment algorithm --
   ent2id/rel2id built in first-seen order scanning rel_triples_1 fully,
   then rel_triples_2 (SHARED counters across both KGs, matching that
   class exactly) -- replicated here so the ELsEA run resolves
   entities/relations the same way DistEA's own training does, i.e. both
   systems solve the same underlying alignment problem.
2. scripts/export_to_divea.py's write_entity_id2uri/write_triple_rel
   output conventions (tab-delimited, (head,rel,tail) order, identity
   id2uri map) -- copied here rather than imported, since export_to_divea
   .py's functions take already-built int lists/tuples, not raw URI
   lines; duplicating the ~15-line writers was simpler than reshaping
   that module's CLI-oriented entrypoint to expose them for reuse.

Skips valid_links (DivEA's UniformData has no validation-split concept).

Usage:
    python scripts/prepare_dbp1m_for_elsea.py \
        --data_dir data/DBP1M_DE \
        --fold_dir data/DBP1M_DE/316_1fold/1 \
        --dst_dir projects/ELsEA/DivEA/data/1m/en_de \
        --kgid1 en --kgid2 de
"""

from __future__ import annotations

import argparse
import os


def read_tab_lines(path: str) -> list[list[str]]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").split("\t") for line in f if line.strip()]


def write_tab_pairs(path: str, pairs: list[tuple[int, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(f"{a}\t{b}\n")


def write_entity_id2uri(path: str, ids: list[int]) -> None:
    write_tab_pairs(path, [(i, i) for i in ids])


def write_triple_rel(path: str, triples: list[tuple[int, int, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="DBP1M_DE dir with rel_triples_1/rel_triples_2")
    p.add_argument("--fold_dir", required=True, help="fold dir with train_links/test_links (e.g. 316_1fold/1)")
    p.add_argument("--dst_dir", required=True, help="Output UniformData directory (DivEA/data/1m/en_de)")
    p.add_argument("--kgid1", default="en")
    p.add_argument("--kgid2", default="de")
    args = p.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    rel_triples_1 = read_tab_lines(os.path.join(args.data_dir, "rel_triples_1"))
    rel_triples_2 = read_tab_lines(os.path.join(args.data_dir, "rel_triples_2"))
    train_links = read_tab_lines(os.path.join(args.fold_dir, "train_links"))
    test_links = read_tab_lines(os.path.join(args.fold_dir, "test_links"))

    # Replicates src/dataload.py's OpenEAData._build_dict exactly: shared
    # ent2id/rel2id counters, first-seen order, KG1 triples scanned fully
    # before KG2 triples.
    ent2id: dict[str, int] = {}
    rel2id: dict[str, int] = {}

    def get_ent_id(e: str) -> int:
        if e not in ent2id:
            ent2id[e] = len(ent2id)
        return ent2id[e]

    def get_rel_id(r: str) -> int:
        if r not in rel2id:
            rel2id[r] = len(rel2id)
        return rel2id[r]

    kg1_int_triples: list[tuple[int, int, int]] = []
    for h, r, t in rel_triples_1:
        hi, ti = get_ent_id(h), get_ent_id(t)
        ri = get_rel_id(r)
        kg1_int_triples.append((hi, ri, ti))

    kg2_int_triples: list[tuple[int, int, int]] = []
    for h, r, t in rel_triples_2:
        hi, ti = get_ent_id(h), get_ent_id(t)
        ri = get_rel_id(r)
        kg2_int_triples.append((hi, ri, ti))

    # Each KG's entity id list is derived from the entities its OWN triples
    # reference, not from numeric ranges (range(ent_num_1) /
    # range(ent_num_1, len(ent2id))). KG1/KG2 ids are NOT two contiguous
    # runs on DBP1M: some entities are referenced from both language
    # editions' triples, and get_ent_id() (like OpenEAData._build_dict's
    # first-seen-wins behaviour) returns the id such an entity already got
    # while scanning rel_triples_1, so a KG1-range id can appear in KG2's
    # triple file. DistEA's own loader never assumes a disjoint numeric
    # split; ELsEA's UniformData does (graph_metrics_main.py builds
    # kg2_entity_dict from whatever de_entity_id2uri.txt lists and raises
    # KeyError on any de_triple_rel.txt entity it does not cover). This
    # keeps the exact global ent2id / shared-id behaviour of _build_dict;
    # it only decides which ids go into which UniformData entity file.
    kg1_entity_ids = sorted({e for h, _, t in kg1_int_triples for e in (h, t)})
    kg2_entity_ids = sorted({e for h, _, t in kg2_int_triples for e in (h, t)})

    kg1_relation_ids = sorted({r for _, r, _ in kg1_int_triples})
    kg2_relation_ids = sorted({r for _, r, _ in kg2_int_triples})

    write_entity_id2uri(os.path.join(args.dst_dir, f"{args.kgid1}_entity_id2uri.txt"), kg1_entity_ids)
    write_entity_id2uri(os.path.join(args.dst_dir, f"{args.kgid1}_relation_id2uri.txt"), kg1_relation_ids)
    write_triple_rel(os.path.join(args.dst_dir, f"{args.kgid1}_triple_rel.txt"), kg1_int_triples)

    write_entity_id2uri(os.path.join(args.dst_dir, f"{args.kgid2}_entity_id2uri.txt"), kg2_entity_ids)
    write_entity_id2uri(os.path.join(args.dst_dir, f"{args.kgid2}_relation_id2uri.txt"), kg2_relation_ids)
    write_triple_rel(os.path.join(args.dst_dir, f"{args.kgid2}_triple_rel.txt"), kg2_int_triples)

    train_pairs = [(ent2id[e1], ent2id[e2]) for e1, e2 in train_links]
    test_pairs = [(ent2id[e1], ent2id[e2]) for e1, e2 in test_links]

    write_tab_pairs(os.path.join(args.dst_dir, "train_alignment.txt"), train_pairs)
    write_tab_pairs(os.path.join(args.dst_dir, "test_alignment.txt"), test_pairs)
    write_tab_pairs(os.path.join(args.dst_dir, "alignment_of_entity.txt"), train_pairs + test_pairs)

    print(f"[prepare_dbp1m_for_elsea] {args.kgid1}: {len(kg1_entity_ids)} entities, "
          f"{len(kg1_relation_ids)} relations, {len(kg1_int_triples)} triples")
    print(f"[prepare_dbp1m_for_elsea] {args.kgid2}: {len(kg2_entity_ids)} entities, "
          f"{len(kg2_relation_ids)} relations, {len(kg2_int_triples)} triples")
    print(f"[prepare_dbp1m_for_elsea] train_alignment.txt: {len(train_pairs)} pairs")
    print(f"[prepare_dbp1m_for_elsea] test_alignment.txt: {len(test_pairs)} pairs")
    print(f"[prepare_dbp1m_for_elsea] wrote UniformData layout -> {args.dst_dir}")


if __name__ == "__main__":
    main()
