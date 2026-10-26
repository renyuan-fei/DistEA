"""
export_to_divea.py -- convert an OpenEA-raw-format export (ent_ids_1/2,
triples_1/2, ref_ent_ids_train/test) into ELsEA/DivEA's UniformData
directory layout, so ELsEA can be run on DistEA's exact split.

Why this exists rather than using DivEA's own run_prepare_data.py: that
script's default path (--unsuper=True, forced on with no CLI way to
disable it -- see DivEA/run_prepare_data.py) generates its OWN train/test
split via surface-form-based unsupervised bootstrap (machine translation
of entity URIs). DistEA's exported ent_ids_1/2 have no real URIs (the
"uri" column is just the id repeated), so that path cannot run anyway, and
even if it could, it would replace DistEA's split with a different,
unsupervised-derived one, whereas a fair comparison requires the identical
split. This script instead copies the split straight through.

Input directory (OpenEA-raw, e.g. an ELsEA-side data_from_distea/ export):
    ent_ids_1, ent_ids_2       "{id} {id}"           (per-KG entity ids)
    triples_1, triples_2       "{head} {rel} {tail}" (OpenEA order: h r t)
    ref_ent_ids_train          "{kg1_id} {kg2_id}"   (train split)
    ref_ent_ids_test           "{kg1_id} {kg2_id}"   (test split)
All space-delimited.

Output directory (DivEA's UniformData, tab-delimited -- read_tab_lines in
DivEA/divea/dataload.py hardcodes "\t"):
    {kgid1}_entity_id2uri.txt, {kgid1}_relation_id2uri.txt,
    {kgid1}_triple_rel.txt     "{head}\t{rel}\t{tail}" (same order as OpenEA
                               triples_1/2 -- straight copy-through; see
                               the comment in write_triple_rel)
    {kgid2}_* (same, for KG2)
    alignment_of_entity.txt    union of train+test
    train_alignment.txt        copy of ref_ent_ids_train
    test_alignment.txt         copy of ref_ent_ids_test

Relation URIs: no separate relation-id file exists in the OpenEA-raw
export (only entity ids were exported), so {kgid}_relation_id2uri.txt is
synthesized as an identity map over the distinct relation ids that
actually appear in that KG's own triples file.

Usage:
    python scripts/export_to_divea.py \
        --src_dir data_from_distea/EN_DE_15K_30pct \
        --dst_dir DivEA/data/IDS15K_V1/en_de \
        --kgid1 en --kgid2 de
"""

from __future__ import annotations

import argparse
import os


def read_space_pairs(path: str) -> list[tuple[int, int]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}: expected 2 columns, got {len(parts)}: {line!r}")
            pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def read_space_triples(path: str) -> list[tuple[int, int, int]]:
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{path}: expected 3 columns, got {len(parts)}: {line!r}")
            h, r, t = (int(x) for x in parts)
            triples.append((h, r, t))
    return triples


def write_tab_pairs(path: str, pairs: list[tuple[int, int]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(f"{a}\t{b}\n")


def write_entity_id2uri(path: str, ids: list[int]) -> None:
    # No real URIs available in the OpenEA-raw export (ent_ids_* already
    # stores "{id} {id}") -- identity map is not a loss of information,
    # it's what the source data already is.
    write_tab_pairs(path, [(i, i) for i in ids])


def write_triple_rel(path: str, triples: list[tuple[int, int, int]]) -> None:
    # UniformData's {kgid}_triple_rel.txt column order is (head, relation,
    # tail) -- SAME as OpenEA's own triples_1/2, a straight copy-through.
    # load_kg()'s own unpacking (`for ent1_id, ent2_id, rel_id in
    # triple_rel_list`) LOOKS like it implies (head, tail, relation), but
    # that line does pure positional int-casting with no semantic check --
    # the variable names are cosmetic and don't constrain file layout.
    # Every actual CONSUMER of the resulting tuples (graph_metrics.py's
    # two loops, components.py's _divide_entities, and dataload.py's own
    # reset_IDs/convert_uniform_to_rrea/convert_uniform_to_openea) unpacks
    # them as (h, r, t) (every "for h, r, t in" site in divea/*.py).
    # Swapping columns to follow load_kg's naming makes graph_metrics and
    # _divide_entities fail with KeyError/IndexError.
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def convert_kg(src_dir: str, dst_dir: str, kgid: str, ent_ids_path: str, triples_path: str) -> None:
    ent_ids = read_space_pairs(os.path.join(src_dir, ent_ids_path))
    triples = read_space_triples(os.path.join(src_dir, triples_path))

    entity_ids = [e[0] for e in ent_ids]
    relation_ids = sorted({r for _, r, _ in triples})

    write_entity_id2uri(os.path.join(dst_dir, f"{kgid}_entity_id2uri.txt"), entity_ids)
    write_entity_id2uri(os.path.join(dst_dir, f"{kgid}_relation_id2uri.txt"), relation_ids)
    write_triple_rel(os.path.join(dst_dir, f"{kgid}_triple_rel.txt"), triples)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_dir", required=True, help="OpenEA-raw source directory (ent_ids_1/2, triples_1/2, ref_ent_ids_train/test)")
    p.add_argument("--dst_dir", required=True, help="Output UniformData directory")
    p.add_argument("--kgid1", default="en", help="KG1 language/source id (matches DivEA's kgids convention)")
    p.add_argument("--kgid2", default="de")
    args = p.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    convert_kg(args.src_dir, args.dst_dir, args.kgid1, "ent_ids_1", "triples_1")
    convert_kg(args.src_dir, args.dst_dir, args.kgid2, "ent_ids_2", "triples_2")

    train_pairs = read_space_pairs(os.path.join(args.src_dir, "ref_ent_ids_train"))
    test_pairs = read_space_pairs(os.path.join(args.src_dir, "ref_ent_ids_test"))

    write_tab_pairs(os.path.join(args.dst_dir, "train_alignment.txt"), train_pairs)
    write_tab_pairs(os.path.join(args.dst_dir, "test_alignment.txt"), test_pairs)
    write_tab_pairs(os.path.join(args.dst_dir, "alignment_of_entity.txt"), train_pairs + test_pairs)

    print(f"[export_to_divea] {args.kgid1}: entities + relations + triples written")
    print(f"[export_to_divea] {args.kgid2}: entities + relations + triples written")
    print(f"[export_to_divea] train_alignment.txt: {len(train_pairs)} pairs")
    print(f"[export_to_divea] test_alignment.txt: {len(test_pairs)} pairs")
    print(f"[export_to_divea] wrote UniformData layout -> {args.dst_dir}")


if __name__ == "__main__":
    main()
