"""
prepare_fb_dbp2m.py -- reshape the FB-DBP2M release into OpenEAData's layout.

Reads:  data/datasets/2m/fb_dbp/{dbp,fb}_triple_rel.txt, alignment_of_entity.txt.
Writes: <out_dir>/rel_triples_1, rel_triples_2, ent_links.

Unlike DBP1M, FB-DBP2M's triples/alignment use integer entity/relation IDs
that are independently 0-indexed *within each KG* (dbp_triple_rel.txt and
fb_triple_rel.txt both start numbering at 0) -- feeding them to OpenEAData
unmodified would silently collide KG1 entity/relation N with KG2 entity/
relation N into the same internal ID. This script prefixes every ID with
its KG's namespace ("dbp_"/"fb_" for entities, "dbpr_"/"fbr_" for relations)
before writing OpenEAData-format triples/links, so IDs are globally unique
strings exactly like DBP1M's/OpenEA's own URI-based entities.

Like DBP1M (see prepare_dbp1m.py), alignment_of_entity.txt references some
entities that were filtered out of {dbp,fb}_triple_rel.txt during LargeEA's
own extraction -- this script drops any alignment pair where either side is
absent from the corresponding triples' entity vocabulary, for the same
reason: OpenEAData's ent2id is built purely from the triples scan.

alignment_of_entity.txt is "<fb id>\t<dbp id>" -- Freebase FIRST. Reading
it as "<dbp id>\t<fb id>" pairs every DBpedia entity with an unrelated
Freebase one (of the KG2 edges whose two ends are both aligned, none then
maps onto a KG1 edge; read Freebase-first, 56% do, vs 45% on DBP1M). The
ent_links written here keep KG1 (dbp) in the first column.

Usage:
    python scripts/prepare_fb_dbp2m.py \
        --src_dir data/datasets/2m/fb_dbp --out_dir data/FB_DBP2M_fix
    # --links_only: write ent_links only (rel_triples_1/2 already in out_dir,
    # e.g. symlinked to an earlier conversion)
"""

import argparse
import os


def load_triples(path: str, ent_prefix: str, rel_prefix: str) -> list[tuple[str, str, str]]:
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            triples.append((f"{ent_prefix}{h}", f"{rel_prefix}{r}", f"{ent_prefix}{t}"))
    return triples


def write_triples(path: str, triples: list[tuple[str, str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def load_alignment(
    path: str, fb_known: set[str], dbp_known: set[str],
) -> list[tuple[str, str]]:
    """(dbp, fb) pairs from alignment_of_entity.txt, whose lines are
    "<fb id>\t<dbp id>" (see the module docstring)."""
    pairs = []
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            fb, dbp = line.split("\t")
            fb_id, dbp_id = f"fb_e{fb}", f"dbp_e{dbp}"
            if fb_id in fb_known and dbp_id in dbp_known:
                pairs.append((dbp_id, fb_id))
    print(f"  ent_links: {total} total, {len(pairs)} both-sides-present ({100 * len(pairs) / total:.1f}%)")
    return pairs


def triple_entities(triples: list[tuple[str, str, str]]) -> set[str]:
    ents = set()
    for h, _, t in triples:
        ents.add(h)
        ents.add(t)
    return ents


def write_links(path: str, pairs: list[tuple[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e1, e2 in pairs:
            f.write(f"{e1}\t{e2}\n")


def main():
    p = argparse.ArgumentParser(description="Reshape FB-DBP2M into OpenEAData layout")
    p.add_argument("--src_dir", required=True,
                   help="Directory containing dbp_triple_rel.txt, fb_triple_rel.txt, alignment_of_entity.txt")
    p.add_argument("--out_dir", required=True, help="Output dataset directory (OpenEAData layout)")
    p.add_argument("--links_only", action="store_true",
                   help="Write ent_links only; rel_triples_1/2 are left as they are in out_dir")
    args = p.parse_args()

    print("Loading + prefixing KG1 (dbp) triples...")
    kg1 = load_triples(os.path.join(args.src_dir, "dbp_triple_rel.txt"), "dbp_e", "dbpr_")
    print(f"  {len(kg1)} triples")

    print("Loading + prefixing KG2 (fb) triples...")
    kg2 = load_triples(os.path.join(args.src_dir, "fb_triple_rel.txt"), "fb_e", "fbr_")
    print(f"  {len(kg2)} triples")

    print("Loading + prefixing alignment (filtering dangling references)...")
    kg1_ents = triple_entities(kg1)
    kg2_ents = triple_entities(kg2)
    links = load_alignment(
        os.path.join(args.src_dir, "alignment_of_entity.txt"), kg2_ents, kg1_ents,
    )

    if not args.links_only:
        write_triples(os.path.join(args.out_dir, "rel_triples_1"), kg1)
        write_triples(os.path.join(args.out_dir, "rel_triples_2"), kg2)
    write_links(os.path.join(args.out_dir, "ent_links"), links)

    print(f"Done. {args.out_dir}/ is ready for OpenEAData (rel_triples_1/2, ent_links).")


if __name__ == "__main__":
    main()
