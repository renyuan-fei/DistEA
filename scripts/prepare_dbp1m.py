"""
prepare_dbp1m.py -- reshape the LargeEA DBP1M release into OpenEAData's
layout.

Reads:  data/mkdata/{lang}_triples_1, {lang}_triples_2, {lang}_ent_links.
Writes: <out_dir>/rel_triples_1, rel_triples_2 (symlinks) and a filtered
        ent_links, so `OpenEAData` can load it completely unmodified.

No content transformation needed here (unlike FB-DBP2M) -- DBP1M's triples
and alignment already use raw, globally-unique URI strings for entities and
relations across both KG1 and KG2, exactly like OpenEA's own format. This
script only reorganizes file names/layout via symlinks (no duplicate disk
usage).

The raw {lang}_ent_links release is known to reference entities that were
degree-filtered out of {lang}_triples_1/2 during LargeEA's own extraction
(observed: only ~37.6% of DBP1M DE-EN's alignment pairs have both entities
present in the triples) -- OpenEAData's ent2id is built purely from the
triples scan, so an unfiltered ent_links would crash with a KeyError on the
first dangling reference. This script filters ent_links down to pairs where
both entities are confirmed present in rel_triples_1/rel_triples_2 before
writing it out, since this pruning is inherent to how the released dataset
was constructed, not a bug to route around.

Usage:
    python scripts/prepare_dbp1m.py --mkdata_dir data/mkdata \
        --lang de --out_dir data/DBP1M_DE
    python scripts/prepare_dbp1m.py --mkdata_dir data/mkdata \
        --lang fr --out_dir data/DBP1M_FR
"""

import argparse
import os


def read_triple_entities(path: str) -> set[str]:
    ents = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            h, r, t = line.split("\t")
            ents.add(h)
            ents.add(t)
    return ents


def filter_ent_links(src: str, dst: str, known_entities: set[str]) -> tuple[int, int]:
    total = 0
    kept = 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # dst may be a symlink pointing straight at src -- opening it "w"
    # without clearing it first would truncate src itself through the link.
    # Remove any pre-existing symlink/file before writing.
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            total += 1
            e1, e2 = line.split("\t")
            if e1 in known_entities and e2 in known_entities:
                fout.write(f"{e1}\t{e2}\n")
                kept += 1
    return total, kept


def main():
    p = argparse.ArgumentParser(description="Reshape DBP1M mkdata/ into OpenEAData layout")
    p.add_argument("--mkdata_dir", required=True, help="Directory containing {lang}_triples_1/2, {lang}_ent_links")
    p.add_argument("--lang", required=True, help="Language prefix used in mkdata filenames (e.g. 'de', 'fr')")
    p.add_argument("--out_dir", required=True, help="Output dataset directory (OpenEAData layout)")
    args = p.parse_args()

    triple_mapping = {
        f"{args.lang}_triples_1": "rel_triples_1",
        f"{args.lang}_triples_2": "rel_triples_2",
    }

    os.makedirs(args.out_dir, exist_ok=True)
    for src_name, dst_name in triple_mapping.items():
        src = os.path.abspath(os.path.join(args.mkdata_dir, src_name))
        dst = os.path.join(args.out_dir, dst_name)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Expected mkdata file not found: {src}")
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        print(f"  {dst} -> {src}")

    print("Scanning triples for known entities (to filter dangling ent_links refs)...")
    known_entities = read_triple_entities(os.path.join(args.mkdata_dir, f"{args.lang}_triples_1"))
    known_entities |= read_triple_entities(os.path.join(args.mkdata_dir, f"{args.lang}_triples_2"))
    print(f"  {len(known_entities)} distinct entities in triples")

    ent_links_src = os.path.join(args.mkdata_dir, f"{args.lang}_ent_links")
    ent_links_dst = os.path.join(args.out_dir, "ent_links")
    total, kept = filter_ent_links(ent_links_src, ent_links_dst, known_entities)
    print(
        f"  ent_links: {total} total, {kept} both-sides-present "
        f"({100 * kept / total:.1f}%) -> {ent_links_dst}"
    )

    print(f"Done. {args.out_dir}/ is ready for OpenEAData (rel_triples_1/2, ent_links).")


if __name__ == "__main__":
    main()
