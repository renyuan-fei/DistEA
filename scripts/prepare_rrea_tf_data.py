#!/usr/bin/env python
"""
prepare_rrea_tf_data.py -- write an OpenEA dataset + fold in the input
layout of the original TensorFlow RREA runner.

Reads:  an OpenEA dataset dir and fold (via src/dataload.py::OpenEAData).
Writes: --dst_dir in the layout of ELsEA/DivEA/RREA/utils.py::load_data, so
        the whole-graph RREA reference can be trained on exactly DistEA's
        split:

    ent_ids_1 / ent_ids_2        "{id}\t{id}"        (new id, old id; identity)
    triples_1 / triples_2        "{h}\t{r}\t{t}"     (ints; the loader adds +1
                                                     to r and reserves 0)
    ref_ent_ids_train            "{e1}\t{e2}"        train_links of the fold
    ref_ent_ids_test             "{e1}\t{e2}"        test_links of the fold
    ref_ent_ids                  train + test
    distea_ids.json              provenance: {uri: id} is DistEA's own
                                 OpenEAData id assignment (KG1 ids first,
                                 then KG2 -- contiguous, as load_data's
                                 node_size = max(id)+1 requires)

Ids are DistEA's (src/dataload.py::OpenEAData), so the exported embeddings
map back to DistEA's global ids by identity (scripts/export_rrea_tf_to_unified.py)
and are scored by scripts/unified_eval.py on the same test_links. The fold's
valid_links are NOT given to RREA (it has no validation concept and DistEA
never trains on them either); they are written to valid_links_distea for
reference only.

Usage:
    python scripts/prepare_rrea_tf_data.py \
        --data_dir data/OpenEA_dataset_v2.0/EN_DE_100K_V1 \
        --fold_dir data/OpenEA_dataset_v2.0/EN_DE_100K_V1/721_5fold/1 \
        --dst_dir projects/ELsEA/DivEA/data/rrea_tf/EN_DE_100K_V1_721_5fold_1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dataload import OpenEAData  # noqa: E402


def write_rows(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--fold_dir", required=True)
    p.add_argument("--dst_dir", required=True)
    a = p.parse_args()

    data = OpenEAData(a.data_dir, a.fold_dir, output_dir=os.path.join(a.dst_dir, "_unused"))
    n1, n2 = data.ent_num_1, data.ent_num_2
    kg1 = data.kg1_triples
    kg2 = data.kg2_triples
    train = data.load_train_alignment()
    valid = data.load_valid_alignment()
    test = data.load_test_alignment()
    assert all(0 <= h < n1 and 0 <= t < n1 for h, _, t in kg1), "KG1 ids must be [0, n1)"
    assert all(n1 <= h < n1 + n2 and n1 <= t < n1 + n2 for h, _, t in kg2), "KG2 ids must be [n1, n1+n2)"
    assert all(0 <= e1 < n1 and n1 <= e2 < n1 + n2 for e1, e2 in train + test)

    os.makedirs(a.dst_dir, exist_ok=True)
    write_rows(os.path.join(a.dst_dir, "ent_ids_1"), [(i, i) for i in range(n1)])
    write_rows(os.path.join(a.dst_dir, "ent_ids_2"), [(i, i) for i in range(n1, n1 + n2)])
    write_rows(os.path.join(a.dst_dir, "triples_1"), kg1)
    write_rows(os.path.join(a.dst_dir, "triples_2"), kg2)
    write_rows(os.path.join(a.dst_dir, "ref_ent_ids_train"), train)
    write_rows(os.path.join(a.dst_dir, "ref_ent_ids_test"), test)
    write_rows(os.path.join(a.dst_dir, "ref_ent_ids"), train + test)
    write_rows(os.path.join(a.dst_dir, "valid_links_distea"), valid)
    with open(os.path.join(a.dst_dir, "distea_ids.json"), "w", encoding="utf-8") as f:
        json.dump({"data_dir": a.data_dir, "fold_dir": a.fold_dir, "ent_num_1": n1, "ent_num_2": n2,
                   "rel_num": data.rel_num, "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
                   "ent2id": data.ent2id}, f)
    print(f"wrote {a.dst_dir}: KG1 {n1} ents / {len(kg1)} triples, KG2 {n2} / {len(kg2)}, "
          f"relations {data.rel_num}, train {len(train)} valid {len(valid)} (held out) test {len(test)}")


if __name__ == "__main__":
    main()
