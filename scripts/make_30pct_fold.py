#!/usr/bin/env python3
"""Make a custom_30pct_fold/1 split (30% train / 10% valid / 60% test) for an
OpenEA dataset (a seeded permutation of ent_links). The EN-DE-15K split of the
thesis has the same proportions but was drawn by a different shuffle.

Reads <data_dir>/ent_links; writes
<data_dir>/custom_30pct_fold/1/{train,valid,test}_links (refuses to
overwrite an existing fold). Lines are copied verbatim from ent_links.

Usage:
    python scripts/make_30pct_fold.py --data_dir data/OpenEA_dataset_v2.0/EN_DE_100K_V1
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--fold_name", default="custom_30pct_fold/1")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--train_frac", type=float, default=0.30)
    p.add_argument("--valid_frac", type=float, default=0.10)
    args = p.parse_args()

    src = os.path.join(args.data_dir, "ent_links")
    out_dir = os.path.join(args.data_dir, args.fold_name)
    if os.path.exists(out_dir):
        print(f"refusing to overwrite existing fold dir {out_dir}", file=sys.stderr)
        return 2
    with open(src, encoding="utf-8") as f:
        links = [ln for ln in f.read().splitlines() if ln.strip()]
    n = len(links)
    rng = random.Random(args.seed)
    rng.shuffle(links)
    n_train = int(round(n * args.train_frac))
    n_valid = int(round(n * args.valid_frac))
    splits = {
        "train_links": links[:n_train],
        "valid_links": links[n_train:n_train + n_valid],
        "test_links": links[n_train + n_valid:],
    }
    os.makedirs(out_dir)
    for name, rows in splits.items():
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        digest = hashlib.md5("\n".join(rows).encode("utf-8")).hexdigest()
        print(f"{path}: {len(rows)} links (md5 {digest})")
    print(f"total {n} = {n_train} train + {n_valid} valid + {n - n_train - n_valid} test, seed {args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
