"""
split_openea.py -- create k-fold train/valid/test splits from an OpenEA ent_links file.

Reads {data_dir}/ent_links (tab-separated entity pairs) and writes:
    {data_dir}/721_5fold/{1..k}/train_links
    {data_dir}/721_5fold/{1..k}/valid_links
    {data_dir}/721_5fold/{1..k}/test_links

Default ratio: 70% train / 10% valid / 20% test  (matching OpenEA's "721" naming).

Usage:
    python scripts/split_openea.py --data_dir data/OpenEA_dataset_v2.0/EN_DE_15K_V1
    python scripts/split_openea.py --data_dir data/MyKG --train 0.8 --valid 0.1 --folds 3
"""

import argparse
import os
import random


def read_links(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Expected 2 tab-separated columns, got {len(parts)}: {line!r}")
            pairs.append((parts[0], parts[1]))
    return pairs


def write_links(path: str, pairs: list[tuple[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e1, e2 in pairs:
            f.write(f"{e1}\t{e2}\n")


def split_once(
    pairs: list[tuple[str, str]],
    train_ratio: float,
    valid_ratio: float,
    rng: random.Random,
) -> tuple[list, list, list]:
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * train_ratio)
    n_valid = round(n * valid_ratio)
    train = shuffled[:n_train]
    valid = shuffled[n_train : n_train + n_valid]
    test = shuffled[n_train + n_valid :]
    return train, valid, test


def main():
    p = argparse.ArgumentParser(description="OpenEA k-fold split generator")
    p.add_argument("--data_dir", required=True,
                   help="Dataset directory containing ent_links")
    p.add_argument("--ent_links", default="ent_links",
                   help="Alignment links filename inside data_dir (default: ent_links)")
    p.add_argument("--out_dir", default=None,
                   help="Output root directory (default: {data_dir}/721_5fold)")
    p.add_argument("--train", type=float, default=0.7, dest="train_ratio",
                   help="Training set ratio (default: 0.7)")
    p.add_argument("--valid", type=float, default=0.1, dest="valid_ratio",
                   help="Validation set ratio (default: 0.1)")
    p.add_argument("--folds", type=int, default=5,
                   help="Number of folds to generate (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed (default: 42)")
    args = p.parse_args()

    if args.train_ratio + args.valid_ratio >= 1.0:
        p.error("--train + --valid must be < 1.0")

    links_path = os.path.join(args.data_dir, args.ent_links)
    if not os.path.exists(links_path):
        p.error(f"ent_links not found: {links_path}")

    pairs = read_links(links_path)
    print(f"Loaded {len(pairs)} alignment pairs from {links_path}")

    # Derive fold directory name from ratios: e.g. 0.7/0.1/0.2 → "721"
    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    ratio_tag = (
        f"{round(args.train_ratio * 10)}"
        f"{round(args.valid_ratio * 10)}"
        f"{round(test_ratio * 10)}"
    )
    fold_name = f"{ratio_tag}_{args.folds}fold"
    out_root = args.out_dir or os.path.join(args.data_dir, fold_name)

    for fold in range(1, args.folds + 1):
        rng = random.Random(args.seed + fold)
        train, valid, test = split_once(pairs, args.train_ratio, args.valid_ratio, rng)

        fold_dir = os.path.join(out_root, str(fold))
        write_links(os.path.join(fold_dir, "train_links"), train)
        write_links(os.path.join(fold_dir, "valid_links"), valid)
        write_links(os.path.join(fold_dir, "test_links"), test)
        print(
            f"  Fold {fold}: train={len(train)}  valid={len(valid)}  test={len(test)}"
            f"  → {fold_dir}/"
        )

    print(f"Done. Splits written to {out_root}/")


if __name__ == "__main__":
    main()
