#!/usr/bin/env python
"""
summarize_rank_split.py -- print the same_rank / cross_rank breakdown that
--eval_rank_split_dir adds to DistEA's metrics.json: the final test numbers
per group (Hit@1, MRR, pair count) and the per-epoch test / valid Hit@1 per
group, so a multi-node run can be compared with an N=1 run that used the same
partition's labels.

Reads:  <root>/*_<jobid>/metrics.json (results directories whose name ends
        with the job id).
Writes: nothing; prints one block per job.

Usage:
    python scripts/summarize_rank_split.py --root outputs/results/EN_DE_100K_V1 \
        --jobs <jobid1>,<jobid2>
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _r(v):
    return None if v is None else round(float(v), 4)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs/results/DBP1M_DE")
    p.add_argument("--jobs", required=True, help="comma-separated job ids (suffix of the results directory names)")
    a = p.parse_args()
    for j in [x for x in a.jobs.split(",") if x]:
        fs = sorted(glob.glob(os.path.join(a.root, f"*_{j}", "metrics.json")))
        if not fs:
            print(f"{j}: no metrics.json under {a.root}")
            continue
        m = json.load(open(fs[-1]))
        its = m.get("iterations") or [{}]
        it = its[-1]
        name = os.path.basename(os.path.dirname(fs[-1]))
        print(f"{name}: world_size={m.get('world_size')} selected_epoch={it.get('selected_epoch')} "
              f"test Hit@1 all={_r(it.get('Hit@1'))} same={_r(it.get('same_rank/Hit@1'))} "
              f"(n={it.get('same_rank/n')}) cross={_r(it.get('cross_rank/Hit@1'))} "
              f"(n={it.get('cross_rank/n')}) | MRR all={_r(it.get('MRR'))} "
              f"same={_r(it.get('same_rank/MRR'))} cross={_r(it.get('cross_rank/MRR'))} | "
              f"valid Hit@1 same={_r(it.get('valid_same_rank/Hit@1'))} cross={_r(it.get('valid_cross_rank/Hit@1'))}")
        for e in it.get("epoch_evals") or []:
            print(f"    ep{e.get('epoch')}: test all={_r(e.get('Hit@1'))} same={_r(e.get('same_rank/Hit@1'))} "
                  f"cross={_r(e.get('cross_rank/Hit@1'))} | valid all={_r(e.get('valid_Hit@1'))} "
                  f"same={_r(e.get('valid_same_rank/Hit@1'))} cross={_r(e.get('valid_cross_rank/Hit@1'))}")


if __name__ == "__main__":
    main()
