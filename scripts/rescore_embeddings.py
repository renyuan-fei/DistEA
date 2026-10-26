#!/usr/bin/env python
"""
rescore_embeddings.py -- re-score an exported embedding bundle under
alternative inference protocols.

Reads:  an export bundle (--export_dir of a DistEA run), optionally the
        run's metrics.json and a partition directory.
Writes: rescore.json (default EXPORT_DIR/rescore.json).

Zero training cost; the parity variant must reproduce the run's
metrics.json before any other number is read.

Usage:
    python scripts/rescore_embeddings.py EXPORT_DIR \
        --variants cosine_csls,l1_csls,bidir,sinkhorn,fuse_sum,fuse_rrf \
        --partition_dir outputs/partitions/DBP1M_DE/316_1fold_1/dist_True/16_metis_kg2locality \
        --metrics_json outputs/results/DBP1M_DE/<run>/metrics.json \
        --device cuda --out rescore.json

Variants: cosine_csls (parity, the recorded protocol) | cosine | l1_csls |
bidir (sparse bidirectional top-K CSLS) | sinkhorn (label-free block-wise)
| fuse_sum | fuse_rrf. Blocks for sinkhorn come from --partition_dir
(DistEA partition files) or --kmeans_blocks N; never from test labels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rescore import load_bundle, rescore  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export_dir")
    p.add_argument("--variants", default="cosine_csls,bidir,sinkhorn,fuse_sum,fuse_rrf")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--csls_k", type=int, default=10)
    p.add_argument("--partition_dir", default=None)
    p.add_argument("--kmeans_blocks", type=int, default=0)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--sinkhorn_iters", type=int, default=100)
    p.add_argument("--max_block_elems", type=int, default=33000 * 33000)
    p.add_argument("--metrics_json", default=None, help="the run's metrics.json for the parity check")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="rescore.json path (default: EXPORT_DIR/rescore.json)")
    p.add_argument("--require_parity", action="store_true",
                   help="exit 2 when the cosine_csls parity check fails")
    a = p.parse_args()

    device = torch.device(a.device)
    b = load_bundle(a.export_dir)
    online = None
    if a.metrics_json and os.path.exists(a.metrics_json):
        with open(a.metrics_json) as f:
            m = json.load(f)
        online = m.get("final") and {
            "Hit@1": m["final"].get("hit1"), "Hit@5": m["final"].get("hit5"),
            "Hit@10": m["final"].get("hit10"), "Hit@50": m["final"].get("hit50"),
            "MRR": m["final"].get("mrr"),
        }
        online = {k: v for k, v in (online or {}).items() if v is not None} or None
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    res = rescore(
        b, variants, device, K=a.topk, csls_k=a.csls_k, partition_dir=a.partition_dir,
        kmeans_blocks=a.kmeans_blocks, tau=a.tau, sinkhorn_iters=a.sinkhorn_iters,
        online_metrics=online, max_block_elems=a.max_block_elems,
    )
    res["export_dir"] = a.export_dir
    res["meta"] = b.meta
    out = a.out or os.path.join(a.export_dir, "rescore.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    for name, d in res["variants"].items():
        keys = [k for k in ("Hit@1", "Hit@5", "Hit@10", "Hit@50", "MRR", "MRR@K", "block_recall", "runtime_s") if k in d]
        print(name + ": " + ", ".join(f"{k}={d[k]:.4f}" if isinstance(d[k], float) else f"{k}={d[k]}" for k in keys)
              + (f", parity={'OK' if d['parity']['ok'] else 'FAIL'}" if "parity" in d else ""))
    print(f"wrote {out}")
    par = res["variants"].get("cosine_csls", {}).get("parity")
    if a.require_parity and par is not None and not par["ok"]:
        print("PARITY FAILED:", par["diffs"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
