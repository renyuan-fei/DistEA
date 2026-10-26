#!/usr/bin/env python
"""
subtask_unified_eval.py -- evaluate ELsEA and DistEA (or any single-space
export) under ONE identical, per-subtask protocol, so the only thing that
differs between the numbers is the embeddings themselves.

Reads:  the ELsEA data dir (partition_*/kg_partition.json,
        {kgid}_entity_id2uri.txt), the ELsEA results dir
        (partition_*/emb.npz) and optional export bundles.
Writes: per-subtask bundles under --out_dir/{elsea,<label>}/partition_*/
        and --out_dir/summary.json.

Why per-subtask and not one merged bundle: ELsEA/DivEA trains N independent
RREA models -> N unrelated embedding spaces. Cross-subtask similarities are
meaningless, so a single merged global ranking is not even well-defined for
ELsEA (which is exactly why DivEA only ever reports per-subtask numbers).
The symmetric fix is to put DistEA (one consistent space) through the SAME
subtask-restricted protocol: for every ELsEA subtask i, build a DistEA
bundle containing exactly that subtask's test KG1 entities and its
valid-test KG2 entities, score both with scripts/unified_eval.run_eval, and
aggregate both with DivEA's own recall weighting.

Protocol details (matches divea/framework.py's aggregation exactly):
  - subtask i owns all test pairs whose KG1 side is in that subtask
    (kg_partition.json: kg1_partition.test_alignment, global ids; N_all_i)
  - of those, the "valid" pairs are the ones whose KG2 side landed in the
    subtask's KG2 candidate set (kg2_partition.test_alignment; N_valid_i);
    recall_i = N_valid_i / N_all_i. Invalid pairs are misses by
    construction (their target is not in play) and are counted in the
    denominator, never dropped.
  - ranking pool ("pool_test", the DivEA-native one): the subtask's
    valid-test KG2 entities only -- RREA's get_hits ranks each test KG1
    entity against the test pairs' KG2 side. For ELsEA an extra "pool_all"
    variant (all KG2 entities in the subtask) is also emitted for reference.
  - aggregate Hit@k = sum_i Hit@k_i * N_valid_i / N_total_test
    (== DivEA's sum_i test_percent_i * Hit@k_i * recall_i), i.e. recall-
    adjusted. "covered_only" (/ sum_i N_valid_i) is reported alongside.

Local->global id mapping: emb.npz stores rows in subtask-LOCAL id order
(row r < n1 is KG1 local id r; row r >= n1 is KG2 local id r-n1);
{kgid}_entity_id2uri.txt in the subtask data dir is the local->global map
(components_base.py writes (idx, oldid) pairs). Global ids are the same
DistEA ent2id ids the export bundles use (the test sets are identical).

Usage:
    python scripts/subtask_unified_eval.py \
        --elsea_data_dir projects/ELsEA/DivEA/data/IDS15K_V1/en_de \
        --elsea_results_dir projects/ELsEA/DivEA/results/rrea/csls/IDS15K_V1/en_de \
        --distea_export export/distea/EN_DE_15K_V1/custom_30pct_fold_1/ws5/c3 \
        --out_dir export/unified/EN_DE_15K_V1/custom_30pct_fold_1/n5 \
        --metric cosine --use_csls --csls_k 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from unified_eval import run_eval  # noqa: E402


def read_id2uri(path: str) -> dict[int, int]:
    m: dict[int, int] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            a, b = line.rstrip("\n").split("\t")
            m[int(a)] = int(b)
    return m


def write_bundle(out_dir: str, emb: np.ndarray, ids: np.ndarray, kg_labels: np.ndarray,
                 test_links: list[tuple[int, int]]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), emb.astype(np.float32))
    np.save(os.path.join(out_dir, "ids.npy"), ids.astype(np.int64))
    np.save(os.path.join(out_dir, "kg_labels.npy"), kg_labels.astype(np.int8))
    with open(os.path.join(out_dir, "test_links.json"), "w") as f:
        json.dump([[int(a), int(b)] for a, b in test_links], f)


def pick(metrics: dict, *names: str) -> float:
    low = {k.lower(): v for k, v in metrics.items()}
    for n in names:
        if n.lower() in low:
            return float(low[n.lower()])
    raise KeyError(f"none of {names} in metrics keys {list(metrics)}")


def load_subtasks(elsea_data_dir: str) -> list[dict]:
    subs = []
    for d in sorted(glob.glob(os.path.join(elsea_data_dir, "partition_*")),
                    key=lambda p: int(p.rsplit("_", 1)[1])):
        with open(os.path.join(d, "kg_partition.json")) as f:
            kp = json.load(f)
        all_test = [(int(a), int(b)) for a, b in kp["kg1_partition"]["test_alignment"]]
        valid_test = [(int(a), int(b)) for a, b in kp["kg2_partition"]["test_alignment"]]
        subs.append({
            "name": os.path.basename(d),
            "data_dir": d,
            "all_test": all_test,
            "valid_test": valid_test,
        })
    return subs


def build_elsea_bundles(sub: dict, results_dir: str, kgid1: str, kgid2: str, out_root: str) -> dict[str, str]:
    # A subtask with zero valid-test pairs (N_valid_i == 0) is one
    # divea/framework.py's own per-client loop SKIPPED entirely:
    # components_base.py's Client.generate_dataset_from_partition() sets
    # ignore=True when its reconstructed local test_alignment.txt is empty
    # (built from kg2_partition.test_alignment, i.e. exactly N_valid_i),
    # and framework.py's run() then `continue`s past train_model/
    # evaluate_model/generate_msg for that index -- no emb.npz, no
    # part_metrics.json, nothing, ever gets written for it (this happens on
    # DBP1M: a few subtasks have N_valid_i=0 but N_all_i>0 -- a handful of
    # real test pairs each, all misses by construction per this module's
    # own docstring -- and their results dirs are empty except for the
    # initial _prepare() mkdir). These are
    # NOT excluded from n_total (load_subtasks/main() already sum
    # all_test unconditionally), only from needing an embedding: write
    # genuinely empty (0-row) bundles instead of trying to load a file
    # that was never created. eval_bundles()'s run_eval() call on a 0-row
    # bundle is itself safe (src/eval.py's _csls_thr's per-row loop never
    # executes when sim.size(0)==0, and _compute_metrics returns zeros
    # for an empty test_links).
    if not sub["valid_test"]:
        empty_embs = np.zeros((0, 1), dtype=np.float32)
        empty_ids = np.zeros((0,), dtype=np.int64)
        empty_labels = np.zeros((0,), dtype=np.int8)
        out = {}
        for pool in ("pool_all", "pool_test"):
            d = os.path.join(out_root, "elsea", sub["name"], pool)
            write_bundle(d, empty_embs, empty_ids, empty_labels, [])
            out[pool] = d
        print(f"  [elsea/{sub['name']}] N_valid=0 (N_all={len(sub['all_test'])}) -- "
              f"ignored by ELsEA's own framework.py (no emb.npz ever written); "
              f"wrote empty bundles, contributes 0 to numerator, its {len(sub['all_test'])} "
              f"pair(s) still counted in the denominator via n_total")
        return out

    z = np.load(os.path.join(results_dir, sub["name"], "emb.npz"))
    embs, ent1_local, ent2_local = z["embs"], z["ent1_ids"], z["ent2_ids"]
    en_map = read_id2uri(os.path.join(sub["data_dir"], f"{kgid1}_entity_id2uri.txt"))
    de_map = read_id2uri(os.path.join(sub["data_dir"], f"{kgid2}_entity_id2uri.txt"))
    n1 = len(en_map)
    assert int(ent1_local.max()) + 1 == n1 and int(ent1_local.min()) == 0, (sub["name"], n1, ent1_local.min(), ent1_local.max())
    assert int(ent2_local.min()) == n1 and int(ent2_local.max()) - n1 + 1 == len(de_map), (sub["name"], n1, len(de_map))
    assert embs.shape[0] == n1 + len(de_map)

    ids = np.array([en_map[r] for r in range(n1)] + [de_map[r - n1] for r in range(n1, embs.shape[0])], dtype=np.int64)
    kg_labels = np.array([0] * n1 + [1] * len(de_map), dtype=np.int8)

    valid_e2 = {e2 for _, e2 in sub["valid_test"]}
    test_e1 = {e1 for e1, _ in sub["all_test"]}
    out = {}

    # pool_all: every KG2 entity ELsEA put in the subtask (reference only).
    d_all = os.path.join(out_root, "elsea", sub["name"], "pool_all")
    write_bundle(d_all, embs, ids, kg_labels, sub["valid_test"])
    out["pool_all"] = d_all

    # pool_test: rows = subtask test KG1 entities + valid-test KG2 entities
    # ONLY -- structurally identical to the DistEA bundle built below, so
    # CSLS thresholds see the same column set on both sides.
    keep = np.array([(kg_labels[i] == 0 and int(ids[i]) in test_e1) or
                     (kg_labels[i] == 1 and int(ids[i]) in valid_e2) for i in range(len(ids))])
    d_test = os.path.join(out_root, "elsea", sub["name"], "pool_test")
    write_bundle(d_test, embs[keep], ids[keep], kg_labels[keep], sub["valid_test"])
    out["pool_test"] = d_test
    return out


def build_export_bundle(sub: dict, export_dir: str, out_root: str, label: str) -> str:
    """Restrict ANY single-space export bundle (DistEA, ClusterEA, ...) to
    exactly ELsEA subtask `sub`'s test KG1 entities + valid-test KG2
    entities -- structurally identical to the ELsEA pool_test bundle."""
    # Same N_valid_i==0 case as build_elsea_bundles() above -- a single-
    # space export (unlike ELsEA's own emb.npz) always covers every
    # entity, so this would otherwise build a bundle with test_e1 KG1 rows
    # but ZERO KG2 rows, which later crashes the same way in run_eval()'s
    # CSLS pass (a 0-width dimension can't satisfy topk(k=10)).
    if not sub["valid_test"]:
        d = os.path.join(out_root, label, sub["name"], "pool_test")
        write_bundle(d, np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64),
                     np.zeros((0,), dtype=np.int8), [])
        print(f"  [{label}/{sub['name']}] N_valid=0 (N_all={len(sub['all_test'])}) -- wrote empty bundle")
        return d

    emb = np.load(os.path.join(export_dir, "embeddings.npy"))
    ids = np.load(os.path.join(export_dir, "ids.npy"))
    kg = np.load(os.path.join(export_dir, "kg_labels.npy"))
    row_of = {int(g): i for i, g in enumerate(ids)}
    test_e1 = {e1 for e1, _ in sub["all_test"]}
    valid_e2 = {e2 for _, e2 in sub["valid_test"]}
    rows = [row_of[e] for e in sorted(test_e1) if e in row_of] + [row_of[e] for e in sorted(valid_e2) if e in row_of]
    missing = sum(1 for e in test_e1 if e not in row_of) + sum(1 for e in valid_e2 if e not in row_of)
    if missing:
        print(f"  [{label}/{sub['name']}] WARNING: {missing} subtask entities absent from export (skipped)")
    rows = np.array(rows, dtype=np.int64)
    d = os.path.join(out_root, label, sub["name"], "pool_test")
    write_bundle(d, emb[rows], ids[rows], kg[rows], sub["valid_test"])
    return d


def aggregate(per: list[dict], n_total: int) -> dict[str, float]:
    n_cov = sum(p["n_scored"] for p in per)
    agg: dict[str, float] = {"n_total_test": float(n_total), "n_covered": float(n_cov),
                             "recall": n_cov / n_total}
    for key in ("hit@1", "hit@5", "hit@10", "hit@50", "mrr"):
        s = sum(p[key] * p["n_scored"] for p in per)
        agg[f"{key}_recall_adjusted"] = s / n_total
        agg[f"{key}_covered_only"] = s / n_cov if n_cov else 0.0
    return agg


def eval_bundles(label: str, dirs: list[tuple[str, str]], subs: list[dict], n_total: int,
                 metric: str, use_csls: bool, csls_k: int) -> dict:
    per = []
    for (name, d), sub in zip(dirs, subs, strict=True):
        m = run_eval(d, metric=metric, use_csls=use_csls, csls_k=csls_k)
        rec = {
            "subtask": name,
            "n_all": len(sub["all_test"]),
            "n_valid": len(sub["valid_test"]),
            "n_scored": int(m["n_scored"]),
            "hit@1": pick(m, "hit@1", "hits@1"),
            "hit@5": pick(m, "hit@5", "hits@5"),
            "hit@10": pick(m, "hit@10", "hits@10"),
            "hit@50": pick(m, "hit@50", "hits@50"),
            "mrr": pick(m, "mrr"),
        }
        per.append(rec)
        print(f"  [{label}/{name}] n_all={rec['n_all']} n_valid={rec['n_valid']} scored={rec['n_scored']} "
              f"hit@1={rec['hit@1']:.4f} hit@10={rec['hit@10']:.4f} mrr={rec['mrr']:.4f}")
    agg = aggregate(per, n_total)
    print(f"  [{label}] AGGREGATE recall={agg['recall']:.4f} "
          f"hit@1 recall-adjusted={agg['hit@1_recall_adjusted']:.4f} covered-only={agg['hit@1_covered_only']:.4f} "
          f"mrr recall-adjusted={agg['mrr_recall_adjusted']:.4f}")
    return {"per_subtask": per, "aggregate": agg}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--elsea_data_dir", required=True)
    p.add_argument("--elsea_results_dir", required=True)
    p.add_argument("--export", action="append", default=[], metavar="LABEL=PATH",
                   help="any single-space export bundle to score under the same subtask protocol "
                        "(repeatable), e.g. --export distea_ws5=export/distea/.../ws5/c3 "
                        "--export clusterea_n1=export/clusterea/.../n1/dual-amn")
    p.add_argument("--distea_export", default=None, help="shorthand for --export distea=PATH")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--kgid1", default="en")
    p.add_argument("--kgid2", default="de")
    p.add_argument("--metric", choices=["l1", "cosine"], default="cosine")
    p.add_argument("--use_csls", action="store_true")
    p.add_argument("--csls_k", type=int, default=10)
    args = p.parse_args()

    subs = load_subtasks(args.elsea_data_dir)
    n_total = sum(len(s["all_test"]) for s in subs)
    n_valid = sum(len(s["valid_test"]) for s in subs)
    print(f"[subtask_unified_eval] {len(subs)} subtasks, total test pairs={n_total}, valid={n_valid} "
          f"(recall={n_valid / n_total:.4f}); protocol metric={args.metric} csls={args.use_csls} k={args.csls_k}")

    results: dict = {"protocol": {"metric": args.metric, "use_csls": args.use_csls, "csls_k": args.csls_k},
                     "n_total_test": n_total}

    elsea_dirs = [build_elsea_bundles(s, args.elsea_results_dir, args.kgid1, args.kgid2, args.out_dir) for s in subs]
    results["elsea_pool_test"] = eval_bundles(
        "elsea/pool_test", [(s["name"], d["pool_test"]) for s, d in zip(subs, elsea_dirs, strict=True)],
        subs, n_total, args.metric, args.use_csls, args.csls_k)
    results["elsea_pool_all"] = eval_bundles(
        "elsea/pool_all", [(s["name"], d["pool_all"]) for s, d in zip(subs, elsea_dirs, strict=True)],
        subs, n_total, args.metric, args.use_csls, args.csls_k)

    exports = [("distea", args.distea_export)] if args.distea_export else []
    for spec in args.export:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--export expects LABEL=PATH, got {spec!r}")
        exports.append((label, path))

    for label, export_dir in exports:
        dirs = [(s["name"], build_export_bundle(s, export_dir, args.out_dir, label)) for s in subs]
        results[f"{label}_pool_test"] = eval_bundles(
            f"{label}/pool_test", dirs, subs, n_total, args.metric, args.use_csls, args.csls_k)
        g = run_eval(export_dir, metric=args.metric, use_csls=args.use_csls, csls_k=args.csls_k)
        results[f"{label}_global"] = g
        print(f"  [{label}/global] hit@1={pick(g, 'hit@1', 'hits@1'):.4f} mrr={pick(g, 'mrr'):.4f} "
              f"(full {int(g['n_test_links'])}-pair global ranking, single-space native protocol)")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[subtask_unified_eval] wrote {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
