"""
show_results.py -- pretty-print saved training metrics.

Reads:  outputs/results/{dataset}/{YYYYMMDD_HHMMSS}/metrics.json (or
        alignment.csv) written by main.py.
Writes: nothing; prints to stdout.

By default shows the LATEST run per dataset. Each run lives in a timestamped
subdirectory: outputs/results/{dataset}/{YYYYMMDD_HHMMSS}/

Usage:
    python scripts/show_results.py                            # latest run per dataset
    python scripts/show_results.py --all                      # all runs per dataset
    python scripts/show_results.py --run 20260610_213200      # specific timestamp
    python scripts/show_results.py outputs/results/EN_DE_15K_V1/20260610_213200/metrics.json
    python scripts/show_results.py outputs/results/EN_DE_15K_V1/20260610_213200/alignment.csv --top 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

_TS_RE = re.compile(r"^\d{8}_\d{6}$")


def is_run_dir(name: str) -> bool:
    return bool(_TS_RE.match(name))


def list_runs(dataset_dir: str) -> list[str]:
    """Return all timestamped run directories under dataset_dir, sorted ascending."""
    if not os.path.isdir(dataset_dir):
        return []
    return sorted(
        os.path.join(dataset_dir, n)
        for n in os.listdir(dataset_dir)
        if is_run_dir(n) and os.path.isdir(os.path.join(dataset_dir, n))
    )


def show_metrics(path: str) -> None:
    with open(path) as f:
        d = json.load(f)

    iters = d.get("iterations", [])
    best = d.get("best", {})

    print(f"\n{'─'*60}")
    print(f"  Dataset    : {d.get('dataset', '?')}")
    print(f"  Run        : {d.get('run_timestamp', '?')}")
    print(f"  Fold       : {d.get('fold', '?')}")
    print(f"  World size : {d.get('world_size', '?')} GPU(s)")
    print(f"  hidden_dim : {d.get('hidden_dim')}  output_dim : {d.get('output_dim')}")
    print(f"  Epochs/iter: {d.get('max_epochs')}  Iterations : {d.get('max_iterations')}")
    print(f"  Pseudo-seeds: {d.get('use_pseudo_seeds', False)}  "
          f"Warm-start: {d.get('warm_start', True)}  "
          f"Eval-on-valid: {d.get('eval_on_valid', False)}")
    print(f"  Final train_links: {d.get('final_train_links', '?')}")
    print(f"  Saved at   : {d.get('timestamp', '?')}")

    mech = d.get("mechanisms")
    if mech:
        print(f"  Mechanisms : MM={mech.get('use_mm', False)}"
              f" (interval={mech.get('mm_sync_interval', '?')})"
              f"  OCD={mech.get('use_ocd', False)}"
              f"  SACA={mech.get('use_saca', False)}"
              f" (threshold={mech.get('saca_threshold', '?')})")
    total_wall = d.get("total_wall_time_s")
    if total_wall is not None:
        print(f"  Total wall time: {total_wall:.2f}s")

    print(f"{'─'*60}")
    header = f"  {'Iter':>4}  {'Hit@1':>7}  {'Hit@10':>7}  {'MRR':>7}"
    has_comm = any(
        "iter_wall_time_s" in m or "comm_bytes" in m or "comm_calls" in m for m in iters
    )
    if has_comm:
        header += f"  {'Wall(s)':>8}  {'Comm(B)':>10}  {'Calls':>6}"
    has_saca = any("saca_total" in m for m in iters)
    if has_saca:
        header += f"  {'SACA skip':>10}"
    print(header)
    print(f"  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}" + ("  " + "─"*8 + "  " + "─"*10 + "  " + "─"*6 if has_comm else "") + ("  " + "─"*10 if has_saca else ""))

    for m in iters:
        tag = " ◀ best" if m.get("Hit@1") == best.get("Hit@1") else ""
        row = (f"  {m['iteration']:>4}  {m.get('Hit@1', 0):.4f}  "
               f"{m.get('Hit@10', 0):.4f}  {m.get('MRR', 0):.4f}")
        if has_comm:
            wall = m.get("iter_wall_time_s")
            wall_s = f"{wall:.2f}" if wall is not None else "-"
            row += f"  {wall_s:>8}  {m.get('comm_bytes', '-'):>10}  {m.get('comm_calls', '-'):>6}"
        if has_saca:
            if "saca_total" in m:
                row += f"  {m.get('saca_skipped', 0):>4}/{m.get('saca_total', 0):<5}"
            else:
                row += f"  {'-':>10}"
        print(row + tag)

    print(f"{'─'*60}")
    print(f"  Best  Hit@1={best.get('Hit@1', 0):.4f}  "
          f"Hit@10={best.get('Hit@10', 0):.4f}  MRR={best.get('MRR', 0):.4f}")

    # Show companion files in the same directory
    run_dir = os.path.dirname(os.path.abspath(path))
    extras = []
    for fname in ("train.log", "alignment.csv", "model.pt"):
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            extras.append(f"{fname} ({size/1024:.1f} KB)")
    if extras:
        print(f"  Files in {run_dir}/:")
        for x in extras:
            print(f"    • {x}")
    print()


def show_alignment(path: str, top: int) -> None:
    with open(path) as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    correct = sum(1 for r in rows if r["correct"].lower() == "true")
    hit1 = correct / total if total else 0.0

    print(f"\n{'─'*72}")
    print(f"  Alignment predictions: {path}")
    print(f"  Total test pairs : {total}   Correct (Hit@1) : {correct}  ({hit1:.1%})")
    print(f"{'─'*72}")
    print(f"  {'KG1':>10}  {'Predicted KG2':>14}  {'True KG2':>10}  "
          f"{'Rank':>5}  {'Score':>7}  Correct")
    print(f"  {'─'*10}  {'─'*14}  {'─'*10}  {'─'*5}  {'─'*7}  {'─'*7}")

    for r in rows[:top]:
        mark = "✓" if r["correct"].lower() == "true" else "✗"
        print(f"  {r['kg1_id']:>10}  {r['predicted_kg2']:>14}  {r['true_kg2']:>10}  "
              f"  {r['rank']:>4}  {r['score']:>7}  {mark}")

    if len(rows) > top:
        print(f"  … ({len(rows) - top} more rows, use --top N to see more)")
    print()


def resolve_metrics_path(path: str, run: str | None, show_all: bool) -> list[str]:
    """Resolve a results path to a list of metrics.json files to display."""
    # Direct file path
    if os.path.isfile(path) and path.endswith(".json"):
        return [path]

    if not os.path.isdir(path):
        return []

    # Path is a dataset directory: e.g. outputs/results/EN_DE_15K_V1
    runs = list_runs(path)
    if runs:
        if run is not None:
            wanted = [r for r in runs if os.path.basename(r) == run]
            return [os.path.join(r, "metrics.json") for r in wanted]
        if show_all:
            return [os.path.join(r, "metrics.json") for r in runs]
        # Default: latest only
        return [os.path.join(runs[-1], "metrics.json")]

    # Path is outputs/results — scan each subdirectory as a dataset
    out = []
    for name in sorted(os.listdir(path)):
        sub = os.path.join(path, name)
        if not os.path.isdir(sub):
            continue
        out.extend(resolve_metrics_path(sub, run, show_all))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Show DistEA alignment results")
    p.add_argument("path", nargs="?", default="outputs/results",
                   help="Path to metrics.json, alignment.csv, dataset dir, or results root")
    p.add_argument("--top", type=int, default=30,
                   help="Number of alignment rows to display (default: 30)")
    p.add_argument("--all", action="store_true", default=False,
                   help="Show all runs, not just the latest")
    p.add_argument("--run", type=str, default=None,
                   help="Specific run timestamp (e.g. 20260610_213200)")
    p.add_argument("--alignment", action="store_true", default=False,
                   help="When path is a directory, show alignment.csv from latest run")
    args = p.parse_args()

    if not os.path.exists(args.path):
        sys.exit(f"Not found: {args.path}")

    # Direct alignment.csv path
    if os.path.isfile(args.path) and args.path.endswith("alignment.csv"):
        show_alignment(args.path, args.top)
        return

    # When --alignment is requested on a directory, find alignment.csv in latest run
    if args.alignment and os.path.isdir(args.path):
        runs = list_runs(args.path)
        if not runs:
            # Try as results root: pick first dataset
            for name in sorted(os.listdir(args.path)):
                sub = os.path.join(args.path, name)
                if os.path.isdir(sub):
                    runs = list_runs(sub)
                    if runs:
                        break
        if not runs:
            sys.exit(f"No timestamped runs found under {args.path}")
        target = runs[-1] if args.run is None else next(
            (r for r in runs if os.path.basename(r) == args.run), None
        )
        if target is None:
            sys.exit(f"Run {args.run} not found")
        ali = os.path.join(target, "alignment.csv")
        if not os.path.exists(ali):
            sys.exit(f"No alignment.csv in {target}")
        show_alignment(ali, args.top)
        return

    # Metrics display
    paths = resolve_metrics_path(args.path, args.run, args.all)
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print(f"No metrics.json found under {args.path}")
        print("No results yet -- train first (see launch/README.md).")
        return

    for mp in paths:
        show_metrics(mp)


if __name__ == "__main__":
    main()
