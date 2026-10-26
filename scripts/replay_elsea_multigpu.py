#!/usr/bin/env python3
"""Replay ELsEA's DivEA running.log to estimate multi-GPU wall-clock
without needing real multi-GPU hardware.

Reads:  ELsEA's running.log (one JSON record per timed process).
Writes: nothing; prints the estimated total time per GPU count G.

Usage:
    python scripts/replay_elsea_multigpu.py outputs/elsea/DBP1M_DE/running.log \
        --g 1 2 4 8 --sacct-elapsed-s 36000

divea/framework.py::ParallelEAFramework.run() structure (from the log and
the source):
  - partition_g1: once, server-only.
  - per iteration: server segment (build_context1, partition_g2,
    build_context2, build_subgraphs) runs once; then each client
    (subtask) runs train -> evaluate -> generate_msg SEQUENTIALLY on
    whichever GPU main.py's --gpu_ids round-robin assigned it to
    (idx % len(gpu_ids)); then communicate.
  - Upstream, all clients run one after another regardless of
    --gpu_ids, i.e. G is always effectively 1. This script estimates
    what G>1 WOULD look like if each GPU's assigned clients ran in
    parallel (each GPU's own clients still execute sequentially on
    that one GPU).

Two schedules per G:
  - "mod": ELsEA's own assignment, client idx -> idx % G.
  - "lpt": longest-processing-time-first greedy bin-packing, the ideal
    lower bound for G cards (not what the mod-patch would actually
    produce, just a ceiling on how much mod-scheduling leaves on the
    table).

Validation: G=1 should reproduce the real (scheduler-reported) wall-clock
within 2%; pass it as --sacct-elapsed-s to check this and to add the
constant overhead that the log does not record.
"""
import argparse
import json
from collections import defaultdict


def load_log(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def schedule_mod(client_times, g):
    """client_times: {client_idx(int): total_seconds}. idx % g assignment."""
    loads = [0.0] * g
    for idx, t in client_times.items():
        loads[idx % g] += t
    return max(loads), loads


def schedule_lpt(client_times, g):
    loads = [0.0] * g
    for _idx, t in sorted(client_times.items(), key=lambda kv: -kv[1]):
        card = loads.index(min(loads))
        loads[card] += t
    return max(loads), loads


def replay(recs, g_list, schedule_fn):
    server = defaultdict(lambda: defaultdict(float))  # iteration -> process -> seconds
    partition_g1 = 0.0
    client = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # iter -> client_idx -> process -> seconds
    communicate = defaultdict(float)
    max_iter = -1

    for r in recs:
        proc = r["process"]
        val = r["value"]
        machine = r["machine"]
        it = r.get("iteration")
        if it is not None:
            max_iter = max(max_iter, it)
        if proc == "partition_g1":
            partition_g1 += val
        elif machine == "server":
            if proc == "communicate":
                communicate[it] += abs(val)
            else:
                server[it][proc] += val
        elif machine.startswith("client_"):
            idx = int(machine.split("_")[1])
            client[it][idx][proc] += val

    results = {}
    for g in g_list:
        total = partition_g1
        iter_breakdown = []
        for it in range(max_iter + 1):
            server_seg = sum(server[it].values())
            client_times = {
                idx: sum(procs.values()) for idx, procs in client[it].items()
            }
            client_seg, loads = schedule_fn(client_times, g)
            comm = communicate[it]
            iter_total = server_seg + client_seg + comm
            total += iter_total
            iter_breakdown.append(
                {
                    "iteration": it,
                    "server_seg_s": round(server_seg, 2),
                    "client_seg_s": round(client_seg, 2),
                    "communicate_s": round(comm, 2),
                    "iter_total_s": round(iter_total, 2),
                }
            )
        results[g] = {"total_s": round(total, 2), "iterations": iter_breakdown}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("running_log", help="path to ELsEA's running.log")
    ap.add_argument("--g", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--sacct-elapsed-s", type=float, default=None,
                     help="measured wall-clock of the run in seconds (e.g. sacct Elapsed), for the G=1 "
                          "validation + constant-overhead correction")
    args = ap.parse_args()

    recs = load_log(args.running_log)

    print("=== mod schedule (ELsEA's own idx %% G assignment) ===")
    mod_results = replay(recs, args.g, schedule_mod)
    for g in args.g:
        print(f"G={g}: total_s={mod_results[g]['total_s']:.1f}  ({mod_results[g]['total_s']/3600:.3f} h)")

    print()
    print("=== lpt schedule (ideal lower bound) ===")
    lpt_results = replay(recs, args.g, schedule_lpt)
    for g in args.g:
        print(f"G={g}: total_s={lpt_results[g]['total_s']:.1f}  ({lpt_results[g]['total_s']/3600:.3f} h)")

    if args.sacct_elapsed_s:
        g1_mod = mod_results[1]["total_s"] if 1 in mod_results else None
        if g1_mod:
            err_pct = 100.0 * (g1_mod - args.sacct_elapsed_s) / args.sacct_elapsed_s
            overhead = args.sacct_elapsed_s - g1_mod
            print()
            print("=== G=1 validation ===")
            print(f"replay G=1 total_s={g1_mod:.1f}  real sacct Elapsed={args.sacct_elapsed_s:.1f}  "
                  f"error={err_pct:+.2f}%  (constant overhead not in log = {overhead:.1f}s)")
            print()
            print("=== mod schedule + constant overhead applied to all G ===")
            for g in args.g:
                corrected = mod_results[g]["total_s"] + overhead
                print(f"G={g}: total_s={corrected:.1f}  ({corrected/3600:.3f} h)")

if __name__ == "__main__":
    main()
