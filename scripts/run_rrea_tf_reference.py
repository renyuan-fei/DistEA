"""
run_rrea_tf_reference.py -- train the original TensorFlow RREA on a whole
graph, as the full-graph reference for DistEA's PyTorch port.

Reads:  --data_dir prepared by scripts/prepare_rrea_tf_data.py.
Writes: --output_dir/emb.npz (Runner.save(); convert with
        scripts/export_rrea_tf_to_unified.py) and reference.json
        (metrics + timings).

Drives ELsEA/DivEA/RREA/runner.py::Runner (TensorFlow/Keras) unchanged and
answers "what does this model family reach with full-graph training".

Usage (from inside projects/ELsEA/DivEA):
    python <DistEA>/scripts/run_rrea_tf_reference.py \
        --data_dir data/rrea_tf/EN_DE_100K_V1_721_5fold_1 \
        --output_dir output/rrea_tf/EN_DE_100K_V1_721_5fold_1 --mode iterative

Modes:
  iterative  Runner(max_train_epoch=E).iterative_training(): turn 0 trains
             E epochs, turns 1-4 train min(100, E) epochs each after adding
             mutual-nearest-neighbour pseudo seeds (RREA's own protocol);
             then CSLS_test(csls=10) on ref_ent_ids_test, save() -> emb.npz.
  single     Runner.train() for E epochs (no pseudo-seed turns), then
             CSLS_test + save -- the smoke / budget-matched variant.

Run from inside ELsEA/DivEA (the RREA package imports divea.util); the
runner's own hyper-parameters are left untouched (node_hidden 128, depth 2,
1 head, gamma 6, dropout 0.2, RMSprop lr 0.005, batch = node_size,
InfoNCE dropped automatically above 40000 nodes -- see create_model).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_train_epoch", type=int, default=1200)
    p.add_argument("--layer_num", type=int, default=2)
    p.add_argument("--mode", choices=["iterative", "single"], default="iterative")
    p.add_argument("--csls", type=int, default=10)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--tf_gpu_id", type=int, default=0)
    a = p.parse_args()

    sys.path.insert(0, os.getcwd())
    # Keras 2.8 (the ELsEA env) only exposes RMSprop as
    # keras.optimizers.rmsprop_v2.RMSprop; runner.py's create_model() does
    # keras.optimizers.RMSprop(lr) -- the same shim run_elsea_divea.py installs.
    import keras  # noqa: E402

    if not hasattr(keras.optimizers, "RMSprop") and hasattr(keras.optimizers, "rmsprop_v2"):
        keras.optimizers.RMSprop = keras.optimizers.rmsprop_v2.RMSprop
    from RREA.runner import Runner  # noqa: E402  (ELsEA/DivEA checkout)

    os.makedirs(a.output_dir, exist_ok=True)
    t0 = time.time()
    runner = Runner(a.data_dir, a.output_dir, max_train_epoch=a.max_train_epoch,
                    depth=a.layer_num, tf_gpu_no=a.tf_gpu_id)
    t_init = time.time() - t0
    print(f"[rrea-tf] node_size={runner.node_size} train_pairs={len(runner.train_pair)} "
          f"test_pairs={len(runner.dev_pair)} init {t_init:.0f}s", flush=True)

    t1 = time.time()
    if a.mode == "iterative":
        runner.iterative_training()
    else:
        runner.train()
    t_train = time.time() - t1

    t2 = time.time()
    hits1, metrics = runner.CSLS_test(thread_number=a.threads, csls=a.csls, accurate=True)
    t_eval = time.time() - t2
    runner.save()
    out = {
        "mode": a.mode, "max_train_epoch": a.max_train_epoch, "layer_num": a.layer_num, "csls": a.csls,
        "node_size": int(runner.node_size), "train_pairs_final": int(len(runner.train_pair)),
        "test_pairs": int(len(runner.dev_pair)),
        "csls_test_hits1": float(hits1), "csls_test_metrics": metrics if isinstance(metrics, (dict, list)) else str(metrics),
        "init_s": round(t_init, 1), "train_s": round(t_train, 1), "eval_s": round(t_eval, 1),
        "total_s": round(time.time() - t0, 1), "data_dir": os.path.abspath(a.data_dir),
    }
    with open(os.path.join(a.output_dir, "reference.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("[rrea-tf] " + json.dumps(out, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
