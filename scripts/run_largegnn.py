#!/usr/bin/env python
"""Launcher for LargeGNN's main_ea.py with a launcher-side evaluation patch.

Reads:  whatever main_ea.py reads (its config.json, dataset and division
        arguments), plus the optional LARGEGNN_* env vars below.
Writes: whatever main_ea.py writes; the patched evaluation only prints.

Same pattern as run_clusterea.py / run_elsea_divea.py / run_largeea.py: fix
upstream incompatibilities here, never in the vendored source.

Why: on DBP1M the upstream evaluation runs out of host memory right after
the first training epoch. The step after each epoch is
ClusterGCN4ea.run() -> self.evaluater.test(source_vec, tgt_vec) with
Dual_AMN/evaluate.py's CSLS_cal(): it computes the test-set similarity in
1024-row batches but APPENDS every batch to L_sim / R_sim, i.e. it
materialises the full |test| x |test| matrix (twice) in host RAM. DBP1M
has 280,173 test pairs -> 280173^2 * 4 B = 314 GB per copy, beyond any
reasonable host memory.

LargeGNN ships a batched faiss evaluator (Dual_AMN/fast_evaluate.py:
test_by_faiss_batch_one_by_one) and clustergcn.py even imports it -- the
call is just commented out in favour of the CSLS one. This launcher swaps
evaluate.test for that faiss path before main_ea.py is loaded. The return
value of evaluater.test() is discarded at the call site (clustergcn.py
~line 380), so returning the faiss tuple is safe.

Protocol note for anyone reporting the number: the faiss path ranks by
L2 on L2-normalised vectors (== cosine), with NO CSLS re-ranking, unlike
the original CSLS_cal. Hits@k is therefore the plain-cosine figure. Test
pairs are position-aligned (rest_set_1[i] <-> rest_set_2[i]), which is
what the one_by_one variant assumes; argument order follows the
commented-out call in clustergcn.py: (source_vec, tgt_vec) -> index on
the target side, query with the source side.

Second issue: ClusterGCN4ea.run() hardcodes `epoch = 150` as a local
variable (clustergcn.py:331) -- it does NOT read config's own
"epochs"/"max_epoch" fields at all ("start_valid"/"eval_freq" are never
referenced either), and evaluates UNCONDITIONALLY every single epoch. At
DBP1M scale that is ~280s/epoch training but ~4.2h/epoch for the faiss
eval loop above, i.e. weeks of evaluation for 150 epochs. This launcher
adds independent, opt-in env-var knobs (all default to the upstream
behaviour, so a bare `python run_largegnn.py` invocation is unaffected):
  LARGEGNN_EVAL_EVERY=<N>       only actually run the expensive faiss
                                evaluation on every Nth call (an epoch
                                is exactly one call); other calls return
                                an instant zero placeholder (discarded by
                                the caller anyway, see above)
  LARGEGNN_MAX_REAL_EVALS=<N>   raise SystemExit(0) right after the Nth
                                REAL evaluation, so the job ends cleanly
                                (exit 0) instead of being killed by a
                                time limit mid-epoch with no controlled
                                final log line
  LARGEGNN_EVAL_BATCH_NUM=<N>   batch_num passed to fast_evaluate.py's
                                test_by_faiss_batch_one_by_one (default
                                16, matching that function's own
                                default). Each batch's index_mat is
                                (batch_size, test_size) int64 -- QUADRATIC
                                in test_size for a fixed batch_num -- so
                                this needs to grow roughly with the
                                square of how much bigger the dataset's
                                test set is than one where batch_num=16
                                fits (DBP1M)
  LARGEGNN_EVAL_RANK_CAP=<N>    cap the exact-rank search at N candidates
                                (see the comment in install_faiss_eval_patch)
Sizing these is the caller's job; this launcher just provides the knobs.

Usage (from LargeGNN/src, exactly as main_ea.py would be invoked):
    python <DistEA>/scripts/run_largegnn.py main_ea.py <config.json> DBP1M_DE 316_1fold/1/
"""

from __future__ import annotations

import os
import runpy
import sys


def install_faiss_eval_patch() -> None:
    import Dual_AMN.evaluate as ev
    import numpy as np
    from Dual_AMN.fast_evaluate import test_by_faiss_batch_one_by_one

    eval_every = int(os.environ.get("LARGEGNN_EVAL_EVERY", "1") or "1")
    max_real_evals = int(os.environ.get("LARGEGNN_MAX_REAL_EVALS", "0") or "0")  # 0 = unlimited
    # LARGEGNN_EVAL_BATCH_NUM (default 16, matching fast_evaluate.py's own
    # default): search_faiss_one_by_one
    # (fast_evaluate.py:92) calls `index.search(query, num)` with
    # num = the FULL target-index size (not top_k!), to locate each
    # query's gold answer's exact rank for MR/MRR. index_mat is therefore
    # shape (batch_size, test_size) int64 -- per-batch memory is
    # (test_size/batch_num) * test_size * 8 bytes, QUADRATIC in test_size
    # for a fixed batch_num. At DBP1M (test_size ~281K) and batch_num=16
    # that's ~40 GB/batch; at FB-DBP2M (test_size ~1.62M, ~5.8x DBP1M) the
    # SAME batch_num=16 blows up ~5.8^2 ~= 34x to ~1.3 TB/batch (the single
    # index_mat allocation itself). Raising batch_num shrinks each batch
    # proportionally; the default is unchanged unless the caller sets the
    # env var.
    eval_batch_num = int(os.environ.get("LARGEGNN_EVAL_BATCH_NUM", "16") or "16")

    # LARGEGNN_EVAL_RANK_CAP (default 0 = off, preserves exact upstream
    # behavior unless the caller opts in). LARGEGNN_EVAL_BATCH_NUM fixes
    # the memory blow-up, but the underlying full-index
    # search (num=embeds1.shape[0], see above) also makes each search
    # call's TIME cost scale with (queries_in_batch * test_size) --
    # independent of batch_num, since splitting into more batches just
    # spreads the same total work over more, individually-smaller search
    # calls. On FB-DBP2M: ~2800s per batch at batch_num=256 (6346 queries/
    # batch against the full 1,624,530-vector index) -> ~200h for ONE
    # evaluation, regardless of the eval-frequency gate above. The
    # exact-rank search is only needed
    # because fast_evaluate.py's search_faiss_one_by_one computes MR/MRR
    # by locating the gold answer's position in a FULL ranking; since we
    # only report top_k in [1,5,10,50], searching a small candidate set
    # (comfortably larger than max(top_k)=50) gives identical hits@k for
    # every k we report -- only MR/MRR become a (standard, widely-used)
    # approximation: a gold answer ranked beyond the cap is treated as
    # "not found" (contributes 0, i.e. ~rank=infinity) rather than its
    # true rank, which undercounts MR and is a negligible MRR effect
    # (1/rank is already ~0 for any rank past a few hundred). Patches
    # fast_evaluate.search_faiss_one_by_one itself (a bare module-level
    # name test_by_faiss_batch_one_by_one calls directly -- Python
    # resolves that via the module's globals at call time, so overwriting
    # the module attribute here is enough, no need to also touch the
    # caller) to clamp its `num` argument.
    eval_rank_cap = int(os.environ.get("LARGEGNN_EVAL_RANK_CAP", "0") or "0")
    if eval_rank_cap > 0:
        import Dual_AMN.fast_evaluate as fe
        _orig_search = fe.search_faiss_one_by_one

        def _capped_search(index, query, num, bj, batch, top_k):
            capped = min(num, eval_rank_cap)
            return _orig_search(index, query, capped, bj, batch, top_k)

        fe.search_faiss_one_by_one = _capped_search
        print(f"[run_largegnn] capped search_faiss_one_by_one's rank search to {eval_rank_cap} "
              f"(MR/MRR become an approximation past this rank; hits@k unaffected for k <= {eval_rank_cap})")

    # fast_evaluate.py's test_by_faiss_batch_one_by_one builds
    # `faiss.IndexFlatL2(dim)` -- a plain CPU index -- and never calls
    # faiss.index_cpu_to_gpu anywhere, so the search always runs on the
    # CPU regardless of the RANK_CAP patch above. Evaluation is required
    # to run on the GPU (a CPU fallback would make the timing comparison
    # meaningless), so the function is replaced wholesale with a GPU-index
    # version instead of patching search_faiss_one_by_one alone -- the
    # index itself, not just the search call, has to move to the GPU.
    # Fails loudly (no try/except CPU fallback) if faiss reports zero
    # GPUs, rather than silently reverting to the slow path.
    import time as _time

    import faiss as _faiss
    import numpy as _np
    from Dual_AMN.fast_evaluate import search_faiss_one_by_one as _search_one_by_one
    from Dual_AMN.fast_evaluate import task_divide as _task_divide
    from sklearn import preprocessing as _preprocessing

    _n_gpus = _faiss.get_num_gpus()
    if _n_gpus < 1:
        raise RuntimeError(
            "[run_largegnn] faiss.get_num_gpus()==0 -- refusing to fall back to "
            "CPU faiss search, which would make the timing incomparable. Install "
            "a GPU build of faiss in LargeGNN's environment."
        )

    def _gpu_test_by_faiss_batch_one_by_one(embeds2, embeds1, top_k=(1, 5, 10, 50),
                                             is_norm=True, batch_num=16):
        start = _time.time()
        if is_norm:
            embeds1 = _preprocessing.normalize(embeds1)
            embeds2 = _preprocessing.normalize(embeds2)
        num = embeds2.shape[0]
        dim = embeds2.shape[1]
        hits = _np.array([0] * len(top_k))
        mr, mrr = 0, 0
        # A plain index_cpu_to_gpu(IndexFlatL2) fails with "cublas failed
        # (13)" on FB-DBP2M's 2.33M-vector index.add() (the GEMM tiling
        # faiss's GPU add/search use at this scale). useFloat16 halves the
        # matmul's memory footprint
        # (fp16 storage + fp16 GEMM), which is the standard faiss-gpu fix for
        # large flat indexes hitting GPU matmul limits; negligible accuracy
        # impact for L2 nearest-neighbour ranking (we only need relative
        # order, not exact distances).
        gpu_res = _faiss.StandardGpuResources()
        flat_config = _faiss.GpuIndexFlatConfig()
        flat_config.device = 0
        flat_config.useFloat16 = True
        index = _faiss.GpuIndexFlatL2(gpu_res, dim, flat_config)
        index.add(embeds1)
        batches = _task_divide(list(range(num)), batch_num)
        for bj, batch in enumerate(batches):
            query = embeds2[batch, :]
            hits_, mrr_, mr_ = _search_one_by_one(index, query, embeds1.shape[0], bj, batch, list(top_k))
            hits += hits_
            mrr += mrr_
            mr += mr_
        mr /= num
        mrr /= num
        hits = hits / num
        print(f"[run_largegnn] GPU faiss eval: hits@{list(top_k)} = {hits}, mr = {mr:.3f}, "
              f"mrr = {mrr:.6f}, total time = {_time.time() - start:.3f} s")
        return hits, mrr, mr

    # Replace the imported upstream function with the GPU version.
    test_by_faiss_batch_one_by_one = _gpu_test_by_faiss_batch_one_by_one  # noqa: F811
    print(f"[run_largegnn] faiss eval index moved to GPU 0 (faiss.get_num_gpus()={_n_gpus}) "
          f"-- test_by_faiss_batch_one_by_one replaced wholesale, not just the search call")

    state = {"calls": 0, "real_evals": 0}

    def _test_faiss(self, Lvec, Rvec):  # noqa: N802 -- upstream name
        state["calls"] += 1
        if eval_every > 1 and (state["calls"] % eval_every) != 1:
            print(f"[run_largegnn] evaluater.test call #{state['calls']}: skipped "
                  f"(LARGEGNN_EVAL_EVERY={eval_every}, real evals so far: {state['real_evals']})")
            return np.zeros(4), 0.0, 0.0
        state["real_evals"] += 1
        print(f"[run_largegnn] evaluater.test call #{state['calls']}: REAL eval #{state['real_evals']} "
              f"-> faiss batched cosine (CSLS_cal skipped: "
              f"dense {len(Lvec)}x{len(Rvec)} matrix would not fit in host RAM)")
        result = test_by_faiss_batch_one_by_one(Lvec, Rvec, top_k=[1, 5, 10, 50], batch_num=eval_batch_num)
        if max_real_evals and state["real_evals"] >= max_real_evals:
            print(f"[run_largegnn] LAST REAL EVAL ({state['real_evals']}/{max_real_evals}) done, "
                  f"result: {result} -- stopping cleanly (LARGEGNN_MAX_REAL_EVALS reached)")
            raise SystemExit(0)
        return result

    ev.evaluate.test = _test_faiss
    print(f"[run_largegnn] patched Dual_AMN.evaluate.evaluate.test "
          f"(eval_every={eval_every}, max_real_evals={max_real_evals or 'unlimited'})")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_largegnn.py main_ea.py <config.json> <dataset> <division>")
    target_script = sys.argv[1]
    sys.argv = sys.argv[1:]
    # main_ea.py does `from Dual_AMN.evaluate import *` etc. relative to
    # LargeGNN/src -- this launcher lives elsewhere, so put cwd first.
    sys.path.insert(0, os.getcwd())
    install_faiss_eval_patch()
    runpy.run_path(target_script, run_name="__main__")


if __name__ == "__main__":
    main()
