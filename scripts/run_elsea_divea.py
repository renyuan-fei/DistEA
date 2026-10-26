"""
run_elsea_divea.py -- launcher for ELsEA/DivEA's Small_DivEA_run_IDS15K_V1.py
(or any other Small/Medium/Large_DivEA_run_*.py) that works around gaps in
the upstream checkout without modifying the checkout itself.

Reads:  whatever the target run script reads (its --data_dir), plus the
        optional env vars ELSEA_KG1_PARTITION_DIR, ELSEA_TF_RESERVE_MB and
        ELSEA_CONCURRENCY_GPUS (see the install_* functions below).
Writes: whatever the target run script writes (its --output_dir,
        running.log).

Patches applied before the target script runs:

1. divea/neural_lightea.py does not exist on disk, but every run script
   imports LightEAModule from it unconditionally at module load time (even
   when --ea_model is rrea/gcn-align/dualamn), so the script cannot even be
   imported without it. Pre-register a stub `divea.neural_lightea` module
   in sys.modules before the real script's own imports run.

2. divea/dataload.py's UniformData.load_weights() sizes each weight tensor
   by relation/entity COUNT but indexes it by the raw relation/entity ID
   VALUE -- correct only if ids are a dense 0..N-1 range, which real data
   isn't (relation ids have gaps, kg2 entity ids are offset past kg1's
   range). Monkey-patch the method with an otherwise-faithful copy that
   sizes by max(ids)+1 instead.

3. Neighbor_Agg_Layer's node-weight gather / ent_num (same id-range
   issue), a Keras RMSprop import shim, an optional KG1 split taken from a
   DistEA partition, an optional TensorFlow GPU memory cap, and an optional
   one-worker-per-GPU concurrency patch -- see each install_* function.

No triple column-order patch is needed: every consumer in divea/*.py
(graph_metrics.py, components.py, and dataload.py's reset_IDs/
convert_uniform_to_rrea/convert_uniform_to_openea) expects (h, r, t),
matching OpenEA's own triples_1/2 order, which is what
scripts/export_to_divea.py writes.

Usage (from inside ELsEA/DivEA/, matching the real script's own relative
default paths for --data_dir/--output_dir):
    python <DistEA>/scripts/run_elsea_divea.py Small_DivEA_run_IDS15K_V1.py --kgids en,de ...
(all args after the script name are passed through unchanged)
"""

import os
import runpy
import sys
import types


def install_neural_lightea_stub() -> None:
    from divea.components_base import NeuralEAModule  # real, present in this checkout

    stub = types.ModuleType("divea.neural_lightea")

    class LightEAModule(NeuralEAModule):
        def __init__(self, conf):
            super().__init__(conf)
            raise NotImplementedError(
                "divea.neural_lightea is a stub (module absent from this "
                "repo checkout) -- LightEAModule cannot actually be used. "
                "Pass --ea_model rrea/gcn-align/dualamn instead."
            )

    stub.LightEAModule = LightEAModule
    sys.modules["divea.neural_lightea"] = stub


def install_load_weights_index_fix() -> None:
    """
    divea/dataload.py's UniformData.load_weights() has an upstream bug:
    it sizes each weight tensor by relation/entity COUNT
    (`torch.zeros(len(self.kg2_relations))`) but then indexes it by the raw
    relation/entity ID VALUE (`self.edge_weights2[r] = ...` for r in
    self.kg2_relations) -- correct only when ids happen to be a dense
    0..N-1 range. Real data isn't: e.g. on EN_DE_15K the relation column
    of {kgid}_triple_rel.txt has 131 distinct values spanning [0, 238], and
    kg2 entity ids span [15000, 29999] against a KG2 entity COUNT of 15000
    -- both hit `IndexError: index 133 is out of bounds for dimension 0
    with size 131`. node_weights1/
    node_weights2 have the identical bug. Fix: size every tensor by
    max(ids)+1 instead of len(ids) -- preserves the exact by-id indexing
    the rest of the method (and every caller of self.edge_weights{1,2}/
    self.node_weights{1,2}) already assumes, just makes the tensor large
    enough to hold real id values, with harmless zero-padding at unused
    indices.
    """
    import divea.dataload as dataload_module
    import torch as torch_module

    def load_weights_fixed(self):
        import json as json_module
        import os as os_module

        with open(os_module.path.join(self.data_dir, "KGs_edge_info.json")) as file:
            obj = json_module.loads(file.read())
        self.edge_weights1 = torch_module.zeros(max(self.kg1_relations) + 1)
        for r in self.kg1_relations:
            self.edge_weights1[r] = obj["KG1"]["r2f"][str(r)] + obj["KG1"]["r2if"][str(r)]
        self.edge_weights2 = torch_module.zeros(max(self.kg2_relations) + 1)
        for r in self.kg2_relations:
            self.edge_weights2[r] = obj["KG2"]["r2f"][str(r)] + obj["KG2"]["r2if"][str(r)]

        with open(os_module.path.join(self.data_dir, "KGs_node_info.json")) as file:
            obj = json_module.loads(file.read())
        self.node_weights1 = torch_module.zeros(max(self.kg1_entities) + 1)
        for k in self.kg1_entities:
            self.node_weights1[k] = obj["KG1"]["in_degree"][str(k)] + obj["KG1"]["out_degree"][str(k)] + 1
        self.node_weights2 = torch_module.zeros(max(self.kg2_entities) + 1)
        for k in self.kg2_entities:
            self.node_weights2[k] = obj["KG2"]["in_degree"][str(k)] + obj["KG2"]["out_degree"][str(k)] + 1

    dataload_module.UniformData.load_weights = load_weights_fixed


def install_neighbor_agg_layer_node_weights_fix() -> None:
    """
    divea/neighborEhance.py's Neighbor_Agg_Layer.__init__ has an upstream
    bug, only reachable once install_load_weights_index_fix()
    stops an earlier crash from masking it: it builds `self.graph` by
    add_edges() using RAW entity ids directly as node indices (triple_arr's
    head/tail columns, plus kg_entities as self-loops) -- DGL auto-sizes
    the graph to max(referenced id)+1 nodes, e.g. 30000 for kg2 entities
    spanning [15000, 29999]. It then computes `n_weights =
    node_weights[np.array(kg_entities)].tolist()` -- a GATHER down to just
    len(kg_entities)=15000 entries -- and assigns that as `self.graph.ndata
    ['x']`, which DGL rejects: "Expect number of features to match number
    of nodes. Got 15000 and 30000 instead." The very next line in the same
    function, `edge_weights[triple_arr[:, 1]]`, indexes directly by raw id
    with NO gather/compaction -- confirming the codebase's actual
    convention is "arrays sized by max(id)+1, indexed directly by raw id",
    and this one gathered line is the anomaly, not the norm. Fix: use
    node_weights directly (already sized to match the graph's own
    auto-sized node count by install_load_weights_index_fix() above) with
    no gather.

    Second, related bug in the same class's forward(): `self.ent_num` is
    stored from the caller's `ent_num` argument, which ctx_builder_onclient
    .py passes as `len(self.data.kg2_entities)` (15000) -- but forward()
    does `h = torch.zeros(size=(self.ent_num,))` then `h[anchors] = 1`
    where anchors are raw ids (e.g. up to 29999), causing CUDA
    out-of-bounds indexing. Fixed the same way: derive self.ent_num from
    self.graph.num_nodes() (30000) instead of trusting the caller's count.
    """
    import dgl as dgl_module
    import divea.neighborEhance as neighbor_enhance_module  # noqa: N813 (upstream module name)
    import numpy as np_module
    import torch as torch_module

    class Neighbor_Agg_Layer_Fixed(neighbor_enhance_module.nn.Module):  # noqa: N801 (mirrors upstream)
        def __init__(self, edge_weights, node_weights, ent_num, kg_entities, kg_triples):
            super().__init__()
            triple_arr = np_module.array(kg_triples)

            self.graph = dgl_module.DGLGraph()
            self.graph.add_edges(triple_arr[:, 0], triple_arr[:, 2])
            self.graph.add_edges(np_module.array(kg_entities), np_module.array(kg_entities))

            e_weights = []
            e_weights += edge_weights[triple_arr[:, 1]].tolist()

            max_weight = torch_module.max(edge_weights)
            max_weight = max_weight + 0.5
            e_weights += max_weight.repeat(len(kg_entities)).tolist()

            n_weights = node_weights.tolist()  # fix: no [np.array(kg_entities)] gather

            self.graph.ndata["x"] = torch_module.tensor(n_weights).float()
            self.graph.edata["w"] = torch_module.tensor(e_weights)
            # fix: the caller passes ent_num=len(kg_entities) (15000), but
            # self.graph itself was built from RAW ids (add_edges above) and
            # DGL auto-sized it to max(id)+1 (30000) nodes -- forward()'s
            # `h = torch.zeros(size=(self.ent_num,))` then `h[anchors] = 1`
            # (anchors are raw ids, e.g. from train_alignment's kg2 side,
            # up to 29999) needs h sized to match, not the entity count.
            # Downstream-safe: _cache_g2_ents's caller reshapes
            # the result to (-1, 1) and torch.cats on the last dim, making
            # no assumption about row count, and the whole codebase's other
            # by-id-indexed arrays (edge_weights, node_weights) already
            # establish "sized to max(id)+1" as the actual convention here.
            self.ent_num = self.graph.num_nodes()

        def forward(self, anchors, device):
            device = torch_module.device(device)
            g = self.graph.to(device)
            h = torch_module.zeros(size=(self.ent_num,), device=device)
            h[anchors] = 1
            h[anchors] += g.ndata["x"][anchors]

            with g.local_scope():
                g.ndata["h"] = h
                g.update_all(
                    neighbor_enhance_module.fn.u_mul_e("h", "w", "m"),
                    neighbor_enhance_module.fn.mean("m", "o"),
                )
                h_o = g.ndata["o"]
                x = g.ndata["x"]
                del g, h
                return h_o, x

    neighbor_enhance_module.Neighbor_Agg_Layer = Neighbor_Agg_Layer_Fixed


def install_keras_rmsprop_shim() -> None:
    """
    RREA/runner.py does `keras.optimizers.RMSprop(lr)` (create_model()),
    matching an older Keras API (<=2.4.x) where RMSprop was exposed at the
    top level of keras.optimizers. Keras 2.8.0 (the version tensorflow-gpu
    ==2.8.1 depends on; Install_requirements' explicit Keras==2.4.3 pin
    conflicts with that dependency and cannot be resolved) only exposes it
    as keras.optimizers.rmsprop_v2.RMSprop, not at the top level --
    keras.optimizers.rmsprop_v2.RMSprop is the same class as
    tf.keras.optimizers.RMSprop. Not a bug in ELsEA's logic,
    just an environment-version gap between what runner.py's API call
    expects and what a resolvable modern dependency set actually provides.
    Fix: re-expose it at the path runner.py expects.
    """
    import keras as keras_module

    keras_module.optimizers.RMSprop = keras_module.optimizers.rmsprop_v2.RMSprop


def install_kg1_partition_override() -> None:
    """ELSEA_KG1_PARTITION_DIR=<DistEA partition dir> (shared-partition
    ablation): replace DivG1Metis's own METIS split of KG1 with DistEA's
    {rank}.npy split -- same global ids on both sides (identical test set),
    so ELsEA and DistEA train on literally the same KG1 partition. This is
    the mirror image of scripts/make_divea_aligned_partition.py --kg2_mode
    locality (DistEA on DivEA's split). Everything after the split -- ctx
    entities, counterpart discovery
    for KG2, per-subtask independent training -- stays DivEA's own code.

    Mirrors DivG1Metis._divide_entities exactly apart from where the split
    comes from: writes each part's FULL entity list to
    partition_{i+1}/tmp_part.json (downstream context/partition components
    read it) and returns per-part test-alignment heads (what
    partition_g1_entities() stores as kg1_partition.entities). Unset env var
    => no-op, DivEA's METIS runs as before."""
    part_dir = os.environ.get("ELSEA_KG1_PARTITION_DIR")
    if not part_dir:
        return
    import glob
    import json

    import divea.components as components
    import numpy as np
    from divea.misc import sub_alignment_with_head

    files = sorted(
        glob.glob(os.path.join(part_dir, "[0-9]*.npy")),
        key=lambda p: int(os.path.basename(p).split(".")[0]),
    )
    if not files:
        raise SystemExit(f"ELSEA_KG1_PARTITION_DIR={part_dir!r} contains no {{rank}}.npy files")

    def _divide_entities(self):
        if len(files) != self.part_n:
            raise SystemExit(
                f"ELSEA_KG1_PARTITION_DIR has {len(files)} parts but --subtask_num is {self.part_n}"
            )
        partitions = [[int(x) for x in np.load(f)] for f in files]
        kg1 = set(self.data.kg1_entities)
        covered = set().union(*(set(p) for p in partitions))
        if covered != kg1:
            raise SystemExit(
                f"DistEA partition covers {len(covered)} KG1 entities "
                f"({len(covered - kg1)} unknown ids), UniformData has {len(kg1)}"
            )
        for i, part in enumerate(partitions):
            os.makedirs(os.path.join(self.data_dir, f"partition_{i + 1}"), exist_ok=True)
            with open(os.path.join(self.data_dir, f"partition_{i + 1}/tmp_part.json"), "w+") as f:
                f.write(json.dumps(part))
        test_alignment = self.data.load_test_alignment()
        out = []
        for part in partitions:
            sub_align = sub_alignment_with_head(part, test_alignment)
            out.append([e1 for e1, _e2 in sub_align])
        print(
            f"[run_elsea_divea] KG1 split taken from {part_dir}: part sizes "
            f"{[len(p) for p in partitions]}, test heads per part {[len(o) for o in out]}"
        )
        return out

    components.DivG1Metis._divide_entities = _divide_entities


def install_tf_memory_cap() -> None:
    """Cap TensorFlow's GPU memory so the torch CSLS evaluator can fit.

    Opt-in via ELSEA_TF_RESERVE_MB=<megabytes to leave free>; unset or 0
    is a strict no-op (not needed at 15K/100K scale).

    Why: ELsEA trains RREA in TensorFlow and then evaluates with the torch
    Evaluator in the SAME process. Training already runs with
    allow_growth / set_memory_growth (RREA/runner.py), but "growth" only
    means TF grabs lazily -- it never gives the peak back. On DBP1M the
    graph-attention concat allocates [3256072, 384] tensors, so after
    training TF is sitting on nearly the whole card; the torch evaluator
    then sees e.g. "39.49 GiB total, 61.56 MiB free" and dies on its first
    558 MiB chunk.

    Mechanism: with Keras 2.8.0 `keras.backend.set_session` still exists,
    so RREA/runner.py's `Runner.__init__` (runner.py:165) takes its FIRST
    branch -- `K.set_session(tf.compat.v1.Session(config=config))` with a
    plain `config.gpu_options.allow_growth = True` -- not the TF2
    `set_memory_growth` fallback in the `except AttributeError` branch.
    A `tf.compat.v1.Session` with its own ConfigProto is TF1's own
    memory-management path and bypasses a TF2 logical-device cap
    (tf.config.set_logical_device_configuration), so that cap does not
    help. Instead, patch `tf.compat.v1.Session` itself so any ConfigProto
    handed to it gets `allow_growth` forced off and `per_process_gpu_memory_
    fraction` set to (total-reserve)/total -- a TF1-native hard cap on
    the code path actually taken. Total memory is read via pynvml (already
    a RREA/runner.py dependency) so no CUDA context is created here.

    Timing: apply lazily at the top of `Runner.__init__`, not eagerly in
    the launcher --
    touching the GPU in the launcher happens before DivEA forks CUDA-
    using ctx-builder worker processes (`proc_n`), and those forks fail
    with "CUDA driver initialization failed" if the parent already holds
    a CUDA context. `Runner` is constructed once per subtask in the same
    process; a module flag makes the patch install (and the cap
    computation) happen once.
    """
    reserve_mb = int(os.environ.get("ELSEA_TF_RESERVE_MB", "0") or "0")
    if reserve_mb <= 0:
        return

    state = {"applied": False}

    def _apply_cap() -> None:
        if state["applied"]:
            return
        state["applied"] = True
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if not gpus:
            print("[run_elsea_divea] ELSEA_TF_RESERVE_MB set but no GPU visible; skipping TF memory cap")
            return
        import nvidia_smi
        nvidia_smi.nvmlInit()
        total_mb = nvidia_smi.nvmlDeviceGetMemoryInfo(nvidia_smi.nvmlDeviceGetHandleByIndex(0)).total // (1024 * 1024)
        limit_mb = total_mb - reserve_mb
        if limit_mb <= 0:
            raise SystemExit(f"[run_elsea_divea] ELSEA_TF_RESERVE_MB={reserve_mb} >= GPU0 total {total_mb} MB")
        fraction = limit_mb / total_mb

        _OrigSession = tf.compat.v1.Session

        class _CappedSession(_OrigSession):
            def __init__(self, *args, config=None, **kwargs):
                if config is not None:
                    config.gpu_options.allow_growth = False
                    config.gpu_options.per_process_gpu_memory_fraction = fraction
                super().__init__(*args, config=config, **kwargs)

        tf.compat.v1.Session = _CappedSession
        print(f"[run_elsea_divea] TF memory cap (v2, per_process_gpu_memory_fraction): "
              f"{limit_mb} MB of {total_mb} MB ({fraction:.3f}), reserving {reserve_mb} MB for the torch evaluator")

    import RREA.runner as rr  # importing TF/keras here does not initialise the GPU
    orig_init = rr.Runner.__init__

    def patched_init(self, *args, **kwargs):
        _apply_cap()
        orig_init(self, *args, **kwargs)

    rr.Runner.__init__ = patched_init
    print(f"[run_elsea_divea] TF memory cap armed (reserve {reserve_mb} MB), applies at Runner.__init__")


def install_elsea_concurrency_patch() -> None:
    """
    divea/framework.py::ParallelEAFramework.run()'s per-client loop already
    has --gpu_ids plumbing (idx % len(gpu_ids) assigns each subtask to a
    card), but execution itself is fully sequential -- the mp.Process lines
    around evaluate_model()/self.evaluate() are commented out. Handing it
    more GPUs via --gpu_ids alone does not make it faster; this patch runs
    the per-client loop concurrently, one worker process per GPU, so a
    multi-GPU number means something.

    Opt-in via ELSEA_CONCURRENCY_GPUS=<comma-separated GPU ids visible to
    this job, e.g. "0,1">; unset or a single id is a strict no-op
    (sequential, unchanged upstream behaviour).

    Timing/safety: the parent process touches CUDA itself inside the
    per-iteration server steps (`build_g1_context`/`build_g2_context` log
    `ctx1/ctx2_gpu_mem_usage` -- DivEA's ctx builder uses the GPU), so
    forking fresh workers on iteration 2+ would fork a process that already
    holds a CUDA context -- the same "CUDA driver initialization failed"
    failure install_tf_memory_cap() above describes for DivEA's own
    `--proc_n` ctx-builder forks. Workers are therefore forked exactly ONCE,
    at the top of run() -- right after `_prepare()`, before
    `partition_g1_entities()` and the outer-iteration loop -- where nothing
    has touched CUDA yet (install_tf_memory_cap() defers its GPU touch to
    Runner.__init__ for the same reason). Each worker stays alive for every
    outer iteration and keeps its own bucket of Client objects resident
    across all `max_iteration` rounds -- the same lifetime those objects
    have in the original single-process code, just split across G
    processes, so whatever a client carries from one round to the next (in
    memory, not only on disk) is preserved. Each worker gets its copy of
    the client objects at fork time, before any of them has run
    generate_dataset_from_partition/train_model/evaluate_model/generate_msg.

    Why fork and not spawn: spawn pickles the target and its args (a local
    function fails with "Can't pickle local object ..._worker_loop") and
    re-imports the parent's __main__ in every child. The launcher runs the
    ELsEA script through runpy.run_path(..., run_name="__main__"), so during
    framework.run() sys.modules['__main__'] IS that script
    (Medium_DivEA_run_IDS100_V1.py etc.), whose top level has no
    `if __name__` guard: every spawned worker would re-execute the whole
    ELsEA program, and none would have this launcher's install_* patches.
    Fork has neither problem: nothing is pickled, and the children inherit
    the patched classes. A guard below refuses to fork if torch already
    initialised CUDA in the parent, so a change that breaks that ordering
    fails loudly instead of handing every worker a broken CUDA context.

    CUDA_VISIBLE_DEVICES is remapped per worker. Each client's own
    conf.gpu_ids/conf.torch_device is not enough for --ea_model rrea:
    neural_rrea.py's train_model() builds `Runner(...,
    tf_gpu_no=self.conf.gpu_ids)`, and RREA/runner.py's Runner.__init__
    builds its `tf.compat.v1.Session(config)` with no `visible_device_list`
    at all -- `tf_gpu_no` is used ONLY for an nvidia_smi memory-usage log
    line, never for device placement. Left alone, every worker's TF session
    lands on the first visible GPU, all of them piling onto one card
    (dualamn/lightea/gcn-align instead run via a subprocess that sets its
    own CUDA_VISIBLE_DEVICES=conf.tf_gpu_id, so they are not affected).

    Fix: right before starting worker g's process, set
    os.environ["CUDA_VISIBLE_DEVICES"] in the parent to that bucket's
    clients' own conf.gpu_ids value (they all agree: the run scripts assign
    `conf.gpu_ids = gpu_ids_list[part_idx % len(gpu_ids_list)]`, constant
    within an idx % G bucket), translated through the job's own
    CUDA_VISIBLE_DEVICES list (conf.gpu_ids is an index into that list, not
    necessarily a physical GPU number -- a job may be handed e.g. "2,3").
    The forked child inherits this environment before its interpreter
    touches CUDA, so that one GPU becomes local index 0 inside the child.
    The parent's own CUDA_VISIBLE_DEVICES is saved and restored around the
    fork loop, so its later GPU use (ctx_builder in build_g1_context/
    build_g2_context) still sees every GPU, unchanged.

    One more consequence of the remap: a worker that only has ONE GPU
    visible needs "cuda:0" for anything that names a device explicitly.
    components_base.py's Client.evaluate_model() already hardcodes
    self.device = "cuda:0", but neural_rrea.py's own evaluate() builds
    `Evaluator(device=self.conf.torch_device)` using the ORIGINAL value
    (e.g. "cuda:1"), which does not exist inside a one-GPU worker, so this
    patch resets every one of its clients' `ea_module.conf.torch_device` to
    torch.device("cuda:0") once, as each worker starts. conf.gpu_ids/
    conf.tf_gpu_id are left at their original values: train_model() only
    reads conf.gpu_ids for that one nvidia_smi log line, which NVML resolves
    by physical index regardless of CUDA_VISIBLE_DEVICES; conf.tf_gpu_id is
    never read on the rrea in-process path (only by the dualamn/gcn-align
    subprocess path).

    Faithfulness to the original per-subtask work: each worker runs its
    assigned subtask indices (idx % G == g) in the ORIGINAL order with the
    ORIGINAL three steps (train_model -> evaluate_model -> generate_msg)
    and the ORIGINAL running.log timing lines (same "process"/"iteration"
    keys scripts/replay_elsea_multigpu.py parses), including the original
    communicate() timing's sign bug (t51 - t52, kept exactly -- the replay
    script takes abs() of it). Only the sequential for-loop itself is
    replaced; _prepare(), partition_g1, build_context1/2, partition_g2,
    build_subgraphs, and the final self.evaluate() still run in the parent,
    exactly as before. ignore_ids contributed by any worker
    (generate_dataset_from_partition() returning True) are merged back via
    a fork-context Queue of small ints before server.communicate() runs
    each round, in the same place the original sequential loop called it.
    """
    gpu_list_env = os.environ.get("ELSEA_CONCURRENCY_GPUS")
    if not gpu_list_env:
        return
    gpu_ids = [g.strip() for g in gpu_list_env.split(",") if g.strip()]
    if len(gpu_ids) <= 1:
        return

    import json as json_module
    import multiprocessing as mp_module
    import queue as queue_module
    import time as time_module

    import divea.framework as framework_module

    ctx = mp_module.get_context("fork")

    def _worker_loop(clients, indices, task_q, result_q, logger, eval_way):
        # CUDA_VISIBLE_DEVICES was already set for this process before it
        # was forked (see patched_run below) -- this worker sees exactly
        # one GPU, as local index 0. Fix up every one of its clients'
        # torch_device accordingly (see docstring: neural_rrea.py's
        # evaluate() uses conf.torch_device explicitly and would otherwise
        # try to address a device ordinal that does not exist here).
        import torch as torch_module
        # torch 1.13 lru_caches device_count(); the parent already filled it
        # with the job's full GPU count (via NVML, without initialising
        # CUDA), so the child's first CUDA call would run _check_capability
        # over devices 1..N-1 that this one-GPU worker cannot see and die
        # with "device >= 0 && device < num_gpus". Re-count under this
        # worker's CUDA_VISIBLE_DEVICES.
        torch_module.cuda.device_count.cache_clear()
        n_visible = torch_module.cuda.device_count()
        if n_visible != 1:
            raise RuntimeError(f"[elsea_concurrency_patch] worker sees {n_visible} GPUs, expected 1 "
                               f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
        for idx in indices:
            clients[idx].ea_module.conf.torch_device = torch_module.device("cuda:0")

        # RREA/runner.py calls nvidia_smi.nvmlInit() once at import, in the
        # parent; a forked child does not inherit a usable NVML session, so
        # Runner.__init__'s memory-usage log line would fail with
        # NVMLError_Uninitialized in every worker. NVML is not CUDA --
        # initialising it here is safe after fork.
        import nvidia_smi
        nvidia_smi.nvmlInit()

        try:
            _worker_rounds(clients, indices, task_q, result_q, logger, eval_way)
        except BaseException:
            # Without this the parent blocks on result_q.get() forever
            # after a worker has died.
            import traceback
            result_q.put(("error", traceback.format_exc()))
            raise

    def _worker_rounds(clients, indices, task_q, result_q, logger, eval_way):
        # Runs for the whole job's lifetime: blocks on task_q for each
        # round's "go" signal (the iteration number), None means shut down.
        for ite in iter(task_q.get, None):
            local_ignored = []
            for idx in indices:
                client = clients[idx]
                t31 = time_module.time()
                ignore = client.generate_dataset_from_partition()
                if ignore:
                    local_ignored.append(idx)
                    continue
                client.train_model()
                t32 = time_module.time()
                logger.log(json_module.dumps({
                    "machine": f"client_{idx + 1}", "msg_type": "running_time",
                    "value": t32 - t31, "process": "train", "iteration": ite,
                }))

                client.evaluate_model(eval_way)
                t33 = time_module.time()
                logger.log(json_module.dumps({
                    "machine": f"client_{idx + 1}", "msg_type": "running_time",
                    "value": t33 - t32, "process": "evaluate", "iteration": ite,
                }))

                client.generate_msg()
                t34 = time_module.time()
                logger.log(json_module.dumps({
                    "machine": f"client_{idx + 1}", "msg_type": "running_time",
                    "value": t34 - t33, "process": "generate_msg", "iteration": ite,
                }))
            result_q.put(local_ignored)

    def patched_run(self):
        self._prepare()
        print(f"index {1}")

        buckets = {g: [] for g in range(len(gpu_ids))}
        for idx in range(len(self.clients)):
            buckets[idx % len(gpu_ids)].append(idx)

        import torch as torch_parent
        if torch_parent.cuda.is_initialized():
            raise RuntimeError(
                "[elsea_concurrency_patch] CUDA is already initialised in the parent before "
                "the worker fork; forked workers would inherit an unusable CUDA context"
            )

        workers = []
        orig_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        # conf.gpu_ids (from --gpu_ids) is an INDEX into the job's own
        # CUDA_VISIBLE_DEVICES list, i.e. torch's "cuda:k" = the k-th
        # entry the scheduler handed this job -- not necessarily a physical
        # GPU number (a partial-node 2-GPU job may be handed "2,3", in which
        # case treating conf.gpu_ids="1" as a literal CUDA_VISIBLE_DEVICES
        # value would point outside this job's own allocation -- either no
        # device under a cgroup constraint, or another job's card without
        # one). Translate through the job's own visible list either way;
        # harmless no-op if the scheduler already renumbers from 0.
        orig_visible_list = orig_cuda_visible.split(",") if orig_cuda_visible else None
        try:
            for _g, indices in buckets.items():
                if not indices:
                    continue
                # All clients in this bucket already agree on conf.gpu_ids
                # (see docstring); that's an index into this job's own
                # original CUDA_VISIBLE_DEVICES list, translated below to
                # what this worker's process needs to see as its own
                # CUDA_VISIBLE_DEVICES, inherited at fork time, before its
                # interpreter ever touches CUDA.
                worker_gpu_id = self.clients[indices[0]].ea_module.conf.gpu_ids
                worker_cuda_visible = (
                    orig_visible_list[int(worker_gpu_id)] if orig_visible_list else str(worker_gpu_id)
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = worker_cuda_visible
                task_q = ctx.Queue()
                result_q = ctx.Queue()
                proc = ctx.Process(
                    target=_worker_loop,
                    args=(self.clients, indices, task_q, result_q, self.logger, self.eval_way),
                )
                proc.start()
                print(
                    f"[run_elsea_divea] worker for conf.gpu_ids={worker_gpu_id!r} (indices "
                    f"{indices}) -> CUDA_VISIBLE_DEVICES={worker_cuda_visible!r} "
                    f"(job's own CUDA_VISIBLE_DEVICES was {orig_cuda_visible!r})"
                )
                workers.append((proc, task_q, result_q))
        finally:
            if orig_cuda_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = orig_cuda_visible
        print(
            f"[run_elsea_divea] ELsEA concurrency patch: {len(workers)} persistent GPU "
            f"workers forked ({','.join(gpu_ids)}), subtasks split idx % {len(gpu_ids)}, "
            f"each worker's CUDA_VISIBLE_DEVICES pinned via its own clients' conf.gpu_ids, "
            f"translated through this job's CUDA_VISIBLE_DEVICES={orig_cuda_visible!r}"
        )

        t11 = time_module.time()
        self.server.g1_partitioner.partition_g1_entities()
        t12 = time_module.time()
        self.logger.log(json_module.dumps({
            "machine": "server", "msg_type": "running_time",
            "value": t12 - t11, "process": "partition_g1",
        }))

        for ite in range(0, self.max_iteration):
            t60 = time_module.time()
            self.server.ctx_builder.build_g1_context()
            t61 = time_module.time()
            self.logger.log(json_module.dumps({
                "machine": "server", "msg_type": "running_time",
                "value": t61 - t60, "process": "build_context1", "iteration": ite,
            }))

            t21 = time_module.time()
            self.server.g2_partitioner.partition_g2_entities()
            t22 = time_module.time()
            self.logger.log(json_module.dumps({
                "machine": "server", "msg_type": "running_time",
                "value": t22 - t21, "process": "partition_g2", "iteration": ite,
            }))

            t41 = time_module.time()
            self.server.ctx_builder.build_g2_context()
            t42 = time_module.time()
            self.logger.log(json_module.dumps({
                "machine": "server", "msg_type": "running_time",
                "value": t42 - t41, "process": "build_context2", "iteration": ite,
            }))

            t51 = time_module.time()
            self.server.g1_partitioner.build_g1_subgraphs()
            self.server.g2_partitioner.build_g2_subgraphs()
            t52 = time_module.time()
            self.logger.log(json_module.dumps({
                "machine": "server", "msg_type": "running_time",
                "value": t52 - t51, "process": "build_subgraphs", "iteration": ite,
            }))

            for _, task_q, _ in workers:
                task_q.put(ite)
            for proc, _, result_q in workers:
                # Poll so a worker killed outright (OOM, segfault -- no
                # chance to report) fails the job instead of hanging it.
                while True:
                    try:
                        result = result_q.get(timeout=60)
                        break
                    except queue_module.Empty:
                        if not proc.is_alive():
                            result = ("error", f"worker exited with code {proc.exitcode} without reporting")
                            break
                if isinstance(result, tuple) and result and result[0] == "error":
                    for p, _, _ in workers:
                        if p.is_alive():
                            p.terminate()
                    raise RuntimeError(
                        f"[elsea_concurrency_patch] worker failed during iteration {ite}:\n{result[1]}"
                    )
                self.ignore_ids.extend(result)
                if not proc.is_alive() and proc.exitcode not in (None, 0):
                    raise RuntimeError(
                        f"[elsea_concurrency_patch] worker died with code {proc.exitcode} "
                        f"during iteration {ite}"
                    )

            t51c = time_module.time()
            self.server.communicate(self.ignore_ids)
            t52c = time_module.time()
            self.logger.log(json_module.dumps({
                "machine": "server", "msg_type": "running_time",
                "value": t51c - t52c, "process": "communicate", "iteration": ite,
            }))

        for _, task_q, _ in workers:
            task_q.put(None)
        for proc, _, _ in workers:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(
                    f"[elsea_concurrency_patch] worker exited with code {proc.exitcode}"
                )

        self.evaluate()

    framework_module.ParallelEAFramework.run = patched_run
    print(
        f"[run_elsea_divea] ELsEA concurrency patch armed: {len(gpu_ids)} GPU workers "
        f"({','.join(gpu_ids)}), subtasks split idx % {len(gpu_ids)}"
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_elsea_divea.py <target_script.py> [args...]")

    target_script = sys.argv[1]
    sys.argv = sys.argv[1:]  # target script sees itself as argv[0], then its own args

    # This launcher lives under DistEA's own scripts/, not inside
    # ELsEA/DivEA/ -- Python sets sys.path[0] to the launcher's own
    # directory, not the caller's cwd, so without this the target script's
    # own `divea` package (and this stub's own `from divea.components_base
    # import ...`) would not resolve even when invoked from inside
    # ELsEA/DivEA/ as intended. Insert cwd, not the launcher's dirname.
    sys.path.insert(0, os.getcwd())

    install_neural_lightea_stub()
    install_load_weights_index_fix()
    install_neighbor_agg_layer_node_weights_fix()
    install_keras_rmsprop_shim()
    install_kg1_partition_override()
    install_tf_memory_cap()
    install_elsea_concurrency_patch()
    runpy.run_path(target_script, run_name="__main__")


if __name__ == "__main__":
    main()
