"""
run_clusterea.py -- launcher for ClusterEA's src/main.py that redirects its
hardcoded data paths, optionally swaps in DistEA's fold, and patches
upstream incompatibilities with current numpy / DGL.

Reads:  the DistEA data copies at MKDATA_DIR / DISTEA_OPENEA_DIR and,
        when CLUSTEREA_FOLD_DIR is set, that fold's train_links/test_links.
Writes: whatever ClusterEA's main.py writes (embeddings.pkl_*, logs).

dataset.py's load_dataset() hardcodes `LargeScaleEAData('../mkdata/', ...)`
for scale='large' with no CLI-exposed way to override the directory. That
path resolves relative to wherever main.py is invoked from (ClusterEA/src/,
per its own README), i.e. inside the third-party clone, which this launcher
leaves untouched.

Fix: monkey-patch dataset.load_dataset's 'large' branch to point at
DistEA's own data/clusterea_mkdata/ (already populated with de_triples1/
de_triples2/de_ent_links, copied from data/DBP1M_DE/{rel_triples_1,
rel_triples_2,ent_links} -- same underlying LargeEA-released DBP1M content
already downloaded for DistEA's own use, just renamed to LargeScaleEAData's
expected {lang}_triples{1,2}/{lang}_ent_links pattern) rather than
reimplementing load_dataset's small/medium branches, which are untouched
and correct as-is.

Usage (from inside ClusterEA/src/, matching main.py's own relative
expectations for small/medium scale datasets):
    CLUSTEREA_FOLD_DIR=<DistEA>/data/DBP1M_DE/316_1fold/1 \
        python <DistEA>/scripts/run_clusterea.py main.py --step 1 --scale large --lang de ...
(all args after the script name are passed through unchanged)
"""

# The ClusterEA env is Python 3.8: without this, the `list[tuple[str, str]]`
# annotation below is evaluated at def time and raises "'type' object is not
# subscriptable".
from __future__ import annotations

import os
import runpy
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DBP1M in LargeEA's mkdata layout, and the OpenEA datasets; override with environment variables.
MKDATA_DIR = os.environ.get("DISTEA_MKDATA_DIR", os.path.join(_REPO, "data", "clusterea_mkdata")) + "/"
DISTEA_OPENEA_DIR = os.environ.get(
    "DISTEA_OPENEA_DIR", os.path.join(_REPO, "data", "OpenEA_dataset_v2.0")) + "/"


def _read_uri_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                a, b = line.rstrip("\n").split("\t")
                pairs.append((a, b))
    return pairs


def apply_fold_override(ea, fold_dir: str) -> None:
    """Replace EAData's own train/test split with an explicit fold's.

    Needed for a like-for-like DistEA/ELsEA/ClusterEA comparison:
    EAData.__init__ builds its split as a PREFIX of ent_links in file order
    (train = link[:int(train_ratio*len)], test = the rest, shuffle_pairs
    False) -- a different split from the fold DistEA and ELsEA train/test
    on (custom_30pct_fold/1 at 15K, 721_5fold/1 at 100K: 20% train, not
    30%), and one whose train prefix overlaps those folds' TEST pairs, so
    evaluating ClusterEA's embeddings on the shared test set would leak.
    Only a retrain on the fold itself is comparable.

    ea.ent1/ea.ent2 are process_one_graph's {uri: id} dicts and ea.link is
    [(id1, id2), ...] with train = link[:train_cnt] -- so the override is
    just: link = fold train pairs + fold test pairs (URIs mapped through
    those dicts), train_cnt = len(train pairs). valid_links is ignored
    (ClusterEA has no validation concept; DistEA uses it only for early
    stopping -- train/test are what make the comparison fair).
    """
    train = _read_uri_pairs(os.path.join(fold_dir, "train_links"))
    test = _read_uri_pairs(os.path.join(fold_dir, "test_links"))

    def to_ids(pairs):
        out, miss = [], 0
        for u1, u2 in pairs:
            if u1 in ea.ent1 and u2 in ea.ent2:
                out.append((ea.ent1[u1], ea.ent2[u2]))
            else:
                miss += 1
        return out, miss

    tr, m1 = to_ids(train)
    te, m2 = to_ids(test)
    ea.link = tr + te
    ea.train_cnt = len(tr)
    if getattr(ea, "semi_link", None) is not None:
        ea.semi_link = None
    print(f"[run_clusterea] fold override from {fold_dir}: train={len(tr)} (unmapped {m1}) "
          f"test={len(te)} (unmapped {m2}) -- replaced EAData's prefix split")


def install_mkdata_path_patch() -> None:
    """Redirect load_dataset's hardcoded data paths.

    Also covers scale='small'/'medium' with ds='ids': the real load_dataset
    does OpenEAData('../OpenEA_dataset_v2.0/EN_{lang}_15K_V1/', ...), a path
    relative to wherever main.py is invoked from (ClusterEA/src/, per its
    own README) -- ClusterEA/OpenEA_dataset_v2.0/ doesn't exist in a fresh
    clone, same class of problem as the 'large' branch. Redirects to
    DistEA's own copy instead of reimplementing OpenEAData itself."""
    import dataset as dataset_module

    real_load_dataset = dataset_module.load_dataset

    def _finish(ea):
        # CLUSTEREA_FOLD_DIR (set by the job scripts): swap EAData's own
        # prefix split for the shared fold's train/test -- see
        # apply_fold_override. Unset => ClusterEA's native behavior.
        fold_dir = os.environ.get("CLUSTEREA_FOLD_DIR")
        if fold_dir:
            apply_fold_override(ea, fold_dir)
        print(ea)
        return ea

    def load_dataset_patched(scale="small", ds="ids", lang="fr", train_ratio=0.3, shuffle=False):
        if scale == "large":
            return _finish(dataset_module.LargeScaleEAData(
                MKDATA_DIR, lang=lang, train_ratio=train_ratio, shuffle_pairs=False
            ))
        if scale == "small" and ds == "ids":
            path = os.path.join(DISTEA_OPENEA_DIR, f"EN_{lang.upper()}_15K_V1") + "/"
            return _finish(dataset_module.OpenEAData(path, train_ratio=train_ratio, shuffle_pairs=shuffle))
        if scale == "medium" and ds == "ids":
            # Same fix as scale='small' above, for the 100K datasets:
            # real load_dataset's medium-scale branch does
            # OpenEAData('../OpenEA_dataset_v2.0/EN_{lang}_100K_V1/', ...),
            # same relative-path problem.
            path = os.path.join(DISTEA_OPENEA_DIR, f"EN_{lang.upper()}_100K_V1") + "/"
            return _finish(dataset_module.OpenEAData(path, train_ratio=train_ratio, shuffle_pairs=shuffle))
        return _finish(real_load_dataset(scale=scale, ds=ds, lang=lang, train_ratio=train_ratio, shuffle=shuffle))

    dataset_module.load_dataset = load_dataset_patched


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_clusterea.py <target_script.py> [args...]")

    target_script = sys.argv[1]
    sys.argv = sys.argv[1:]

    # Same reasoning as run_elsea_divea.py: this launcher lives under
    # DistEA's own scripts/, not inside ClusterEA/src/, so Python's
    # default sys.path[0] (the launcher's own directory) would not resolve
    # ClusterEA's own `dataset`/`framework`/`utils`/`dto` modules -- insert
    # cwd (ClusterEA/src/, where this is meant to be invoked from) instead.
    sys.path.insert(0, os.getcwd())

    # Upstream incompatibility (hit by --step 3): align_batch.py's
    # make_assoc() calls np.int, an alias numpy deprecated in 1.20 and
    # removed in later versions. numpy's own deprecation message says
    # restoring `np.int = int` is a safe, behavior-preserving fix (that is
    # what the alias was). --step 1 never touches align_batch.py.
    import numpy as _np
    if not hasattr(_np, "int"):
        _np.int = int

    install_mkdata_path_patch()

    # Upstream incompatibility (DBP1M, --model rrea-large dispatching to
    # the dual-amn mini-batch path): prev_models/duala_sample/wrapper.py's
    # train1step() calls MultiLayerNeighborSampler(fanouts, return_eids=True)
    # -- a kwarg DGL 0.9.0 no longer accepts (its __init__ dropped
    # return_eids entirely; sampled blocks carry dgl.EID unconditionally
    # now, so the flag became meaningless rather than being renamed).
    # No replacement kwarg needed, just drop it. Patched at the constructor rather than editing
    # ClusterEA's vendored wrapper.py, same pattern as the np.int shim
    # above: wrap the callable so any return_eids kwarg is silently
    # dropped, behavior otherwise identical (edge IDs were always going to
    # be present either way in this DGL version).
    import inspect as _inspect

    import dgl.dataloading as _dgl_dl
    _OrigSampler = _dgl_dl.MultiLayerNeighborSampler
    if "return_eids" not in _inspect.signature(_OrigSampler.__init__).parameters:
        # Second API drift in the same call site ("'list' object has no
        # attribute 'to'" at wrapper.py:330): DGL 0.9's sample_blocks()
        # returns the 3-tuple (input_nodes, output_nodes, blocks),
        # whereas the DGL that wrapper.py was written against
        # returned just `blocks`. wrapper.py iterates the return value and
        # calls .to(device) on each element; the two node-id tensors
        # survive that, the trailing `blocks` list doesn't. Return only
        # the blocks list, in the same outermost-first order (the loop
        # does blocks.insert(0, block), identical to the old ordering).
        class _CompatSampler(_OrigSampler):
            def sample_blocks(self, g, seed_nodes, exclude_eids=None):
                out = super().sample_blocks(g, seed_nodes, exclude_eids)
                return out[2] if isinstance(out, tuple) and len(out) == 3 else out

        def _sampler_compat(fanouts, *args, return_eids=None, **kwargs):
            return _CompatSampler(fanouts, *args, **kwargs)
        _dgl_dl.MultiLayerNeighborSampler = _sampler_compat

        # Third drift, same DGL 0.9 upgrade (at dgl_rrea.py:56): the model's forward()
        # does MultiLayerFullNeighborSampler(1).sample_blocks(g, src_nodes)[0]
        # at six call sites across dgl_rrea.py / dgl_sample_gcn.py, where
        # `g` is the full CPU graph and `src_nodes` comes off a block that
        # wrapper.py already moved to the GPU. Two things break there:
        #   (1) DGL 0.9's sample_neighbors() enforces seed-tensor device ==
        #       graph device ("Expect argument nodes["_N"] to have device
        #       cpu. But got cuda:0.") where the old DGL silently moved it;
        #   (2) the trailing [0] assumed the old "returns blocks" API; under
        #       the 3-tuple return it would pick input_nodes (a tensor) and
        #       the next line's temp_block.srcdata would fail.
        # Fix both in one subclass: move the seeds (tensor or dict of
        # tensors, some call sites pass {'entity': ...} / {'index': ...})
        # to g.device, then return only the blocks list so [0] is the first
        # block again. dtype is left alone -- the code already casts to
        # int32 where the hetero graph's idtype needs it. dgl_rrea.py binds
        # the name via `from dgl.dataloading import MultiLayerFullNeighborSampler`
        # when it is first imported inside runpy.run_path() below, i.e.
        # after this patch, so patching the module attribute is enough.
        _OrigFull = _dgl_dl.MultiLayerFullNeighborSampler

        def _to_graph_device(seed, dev):
            import torch as _torch
            if isinstance(seed, dict):
                return {k: (v.to(dev) if isinstance(v, _torch.Tensor) else v) for k, v in seed.items()}
            return seed.to(dev) if isinstance(seed, _torch.Tensor) else seed

        class _CompatFullSampler(_OrigFull):
            def sample_blocks(self, g, seed_nodes, exclude_eids=None):
                seed_nodes = _to_graph_device(seed_nodes, g.device)
                out = super().sample_blocks(g, seed_nodes, exclude_eids)
                return out[2] if isinstance(out, tuple) and len(out) == 3 else out

        _dgl_dl.MultiLayerFullNeighborSampler = _CompatFullSampler
    # else: this DGL version already accepts return_eids -- no patch needed.
    # Patched here (before target_script is loaded via runpy below), not at
    # wrapper.py's own `from dgl.dataloading import *` -- that statement
    # binds the name into wrapper.py's namespace at import time, and
    # wrapper.py is only first imported once main.py's own import chain
    # runs inside runpy.run_path(), i.e. strictly after this point.

    # Upstream bug: main.py's own module-level code sets
    # train_epoch = global_arguments.epoch (a bare int) whenever --epoch
    # is passed as anything other than its -1 default; only the -1/default
    # branch produces the correctly-shaped per-model list. step1_training()
    # unconditionally does train_epoch[ep] -- TypeError for any explicit
    # --epoch override. main.py has no __main__ guard around this
    # computation (only around the final run() call, confirmed via its
    # last two lines), so loading it with a non-"__main__" run_name
    # executes everything except that final call, letting us patch
    # train_epoch into a list here before invoking run() ourselves --
    # fixes it without touching ClusterEA's own source, and is a no-op
    # when --epoch is left at -1 (train_epoch is already a list then).
    #
    # Gotcha: runpy.run_path() does NOT
    # return the live module namespace that exec()'d code and the
    # functions defined in it actually reference. Internally it execs into
    # a temporary module's __dict__, then returns mod_globals.copy() --
    # a detached copy -- specifically because the temp module (and its
    # __dict__) may get torn down once run_path() returns. Mutating `ns`
    # itself is therefore a no-op as far as `run()`/`step1_training()` are
    # concerned; they still see the ORIGINAL train_epoch via their own
    # __globals__, which is NOT `ns` (ns is ns['run'].__globals__ -> False).
    # Fix: patch through ns['run'].__globals__ (the live dict all
    # module-level functions actually share), not through `ns` itself.
    ns = runpy.run_path(target_script, run_name="clusterea_patched")
    live_globals = ns["run"].__globals__
    if not isinstance(live_globals["train_epoch"], list):
        live_globals["train_epoch"] = [live_globals["train_epoch"]] * live_globals["n_semi_iter"]

    # Optional rrea-large embedding width. main.py's module-level
    # `model_dims` dict only changes what the DOWNSTREAM step
    # (clustering/partitioning, main.py's four
    # `model_dim=model_dims[model_name]` call sites, all inside
    # framework.get_partition_similarity_matrix/get_partition_nodes) is
    # TOLD to expect -- it does not control what width step 1's rrea-large
    # ENCODER actually produces (changing it alone makes step 2's
    # clustering GCN fail with "mat1 and mat2 shapes cannot be multiplied
    # (Nx600 and 128x16)"). That width is set by a literal hardcoded
    # inside DualASamplingWrapper.__init__ (prev_models/duala_sample/
    # wrapper.py): `if model_name == 'rrea-large': node_hidden = 100`,
    # then `overAllRREA(node_size=..., node_hidden=node_hidden, ...)`.
    # overAllRREA concatenates two RREA sub-encoders (entity + relation),
    # each outputting node_hidden*(depth+1) (depth=2 -> node_hidden*3) --
    # (overAllRREA.forward's `torch.cat((out_feature_ent,
    # out_feature_rel), dim=-1)`; observed width 600 = 100*3*2) -- so
    # real_dim = node_hidden * (depth+1) * 2.
    #
    # Fix: patch the real knob. overAllRREA always receives node_hidden
    # as an explicit keyword from DualASamplingWrapper.__init__ (one call
    # site, only for model_name == 'rrea-large' -- model_name == 'dual-
    # large' goes through the sibling `overAll` class instead, untouched
    # by this), so wrapping overAllRREA.__init__ to override that one
    # kwarg is a one-line interception, not a copy of the ~150-line
    # DualASamplingWrapper constructor. Importing prev_models.duala_
    # sample.dgl_rrea here (before ns["run"]() ever triggers
    # DualASamplingWrapper's own lazy `from .duala_sample.wrapper import
    # ...`) and patching the class object works regardless of import
    # order -- classes are shared mutable objects via sys.modules, so
    # whichever module later does `from .dgl_rrea import overAllRREA`
    # binds the SAME already-patched class.
    #
    # Opt-in via CLUSTEREA_RREA_LARGE_DIM=<desired total embedding width>
    # (e.g. "128"); unset => strict no-op (node_hidden stays 100, dim
    # stays 600, ClusterEA's default). Since real_dim = node_hidden*6 for
    # depth=2, an exact 128 isn't reachable (128/6 is not an integer) --
    # node_hidden is rounded to the nearest integer and the ACHIEVED dim
    # (not the requested one) is what gets written into model_dims for
    # step 2's consistency, so the clustering stage's expectation always
    # matches what step 1 actually produced, whatever that rounds to.
    rrea_large_dim = os.environ.get("CLUSTEREA_RREA_LARGE_DIM")
    if rrea_large_dim:
        requested_dim = int(rrea_large_dim)
        depth = 2  # matches wrapper.py's hardcoded depth=2 for rrea-large
        node_hidden_override = round(requested_dim / (2 * (depth + 1)))
        if node_hidden_override <= 0:
            raise SystemExit(
                f"[run_clusterea] CLUSTEREA_RREA_LARGE_DIM={requested_dim} too small "
                f"(rounds to node_hidden <= 0)"
            )
        achieved_dim = node_hidden_override * 2 * (depth + 1)

        import prev_models.duala_sample.dgl_rrea as dgl_rrea_module

        orig_overallrrea_init = dgl_rrea_module.overAllRREA.__init__

        def patched_overallrrea_init(self, *init_args, **init_kwargs):
            if "node_hidden" in init_kwargs:
                init_kwargs["node_hidden"] = node_hidden_override
            orig_overallrrea_init(self, *init_args, **init_kwargs)

        dgl_rrea_module.overAllRREA.__init__ = patched_overallrrea_init

        if "rrea-large" not in live_globals["model_dims"]:
            raise SystemExit(
                "[run_clusterea] CLUSTEREA_RREA_LARGE_DIM set but 'rrea-large' not in "
                f"model_dims; known: {sorted(live_globals['model_dims'])}"
            )
        old_dim = live_globals["model_dims"]["rrea-large"]
        live_globals["model_dims"]["rrea-large"] = achieved_dim
        print(
            f"[run_clusterea] rrea-large node_hidden overridden: 100 -> {node_hidden_override} "
            f"(requested dim {requested_dim}, achieved dim {achieved_dim}, exact not reachable "
            f"since dim = node_hidden*6); model_dims['rrea-large']: {old_dim} -> {achieved_dim}"
        )

    ns["run"]()


if __name__ == "__main__":
    main()
