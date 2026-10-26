"""
framework.py — DistEAFramework: multi-iteration refinement orchestrator.

--shared_h_init is an architectural alternative to
Steps 3/3a/3d/8 below: every rank holds the FULL entity table and
all_reduce()s it every epoch (Step 6) instead of a per-rank partitioned copy
patched at iteration boundaries — Steps 3/3a/3d/8 are skipped entirely in
this mode (nothing to refresh/regularize/push when there's no separate
rank-owned copy), and Step 2 builds h_init over the full global entity range
rather than just this rank's local subset. See dist_trainer.py's module
docstring and main.py's --shared_h_init help text for the full rationale.

Each iteration (default, non-shared-h_init path):
  1. DistGraphLoader.load()      → local graph with Halo nodes (halo_kg1
                                    AND halo_kg2 — see dist_graph.py)
  2. build_entity_features()     → deterministic h_init
  3. refresh halo_kg1 rows         → file-based, or OCD point-to-point (--use_ocd)
  3a. refresh halo_kg2 rows       → file-based only (restores KG2-KG2
                                    cross-partition edges' message passing;
                                    always on, not gated behind a flag)
  3c. dist_ctx_kg2.select_halo_kg2() → cross-rank KG2 negative-sampling
                                    candidates (--use_halo_kg2, requires
                                    --cache_dir; see dist_ctx_kg2.py) — a
                                    separate, complementary mechanism from
                                    3a: extra loss-only candidates vs. real
                                    graph edges
  3d. build consistency target    → (--consistency_lambda) reuses 3/3a's
                                    halo snapshots to
                                    build a fixed per-epoch gradient anchor
                                    for DistTrainer, on top of (not instead
                                    of) 3/3a's hard overwrite
  4. _map_links_to_local()       → train_links → local index pairs
  4b. compute_and_gather_vbip()  → periodic VBIP pair-weights (--use_mm)
  5. GCN + DistTrainer           → model + optimiser
  6. trainer.train_epoch() × N   → max_epochs training rounds (negative
                                    sampling extended with halo_kg2_emb when
                                    --use_halo_kg2 is set; consistency
                                    regularizer added to the loss every
                                    epoch when --consistency_lambda > 0)
  7. selective_eval()            → Hit@1 / Hit@10 / MRR
  7b. generate_pseudo_seeds()    → augment train_links (if --use_pseudo_seeds)
  8. write_rank_embeddings()     → write h_init state to shared filesystem
                                    (staleness-filtered when --use_saca;
                                    always includes local_kg2, needed by
                                    halo_kg2 refresh; --use_halo_kg2 also
                                    writes a separate model-output-width
                                    kg2_full/ shard)
  9. dist.barrier()              → all ranks must finish writing before next iter reads

The MM (VBIP pair weights), OCD (owner-compute dispatch) and SACA
(staleness-aware communication avoiding) mechanisms plug into steps 4b, 3
and 8 above respectively.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

import dist_ctx_kg2
import dist_halo_sync
import dist_ocd
from comm_stats import CommStats
from counterpart_disc import assign_kg2_to_ranks_locality
from dataload import Data
from dist_graph import DistGraphLoader, build_rrea_graph, sample_halo_rrea_graph
from dist_trainer import DistTrainer
from dist_utils import all_reduce_avg
from dist_vbip import compute_and_gather_vbip
from epoch_select import EpochSelector, parse_eval_splits, pick_best_iteration, split_prefix
from eval import generate_pseudo_seeds, selective_eval
from minibatch_eval import minibatch_generate_pseudo_seeds, minibatch_selective_eval
from minibatch_loss import describe as describe_minibatch_loss
from minibatch_loss import extend_seeds_with_negatives, minibatch_loss
from minibatch_ocd import build_full_owner_map, gather_minibatch_features
from models.gcn import build_entity_features
from models.rrea import RREA
from rrea_minibatch import RREAMinibatchIndex, prepare_rrea_batch
from sparse_table import SparseEntityTable
from train_opt import build_lr_scheduler, build_optimizer

# ---------------------------------------------------------------------------
#  Shared-filesystem helpers (module-level for direct testability)
# ---------------------------------------------------------------------------


def write_rank_embeddings(
    cache_dir: str, rank: int, ids: np.ndarray, emb: np.ndarray
) -> None:
    """Write one rank's KG1 embeddings to the shared filesystem."""
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, f"ids_{rank}.npy"), ids.astype(np.int64))
    np.save(os.path.join(cache_dir, f"emb_{rank}.npy"), emb.astype(np.float32))


def push_final_embeddings(
    cache_backend: str,
    cache_dir: str,
    partition_dir: str,
    world_size: int,
    rank: int,
    all_global_ids: np.ndarray,
    local_kg1_idx: np.ndarray,
    local_kg2_idx: np.ndarray,
    h_init_for_eval,
    grpc_addrs: str = "",
) -> int:
    """
    Stage C: push this rank's final-iteration local KG1+KG2 embeddings to a
    post-training lookup service (leapgnn_cache.py's EmbeddingCache).

    No-ops (returns 0) when cache_backend == "none". Extracted as a
    standalone function (matching write_rank_embeddings/_patch_halo_rows'
    module-level pattern) so it's directly testable without a full
    DistEAFramework.run() end-to-end call.

    Returns the number of embeddings pushed.
    """
    if cache_backend == "none":
        return 0

    from leapgnn_cache import make_embedding_cache

    addrs = [a for a in grpc_addrs.split(",") if a]
    cache = make_embedding_cache(
        backend=cache_backend,
        cache_dir=cache_dir,
        world_size=world_size,
        grpc_addrs=addrs,
        partition_dir=partition_dir,
    )
    push_ids = np.concatenate(
        [all_global_ids[local_kg1_idx], all_global_ids[local_kg2_idx]]
    )
    push_emb = np.concatenate(
        [
            h_init_for_eval[local_kg1_idx].detach().cpu().numpy(),
            h_init_for_eval[local_kg2_idx].detach().cpu().numpy(),
        ],
        axis=0,
    )
    cache.push(push_ids, push_emb, rank)
    cache.barrier()
    return len(push_ids)


def _patch_halo_rows(
    h_init: torch.Tensor,
    halo_kg1_idx: np.ndarray,
    halo_global_ids: np.ndarray,
    id_to_emb: dict[int, np.ndarray],
    alpha: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """
    Shared row-patching logic for both the file-based and OCD halo-refresh
    paths. h_init and the pushed embeddings are both output_dim-wide by
    construction — Step 8 pushes h_init's own values, not the (wider, for
    RREA) model forward-pass output — so this is a direct copy.

    alpha=1.0 (default) is a hard overwrite — matches all existing callers'
    behaviour exactly. alpha<1.0 blends toward the remote value instead
    (h_init[li] = (1-alpha)*h_init[li] + alpha*remote) — a gentler
    alternative for --halo_refresh_interval, since hard mid-epoch
    overwrites were found to hurt accuracy rather than help: frequent hard overwrites may disrupt --persist_optimizer's
    accumulated Adam momentum more than the staleness they fix.
    """
    if not id_to_emb:
        return h_init, 0
    refreshed = 0
    for li, gid in zip(halo_kg1_idx, halo_global_ids, strict=False):
        feat = id_to_emb.get(int(gid))
        if feat is not None:
            remote = torch.from_numpy(feat).to(h_init.device)
            if alpha >= 1.0:
                h_init[li] = remote
            else:
                h_init[li] = (1.0 - alpha) * h_init[li] + alpha * remote
            refreshed += 1
    return h_init, refreshed


def _reapply_halo_patches(
    h_data: torch.Tensor,
    halo_kg1_idx: np.ndarray,
    halo_kg2_idx: np.ndarray,
    all_global_ids: np.ndarray,
    halo_id_to_emb: dict[int, np.ndarray],
    halo_kg2_id_to_emb: dict[int, np.ndarray],
) -> int:
    """
    Re-apply Step 3/3a's halo-row patches ON TOP of a table that was just
    restored wholesale (--warm_start's `h_init.data = self._h_state`).

    Step 3/3a patch h_init_raw, a fresh build_entity_features() tensor;
    Step 3b wraps it as the Parameter; the warm-start restore then replaces
    the whole table with the previous iteration's locally-trained copy —
    halo rows included, i.e. this rank's own stale, locally-drifted copies
    of other ranks' entities, not the owners' freshly pushed values. On the
    default path (warm_start on, no --persist_optimizer) that silently
    discards the cross-rank refresh after iteration 0 — the same failure
    the persist_optimizer branch guards against by patching its reused
    parameter in place. Gated behind --halo_refresh_after_warm_start
    (main.py; default off) so the default path keeps its original semantics.

    In place on h_data. Returns the number of rows actually patched (ids
    missing from the cache are skipped, as in _patch_halo_rows). The
    optimizer is rebuilt every iteration on this path, so no Adam-moment
    reset (cf. --reset_halo_momentum) is needed here.
    """
    n = 0
    with torch.no_grad():
        if halo_id_to_emb and len(halo_kg1_idx) > 0:
            _, r = _patch_halo_rows(
                h_data, halo_kg1_idx, all_global_ids[halo_kg1_idx], halo_id_to_emb,
            )
            n += r
        if halo_kg2_id_to_emb and len(halo_kg2_idx) > 0:
            _, r = _patch_halo_rows(
                h_data, halo_kg2_idx, all_global_ids[halo_kg2_idx], halo_kg2_id_to_emb,
            )
            n += r
    return n


def _merge_pseudo_seeds(
    train_links: list[tuple[int, int]],
    new_seeds: list[tuple[int, int]],
    test_links_set: set[tuple[int, int]],
    known_pairs: set[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], int, int]:
    """
    Merge pseudo-seeds into train_links with entity-level freezing —
    matches RREA/runner.py's reference:
    `Runner.iterative_training()`/`train()` remove a matched entity from
    `rest_set_1`/`rest_set_2` the moment it's consumed, so it can never
    receive a second, potentially contradictory pseudo-seed in a later
    round. DivEA's own port has the same safeguard (`pseu_exist_map`,
    components_base.py). Deduplicating only by exact (kg1_id, kg2_id) tuple
    would let the same kg1_id (or kg2_id) accumulate multiple, contradictory
    pairs across iterations, corrupting both alignment_loss's positive-pair
    signal for that entity and, with --reassign_kg2_every, the seed_links
    fed into assign_kg2_to_ranks_locality.

    Note: this merge also rejects any mined pair that equals a test pair
    (test_links_set); --pseudo_seed_policy testside/unaligned avoids this.

    Returns (updated_train_links, added_count, correct_estimate) --
    correct_estimate counts accepted pairs found in known_pairs (train+valid,
    used only for logging estimated pseudo-seed precision).
    """
    updated = list(train_links)
    used_kg1 = {int(a) for a, b in updated}
    used_kg2 = {int(b) for a, b in updated}
    known_pairs = known_pairs or set()
    added = 0
    correct_estimate = 0
    for pair in new_seeds:
        p = (int(pair[0]), int(pair[1]))
        if p[0] not in used_kg1 and p[1] not in used_kg2 and p not in test_links_set:
            updated.append(p)
            used_kg1.add(p[0])
            used_kg2.add(p[1])
            added += 1
            if p in known_pairs:
                correct_estimate += 1
    return updated, added, correct_estimate


def _pseudo_seed_candidates(
    policy: str,
    refresh: str,
    train_links,
    orig_train_links,
    valid_links,
    test_links,
    kg1_entities,
    kg2_entities,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Base links + candidate entity ids for one
    pseudo-seed round under --pseudo_seed_policy unaligned|testside.

    refresh="accumulate" (RREA's rest_set protocol): the base is the current
    train_links, earlier rounds' pseudo-seeds included, so their entities
    leave the pool for good. refresh="replace": the base is the ORIGINAL
    train links and every round re-mines the whole pool with the newer model;
    the previous round's pseudo-seeds are dropped, so an early wrong pair can
    be corrected and later rounds are not limited to the hardest leftovers.
    The pool is always minus train + valid entities; testside restricts it to
    the entity sets of the test links (never their pairing)."""
    base = list(orig_train_links) if refresh == "replace" else list(train_links)
    known = base + list(valid_links)
    k1 = {int(a) for a, b in known}
    k2 = {int(b) for a, b in known}
    if policy == "testside":
        pool1 = {int(a) for a, b in test_links}
        pool2 = {int(b) for a, b in test_links}
    else:
        pool1 = {int(x) for x in kg1_entities}
        pool2 = {int(x) for x in kg2_entities}
    return (
        base,
        np.asarray(sorted(pool1 - k1), dtype=np.int64),
        np.asarray(sorted(pool2 - k2), dtype=np.int64),
    )


def _reset_adam_momentum_for_rows(
    optimizer: optim.Optimizer,
    param: nn.Parameter,
    row_idx: np.ndarray,
) -> bool:
    """
    Zero Adam's exp_avg/exp_avg_sq for specific rows of `param`, so the
    NEXT optimizer.step() treats those rows as freshly-initialized (no
    momentum) instead of continuing to apply momentum accumulated for the
    value those rows held BEFORE an external overwrite (e.g. halo-row
    patching).

    Motivation: attempts at "sync h_init more often" made accuracy worse,
    not better, even without mid-epoch disruption. --persist_optimizer
    keeps the SAME optimizer object (and its Adam moment estimates) alive
    across iterations, and _patch_halo_rows writes directly into
    h_init.data (bypassing optimizer.step() entirely) — so after a patch,
    Adam's exp_avg/exp_avg_sq for those rows reflect a gradient history
    computed against the OLD value, not the newly-arrived one. (The
    optimizer is built from `list(model.parameters()) + [h_init,
    loss_weight]`, so halo rows are NOT excluded from h_init's gradient
    updates despite being read-only in the graph-structural sense; they
    get real per-epoch Adam updates like any other row.)

    Adam-only: SGD/RMSprop use different state keys and aren't handled
    here; the h_init-consistency experiments all used --optimizer adam,
    matching ELsEA's own choice.

    Returns False (no-op) on the first iteration a parameter is optimized,
    when the optimizer has no state for it yet — nothing to reset.
    """
    state = optimizer.state.get(param)
    if not state or "exp_avg" not in state:
        return False
    idx = torch.as_tensor(row_idx, dtype=torch.long, device=param.device)
    if len(idx) == 0:
        return False
    with torch.no_grad():
        state["exp_avg"][idx] = 0
        state["exp_avg_sq"][idx] = 0
    return True


def load_id_to_emb(cache_dir: str, world_size: int) -> dict[int, np.ndarray]:
    """
    Load every rank's pushed {ids,emb}_{rank}.npy shards into one
    global_id -> embedding lookup. Shared by halo_kg1 row-patching
    (refresh_halo_rows) and halo_kg2 candidate fetch (dist_ctx_kg2), so both
    read the same on-disk shards once per iteration instead of twice.

    Must be called after dist.barrier() so all ranks' files are visible.
    """
    id_to_emb: dict[int, np.ndarray] = {}
    for r in range(world_size):
        ids_path = os.path.join(cache_dir, f"ids_{r}.npy")
        emb_path = os.path.join(cache_dir, f"emb_{r}.npy")
        if not os.path.exists(ids_path):
            continue
        ids = np.load(ids_path)
        emb = np.load(emb_path)
        for i, gid in enumerate(ids):
            id_to_emb[int(gid)] = emb[i]
    return id_to_emb


def refresh_halo_rows(
    h_init: torch.Tensor,
    halo_kg1_idx: np.ndarray,
    halo_global_ids: np.ndarray,
    cache_dir: str,
    world_size: int,
) -> torch.Tensor:
    """
    Patch halo rows in h_init from embeddings on the shared filesystem.

    Must be called after dist.barrier() so all ranks' files are visible.
    """
    id_to_emb = load_id_to_emb(cache_dir, world_size)
    h_init, refreshed = _patch_halo_rows(
        h_init, halo_kg1_idx, halo_global_ids, id_to_emb
    )
    logging.debug(f"refresh_halo_rows: {refreshed}/{len(halo_kg1_idx)} rows updated")
    return h_init


def fetch_halo_kg2_emb(
    halo_kg2_ids: list[int],
    id_to_emb: dict[int, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """
    Build a (M, D) tensor of halo_kg2 embeddings from id_to_emb (must be the
    kg2_full/ channel — model-output width, e.g. RREA's
    output_dim*2*(num_layers+1) — NOT the h_init-width halo_kg1 channel;
    dist_trainer.py concatenates the result onto emb_for_loss along dim=0,
    which needs matching width along dim=1). IDs with no pushed embedding
    yet (e.g. iteration 0, or never written by their owning rank) are
    simply dropped — halo_kg2 is a best-effort negative-sampling aid, not a
    correctness requirement, so a partial or empty result is always valid.

    Returns:
        (tensor of shape (M_found, D), M_found)
    """
    rows = [id_to_emb[gid] for gid in halo_kg2_ids if gid in id_to_emb]
    if not rows:
        return torch.zeros(0, 0, device=device), 0
    arr = np.stack(rows, axis=0).astype(np.float32)
    return torch.from_numpy(arr).to(device), len(rows)


def saca_filter(
    prev_emb: dict[int, np.ndarray],
    ids: np.ndarray,
    emb: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Staleness-Aware Communication Avoiding (SACA): drop entities whose
    embedding hasn't changed enough (relative-L2) to be worth rewriting/
    re-broadcasting this iteration. Entities with no prior value (first
    write) are always kept. Sender-side decision: the owner holds both the
    previous and the new value, so receivers need no extra state.

    Returns:
        (filtered_ids, filtered_emb, skipped_count)
    """
    if not prev_emb or threshold <= 0:
        return ids, emb, 0

    keep_mask = np.ones(len(ids), dtype=bool)
    for i, gid in enumerate(ids):
        prev = prev_emb.get(int(gid))
        if prev is None:
            continue
        denom = np.linalg.norm(prev) + 1e-12
        rel_change = np.linalg.norm(emb[i] - prev) / denom
        if rel_change < threshold:
            keep_mask[i] = False

    skipped = int((~keep_mask).sum())
    return ids[keep_mask], emb[keep_mask], skipped


# ---------------------------------------------------------------------------
#  Framework
# ---------------------------------------------------------------------------


class DistEAFramework:
    """
    Top-level distributed entity alignment training loop.

    Args:
        rank:          this process's rank
        world_size:    total number of ranks
        data:          Data instance (loaded dataset)
        partition_dir: directory with {rank}.npy + kg2_assignments.json
        cfg:           argparse.Namespace with training hyperparameters
        device:        torch.device
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        data: Data,
        partition_dir: str,
        cfg,
        device: torch.device,
    ):
        self.rank = rank
        self.world_size = world_size
        self.data = data
        self.partition_dir = partition_dir
        self.cfg = cfg
        self.device = device
        self.cache_dir: str | None = getattr(cfg, "cache_dir", None) or None

        hidden_dim = getattr(cfg, "hidden_dim", None)
        if rank == 0 and hidden_dim is not None and hidden_dim != cfg.output_dim:
            logging.warning(
                f"--hidden_dim={hidden_dim} is ignored: RREA (models/rrea.py, "
                f"ELsEA's actual primary model) has no per-layer learnable "
                f"transform on the entity-table side either, so h_init "
                f"operates entirely at --output_dim={cfg.output_dim}. Set "
                f"--hidden_dim={cfg.output_dim} to silence this warning."
            )

        if (getattr(cfg, "pseudo_seed_refresh", "accumulate") == "replace"
                and (getattr(cfg, "pseudo_seed_policy", "legacy") or "legacy") == "legacy"):
            raise ValueError(
                "--pseudo_seed_refresh replace needs --pseudo_seed_policy "
                "unaligned|testside (the legacy merge filters by test labels)."
            )

        # --reassign_kg2_every can change a rank's
        # local_kg2/halo_kg2 membership between iterations. Under
        # --persist_optimizer, h_init and optimizer.state[h_init] are a
        # single fixed-shape tensor/state reused verbatim across iterations
        # with no shape check at all (see Step 3d/persist_optimizer reuse
        # below) -- a membership change there would silently misalign rows
        # or crash. Only the non-persist_optimizer path is supported, where
        # h_init is rebuilt from scratch every iteration regardless (Step 2)
        # and a changed assignment "just works" mechanically. Fail fast
        # rather than let this corrupt a long run.
        if getattr(cfg, "reassign_kg2_every", 0) > 0 and getattr(cfg, "persist_optimizer", False):
            raise ValueError(
                "--reassign_kg2_every > 0 is not yet supported together with "
                "--persist_optimizer: persisted h_init/optimizer state assumes "
                "a fixed local_kg2/halo_kg2 layout across iterations, which "
                "reassignment breaks. Drop "
                "--persist_optimizer to use --reassign_kg2_every for now."
            )

        if getattr(cfg, "cache_backend", "none") != "none" and not self.cache_dir:
            raise ValueError(
                "--cache_backend requires --cache_dir (push() writes "
                "ids_{rank}.npy/emb_{rank}.npy there for the Stage C "
                "embedding-cache service to read)."
            )

        self.results_dir: str | None = getattr(cfg, "results_dir", None) or None

        self.train_links: list[tuple[int, int]] = data.load_train_alignment()
        self.valid_links: list[tuple[int, int]] = data.load_valid_alignment()
        self.test_links  = data.load_test_alignment()
        # --pseudo_seed_refresh replace re-mines
        # every round on top of the ORIGINAL train links.
        self._orig_train_links: list[tuple[int, int]] = list(self.train_links)
        self._iter_metrics: list[dict] = []

        # Pre-built sets for O(1) look-ups
        self._test_links_set: set[tuple[int, int]] = {
            (int(a), int(b)) for a, b in self.test_links
        }
        # Known correct pairs (train + valid) used for pseudo-seed quality estimation
        self._known_pairs: set[tuple[int, int]] = {
            (int(a), int(b)) for a, b in self.train_links + self.valid_links
        }

        # Warm-start state: model weights from the previous iteration
        self._model_state: dict | None = None

        # Warm-start state: learnable entity embeddings from the previous iteration
        self._h_state: torch.Tensor | None = None

        # Iteration-invariant setup reused across
        # outer iterations (pseudo-seed runs): the loaded local graph and the
        # ID-seeded feature table (build_entity_features is a per-entity
        # Python loop, ~24 s on DBP1M N=1). Same values as a rebuild; off when
        # --reassign_kg2_every can change the partition between iterations.
        self._setup_cache: tuple | None = None
        self._feat_cache: torch.Tensor | None = None

        # GAS (cluster-wise training) state, built lazily on first epoch when
        # --gas_clusters > 0; graph structure is iteration-invariant so
        # these persist for the whole run (histories carry across
        # iterations by design -- GAS warm-up, staleness shrinks as
        # clusters revisit).
        self._gas_clusters = None
        self._gas_history = None
        self._gas_final: torch.Tensor | None = None

        # Valid-set early-stopping state
        self._best_valid_hit1: float = 0.0
        self._patience_counter: int = 0

        # Best-so-far checkpoint (by valid Hit@1, only populated when
        # --eval_on_valid is set) — separate from _model_state/_h_state
        # above, which are the WARM-START snapshot and get overwritten every
        # iteration regardless of quality. Without this, a run that peaks
        # early and then degrades (e.g. from the fresh-optimizer-per-iteration
        # effect) would save the LAST iteration's
        # weights to model.pt even though an earlier iteration scored higher.
        self._best_model_state: dict | None = None
        self._best_h_state: torch.Tensor | None = None

        # --persist_optimizer: keep the SAME model/optimizer/h_init/loss_weight
        # objects alive across iterations instead of rebuilding + warm-start-
        # loading each time. RREA/runner.py's iterative_training() trains one
        # continuous self.model/optimizer across all 5 turns and never resets
        # either — DistEA's own per-iteration rebuild (even with warm_start
        # copying weights across) still discards Adam's moment estimates every
        # iteration, which diverges from RREA's algorithm and causes
        # accuracy to peak at iteration 2 and decline afterwards. halo_kg1 rows are patched onto the reused h_init
        # in place (see where it's applied below) so this is safe at
        # world_size>1 too, not just world_size=1.
        self._persistent_model = None
        self._persistent_optimizer = None
        self._persistent_h_init: nn.Parameter | None = None
        self._persistent_loss_weight: nn.Parameter | None = None

        # This rank's own KG1 entities' last-computed embeddings. Doubles as
        # (a) SACA's staleness-comparison baseline and (b) the data OCD
        # serves to peers that request these entities — both need "what did
        # I last compute for entities I own", so they share one dict.
        self._prev_written_emb: dict[int, np.ndarray] = {}
        self._saca_skipped: int = 0
        self._saca_total: int = 0

        # OCD: entity_id -> owner_rank, static for the whole run, built lazily.
        self._owner_map: dict[int, int] | None = None

        # OCD's isend/irecv/all_gather_object need a CPU-capable (gloo)
        # process group regardless of the world's default backend. Under
        # nccl (the usual multi-node backend), point-to-point ops lazily
        # create a separate per-rank-pair NCCL sub-communicator the first
        # time two ranks exchange directly, and that bootstrap can fail
        # across physical nodes (`ncclInternalError` in ncclSocketInit). A
        # dedicated gloo group uses a P2P path that works on one node and
        # across nodes alike. Created here (unconditionally, identically on every
        # rank) rather than lazily inside the per-iteration halo-refresh
        # branch: dist.new_group() is itself a collective call and must be
        # issued by every rank in the same order, which per-rank
        # data-dependent conditions (e.g. len(halo_kg1_idx) > 0) can't
        # guarantee.
        # --minibatch always needs the gloo P2P group for its
        # per-batch cross-rank fetch, same as --use_ocd — see
        # minibatch_ocd.py/gather_minibatch_features.
        self._ocd_gloo_group = (
            dist.new_group(backend="gloo")
            if (getattr(cfg, "use_ocd", False) or getattr(cfg, "minibatch", False))
            else None
        )

        # MM: cached VBIP pair-weights, refreshed every --mm_sync_interval iterations.
        self._vbip_pair_weights: torch.Tensor | None = None

        # Halo refresh backend (Step 3/8, full-graph path only): unifies onto
        # dist_halo_sync.py's HaloSync abstraction instead of hardcoding a
        # raw shared-filesystem read (see main.py's --cache_backend help and
        # dist_halo_sync.py's own module docstring). Constructed once and
        # reused across every iteration — GrpcHaloSync tracks its own spawned
        # subprocess list internally, so a fresh instance per iteration would
        # lose track of previously-started servers and leak them. A single
        # persistent HaloSync object works correctly on a single node too
        # (--cache_backend grpc doesn't require multi-node — GrpcHaloSync's
        # server addresses are just whatever --grpc_addrs names, commonly
        # 127.0.0.1:{port+rank} for one node).
        self._halo_sync = dist_halo_sync.make_halo_sync(
            cfg, rank, world_size, partition_dir
        )

        # KG2 Counterpart Halo (--use_halo_kg2): global KG2-KG2 adjacency for
        # BFS candidate selection, built once (structure is static across
        # iterations; only which entities are *anchors* changes as
        # train_links grows via pseudo-seeds — see dist_ctx_kg2.py).
        self._kg2_adj: dict[int, set[int]] | None = None
        if getattr(cfg, "use_halo_kg2", False):
            self._kg2_adj = dist_ctx_kg2.build_kg2_adjacency(data.kg2_triples)

        # Per-iteration instrumentation (reset every iteration; see CommStats).
        self._comm_stats: CommStats | None = None
        self._total_wall_time: float | None = None
        # Per-rank stats gathered each iteration.
        self._per_rank: list | None = None

    def run(self):
        if getattr(self.cfg, "minibatch", False):
            return self._run_minibatch()

        cfg = self.cfg
        rank = self.rank
        run_start = time.perf_counter()

        use_mm = getattr(cfg, "use_mm", False)
        use_ocd = getattr(cfg, "use_ocd", False)
        use_saca = getattr(cfg, "use_saca", False)
        shared_h_init = getattr(cfg, "shared_h_init", False)
        # --defer_test_eval: with the iteration chosen
        # by valid (--select_by valid, --eval_on_valid), only the chosen
        # iteration's test numbers are reported, so the per-iteration test
        # eval (DBP1M N=1: 27-31 s each) is skipped and run once after the
        # loop on that iteration's state, which every rank keeps (the eval
        # is collective). The reported test metrics are those of the same
        # model; the unchosen iterations carry no test numbers.
        _defer_test = bool(getattr(cfg, "defer_test_eval", False))
        if _defer_test and not (
            getattr(cfg, "eval_on_valid", False) and self.valid_links
            and (getattr(cfg, "select_by", "test") or "test") == "valid"
            and not (getattr(cfg, "reassign_kg2_every", 0) or 0) > 0
        ):
            if rank == 0:
                logging.warning("--defer_test_eval needs --eval_on_valid, valid links, "
                                "--select_by valid and a graph that stays fixed across "
                                "iterations (no --reassign_kg2_every); running the "
                                "per-iteration test eval instead")
            _defer_test = False
        _defer_best: tuple | None = None  # (iteration, model state, h_init) on this rank

        for iteration in range(cfg.max_iterations):
            iter_start = time.perf_counter()
            self._comm_stats = CommStats()
            iter_entry: dict | None = None

            # Master node
            if rank == 0:
                logging.info(
                    f"\n{'=' * 60}\n Iteration {iteration + 1}/{cfg.max_iterations}\n{'=' * 60}"
                )

            # ── Step 1: Load distributed graph ──────────────────────────
            # Setup timing (measurement only): the
            # per-iteration rebuild is ~37 s on DBP1M N=1, which pseudo-seed
            # runs pay once per extra iteration.
            _t_setup = [time.perf_counter()]
            _reuse_setup = not (getattr(cfg, "reassign_kg2_every", 0) > 0)
            _setup_cached = _reuse_setup and self._setup_cache is not None
            if _setup_cached:
                (
                    global2local,
                    all_global_ids,
                    local_kg1_idx,
                    halo_kg1_idx,
                    local_kg2_idx,
                    halo_kg2_idx,
                    g,
                ) = self._setup_cache
                _t_setup += [time.perf_counter(), time.perf_counter()]
            else:
                loader = DistGraphLoader(
                    self.data, self.partition_dir, rank, self.world_size, self.device,
                    no_halo_kg1=getattr(cfg, "no_halo_kg1", False),
                    no_halo_kg2=getattr(cfg, "no_halo_kg2", False),
                    full_graph=bool(getattr(cfg, "ablate_full_graph", False)),
                )
                (
                    _,
                    global2local,
                    all_global_ids,
                    local_kg1_idx,
                    halo_kg1_idx,
                    local_kg2_idx,
                    halo_kg2_idx,
                ) = loader.load()
                _t_setup.append(time.perf_counter())
                # --drop_halo_halo_edges: keep only triples touching an owned
                # row (the kwarg is passed only when on, so the default call
                # is unchanged).
                _rg_kw = {}
                if getattr(cfg, "drop_halo_halo_edges", False):
                    _own = np.zeros(len(all_global_ids), dtype=bool)
                    _own[np.asarray(local_kg1_idx, dtype=np.int64)] = True
                    _own[np.asarray(local_kg2_idx, dtype=np.int64)] = True
                    _rg_kw["local_mask"] = _own
                g = build_rrea_graph(
                    self.data, global2local, all_global_ids, self.data.rel_num, self.device,
                    **_rg_kw,
                )
                if _rg_kw and rank == 0:
                    logging.info(
                        f"[Iter {iteration + 1}] --drop_halo_halo_edges: RREA graph "
                        f"{int(g.edge_index.size(0))} directed entries (halo-halo triples left out)"
                    )
                _t_setup.append(time.perf_counter())
                if _reuse_setup:
                    self._setup_cache = (
                        global2local, all_global_ids, local_kg1_idx, halo_kg1_idx,
                        local_kg2_idx, halo_kg2_idx, g,
                    )

            # ── Step 2: Build initial features (deterministic random) ────
            # Width = output_dim: h_init is RREA's entity embedding table
            # (see models/rrea.py) — relation embeddings are separate and
            # live inside the model itself (self.rel_emb), kept in sync via
            # ordinary all_reduce like any other parameter.
            #
            # --shared_h_init: build over the FULL
            # global entity range instead of just this rank's local subset,
            # identical on every rank (build_entity_features is ID-seeded,
            # order-independent) — this is the table every rank all_reduces
            # every epoch (see Step 6), replacing the halo-patch mechanism
            # entirely rather than reconciling it after the fact.
            if shared_h_init:
                full_ids = np.arange(self.data.ent_num_1 + self.data.ent_num_2)
                h_init_raw = build_entity_features(full_ids, cfg.output_dim, self.device)
                local_idx = torch.from_numpy(all_global_ids).to(self.device)
            else:
                if _reuse_setup and self._feat_cache is not None:
                    h_init_raw = self._feat_cache.to(self.device, copy=True)
                else:
                    _feat = build_entity_features(
                        all_global_ids, cfg.output_dim, torch.device("cpu")
                    )
                    if _reuse_setup:
                        self._feat_cache = _feat
                    h_init_raw = _feat.to(self.device, copy=True)
                local_idx = None
            # --init_scale: build_entity_features gives
            # unit-norm rows; the original RREA's Keras TokenEmbedding starts
            # at U(-0.05, 0.05) (row norm ~0.33 at dim 128), and the L1
            # margin loss is an absolute quantity, so the embedding scale
            # changes the hinge regime. 1.0 (default) = unchanged.
            _init_scale = float(getattr(cfg, "init_scale", 1.0) or 1.0)
            if _init_scale != 1.0:
                h_init_raw = h_init_raw * _init_scale
            _t_setup.append(time.perf_counter())

            # ── Step 3: Refresh halo rows — file-based, or OCD point-to-point ──
            # Skipped entirely under --shared_h_init: there's no separate
            # rank-owned halo copy to refresh when every rank already holds
            # (and all_reduces every epoch — Step 6) the full table.
            # halo_id_to_emb is kept in the outer scope (not just inside the
            # file-based branch) so Step 3c (KG2 Counterpart Halo) can reuse
            # it below without re-reading the same on-disk shards.
            # Width: Step 8 pushes h_init's own (output_dim-wide) values, not
            # the wider RREA model output, so the cached rows patch h_init
            # directly. halo_kg1_idx is empty at world_size=1, so this block
            # only runs for world_size>1.
            halo_id_to_emb: dict[int, np.ndarray] = {}
            if not shared_h_init and iteration > 0 and len(halo_kg1_idx) > 0 and (self.cache_dir or use_ocd):
                halo_global_ids = all_global_ids[halo_kg1_idx]
                with self._comm_stats.timed("iter_halo_refresh"):
                    if use_ocd:
                        if self._owner_map is None:
                            self._owner_map = dist_ocd.build_owner_map(
                                self.partition_dir, self.world_size
                            )
                        halo_id_to_emb = dist_ocd.request_and_fulfill_halo(
                            halo_global_ids,
                            self._owner_map,
                            self._prev_written_emb,
                            cfg.output_dim,
                            rank,
                            self.world_size,
                            self.device,
                            comm_stats=self._comm_stats,
                            group=self._ocd_gloo_group,
                        )
                    else:
                        # Goes through HaloSync (dist_halo_sync.py):
                        # cfg.cache_backend picks Noop/File/Grpc uniformly,
                        # the same flag Stage C's push_final_embeddings uses.
                        # GrpcHaloSync's
                        # start() is rank-0-only (spawns/waits-ready one Go
                        # server subprocess per rank); the barrier makes every
                        # OTHER rank wait for that before calling fetch(),
                        # which start()'s own docstring doesn't guarantee by
                        # itself. No-ops for Noop/File backends.
                        #
                        # Fetches halo_kg1 AND halo_kg2 ids together (not just
                        # halo_kg1) so Step 3a below can reuse this SAME dict
                        # instead of a second fetch/server-restart round trip
                        # (Step 8 always pushes both KG1 and KG2 rows; Step
                        # 3a's "reuse halo_id_to_emb when not use_ocd" logic
                        # depends on this dict covering halo_kg2 too).
                        combined_ids = np.concatenate(
                            [halo_global_ids, all_global_ids[halo_kg2_idx]]
                        )
                        self._halo_sync.start()
                        dist.barrier()
                        halo_id_to_emb = self._halo_sync.fetch(combined_ids)
                        self._halo_sync.stop()
                    h_init_raw, _ = _patch_halo_rows(
                        h_init_raw, halo_kg1_idx, halo_global_ids, halo_id_to_emb,
                    )
                if rank == 0:
                    logging.info(
                        f"[Iter {iteration + 1}] Halo rows refreshed: "
                        f"{len(halo_kg1_idx)} (ocd={use_ocd}, "
                        f"cache_backend={getattr(cfg, 'cache_backend', 'none')})"
                    )

            # ── Step 3a: Refresh halo_kg2 rows (file-cache only) ────────
            # Mirrors Step 3 above for the KG2-side halo introduced by
            # DistGraphLoader — restores message passing across the
            # KG2-KG2 edges that counterpart_disc's round-robin assignment
            # would otherwise scatter across ranks. The OCD path's
            # owner_map is KG1-only (see dist_ocd.build_owner_map), so this
            # only refreshes via the shared file cache, same scoping
            # --use_halo_kg2 (Step 3c below) already uses. `halo_id_to_emb`
            # is reused as-is when it already came from the file-cache path
            # (the common case); if OCD was used for halo_kg1 instead, its
            # dict only has KG1 entries, so load the file cache separately
            # here rather than silently skipping halo_kg2.
            halo_kg2_id_to_emb: dict[int, np.ndarray] = {}
            if not shared_h_init and iteration > 0 and len(halo_kg2_idx) > 0 and self.cache_dir:
                halo_kg2_id_to_emb = (
                    halo_id_to_emb if (halo_id_to_emb and not use_ocd)
                    else load_id_to_emb(self.cache_dir, self.world_size)
                )
                halo_kg2_global_ids = all_global_ids[halo_kg2_idx]
                h_init_raw, _ = _patch_halo_rows(
                    h_init_raw, halo_kg2_idx, halo_kg2_global_ids, halo_kg2_id_to_emb,
                )
                if rank == 0:
                    logging.info(
                        f"[Iter {iteration + 1}] Halo KG2 rows refreshed: "
                        f"{len(halo_kg2_idx)}"
                    )
            _t_setup.append(time.perf_counter())
            if rank == 0:
                _d = np.diff(_t_setup)
                logging.info(
                    f"[Iter {iteration + 1}] setup: graph load {_d[0]:.1f}s, "
                    f"rrea graph {_d[1]:.1f}s, features {_d[2]:.1f}s, "
                    f"halo refresh {_d[3]:.1f}s"
                    + (" (graph + features reused from iteration 1)" if _setup_cached else "")
                )

            # ── Step 3d: Build h_init consistency-regularizer target ────
            # (--consistency_lambda). Reuses the exact
            # same halo_id_to_emb/halo_kg2_id_to_emb snapshots Step 3/3a
            # already loaded to hard-overwrite halo rows — no new I/O.
            # consistency_idx/consistency_target let DistTrainer add a
            # continuous, every-epoch gradient term pulling those rows
            # toward this snapshot, on top of (not instead of) the hard
            # overwrite already applied above. Only entities actually found
            # in the cache are included (mirrors _patch_halo_rows' own
            # missing-id skip) — iteration 0 has no snapshot at all, so both
            # dicts are empty there and this naturally produces an empty
            # (no-op) target, same as the halo refresh blocks above.
            consistency_idx_list: list[int] = []
            consistency_target_rows: list[np.ndarray] = []
            for idx_arr, id_to_emb_map in (
                (halo_kg1_idx, halo_id_to_emb),
                (halo_kg2_idx, halo_kg2_id_to_emb),
            ):
                if not id_to_emb_map or len(idx_arr) == 0:
                    continue
                for li, gid in zip(idx_arr, all_global_ids[idx_arr], strict=False):
                    feat = id_to_emb_map.get(int(gid))
                    if feat is not None:
                        consistency_idx_list.append(int(li))
                        consistency_target_rows.append(feat)
            if consistency_idx_list:
                consistency_idx = torch.tensor(
                    consistency_idx_list, dtype=torch.long, device=self.device
                )
                consistency_target = torch.from_numpy(
                    np.stack(consistency_target_rows, axis=0).astype(np.float32)
                ).to(self.device)
            else:
                consistency_idx = None
                consistency_target = None

            # ── Step 3c: KG2 Counterpart Halo (--use_halo_kg2) ──────────
            # Extra, possibly cross-rank KG2 negative-sampling candidates —
            # see dist_ctx_kg2.py for why this exists (alignment_loss's
            # negatives are otherwise capped at this rank's local ~1/world_size
            # slice of KG2, which selective_eval's globally-gathered ranking
            # never respects). Requires --cache_dir (the OCD path's owner_map
            # is KG1-only — see build_owner_map — so it can't resolve KG2 IDs
            # and is intentionally not wired up here).
            halo_kg2_emb: torch.Tensor | None = None
            use_halo_kg2 = getattr(cfg, "use_halo_kg2", False)
            if use_halo_kg2 and iteration > 0 and self.cache_dir and self._kg2_adj is not None:
                # Separate, model-output-width channel — see Step 8's write
                # side for why this can't reuse halo_id_to_emb (h_init-width,
                # wrong space for concatenation with emb_for_loss).
                kg2_full_id_to_emb = load_id_to_emb(
                    os.path.join(self.cache_dir, "kg2_full"), self.world_size
                )
                local_kg1_global = {int(g) for g in all_global_ids[local_kg1_idx]}
                local_kg2_global = {int(g) for g in all_global_ids[local_kg2_idx]}
                anchors = {
                    int(kg2_id) for kg1_id, kg2_id in self.train_links
                    if int(kg1_id) in local_kg1_global
                }
                max_hop = getattr(cfg, "halo_kg2_max_hop", None) or (2 * cfg.num_layers)
                top_k = int(len(local_kg2_global) * getattr(cfg, "halo_kg2_ratio", 0.5))
                halo_kg2_ids = dist_ctx_kg2.select_halo_kg2(
                    anchors, self._kg2_adj, self.data.node_weights2,
                    local_kg2_global, max_hop=max_hop, top_k=top_k,
                )
                halo_kg2_emb, n_found = fetch_halo_kg2_emb(
                    halo_kg2_ids, kg2_full_id_to_emb, self.device,
                )
                if rank == 0:
                    logging.info(
                        f"[Iter {iteration + 1}] KG2 Counterpart Halo: "
                        f"{n_found}/{len(halo_kg2_ids)} candidates found in cache "
                        f"(anchors={len(anchors)}, top_k={top_k})"
                    )

            # ── Step 3b: Wrap features as learnable parameter ───────────
            # Random init alone produces poor results because the encoder can only
            # transform fixed noise. Making h_init learnable (N_local × output_dim
            # params) lets the optimiser align entity representations directly.
            persist_optimizer = getattr(cfg, "persist_optimizer", False)
            reset_halo_momentum = getattr(cfg, "reset_halo_momentum", False)
            if persist_optimizer and self._persistent_h_init is not None:
                h_init = self._persistent_h_init
                # halo_id_to_emb (Step 3) was computed against h_init_raw, a
                # throwaway tensor persist_optimizer otherwise never looks
                # at — apply it to the reused parameter directly instead, or
                # halo rows would silently stop refreshing after iteration 0.
                # Same reasoning applies to halo_kg2_id_to_emb (Step 3a).
                n_momentum_reset = 0
                if halo_id_to_emb:
                    with torch.no_grad():
                        _patch_halo_rows(
                            h_init.data, halo_kg1_idx,
                            all_global_ids[halo_kg1_idx], halo_id_to_emb,
                        )
                    if reset_halo_momentum and self._persistent_optimizer is not None:
                        if _reset_adam_momentum_for_rows(
                            self._persistent_optimizer, h_init, halo_kg1_idx,
                        ):
                            n_momentum_reset += len(halo_kg1_idx)
                if halo_kg2_id_to_emb:
                    with torch.no_grad():
                        _patch_halo_rows(
                            h_init.data, halo_kg2_idx,
                            all_global_ids[halo_kg2_idx], halo_kg2_id_to_emb,
                        )
                    if reset_halo_momentum and self._persistent_optimizer is not None:
                        if _reset_adam_momentum_for_rows(
                            self._persistent_optimizer, h_init, halo_kg2_idx,
                        ):
                            n_momentum_reset += len(halo_kg2_idx)
                if rank == 0:
                    logging.info(
                        f"[Iter {iteration + 1}] persist_optimizer: reusing learnable "
                        f"embeddings (no rebuild, no state reload"
                        + (", halo rows patched in place)" if (halo_id_to_emb or halo_kg2_id_to_emb) else ")")
                    )
                    if reset_halo_momentum and n_momentum_reset:
                        logging.info(
                            f"[Iter {iteration + 1}] reset_halo_momentum: cleared Adam "
                            f"exp_avg/exp_avg_sq for {n_momentum_reset} patched halo rows"
                        )
                h_init_reused = True
            else:
                h_init_reused = False
                h_init = nn.Parameter(h_init_raw)
            if not h_init_reused and (
                getattr(cfg, "warm_start", True)
                and self._h_state is not None
                and self._h_state.shape == h_init.data.shape
            ):
                h_init.data = self._h_state.to(self.device)
                if rank == 0:
                    logging.info(
                        f"[Iter {iteration + 1}] Warm-start: restored learnable embeddings"
                    )
                # The restore above replaced the WHOLE table, including the
                # halo rows Step 3/3a just patched into h_init_raw — see
                # _reapply_halo_patches' docstring. Opt-in re-apply; on the
                # default path say so loudly instead of silently training on
                # this rank's own stale copies of other ranks' entities.
                if getattr(cfg, "halo_refresh_after_warm_start", False):
                    n_reapplied = _reapply_halo_patches(
                        h_init.data, halo_kg1_idx, halo_kg2_idx, all_global_ids,
                        halo_id_to_emb, halo_kg2_id_to_emb,
                    )
                    if rank == 0:
                        logging.info(
                            f"[Iter {iteration + 1}] halo_refresh_after_warm_start: "
                            f"re-applied {n_reapplied} halo rows on top of the "
                            f"restored table"
                        )
                elif (halo_id_to_emb or halo_kg2_id_to_emb) and rank == 0:
                    logging.warning(
                        f"[Iter {iteration + 1}] warm-start restore overwrote this "
                        f"iteration's halo-row refresh (Step 3/3a patched h_init_raw, "
                        f"not the restored table): the cross-rank halo values have NO "
                        f"effect on this path. Set --halo_refresh_after_warm_start "
                        f"(or --persist_optimizer) to keep them."
                    )

            # ── Step 4: Map train_links to local indices ─────────────────
            train_pairs = self._map_links_to_local(
                global2local,
                owned_kg1_n=(len(local_kg1_idx)
                             if getattr(cfg, "ablate_full_graph", False) else None),
            )
            if rank == 0:
                logging.info(
                    f"[Iter {iteration + 1}] Local train pairs: {len(train_pairs)}"
                )

            if isinstance(train_pairs, list):
                train_pairs_t = torch.tensor(
                    train_pairs, dtype=torch.long, device=self.device
                )
            else:
                train_pairs_t = train_pairs

            # ── Step 4b: MM — periodic VBIP pair-weighting sync ──────────
            # compute_and_gather_vbip does local diffusion + a compact
            # all_gather of per-entity scalars (dist_vbip.py) — exactly the
            # "compute locally, sync only compact stats, periodically" shape
            # Model Migration calls for.
            pair_weights = None
            if use_mm:
                mm_interval = max(1, getattr(cfg, "mm_sync_interval", 5))
                if self._vbip_pair_weights is None or iteration % mm_interval == 0:
                    with self._comm_stats.timed("mm_vbip_sync"):
                        pw, _scores = compute_and_gather_vbip(
                            g, self.train_links, global2local, all_global_ids,
                            local_kg1_idx, train_pairs_t,
                            rank, self.world_size, self.device,
                            comm_stats=self._comm_stats,
                            impl=getattr(cfg, "vbip_impl", "simplified"),
                            data=self.data,
                        )
                    self._vbip_pair_weights = pw
                    if rank == 0:
                        status = "refreshed" if pw is not None else "unavailable (uniform)"
                        logging.info(f"[Iter {iteration + 1}] MM: VBIP pair-weights {status}")
                pair_weights = self._vbip_pair_weights

            # ── Step 5: Build model + optimizer (warm-start if available) ──
            # persist_optimizer (see __init__ comment):
            # reuse the SAME model/optimizer/loss_weight objects instead of
            # rebuilding, so Adam's moment estimates carry over between
            # iterations exactly like RREA's own continuous training loop.
            kg1_size = len(local_kg1_idx) + len(halo_kg1_idx)
            if persist_optimizer and self._persistent_model is not None:
                model = self._persistent_model
                optimizer = self._persistent_optimizer
                loss_weight = self._persistent_loss_weight
                if rank == 0:
                    logging.info(
                        f"[Iter {iteration + 1}] persist_optimizer: reusing model + "
                        f"optimizer (Adam moments carried over, no reset)"
                    )
            else:
                model = RREA(
                    rel_num=self.data.rel_num,
                    output_dim=cfg.output_dim,
                    num_layers=cfg.num_layers,
                    dropout=cfg.dropout,
                    attn_heads=getattr(cfg, "attn_heads", 1),
                    edge_chunk=int(getattr(cfg, "rrea_edge_chunk", 0) or 0),
                    rrea_dropout=float(getattr(cfg, "rrea_dropout", 0.0) or 0.0),
                    init_scale=float(getattr(cfg, "init_scale", 1.0) or 1.0),
                    attn_softmax=getattr(cfg, "attn_softmax", "dst") or "dst",
                    input_relu=bool(getattr(cfg, "rrea_input_relu", False)),
                ).to(self.device)
                if rank == 0 and iteration == 0 and float(getattr(cfg, "dropout", 0.0) or 0.0) > 0:
                    logging.warning(
                        f"--dropout {cfg.dropout} has no effect on the RREA encoder "
                        f"(NRGraphAttention stores it but never applies it); the "
                        f"original RREA output dropout is --rrea_dropout "
                        f"(currently {getattr(cfg, 'rrea_dropout', 0.0)})."
                    )

                if getattr(cfg, "warm_start", True) and self._model_state is not None:
                    model.load_state_dict(self._model_state)
                    if rank == 0:
                        logging.info(f"[Iter {iteration + 1}] Warm-start from iteration {iteration} weights")

                # Learnable weight (2,) for combined loss (ELsEA-style softmax mix)
                loss_weight = nn.Parameter(
                    torch.tensor([0.5, 0.5], device=self.device)
                )

                # Optimizer updates GCN weights + h_init + loss-mix weight
                # --fused_adam: one fused CUDA kernel per step instead of
                # the foreach implementation; same update (fp32 rounding
                # aside). h_init alone is 1.68M x 128 at DBP1M/N=1, where
                # Adam took about 11% of an epoch.
                # train_opt.build_optimizer: with
                # --weight_decay / --weight_decay_h_init > 0 this becomes
                # AdamW with model / h_init / loss_weight groups (decoupled
                # decay, loss weight never decayed); at 0 it is a plain
                # flat-list Adam.
                _fused = bool(getattr(cfg, "fused_adam", False)) and self.device.type == "cuda"
                optimizer = build_optimizer(
                    model.parameters(), h_init, loss_weight,
                    name=getattr(cfg, "optimizer", "adam"), lr=cfg.lr,
                    weight_decay=float(getattr(cfg, "weight_decay", 0.0) or 0.0),
                    weight_decay_h_init=getattr(cfg, "weight_decay_h_init", None),
                    fused=_fused,
                )

                if persist_optimizer:
                    self._persistent_model = model
                    self._persistent_optimizer = optimizer
                    self._persistent_h_init = h_init
                    self._persistent_loss_weight = loss_weight

            # --compile wraps ONLY the full-graph training forward
            # (fixed shapes across epochs -> one compile, then cache hits;
            # the minibatch path's variable-shape sampled blocks would
            # recompile every batch and are deliberately left alone).
            # `model` itself stays the source of truth for state_dict /
            # warm-start / eval -- torch.compile's wrapper shares the same
            # parameters, so the optimizer and _allreduce_sync see
            # identical tensors either way.
            train_model = model
            if getattr(cfg, "compile", False):
                train_model = torch.compile(model, dynamic=False)
                if rank == 0 and iteration == 0:
                    logging.info("torch.compile enabled for training forward")

            trainer = DistTrainer(
                train_model,
                optimizer,
                rank,
                self.world_size,
                neg_k=cfg.neg_k,
                margin=cfg.margin,
                use_infonce=getattr(cfg, "use_infonce", True),
                infonce_temp=getattr(cfg, "infonce_temp", 0.1),
                loss_weight=loss_weight,
                neg_resample_every=getattr(cfg, "neg_resample_every", 10),
                consistency_lambda=getattr(cfg, "consistency_lambda", 0.0),
                weighted_shared_h_init=getattr(cfg, "shared_h_init_weighted_avg", False),
                comm_stats=self._comm_stats,
                loss_fn=getattr(cfg, "loss", "margin"),
                nhsm_gamma=getattr(cfg, "nhsm_gamma", 1.0),
                amp=getattr(cfg, "amp", "none"),
            )

            # ── Step 6: Training loop ────────────────────────────────────
            later_iter_epochs = getattr(cfg, "later_iter_epochs", None)
            if iteration == 0 or later_iter_epochs is None:
                epochs_this_iter = cfg.max_epochs
            else:
                epochs_this_iter = min(later_iter_epochs, cfg.max_epochs)

            halo_refresh_interval = getattr(cfg, "halo_refresh_interval", 0)
            mid_epoch_halo_active = (
                not shared_h_init
                and halo_refresh_interval > 0
                and bool(self.cache_dir)
                and (len(halo_kg1_idx) > 0 or len(halo_kg2_idx) > 0)
            )

            # --table_consensus_every E (multi-node accuracy fix):
            # every E epochs (and after the last one) each h_init row held by
            # more than one rank -- the owner copy plus its halo copies -- is
            # merged across all copies (table_consensus.py), so the alignment
            # signal a halo copy picks up is not lost. Synced before the
            # periodic eval of the same epoch. Off by default.
            _tc_every = int(getattr(cfg, "table_consensus_every", 0) or 0)
            _consensus = None
            if _tc_every > 0 and self.world_size > 1 and not shared_h_init:
                from table_consensus import TableConsensus
                _consensus = TableConsensus(
                    all_global_ids, h_init.device,
                    mode=getattr(cfg, "table_consensus_mode", "mean") or "mean",
                )
                _consensus.sync(h_init.data)  # copies start equal; sets the delta base
                from table_consensus import adamw_decay_factor
                _tc_step = adamw_decay_factor(optimizer, h_init, 0.0)[1]
                if rank == 0:
                    logging.info(
                        f"[Iter {iteration + 1}] table consensus every {_tc_every} epochs "
                        f"({_consensus.mode}): {_consensus.n_shared} rows held by >1 rank, "
                        f"{_consensus.n_local_shared} of them on rank 0"
                    )

            # Per-epoch forward/backward timing, mean rolled into this
            # iteration's metrics.json entry below (see the "Instrumentation"
            # block) -- deliberately just wall-clock around the whole
            # train_epoch() call rather than instrumenting inside DistTrainer
            # itself, so it cannot change training numerics.
            # BNS-GCN halo sampling: p < 1 redraws a halo subsample
            # every epoch (training only -- eval always sees the full g).
            halo_p_kg1 = getattr(cfg, "halo_sample_p_kg1", 1.0)
            halo_p_kg2 = getattr(cfg, "halo_sample_p_kg2", 1.0)
            halo_sampling = halo_p_kg1 < 1.0 or halo_p_kg2 < 1.0
            if halo_sampling and getattr(cfg, "compile", False) and rank == 0 and iteration == 0:
                logging.warning(
                    "halo sampling changes edge-set shape every epoch; "
                    "torch.compile will recompile repeatedly -- do not combine "
                    "--halo_sample_p* with --compile for timing runs."
                )

            # --gas_clusters K > 0: GNNAutoScale-style
            # training epochs -- METIS clusters as batches, exact 1-hop
            # aggregation, out-of-batch neighbors from history (gas.py).
            # Graph structure is identical across iterations (only pairs
            # and h_init change), so clusters/history build once per run.
            # Eval below always uses the full-graph forward.
            gas_k = int(getattr(cfg, "gas_clusters", 0) or 0)
            # gas_train_epoch builds its per-cluster forward from
            # model.encoder's kernels directly, so it never runs through
            # the torch.compile wrapper above (attribute access on an
            # OptimizedModule resolves to the original submodule) nor
            # through DistTrainer.train_epoch's autocast block. Both
            # switches are silent no-ops on this path; say so once rather
            # than let a timing run be read as "bf16/compile don't help".
            if gas_k > 0 and rank == 0 and iteration == 0 and (
                getattr(cfg, "compile", False) or getattr(cfg, "amp", "none") != "none"
            ):
                logging.warning(
                    "--gas_clusters: --amp/--compile only cover the "
                    "full-graph forward; the GAS cluster forward runs fp32 "
                    "eager regardless (no-op here, not a speed measurement)."
                )
            if gas_k > 0 and self._gas_clusters is None:
                from gas import GASHistory, build_gas_clusters
                _t0 = time.perf_counter()
                _smh = bool(getattr(cfg, "gas_softmax_history", False))
                _l0h = bool(getattr(cfg, "gas_layer0_history", False))
                self._gas_clusters = build_gas_clusters(
                    g, gas_k,
                    method=getattr(cfg, "gas_cluster_method", "metis") or "metis",
                    closure_cap=int(getattr(cfg, "gas_closure_cap", 0) or 0),
                    hub_degree=int(getattr(cfg, "gas_hub_degree", 1000) or 0),
                    softmax_history=_smh,
                    layer0_history=_l0h,
                    csr_spmm=bool(getattr(cfg, "gas_csr_spmm", False)),
                    csr_gather=bool(getattr(cfg, "gas_csr_gather", False)),
                    csr_device=(
                        self.device
                        if getattr(cfg, "gas_clusters_on_device", False) and self.device.type == "cuda"
                        else None
                    ),
                    # DISTEA_GAS_SEGMENT_SOFTMAX=1:
                    # under --attn_softmax src, order each cluster's edges by
                    # src so the softmax runs as contiguous segments.
                    src_sorted_edges=(
                        (getattr(cfg, "attn_softmax", "dst") or "dst") == "src"
                        and os.environ.get("DISTEA_GAS_SEGMENT_SOFTMAX", "0").lower() in ("1", "true", "yes")
                    ),
                )
                self._gas_history = GASHistory(
                    g.adj_self_norm.size(0), cfg.output_dim, cfg.num_layers,
                    device=(
                        self.device
                        if getattr(cfg, "gas_history_on_device", False)
                        else None
                    ),
                    edge_dst=(g.edge_index[:, 1] if _smh else None),
                    attn_heads=int(getattr(cfg, "attn_heads", 1) or 1),
                    layer0=_l0h,
                )
                if _l0h:
                    # Exact whole-graph layer-0 rows once (one spmm per
                    # channel, no grad) so epoch 1 does not read zeros.
                    self._gas_history.warm_layer0(g, h_init, model.rel_emb, self.device)
                if getattr(cfg, "gas_clusters_on_device", False):
                    for _c in self._gas_clusters:
                        _c.to_(self.device)
                self._gas_final = torch.zeros(
                    g.adj_self_norm.size(0), model.output_width, device=self.device)
                if rank == 0:
                    logging.info(
                        f"GAS: {len(self._gas_clusters)} clusters built "
                        f"in {time.perf_counter() - _t0:.1f}s"
                    )

            # Per-epoch lr schedule (rebuilt every iteration over this
            # iteration's epoch count; None with --lr_schedule none). Stepped once after each epoch below.
            _lr_sched = build_lr_scheduler(
                optimizer,
                schedule=getattr(cfg, "lr_schedule", "none") or "none",
                total_epochs=epochs_this_iter,
                warmup_epochs=int(getattr(cfg, "lr_warmup_epochs", 0) or 0),
                min_ratio=float(getattr(cfg, "lr_min_ratio", 0.0) or 0.0),
                step_size=int(getattr(cfg, "lr_step_size", 50) or 50),
                gamma=float(getattr(cfg, "lr_gamma", 0.5) or 0.5),
            )
            if _lr_sched is not None and rank == 0:
                logging.info(
                    f"[Iter {iteration + 1}] lr schedule {getattr(cfg, 'lr_schedule', 'none')}"
                    f" over {epochs_this_iter} epochs (warmup {getattr(cfg, 'lr_warmup_epochs', 0)},"
                    f" min_ratio {getattr(cfg, 'lr_min_ratio', 0.0)})"
                )
            epoch_times: list[float] = []
            epoch_evals: list[dict] = []
            # Per-epoch split evals + epoch selection.
            # `_eval_splits` lists what each --eval_every_epochs point
            # scores (test keys bare, valid_*/train_* prefixed); the
            # selector (--keep_best_epoch / --epoch_patience) watches the
            # --select_by split's Hit@1, snapshots model + h_init on every
            # improvement (each rank its OWN rows -- h_init is partitioned)
            # and the snapshot is restored before Step 7 so the reported
            # test numbers belong to the selected epoch. All decisions are
            # taken on rank 0 and broadcast (same pattern as Step 7d).
            _eval_every = int(getattr(cfg, "eval_every_epochs", 0) or 0)
            _select_by = getattr(cfg, "select_by", "test") or "test"
            _select_active = bool(
                getattr(cfg, "keep_best_epoch", False)
                or int(getattr(cfg, "epoch_patience", 0) or 0) > 0
            )
            _eval_splits = parse_eval_splits(
                getattr(cfg, "eval_every_splits", "test"),
                select_by=_select_by if _select_active else "test",
            )
            _selector: EpochSelector | None = None
            _best_snapshot: tuple[dict, torch.Tensor] | None = None
            if _select_active:
                if _eval_every <= 0:
                    if rank == 0:
                        logging.warning(
                            "--keep_best_epoch/--epoch_patience need "
                            "--eval_every_epochs > 0; epoch selection is OFF "
                            "for this run (last epoch reported)."
                        )
                    _select_active = False
                else:
                    _selector = EpochSelector(
                        patience=int(getattr(cfg, "epoch_patience", 0) or 0),
                        min_epochs=0,
                        metric=f"{split_prefix(_select_by)}Hit@1",
                    )
            _split_links = {
                "test": self.test_links,
                "valid": self.valid_links,
                "train": self.train_links,
            }
            # Hard negatives: pools re-mined from the GAS
            # final history every --hard_neg_every epochs from epoch
            # --hard_neg_warmup on, for exactly train_pairs_t (same row
            # order the epoch indexes them by). Per iteration (pairs may
            # grow with pseudo-seeds). Mining time is kept apart from the
            # epoch times so s/epoch stays comparable.
            _hn_pool = int(getattr(cfg, "hard_neg_pool", 0) or 0)
            _hn_k = int(getattr(cfg, "hard_neg_k", 1) or 0)
            _hn_every = max(1, int(getattr(cfg, "hard_neg_every", 25) or 1))
            _hn_warmup = max(1, int(getattr(cfg, "hard_neg_warmup", 10) or 1))
            _hn_pools = None
            _hn_time = 0.0
            _hn_mines = 0
            if _hn_pool > 0 and gas_k <= 0:
                if rank == 0:
                    logging.warning("--hard_neg_pool is wired for the GAS path only "
                                    "(--gas_clusters > 0); ignored on this run.")
                _hn_pool = 0
            if (rank == 0 and gas_k > 0 and getattr(cfg, "loss", "margin") == "nhsm"
                    and (getattr(cfg, "nhsm_negs", "full") or "full") == "full"
                    and self._gas_final is not None and self._gas_final.size(0) > 500_000):
                logging.warning(
                    f"--loss nhsm with --nhsm_negs full on {self._gas_final.size(0)} rows: "
                    f"each cluster step materialises (P, N) temporaries (~15 s/epoch, OOM risk); "
                    f"use --nhsm_negs compact."
                )
            if _hn_pool > 0 and _hn_every < 10 and rank == 0:
                logging.warning(f"--hard_neg_every {_hn_every} < 10: mining cost "
                                f"(~8-12 s each at DBP1M) will dominate.")
            for epoch in range(epochs_this_iter):
                if (
                    _hn_pool > 0 and _hn_k > 0
                    and (epoch + 1) >= _hn_warmup
                    and ((epoch + 1 - _hn_warmup) % _hn_every == 0 or _hn_pools is None)
                ):
                    from hard_neg import mine_hard_negatives
                    _t_hn = time.perf_counter()
                    _hn_pools = mine_hard_negatives(
                        self._gas_final, train_pairs_t, kg1_size,
                        pool=_hn_pool,
                        side=getattr(cfg, "hard_neg_side", "other"),
                        dtype=(torch.bfloat16
                               if getattr(cfg, "hard_neg_dtype", "bf16") == "bf16"
                               and self.device.type == "cuda" else torch.float32),
                        epoch=epoch + 1,
                    )
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    _hn_time += time.perf_counter() - _t_hn
                    _hn_mines += 1
                    if rank == 0 and _hn_pools is not None:
                        logging.info(
                            f"[Iter {iteration + 1}] mined hard negatives at epoch "
                            f"{epoch + 1}: pool {_hn_pools.pool_size} x {len(_hn_pools)} pairs "
                            f"x 2 sides in {time.perf_counter() - _t_hn:.1f}s"
                        )
                _epoch_t0 = time.perf_counter()
                if gas_k > 0:
                    from gas import gas_train_epoch
                    loss = gas_train_epoch(
                        trainer, self._gas_clusters, self._gas_history,
                        self._gas_final, h_init, train_pairs_t, kg1_size,
                        self.device,
                        hard_neg=_hn_pools, hard_neg_k=_hn_k if _hn_pools is not None else 0,
                        nhsm_negs=getattr(cfg, "nhsm_negs", "full") or "full",
                        nhsm_pool=int(getattr(cfg, "nhsm_pool", 10000) or 10000),
                        neg_pool_frac=float(getattr(cfg, "neg_pool_frac", 1.0) or 1.0),
                    )
                else:
                    g_epoch = (
                        sample_halo_rrea_graph(g, halo_kg1_idx, halo_kg2_idx,
                                               halo_p_kg1, halo_p_kg2)
                        if halo_sampling else g
                    )
                    loss = trainer.train_epoch(
                        g_epoch, h_init, train_pairs_t, kg1_size,
                        pair_weights=pair_weights, halo_kg2_emb=halo_kg2_emb,
                        consistency_idx=consistency_idx, consistency_target=consistency_target,
                        local_idx=local_idx,
                    )
                epoch_times.append(time.perf_counter() - _epoch_t0)
                if _lr_sched is not None:
                    _lr_sched.step()
                if _consensus is not None and (
                    (epoch + 1) % _tc_every == 0 or (epoch + 1) == epochs_this_iter
                ):
                    with self._comm_stats.timed("table_consensus"):
                        _tc_f = 1.0
                        if _consensus.mode == "delta_dc":
                            _tc_f, _tc_step = adamw_decay_factor(optimizer, h_init, _tc_step)
                        _consensus.sync(h_init.data, decay_factor=_tc_f)
                if rank == 0 and (epoch + 1) % max(1, epochs_this_iter // 10) == 0:
                    logging.info(
                        f"  Epoch {epoch + 1}/{epochs_this_iter}  loss={loss:.4f}"
                    )

                # --eval_every_epochs k: a time-vs-accuracy
                # curve from ONE iteration (no per-iteration graph reload,
                # no table/optimizer restart): test eval every k epochs,
                # logged and kept under the iteration's "epoch_evals" in
                # metrics.json with the training seconds elapsed so far.
                # Collective (every rank hits the same condition); the
                # final epoch is left to the regular Step 7 eval. Off by
                # default.
                _is_last_epoch = (epoch + 1) == epochs_this_iter
                _periodic = (
                    _eval_every > 0
                    and (epoch + 1) % _eval_every == 0
                    and not _is_last_epoch
                )
                # The final epoch is scored on the selection split only
                # (its test numbers come from Step 7 as always) so that it
                # competes with the earlier snapshots.
                _splits_now = (
                    list(_eval_splits) if _periodic
                    else ([_select_by] if (_select_active and _is_last_epoch) else [])
                )
                if _splits_now and _is_last_epoch and not _periodic:
                    _splits_now = [sp for sp in _splits_now if sp != "test"]
                if _splits_now:
                    _h_eval = h_init[local_idx] if local_idx is not None else h_init
                    _owned = np.concatenate(
                        [np.asarray(local_kg1_idx), np.asarray(local_kg2_idx)]
                    )
                    _t0e = time.perf_counter()
                    _entry: dict = {}
                    for _sp in _splits_now:
                        _links = _split_links.get(_sp) or []
                        if not _links:
                            if rank == 0:
                                logging.warning(f"split '{_sp}' has no links; skipped")
                            continue
                        _m = selective_eval(
                            model, g, _h_eval, all_global_ids, _links,
                            rank, self.world_size, self.device,
                            use_csls=getattr(cfg, "use_csls", True),
                            csls_k=getattr(cfg, "csls_k", 10),
                            metric=getattr(cfg, "eval_metric", "cosine"),
                            top_k=(1, 5, 10, 50),
                            owned_local=_owned,
                            link_groups=self._eval_group_labels(_links),
                        )
                        if rank == 0 and _m:
                            _pf = split_prefix(_sp)
                            _entry.update({f"{_pf}{k}": v for k, v in _m.items()})
                    model.train()
                    # Cheap table diagnostics for regularisation studies:
                    # current lr and the mean L2 norm of this rank's
                    # owned h_init rows (reduced to a global mean).
                    with torch.no_grad():
                        _tab = h_init.data[local_idx] if local_idx is not None else h_init.data
                        _norm_sum = _tab[_owned].float().norm(dim=1).sum()
                        _norm_cnt = torch.tensor(
                            [float(len(_owned))], device=h_init.device
                        )
                        _ns = torch.stack([_norm_sum.to(h_init.device), _norm_cnt[0]])
                        if self.world_size > 1:
                            dist.all_reduce(_ns)
                        _h_norm_mean = float(_ns[0] / max(_ns[1].item(), 1.0))
                    _lr_now = float(optimizer.param_groups[0].get("lr", cfg.lr))
                    # Selection decision (rank 0) -> broadcast [is_best, stop]
                    _is_best = 0
                    _stop_now = 0
                    if rank == 0 and _entry:
                        _entry = {
                            "epoch": epoch + 1,
                            "train_time_s": round(sum(epoch_times), 2),
                            "eval_time_s": round(time.perf_counter() - _t0e, 2),
                            "lr": _lr_now,
                            "h_init_norm_mean": round(_h_norm_mean, 6),
                        } | _entry
                        if _periodic or _is_last_epoch:
                            epoch_evals.append(_entry)
                        logging.info(
                            f"[Iter {iteration + 1}] Epoch {epoch + 1} eval "
                            f"({_entry['train_time_s']:.0f}s of training so far): "
                            + ", ".join(
                                f"{k}={v:.4f}" for k, v in _entry.items()
                                if isinstance(v, float) and ("Hit@" in k or "MRR" in k)
                            )
                        )
                        if _selector is not None:
                            _score = _entry.get(_selector.metric)
                            if _score is not None:
                                _is_best = int(_selector.observe(epoch + 1, float(_score)))
                                _stop_now = int(_selector.should_stop() and not _is_last_epoch)
                                if _is_best:
                                    logging.info(
                                        f"[Iter {iteration + 1}] new best "
                                        f"{_selector.metric}={_score:.4f} at epoch {epoch + 1}"
                                    )
                                elif _stop_now:
                                    logging.info(
                                        f"[Iter {iteration + 1}] epoch patience "
                                        f"({_selector.patience}) exhausted at epoch "
                                        f"{epoch + 1}; best epoch {_selector.best_epoch}"
                                    )
                    if _selector is not None:
                        _flags = torch.tensor([_is_best, _stop_now], device=self.device)
                        if self.world_size > 1:
                            dist.broadcast(_flags, src=0)
                        _is_best, _stop_now = int(_flags[0].item()), int(_flags[1].item())
                        if _is_best:
                            _best_snapshot = (
                                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                h_init.data.detach().cpu().clone(),
                            )
                        if _stop_now:
                            break

                # ── Mid-epoch halo refresh (see --halo_refresh_interval) ──
                # Covers halo_kg2 as well as halo_kg1. Cross-rank pairs score
                # below same-rank pairs even with the per-iteration halo
                # refresh, and h_init has no other cross-rank consistency
                # mechanism here; this refreshes more often within an
                # iteration.
                if mid_epoch_halo_active and (epoch + 1) % halo_refresh_interval == 0:
                    with self._comm_stats.timed("mid_epoch_halo_refresh"):
                        # Goes through HaloSync (see Step 3/8's comments). Caution
                        # for --cache_backend grpc specifically: this runs up
                        # to max_epochs/halo_refresh_interval times PER
                        # ITERATION, and GrpcHaloSync.start()/stop() spawn and
                        # kill real subprocesses (with a gRPC-readiness wait)
                        # every call — a small --halo_refresh_interval with
                        # grpc could dominate wall time with server-restart
                        # overhead. file/none backends have no such cost.
                        self._halo_sync.push(
                            np.concatenate(
                                [all_global_ids[local_kg1_idx], all_global_ids[local_kg2_idx]]
                            ),
                            np.concatenate([
                                h_init.data[local_kg1_idx].cpu().numpy(),
                                h_init.data[local_kg2_idx].cpu().numpy(),
                            ], axis=0),
                            rank,
                        )
                        dist.barrier()
                        self._halo_sync.start()
                        dist.barrier()
                        all_local_ids = np.concatenate(
                            [all_global_ids[halo_kg1_idx], all_global_ids[halo_kg2_idx]]
                        )
                        id_to_emb = self._halo_sync.fetch(all_local_ids)
                        self._halo_sync.stop()
                        blend_alpha = getattr(cfg, "halo_blend_alpha", 1.0)
                        # reset_halo_momentum applies here too, not just Step
                        # 3b's iteration-boundary refresh: the same Adam
                        # exp_avg/exp_avg_sq-vs-value mismatch arises within a single iteration's own epoch
                        # history whenever a mid-epoch patch overwrites rows
                        # the optimizer has been accumulating momentum for —
                        # this doesn't depend on --persist_optimizer carrying
                        # state ACROSS iterations, only on `optimizer` (in
                        # scope here regardless) having trained epochs 1..N
                        # before this patch.
                        reset_halo_momentum = getattr(cfg, "reset_halo_momentum", False)
                        n_refreshed = 0
                        with torch.no_grad():
                            if len(halo_kg1_idx) > 0:
                                _, r1 = _patch_halo_rows(
                                    h_init.data, halo_kg1_idx,
                                    all_global_ids[halo_kg1_idx], id_to_emb,
                                    alpha=blend_alpha,
                                )
                                n_refreshed += r1
                                if reset_halo_momentum:
                                    _reset_adam_momentum_for_rows(optimizer, h_init, halo_kg1_idx)
                            if len(halo_kg2_idx) > 0:
                                _, r2 = _patch_halo_rows(
                                    h_init.data, halo_kg2_idx,
                                    all_global_ids[halo_kg2_idx], id_to_emb,
                                    alpha=blend_alpha,
                                )
                                n_refreshed += r2
                                if reset_halo_momentum:
                                    _reset_adam_momentum_for_rows(optimizer, h_init, halo_kg2_idx)
                    if rank == 0:
                        logging.info(
                            f"  Epoch {epoch + 1}: mid-epoch halo refresh "
                            f"({n_refreshed}/{len(halo_kg1_idx) + len(halo_kg2_idx)} rows)"
                        )

            # Restore the selected epoch's snapshot (all ranks) so the
            # Step 7 test eval and the warm-start state (7c) carry it.
            _selected_epoch = None
            if _selector is not None and _best_snapshot is not None:
                _selected_epoch = _selector.best_epoch
                if _selected_epoch != len(epoch_times):
                    _snap_model, _snap_h = _best_snapshot
                    model.load_state_dict(_snap_model)
                    with torch.no_grad():
                        h_init.data.copy_(_snap_h.to(h_init.device))
                    if rank == 0:
                        logging.info(
                            f"[Iter {iteration + 1}] restored epoch "
                            f"{_selected_epoch} (of {len(epoch_times)} trained) "
                            f"for the final eval"
                        )
                _best_snapshot = None

            # ── Step 7: Evaluate ─────────────────────────────────────────
            # eval.py's selective_eval/distributed_eval/generate_pseudo_seeds
            # all call model(g, h_init) expecting h_init shaped to match g
            # (N_local rows) — under --shared_h_init, h_init is the FULL
            # (N_global, D) table, so gather this rank's local view once here
            # (same differentiable-gather-shaped operation dist_trainer.py's
            # train_epoch does, just without needing gradients — eval is
            # no_grad throughout) rather than passing the full table in.
            h_init_for_eval = h_init[local_idx] if local_idx is not None else h_init
            # Only this rank's OWN entities go into the eval gather; halo
            # copies (stale row, partial neighborhood) must not end up as the
            # scored target (eval.py _select_owned_rows). Local node order is
            # [local_kg1 | halo_kg1 | local_kg2 | halo_kg2].
            owned_local = np.concatenate(
                [np.asarray(local_kg1_idx), np.asarray(local_kg2_idx)]
            )

            is_last_iter = (iteration + 1 == cfg.max_iterations)
            if (iteration + 1) % cfg.eval_interval == 0:
                save_csv = None
                if rank == 0 and self.results_dir:
                    # Written every eval, not just when is_last_iter — that
                    # check only matches the nominal --max_iterations, so a
                    # run that stops early (--patience or pseudo-seed
                    # convergence) would never save alignment.csv.
                    # Overwriting each iteration means whichever iteration
                    # actually turns out to be the last one leaves its
                    # predictions behind.
                    save_csv = os.path.join(self._run_dir(), "alignment.csv")
                # Same "write every eval, let the truly-last one win" logic
                # as save_csv above, for the unified_eval.py export bundle.
                export_dir = (
                    (getattr(cfg, "export_dir", "") or None) if rank == 0 else None
                )
                _eval_t0 = time.perf_counter()
                metrics = {} if _defer_test else selective_eval(
                    model,
                    g,
                    h_init_for_eval,
                    all_global_ids,
                    self.test_links,
                    rank,
                    self.world_size,
                    self.device,
                    save_path=save_csv,
                    use_csls=getattr(cfg, "use_csls", True),
                    csls_k=getattr(cfg, "csls_k", 10),
                    metric=getattr(cfg, "eval_metric", "cosine"),
                    top_k=(1, 5, 10, 50),
                    export_dir=export_dir,
                    owned_local=owned_local,
                    link_groups=self._eval_group_labels(self.test_links),
                    export_dtype=getattr(cfg, "export_dtype", "fp32") or "fp32",
                    export_meta={"task_id": getattr(cfg, "task_id", ""),
                                 "run_timestamp": getattr(cfg, "run_timestamp", ""),
                                 "iteration": iteration + 1,
                                 "selected_epoch": _selected_epoch or len(epoch_times),
                                 "seed": getattr(cfg, "seed", None)},
                )
                eval_time_s = time.perf_counter() - _eval_t0
                if rank == 0 and (metrics or _defer_test):
                    if metrics:
                        logging.info(f"[Iter {iteration + 1}] Test eval: {metrics}")
                    iter_entry = {"iteration": iteration + 1} | metrics
                    if _defer_test:
                        iter_entry["test_deferred"] = True
                    if epoch_times:
                        iter_entry["epoch_time_mean_s"] = round(
                            sum(epoch_times) / len(epoch_times), 4
                        )
                    if not _defer_test:
                        iter_entry["eval_time_s"] = round(eval_time_s, 4)
                    if _consensus is not None:
                        iter_entry["table_consensus"] = _consensus.summary()
                    if epoch_evals:
                        iter_entry["epoch_evals"] = epoch_evals
                    if _selector is not None:
                        iter_entry["epoch_selection"] = _selector.summary()
                        iter_entry["selected_epoch"] = _selected_epoch or len(epoch_times)
                        iter_entry["epochs_trained"] = len(epoch_times)
                    if _hn_mines:
                        iter_entry["hard_neg_time_s"] = round(_hn_time, 3)
                        iter_entry["hard_neg_mines"] = _hn_mines
                    self._iter_metrics.append(iter_entry)

            # ── Step 7c: Save model + embedding state for warm-start ────
            self._model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self._h_state = h_init.data.detach().cpu().clone()

            # ── Step 7d: Validation-set eval + patience-based early stop ─
            _early_stop = False
            _new_best = False
            if getattr(cfg, "eval_on_valid", False) and self.valid_links:
                valid_metrics = selective_eval(
                    model, g, h_init_for_eval, all_global_ids,
                    self.valid_links, rank, self.world_size, self.device,
                    use_csls=getattr(cfg, "use_csls", True),
                    csls_k=getattr(cfg, "csls_k", 10),
                    metric=getattr(cfg, "eval_metric", "cosine"),
                    owned_local=owned_local,
                    link_groups=self._eval_group_labels(self.valid_links),
                )
                if rank == 0 and valid_metrics:
                    valid_hit1 = valid_metrics.get("Hit@1", 0.0)
                    logging.info(
                        f"[Iter {iteration + 1}] Valid eval: {valid_metrics}"
                    )
                    # Keep the valid numbers next to the test ones so
                    # --select_by valid can pick the iteration too.
                    if self._iter_metrics and self._iter_metrics[-1].get("iteration") == iteration + 1:
                        self._iter_metrics[-1].update(
                            {f"valid_{k}": v for k, v in valid_metrics.items()}
                        )
                    if valid_hit1 > self._best_valid_hit1:
                        self._best_valid_hit1 = valid_hit1
                        _new_best = True
                        self._patience_counter = 0
                        self._best_model_state = {
                            k: v.cpu().clone() for k, v in model.state_dict().items()
                        }
                        self._best_h_state = h_init.data.detach().cpu().clone()
                        logging.info(
                            f"[Iter {iteration + 1}] New best valid Hit@1: {valid_hit1:.4f}"
                        )
                    else:
                        self._patience_counter += 1
                        logging.info(
                            f"[Iter {iteration + 1}] Valid Hit@1 did not improve "
                            f"({self._patience_counter}/{cfg.patience})"
                        )
                        min_iters = getattr(cfg, "min_iterations", 2)
                        if self._patience_counter >= cfg.patience and (iteration + 1) >= min_iters:
                            logging.info(
                                f"[Iter {iteration + 1}] Early stop: patience exhausted."
                            )
                            _early_stop = True
                # Broadcast early-stop flag to all ranks
                stop_t = torch.tensor([int(_early_stop)], device=self.device)
                dist.broadcast(stop_t, src=0)
                _early_stop = bool(stop_t.item())
                if _defer_test:
                    # Every rank keeps its own copy of the best-valid state
                    # (rank 0 already holds it in _best_*_state).
                    nb_t = torch.tensor([int(_new_best)], device=self.device)
                    dist.broadcast(nb_t, src=0)
                    if bool(nb_t.item()):
                        if rank == 0:
                            _defer_best = (iteration, self._best_model_state, self._best_h_state)
                        else:
                            _defer_best = (
                                iteration,
                                {k: v.cpu().clone() for k, v in model.state_dict().items()},
                                h_init.data.detach().cpu().clone(),
                            )

            # ── Step 7b: Pseudo-seed augmentation ───────────────────────
            # Runs on every non-final iteration when --use_pseudo_seeds is set.
            # generate_pseudo_seeds uses all_gather internally, so all ranks participate.
            _converged = False
            if getattr(cfg, "use_pseudo_seeds", False) and not is_last_iter:
                # --pseudo_seed_policy unaligned: RREA's protocol --
                # mutual-NN among entities not in the train (incl. earlier
                # pseudo-seeds) or valid links, merged WITHOUT the test-link
                # filter. The legacy merge rejects any candidate that equals
                # a test pair, i.e. it uses the test labels to drop exactly
                # the correct predictions and keeps the wrong ones, so test
                # accuracy falls every round. legacy (default) is kept as is.
                _ps_policy = getattr(cfg, "pseudo_seed_policy", "legacy") or "legacy"
                _ps_refresh = getattr(cfg, "pseudo_seed_refresh", "accumulate") or "accumulate"
                _base_links = self.train_links
                _cand1 = _cand2 = None
                if _ps_policy in ("unaligned", "testside"):
                    # testside: entity SETS of the test links (the pool eval
                    # already ranks against), never their pairing; DBP1M has
                    # ~720K KG1 entities with no counterpart at all.
                    _base_links, _cand1, _cand2 = _pseudo_seed_candidates(
                        _ps_policy, _ps_refresh, self.train_links,
                        self._orig_train_links, self.valid_links, self.test_links,
                        self.data.kg1_entities, self.data.kg2_entities,
                    )
                _prev_ps = (
                    {(int(a), int(b)) for a, b in self.train_links}
                    - {(int(a), int(b)) for a, b in _base_links}
                )
                new_seeds = generate_pseudo_seeds(
                    model, g, h_init_for_eval, all_global_ids,
                    local_kg1_idx, local_kg2_idx,
                    rank, self.world_size, self.device,
                    threshold=cfg.seed_threshold,
                    metric=getattr(cfg, "eval_metric", "cosine"),
                    use_csls=getattr(cfg, "use_csls", True),
                    csls_k=getattr(cfg, "csls_k", 10),
                    candidate_kg1=_cand1, candidate_kg2=_cand2,
                    sim_on_device=(_ps_policy != "legacy"),
                    top_frac=(
                        float(getattr(cfg, "pseudo_seed_top_frac", 1.0) or 1.0)
                        if _ps_policy != "legacy" else 1.0
                    ),
                )
                self.train_links, added, correct_estimate = _merge_pseudo_seeds(
                    _base_links, new_seeds,
                    self._test_links_set if _ps_policy == "legacy" else set(),
                    self._known_pairs if _ps_policy == "legacy" else self._test_links_set,
                )
                if rank == 0 and iter_entry is not None:
                    # Per-iteration seed count + estimated precision, the
                    # same numbers the log line below reports, also
                    # persisted to metrics.json.
                    iter_entry["pseudo_seed_n"] = added
                    iter_entry["pseudo_seed_precision"] = (
                        round(correct_estimate / added, 4) if added > 0 else None
                    )
                    if _prev_ps:
                        iter_entry["pseudo_seed_kept"] = len(
                            _prev_ps & set(self.train_links[len(_base_links):])
                        )
                if rank == 0:
                    if added > 0:
                        prec_str = (
                            f", est. precision ≥{correct_estimate/added:.1%}"
                            f" ({correct_estimate}/{added} in "
                            f"{'train+valid' if _ps_policy == 'legacy' else 'test links'})"
                        )
                        if _prev_ps:
                            prec_str += (
                                f", replace: {len(_prev_ps & set(self.train_links[len(_base_links):]))}"
                                f" of the previous {len(_prev_ps)} kept"
                            )
                        logging.info(
                            f"[Iter {iteration + 1}] Pseudo-seeds: "
                            f"+{added} added{prec_str}, "
                            f"total train_links: {len(self.train_links)}"
                        )
                    else:
                        min_iters = getattr(cfg, "min_iterations", 2)
                        if (iteration + 1) >= min_iters:
                            logging.info(
                                f"[Iter {iteration + 1}] Pseudo-seeds: 0 new seeds above "
                                f"threshold {cfg.seed_threshold} — converged."
                            )
                            _converged = True
                        else:
                            logging.info(
                                f"[Iter {iteration + 1}] Pseudo-seeds: 0 new seeds above "
                                f"threshold {cfg.seed_threshold} "
                                f"(min_iterations={min_iters} not reached, continuing)"
                            )
                # Broadcast convergence flag to all ranks
                conv_t = torch.tensor([int(_converged)], device=self.device)
                dist.broadcast(conv_t, src=0)
                _converged = bool(conv_t.item())
                # --eval_pseudo_split (test-leak control): the
                # model that just mined the pseudo-seeds (before any training on
                # them), scored on the same ps_pair / ps_entity / ps_clean groups
                # the final eval reports, so the round-2 change per group can be
                # read off. All ranks take part; rank 0 stores the result.
                if getattr(cfg, "eval_pseudo_split", False):
                    _ps_groups = self._pseudo_split_labels(self.test_links)
                    if _ps_groups is not None:
                        _pre = selective_eval(
                            model, g, h_init_for_eval, all_global_ids,
                            self.test_links, rank, self.world_size, self.device,
                            use_csls=getattr(cfg, "use_csls", True),
                            csls_k=getattr(cfg, "csls_k", 10),
                            metric=getattr(cfg, "eval_metric", "cosine"),
                            top_k=(1, 5, 10, 50),
                            owned_local=owned_local,
                            link_groups=_ps_groups,
                        )
                        if rank == 0 and iter_entry is not None and _pre:
                            iter_entry["pre_round2_split"] = {
                                k: (round(v, 6) if isinstance(v, float) else v)
                                for k, v in _pre.items()
                            }
                            logging.info(f"[Iter {iteration + 1}] pre-round-2 pseudo split: {_pre}")

            # ── Step 7e: Periodic KG2 re-assignment (--reassign_kg2_every) ──
            # assign_kg2_to_ranks_locality's BFS-layer propagation is only as
            # good as its seed set, which partition.py froze at the fold's
            # original train_links (~4500 pairs, never grows). By this point
            # self.train_links has already absorbed this run's pseudo-seeds
            # (Step 7b above) via an all_gather+broadcast every rank already
            # participated in identically, so every rank can independently
            # recompute the SAME deterministic assignment from the SAME
            # inputs with no new communication of its own -- only rank 0
            # writes, to avoid a filesystem race, and every rank barriers
            # before the next iteration's DistGraphLoader.load() (which
            # already re-reads kg2_assignments.json from scratch every call)
            # picks up the richer assignment.
            reassign_every = getattr(cfg, "reassign_kg2_every", 0)
            if reassign_every > 0 and not is_last_iter and (iteration + 1) % reassign_every == 0:
                if rank == 0:
                    partitions = [
                        sorted(np.load(os.path.join(self.partition_dir, f"{r}.npy")).tolist())
                        for r in range(self.world_size)
                    ]
                    assign_kg2_to_ranks_locality(
                        self.data, partitions, self.partition_dir,
                        seed_links=self.train_links,
                    )
                    logging.info(
                        f"[Iter {iteration + 1}] reassign_kg2_every: "
                        f"kg2_assignments.json refreshed with "
                        f"{len(self.train_links)} seeds"
                    )
                dist.barrier()

            # ── Step 8: Write h_init state — file cache and/or OCD source-of-truth ──
            # Pushes h_init's OWN current values (output_dim-wide), not the
            # model's full forward-pass output (output_dim*2*(num_layers+1)
            # -wide for RREA — see models/rrea.py). halo_kg1's and halo_kg2's
            # _patch_halo_rows both do a direct row-assignment into h_init, so
            # the pushed width must equal h_init's width exactly or it
            # raises a shape-mismatch error the moment halo_kg1_idx/
            # halo_kg2_idx is non-empty (world_size>1) and this cache
            # actually has data to read back (both are always empty at
            # world_size=1). Step 3c (--use_halo_kg2) needs model-output-width
            # rows instead and gets them from its own kg2_full/ channel below.
            #
            # local_kg2 is included unconditionally (not just under
            # --use_halo_kg2) because DistGraphLoader's halo_kg2 nodes (the
            # graph-level KG2 halo — see dist_graph.py) need every rank's
            # local_kg2 h_init rows readable next iteration, exactly like
            # halo_kg1 needs KG1 entities readable.
            #
            # Skipped entirely under --shared_h_init: (a) unnecessary — every
            # rank already holds an in-sync full table, nothing to push for
            # next iteration's halo refresh; (b) would be actively WRONG if
            # it ran — local_kg1_idx/local_kg2_idx are LOCAL subgraph
            # positions, not global ids, and h_init here is the FULL
            # (N_global, D) table, so indexing it with local positions would
            # silently grab the wrong rows.
            if not shared_h_init and (self.cache_dir or use_ocd):
                kg1_ids = all_global_ids[local_kg1_idx]
                kg1_emb_np = h_init.data[local_kg1_idx].cpu().numpy()
                owned_ids = np.concatenate(
                    [kg1_ids, all_global_ids[local_kg2_idx]]
                )
                owned_emb_np = np.concatenate(
                    [kg1_emb_np, h_init.data[local_kg2_idx].cpu().numpy()], axis=0
                )

                # KG2 Counterpart Halo needs candidate embeddings in the
                # SAME (model-output) space as emb_for_loss in dist_trainer.py
                # — dim=0 torch.cat requires matching width — not h_init's
                # space (output_dim only). h_init-width owned_emb_np above is
                # correct for halo_kg1 refresh (matches h_init exactly by
                # construction) but wrong for this: the widths differ (e.g.
                # 768 vs 128 for RREA's output_dim*2*(num_layers+1) vs
                # output_dim), so the concat would fail. Written to a
                # separate cache_dir/kg2_full/ subdirectory, only when the
                # feature is in use, so it costs nothing otherwise.
                if use_halo_kg2:
                    model.eval()
                    with torch.no_grad():
                        full_emb = model(g, h_init)
                    kg2_full_ids = all_global_ids[local_kg2_idx]
                    kg2_full_emb_np = full_emb[local_kg2_idx].cpu().numpy()

                if use_saca:
                    write_ids, write_emb, skipped = saca_filter(
                        self._prev_written_emb, owned_ids, owned_emb_np,
                        getattr(cfg, "saca_threshold", 0.01),
                    )
                    self._saca_skipped, self._saca_total = skipped, len(owned_ids)
                    if rank == 0:
                        logging.info(
                            f"[Iter {iteration + 1}] SACA: skipped "
                            f"{skipped}/{len(owned_ids)} unchanged entities"
                        )
                else:
                    write_ids, write_emb = owned_ids, owned_emb_np
                    self._saca_skipped, self._saca_total = 0, len(owned_ids)

                # Always update with the FULL current values — this is "what
                # I actually own now", used by next iteration's SACA compare
                # and OCD serving, independent of what got filtered above.
                for gid, e in zip(owned_ids, owned_emb_np, strict=False):
                    self._prev_written_emb[int(gid)] = e

                if self.cache_dir:
                    # HaloSync.push (dist_halo_sync.py) writes the same
                    # ids_{rank}.npy/emb_{rank}.npy files as
                    # write_rank_embeddings (FileHaloSync/GrpcHaloSync.push
                    # both use that format) and respects --cache_backend.
                    self._halo_sync.push(write_ids, write_emb, rank)
                    if use_halo_kg2:
                        # halo_kg2's separate model-output-width channel
                        # (see the comment above on why it can't share
                        # owned_emb_np's space) has no HaloSync-level
                        # concept — kept on the raw file path, gated on
                        # self.cache_dir alone (independent of
                        # --cache_backend).
                        write_rank_embeddings(
                            os.path.join(self.cache_dir, "kg2_full"), rank,
                            kg2_full_ids, kg2_full_emb_np,
                        )
                dist.barrier()
                if rank == 0:
                    logging.info(
                        f"[Iter {iteration + 1}] Embeddings written to {self.cache_dir}"
                    )

            # ── Per-rank stats (metrics.json per_rank field) ──
            # Collective (all_gather_object), so it runs UNCONDITIONALLY on
            # every rank every iteration -- placed before the early-stop
            # break below, which every rank reaches in lock-step. Overwritten
            # each iteration so the final iteration's counts stand.
            # peak_mem_mb is cumulative CUDA max_memory_allocated for this
            # rank's process (the honest per-run peak; 0 on CPU).
            # epoch_time_mean_s (per epoch) and all_reduce_time_s (summed
            # over this iteration's epochs, same convention as
            # iterations[*].comm_time_s) are this rank's own numbers
            # (CommStats is rebuilt per iteration). The per-epoch
            # all_reduce is a barrier, so every rank's epoch wall converges
            # to the slowest rank's. Under gloo the straggler shows up as
            # the fast ranks' all_reduce_time_s; under nccl that host timer
            # only sees the enqueue (the wait lands in the next epoch), so
            # all_reduce_wait_gpu_s times it with CUDA events instead.
            _snap = self._comm_stats.snapshot()
            _my_rank_stats = {
                "rank": rank,
                "n_local_kg1": int(len(local_kg1_idx)),
                "n_halo_kg1": int(len(halo_kg1_idx)),
                "n_local_kg2": int(len(local_kg2_idx)),
                "n_halo_kg2": int(len(halo_kg2_idx)),
                "peak_mem_mb": round(
                    torch.cuda.max_memory_allocated(self.device) / 2**20, 1
                ) if self.device.type == "cuda" else 0,
                "epoch_time_mean_s": round(
                    sum(epoch_times) / len(epoch_times), 4
                ) if epoch_times else 0.0,
                "all_reduce_time_s": round(
                    _snap["time_s"].get("all_reduce", 0.0), 4
                ),
            }
            _ar_wait = trainer.allreduce_gpu_wait_s()
            _my_rank_stats["all_reduce_wait_gpu_s"] = (
                round(_ar_wait, 4) if _ar_wait is not None else None
            )
            _gathered_rank_stats: list = [None] * self.world_size
            dist.all_gather_object(_gathered_rank_stats, _my_rank_stats)
            self._per_rank = _gathered_rank_stats

            # ── Instrumentation: attach this iteration's timing/comm/SACA
            # stats to its metrics entry, if eval recorded one at Step 7 ──
            # Runs even on the iteration where early-stop/convergence just
            # triggered (the break comes after this), so metrics.json's last
            # iteration always has wall_time_s/comm_bytes/comm_calls.
            if rank == 0 and iter_entry is not None:
                iter_entry["iter_wall_time_s"] = round(time.perf_counter() - iter_start, 4)
                snap = self._comm_stats.snapshot()
                iter_entry["comm_bytes"] = snap["total_bytes"]
                iter_entry["comm_calls"] = snap["total_calls"]
                iter_entry["comm_time_s"] = round(snap["total_time_s"], 4)
                if use_saca:
                    iter_entry["saca_skipped"] = self._saca_skipped
                    iter_entry["saca_total"] = self._saca_total

            # Incremental metrics.json write: every iteration completed so
            # far survives a scheduler timeout/preemption/OOM kill, instead
            # of living only in the un-flushed in-memory self._iter_metrics.
            # complete=False marks it as a partial result.
            if rank == 0 and self.results_dir and self._iter_metrics:
                self._total_wall_time = time.perf_counter() - run_start
                self._save_metrics_json(complete=False)

            if _early_stop or _converged:
                if rank == 0:
                    reason = "patience" if _early_stop else "no new pseudo-seeds"
                    logging.info(f"Stopping after iteration {iteration + 1} ({reason}).")
                break

        # ── Stage C: push final-iteration embeddings (--cache_backend) ──────
        # h_init_for_eval/local_kg1_idx/local_kg2_idx/all_global_ids are the
        # final iteration's loop-local values (Python for-loop bodies don't
        # scope), same values Step 7's eval already used above.
        cache_backend = getattr(cfg, "cache_backend", "none")
        n_pushed = push_final_embeddings(
            cache_backend, self.cache_dir, self.partition_dir, self.world_size,
            rank, all_global_ids, local_kg1_idx, local_kg2_idx, h_init_for_eval,
            getattr(cfg, "grpc_addrs", ""),
        )
        if rank == 0 and n_pushed:
            logging.info(
                f"[Stage C] pushed {n_pushed} embeddings via "
                f"--cache_backend {cache_backend} to {self.cache_dir}"
            )

        if _defer_test:
            self._deferred_test_eval(
                _defer_best, model, g, h_init, local_idx, all_global_ids, owned_local, iteration,
            )

        self._total_wall_time = time.perf_counter() - run_start

        if rank == 0:
            logging.info(
                f"DistEAFramework.run() complete. Total wall time: "
                f"{self._total_wall_time:.1f}s ({self._total_wall_time / 3600:.2f}h)."
            )
            if self.results_dir and self._iter_metrics:
                self._save_metrics_json()
                self._save_checkpoint()

    def _deferred_test_eval(self, defer_best, model, g, h_init, local_idx, all_global_ids,
                            owned_local, last_iteration: int) -> None:
        """--defer_test_eval: restore the best-valid iteration's model and
        h_init (every rank holds its own copy) and run the test eval once,
        filling that iteration's metrics entry. Collective: every rank calls
        it. Without a recorded best (valid Hit@1 never above 0) the last
        iteration's state -- still loaded -- is evaluated."""
        cfg, rank = self.cfg, self.rank
        it = last_iteration
        if defer_best is not None:
            it, model_state, h_state = defer_best
            model.load_state_dict(model_state)
            with torch.no_grad():
                h_init.data.copy_(h_state.to(h_init.device))
        h_eval = h_init[local_idx] if local_idx is not None else h_init
        entry = None
        if rank == 0:
            entry = next((e for e in self._iter_metrics if e.get("iteration") == it + 1), None)
        save_csv = (os.path.join(self._run_dir(), "alignment.csv")
                    if rank == 0 and self.results_dir else None)
        export_dir = (getattr(cfg, "export_dir", "") or None) if rank == 0 else None
        t0 = time.perf_counter()
        metrics = selective_eval(
            model, g, h_eval, all_global_ids, self.test_links, rank, self.world_size, self.device,
            save_path=save_csv,
            use_csls=getattr(cfg, "use_csls", True),
            csls_k=getattr(cfg, "csls_k", 10),
            metric=getattr(cfg, "eval_metric", "cosine"),
            top_k=(1, 5, 10, 50),
            export_dir=export_dir,
            owned_local=owned_local,
            link_groups=self._eval_group_labels(self.test_links),
            export_dtype=getattr(cfg, "export_dtype", "fp32") or "fp32",
            export_meta={"task_id": getattr(cfg, "task_id", ""),
                         "run_timestamp": getattr(cfg, "run_timestamp", ""),
                         "iteration": it + 1,
                         "selected_epoch": (entry or {}).get("selected_epoch"),
                         "seed": getattr(cfg, "seed", None)},
        )
        dt = time.perf_counter() - t0
        if rank == 0 and metrics:
            logging.info(f"[Iter {it + 1}] Test eval (deferred; iteration chosen by valid): {metrics}")
            if entry is not None:
                entry.update(metrics)
                entry["eval_time_s"] = round(dt, 4)
                entry.pop("test_deferred", None)

    def _run_minibatch(self):
        """
        Mini-batch neighbor-sampling training path (--minibatch): replaces
        the full-graph per-iteration halo-refresh loop with per-batch
        sampled-neighborhood training + cross-rank on-demand fetch
        (minibatch_ocd.py), for partitions too large to hold a full
        local+halo forward pass in memory at once (e.g. DBP1M at small
        world sizes).

        Deliberately reuses only what generalizes (data already loaded in
        __init__, alignment_loss, eval's metric helpers via
        minibatch_eval.py) and skips the halo-specific machinery entirely
        (SACA / MM-VBIP / consistency-reg / halo_kg2-counterpart /
        shared_h_init / persist_optimizer / warm_start) rather than
        threading a conditional through run()'s long body. These are
        independent, not-yet-validated-for-mini-batch mechanisms; adding
        them back is future work, not a correctness gap in what's here.

        Ownership: a train_links pair is trained by whichever rank owns
        its KG1 endpoint (build_full_owner_map) — counterpart_disc.py
        already assigns a train-linked KG2 entity to its paired KG1
        entity's rank, so this is a
        non-overlapping partition of train_links across ranks in the
        common case, not an arbitrary choice.
        """
        cfg = self.cfg
        rank = self.rank
        world_size = self.world_size
        device = self.device
        run_start = time.perf_counter()

        batch_size = getattr(cfg, "batch_size", 512)
        neg_extra = int(getattr(cfg, "minibatch_neg_extra", 0) or 0)
        neg_rng = np.random.default_rng(1234 + rank)
        fanout = getattr(cfg, "fanout", None)
        num_layers = cfg.num_layers
        dim = cfg.output_dim
        use_csls = getattr(cfg, "use_csls", True)
        csls_k = getattr(cfg, "csls_k", 10)
        eval_metric = getattr(cfg, "eval_metric", "cosine")
        # Loss selection mirrors DistTrainer (see minibatch_loss.py),
        # honouring --use_infonce / --loss.
        loss_fn = getattr(cfg, "loss", "margin")
        use_infonce = getattr(cfg, "use_infonce", True)
        infonce_temp = getattr(cfg, "infonce_temp", 0.1)
        nhsm_gamma = getattr(cfg, "nhsm_gamma", 1.0)
        # Eval-time sampling knobs: --eval_fanout (-1 = full
        # neighbourhood, None = same as --fanout) and --eval_batch_size
        # (None = 4x --batch_size). Inference does not need the training
        # batch's memory bound at 15K-100K scale, and nested active sets
        # make every batch touch most of the graph anyway, so fewer,
        # larger, un-sampled eval batches are both faster and less noisy.
        _ef = getattr(cfg, "eval_fanout", None)
        eval_fanout = fanout if _ef is None else (None if _ef < 0 else _ef)
        eval_batch_size = getattr(cfg, "eval_batch_size", None) or 4 * batch_size

        triples = self.data.kg1_triples + self.data.kg2_triples
        index = RREAMinibatchIndex.build(
            triples, self.data.rel_num,
            backend=getattr(cfg, "minibatch_sampler", "vectorized"),
            n_entities=self.data.ent_num_1 + self.data.ent_num_2,
        )
        if rank == 0:
            logging.info(f"[minibatch] sampler backend: {index.backend}")
        owner_map = build_full_owner_map(self.partition_dir, world_size)
        group = self._ocd_gloo_group

        local_kg1_ids = np.array(
            sorted(g for g, r in owner_map.items() if r == rank and g < self.data.ent_num_1),
            dtype=np.int64,
        )
        local_kg2_ids = np.array(
            sorted(g for g, r in owner_map.items() if r == rank and g >= self.data.ent_num_1),
            dtype=np.int64,
        )
        owned_all_ids = np.concatenate([local_kg1_ids, local_kg2_ids])  # --minibatch_neg_extra pool
        if rank == 0:
            logging.info(
                f"[minibatch] {len(owner_map)} entities assigned across "
                f"{world_size} ranks; this rank owns "
                f"{len(local_kg1_ids)} KG1 + {len(local_kg2_ids)} KG2"
            )

        table = SparseEntityTable(dim=dim, device=device, lr=cfg.lr)
        model = RREA(
            rel_num=self.data.rel_num, output_dim=dim, num_layers=num_layers,
            dropout=cfg.dropout, attn_heads=getattr(cfg, "attn_heads", 1),
            edge_chunk=int(getattr(cfg, "rrea_edge_chunk", 0) or 0),
            rrea_dropout=float(getattr(cfg, "rrea_dropout", 0.0) or 0.0),
            attn_softmax=getattr(cfg, "attn_softmax", "dst") or "dst",
            input_relu=bool(getattr(cfg, "rrea_input_relu", False)),
        ).to(device)
        # Learnable (2,) InfoNCE/margin mix, same init as the full-graph path.
        loss_weight = nn.Parameter(torch.tensor([0.5, 0.5], device=device))
        opt_params = list(model.parameters())
        if loss_fn == "margin" and use_infonce:
            opt_params.append(loss_weight)
        optimizer_name = getattr(cfg, "optimizer", "adam")
        if optimizer_name == "sgd":
            model_opt = optim.SGD(opt_params, lr=cfg.lr)
        elif optimizer_name == "rmsprop":
            model_opt = optim.RMSprop(opt_params, lr=cfg.lr)
        else:
            model_opt = optim.Adam(opt_params, lr=cfg.lr)
        if rank == 0:
            logging.info(
                f"[minibatch] loss={describe_minibatch_loss(loss_fn, use_infonce)} "
                f"neg_k={cfg.neg_k} margin={cfg.margin} batch_size={batch_size} neg_extra={neg_extra} "
                f"fanout={fanout} eval_fanout={eval_fanout} eval_batch_size={eval_batch_size}"
            )

        serve_table: SparseEntityTable | None = None  # --minibatch_halo_snapshot (see loop)
        for iteration in range(cfg.max_iterations):
            iter_start = time.perf_counter()
            self._comm_stats = CommStats()
            is_last_iter = (iteration + 1 == cfg.max_iterations)

            if rank == 0:
                logging.info(
                    f"\n{'=' * 60}\n Iteration {iteration + 1}/{cfg.max_iterations} "
                    f"(mini-batch)\n{'=' * 60}"
                )

            # --minibatch_reset_table: mirror the full-graph path's default
            # per-iteration semantics (h_init rebuilt from build_entity_features,
            # optimizer + loss mix weight recreated, model weights warm-started)
            # instead of carrying the entity table and its Adam state across
            # iterations. Motivation: a converged table barely moves when new
            # pseudo-seeds arrive, whereas the full-graph path re-learns the
            # table against the enlarged seed set (iteration 2 then lags the
            # full-graph path by several Hit@1 points).
            # --minibatch_halo_snapshot: the full-graph path's actual
            # iteration semantics, where the iteration-2 accuracy jump comes
            # from halo rows carrying the OTHER rank's
            # trained values while this rank's own rows restart from the
            # deterministic init. Here: freeze the previous iteration's table
            # as what this rank SERVES to peers (serve_table), and re-learn a
            # fresh local table. Eval / pseudo-seeds at the end of the
            # iteration serve the newly trained table (like the full-graph
            # push). Implies the reset semantics of --minibatch_reset_table.
            halo_snapshot = getattr(cfg, "minibatch_halo_snapshot", False)
            if iteration > 0 and halo_snapshot:
                serve_table = table
                if rank == 0:
                    logging.info(
                        f"[minibatch] iter {iteration + 1}: serving peers the frozen iteration-"
                        f"{iteration} table ({len(serve_table)} rows) (--minibatch_halo_snapshot)"
                    )
            if iteration > 0 and (getattr(cfg, "minibatch_reset_table", False) or halo_snapshot):
                table = SparseEntityTable(dim=dim, device=device, lr=cfg.lr)
                loss_weight = nn.Parameter(torch.tensor([0.5, 0.5], device=device))
                opt_params = list(model.parameters())
                if loss_fn == "margin" and use_infonce:
                    opt_params.append(loss_weight)
                if optimizer_name == "sgd":
                    model_opt = optim.SGD(opt_params, lr=cfg.lr)
                elif optimizer_name == "rmsprop":
                    model_opt = optim.RMSprop(opt_params, lr=cfg.lr)
                else:
                    model_opt = optim.Adam(opt_params, lr=cfg.lr)
                if rank == 0:
                    logging.info(
                        f"[minibatch] iter {iteration + 1}: entity table + optimizer reset "
                        f"(--minibatch_reset_table); model weights warm-started"
                    )

            local_pairs = np.array(
                [
                    (int(e1), int(e2)) for e1, e2 in self.train_links
                    if owner_map.get(int(e1), rank) == rank
                ],
                dtype=np.int64,
            ).reshape(-1, 2)
            n_batches = max(1, (len(local_pairs) + batch_size - 1) // batch_size) if len(local_pairs) else 0
            if world_size > 1:
                t = torch.tensor([n_batches], dtype=torch.long, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                n_batches = int(t.item())
            if rank == 0:
                logging.info(
                    f"[minibatch] iter {iteration + 1}: {len(local_pairs)} local pairs -> "
                    f"{n_batches} batches/epoch x {cfg.max_epochs} epochs = "
                    f"{n_batches * cfg.max_epochs} optimizer steps"
                )

            model.train()
            epoch_losses = []
            for epoch in range(cfg.max_epochs):
                perm = np.random.permutation(len(local_pairs)) if len(local_pairs) else np.array([], dtype=np.int64)
                epoch_loss = 0.0
                n_real_batches = 0
                for b in range(n_batches):
                    batch_pairs = local_pairs[perm[b * batch_size : (b + 1) * batch_size]] if len(local_pairs) else np.zeros((0, 2), dtype=np.int64)

                    if len(batch_pairs):
                        seed_ids = np.unique(batch_pairs.reshape(-1))
                        if neg_extra > 0:
                            # --minibatch_neg_extra: non-seed entities as negatives
                            # (see minibatch_loss.extend_seeds_with_negatives)
                            seed_ids = extend_seeds_with_negatives(
                                seed_ids, owned_all_ids, neg_extra, neg_rng,
                            )
                        mb = prepare_rrea_batch(seed_ids, index, num_layers=num_layers, fanout=fanout, device=device)
                        outer_ids = mb.outer_ids
                    else:
                        mb = None
                        outer_ids = np.zeros(0, dtype=np.int64)

                    gathered = gather_minibatch_features(
                        outer_ids, table, owner_map, rank, world_size, device, dim,
                        self._comm_stats, group, serve_table=serve_table,
                    )

                    if mb is None:
                        continue

                    emb = model.forward_blocks(mb, index, gathered.rows)
                    pos_local = {int(g): i for i, g in enumerate(mb.seed_ids)}
                    pairs_local = torch.tensor(
                        [[pos_local[int(e1)], pos_local[int(e2)]] for e1, e2 in batch_pairs],
                        dtype=torch.long, device=device,
                    )
                    kg1_size = int(sum(1 for g in mb.seed_ids if int(g) < self.data.ent_num_1))

                    model_opt.zero_grad()
                    loss = minibatch_loss(
                        emb, pairs_local, kg1_size, loss_fn=loss_fn,
                        use_infonce=use_infonce, loss_weight=loss_weight,
                        neg_k=cfg.neg_k, margin=cfg.margin,
                        infonce_temp=infonce_temp, nhsm_gamma=nhsm_gamma,
                    )
                    loss.backward()
                    model_opt.step()

                    if len(gathered.local_ids) > 0 and gathered.rows.grad is not None:
                        local_mask = torch.from_numpy(np.isin(outer_ids, gathered.local_ids))
                        local_rows = gathered.rows.detach()[local_mask].clone()
                        local_rows.grad = gathered.rows.grad[local_mask]
                        table.apply_gradients(gathered.local_ids, local_rows)

                    epoch_loss += float(loss.item())
                    n_real_batches += 1

                if world_size > 1:
                    with self._comm_stats.timed("minibatch_model_allreduce"):
                        stage = torch.device("cpu") if dist.get_backend() == "gloo" else device
                        for param in model.parameters():
                            p = param.data.to(stage)
                            all_reduce_avg(p)
                            param.data.copy_(p.to(device))

                mean_loss = epoch_loss / n_real_batches if n_real_batches else 0.0
                epoch_losses.append(mean_loss)
                if rank == 0 and (epoch + 1) % max(1, cfg.max_epochs // 10) == 0:
                    logging.info(f"  Epoch {epoch + 1}/{cfg.max_epochs}  loss={mean_loss:.4f}")

            iter_entry = None
            if (iteration + 1) % cfg.eval_interval == 0:
                _eval_t0 = time.perf_counter()
                metrics = minibatch_selective_eval(
                    model, index, table, owner_map, self.test_links,
                    rank, world_size, device, dim, num_layers, eval_fanout, eval_batch_size,
                    use_csls=use_csls, csls_k=csls_k, top_k=(1, 5, 10, 50),
                    metric=eval_metric, comm_stats=self._comm_stats, group=group,
                )
                eval_time_s = time.perf_counter() - _eval_t0
                if rank == 0 and metrics:
                    logging.info(f"[Iter {iteration + 1}] Test eval: {metrics}")
                    iter_entry = {"iteration": iteration + 1} | metrics
                    if epoch_losses:
                        iter_entry["final_loss"] = round(epoch_losses[-1], 4)
                    iter_entry["eval_time_s"] = round(eval_time_s, 4)
                    self._iter_metrics.append(iter_entry)

            self._model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self._h_state = None  # SparseEntityTable has no single dense tensor to snapshot

            _converged = False
            if getattr(cfg, "use_pseudo_seeds", False) and not is_last_iter:
                new_seeds = minibatch_generate_pseudo_seeds(
                    model, index, table, owner_map, local_kg1_ids, local_kg2_ids,
                    rank, world_size, device, dim, num_layers, eval_fanout, eval_batch_size,
                    metric=eval_metric, use_csls=use_csls, csls_k=csls_k,
                )
                self.train_links, added, correct_estimate = _merge_pseudo_seeds(
                    self.train_links, new_seeds, self._test_links_set, self._known_pairs
                )
                if rank == 0:
                    if iter_entry is not None:
                        iter_entry["pseudo_seed_n"] = added
                        iter_entry["pseudo_seed_precision"] = (
                            round(correct_estimate / added, 4) if added > 0 else None
                        )
                    if added > 0:
                        logging.info(
                            f"[Iter {iteration + 1}] Pseudo-seeds: +{added} added, "
                            f"total train_links: {len(self.train_links)}"
                        )
                    else:
                        min_iters = getattr(cfg, "min_iterations", 2)
                        if (iteration + 1) >= min_iters:
                            logging.info(
                                f"[Iter {iteration + 1}] Pseudo-seeds: 0 new seeds — converged."
                            )
                            _converged = True
                conv_t = torch.tensor([int(_converged)], device=device)
                dist.broadcast(conv_t, src=0)
                _converged = bool(conv_t.item())

            if rank == 0 and iter_entry is not None:
                iter_entry["iter_wall_time_s"] = round(time.perf_counter() - iter_start, 4)
                snap = self._comm_stats.snapshot()
                iter_entry["comm_bytes"] = snap["total_bytes"]
                iter_entry["comm_calls"] = snap["total_calls"]
                iter_entry["comm_time_s"] = round(snap["total_time_s"], 4)

            # Incremental metrics.json write; see the matching comment in run().
            if rank == 0 and self.results_dir and self._iter_metrics:
                self._total_wall_time = time.perf_counter() - run_start
                self._save_metrics_json(complete=False)

            if _converged:
                if rank == 0:
                    logging.info(f"Stopping after iteration {iteration + 1} (no new pseudo-seeds).")
                break

        self._total_wall_time = time.perf_counter() - run_start
        if rank == 0:
            logging.info(
                f"DistEAFramework._run_minibatch() complete. Total wall time: "
                f"{self._total_wall_time:.1f}s ({self._total_wall_time / 3600:.2f}h)."
            )
            if self.results_dir and self._iter_metrics:
                self._save_metrics_json()
                self._save_checkpoint()

    def _run_dir(self) -> str:
        """Per-run directory: results_dir/{dataset}/{run_timestamp}/"""
        dataset_name = os.path.basename(os.path.normpath(self.data.data_dir))
        run_id = getattr(self.cfg, "run_timestamp", "")
        out = os.path.join(self.results_dir, dataset_name)
        if run_id:
            out = os.path.join(out, run_id)
        os.makedirs(out, exist_ok=True)
        return out

    def _save_checkpoint(self) -> None:
        """
        Save final model weights + learnable entity embeddings.

        Prefers the best-valid-Hit@1 snapshot (--eval_on_valid) over the last
        iteration's state, so a run that peaks early and degrades afterward
        doesn't save degraded weights just because
        they happened to run last.
        """
        use_best = (
            getattr(self.cfg, "eval_on_valid", False)
            and self._best_model_state is not None
        )
        model_state = self._best_model_state if use_best else self._model_state
        h_state = self._best_h_state if use_best else self._h_state

        if getattr(self.cfg, "no_save_model", False):
            # model.pt is about 0.9 GB at DBP1M/N=1 (the whole h_init table);
            # skipping it keeps many diagnostic runs from filling a shared
            # disk quota.
            logging.info("Checkpoint skipped (--no_save_model)")
            return

        ckpt_path = os.path.join(self._run_dir(), "model.pt")
        torch.save(
            {
                "model_state": model_state,
                "h_state": h_state,
                "hidden_dim": self.cfg.hidden_dim,
                "output_dim": self.cfg.output_dim,
                "num_layers": self.cfg.num_layers,
            },
            ckpt_path,
        )
        source = f"best valid Hit@1={self._best_valid_hit1:.4f}" if use_best else "last iteration"
        logging.info(f"Checkpoint saved ({source}) → {ckpt_path}")

    def _save_metrics_json(self, complete: bool = True) -> None:
        """Write metrics.json for this run.

        complete=False: called mid-loop, once per finished outer
        iteration (see the two `_iter_metrics.append` call sites above),
        so a run killed by a scheduler timeout/preemption/OOM still leaves
        every iteration it actually completed and evaluated on disk instead
        of losing all of it. `complete` is stamped into the payload so
        downstream readers (e.g. scripts/show_results.py) can tell a genuine
        final result from a partial one left behind by an interrupted run.
        The final call after the loop (both
        run() and _run_minibatch()) uses the default complete=True.
        """
        cfg = self.cfg
        out_dir = self._run_dir()
        path = os.path.join(out_dir, "metrics.json")
        dataset_name = os.path.basename(os.path.normpath(self.data.data_dir))

        # --select_by valid picks the iteration by valid_Hit@1 (falls
        # back to test for entries without one); default keeps test.
        best = pick_best_iteration(self._iter_metrics, getattr(cfg, "select_by", "test"))
        # Full CLI args + a short hash of them, so a results aggregator
        # (or anyone re-deriving a table/figure later) can
        # group/dedupe runs by exact configuration without re-deriving it
        # from the individual top-level fields already dumped below (kept
        # for backward compat with scripts/show_results.py, which reads
        # those directly rather than through "args").
        args_dict = {
            k: v for k, v in vars(cfg).items()
            if isinstance(v, (str, int, float, bool)) or v is None
        }
        config_hash = hashlib.sha256(
            json.dumps(args_dict, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        # Common results-schema fields. `final` mirrors the last
        # iteration's metrics under the schema's key names; coverage is
        # unified_eval.py's to fill and stays None here.
        last = self._iter_metrics[-1] if self._iter_metrics else {}
        final_block = {
            "hit1": last.get("Hit@1"), "hit5": last.get("Hit@5"),
            "hit10": last.get("Hit@10"), "hit50": last.get("Hit@50"),
            "mrr": last.get("MRR"), "coverage": None,
        }
        payload = {
            "task_id": getattr(cfg, "task_id", ""),
            "sim": getattr(cfg, "sim", False),
            "config_hash": config_hash,
            "args": args_dict,
            "final": final_block,
            "per_rank": self._per_rank,
            "dataset": dataset_name,
            "fold": os.path.basename(os.path.normpath(self.data.fold_dir)),
            "world_size": self.world_size,
            "hidden_dim": cfg.hidden_dim,
            "output_dim": cfg.output_dim,
            "max_epochs": cfg.max_epochs,
            "later_iter_epochs": getattr(cfg, "later_iter_epochs", None),
            "max_iterations": cfg.max_iterations,
            "warm_start": getattr(cfg, "warm_start", True),
            "persist_optimizer": getattr(cfg, "persist_optimizer", False),
            "use_pseudo_seeds": getattr(cfg, "use_pseudo_seeds", False),
            "seed_threshold": getattr(cfg, "seed_threshold", 0.85),
            "eval_metric": getattr(cfg, "eval_metric", "cosine"),
            "use_csls": getattr(cfg, "use_csls", True),
            "csls_k": getattr(cfg, "csls_k", 10),
            "eval_on_valid": getattr(cfg, "eval_on_valid", False),
            "halo_refresh_interval": getattr(cfg, "halo_refresh_interval", 0),
            "final_train_links": len(self.train_links),
            "run_timestamp": getattr(cfg, "run_timestamp", ""),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mechanisms": {
                "use_mm": getattr(cfg, "use_mm", False),
                "mm_sync_interval": getattr(cfg, "mm_sync_interval", 5),
                "use_ocd": getattr(cfg, "use_ocd", False),
                "use_saca": getattr(cfg, "use_saca", False),
                "saca_threshold": getattr(cfg, "saca_threshold", 0.01),
                "use_halo_kg2": getattr(cfg, "use_halo_kg2", False),
                "halo_kg2_ratio": getattr(cfg, "halo_kg2_ratio", 0.5),
                "halo_kg2_max_hop": getattr(cfg, "halo_kg2_max_hop", None),
            },
            "total_wall_time_s": (
                round(self._total_wall_time, 4) if self._total_wall_time is not None else None
            ),
            "best": best,
            "iterations": self._iter_metrics,
            "complete": complete,
        }
        # Atomic write (temp file + os.replace): this runs once per
        # completed iteration (see the two incremental-write call sites
        # above), not just once at the very end, so a reader (a results
        # script, or a human checking progress) racing a write could
        # otherwise observe a truncated/invalid JSON file. os.replace is
        # atomic on the same filesystem on POSIX, which _run_dir() (under
        # results_dir) always is for this project's runs.
        tmp_path = path + f".tmp{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
        logging.info(f"Results saved → {path}")

    # ------------------------------------------------------------------

    def _pseudo_split_labels(self, links):
        """--eval_pseudo_split (test-leak check): per-link labels
        ps_pair / ps_entity / ps_clean against the pseudo-seeds merged so far
        (train_links minus the original train links). None when the flag is
        unset or no pseudo-seed has been merged yet."""
        if not getattr(self.cfg, "eval_pseudo_split", False) or not links:
            return None
        orig = {(int(a), int(b)) for a, b in self._orig_train_links}
        ps = {(int(a), int(b)) for a, b in self.train_links} - orig
        if not ps:
            return None
        ps1 = {a for a, _ in ps}
        ps2 = {b for _, b in ps}
        labels = []
        for a, b in links:
            a, b = int(a), int(b)
            if (a, b) in ps:
                labels.append("ps_pair")
            elif a in ps1 or b in ps2:
                labels.append("ps_entity")
            else:
                labels.append("ps_clean")
        if self.rank == 0:
            logging.info(
                f"[eval] pseudo split over {len(ps)} pseudo-seeds: "
                f"{labels.count('ps_pair')} ps_pair, {labels.count('ps_entity')} ps_entity, "
                f"{labels.count('ps_clean')} ps_clean of {len(labels)} pairs"
            )
        return labels

    def _eval_group_labels(self, links):
        """Rank-split labels, pseudo-split labels, or both joined as
        '<rank>|<ps>'; None when neither flag is set."""
        rk = self._rank_split_labels(links)
        ps = self._pseudo_split_labels(links)
        if rk is None:
            return ps
        if ps is None:
            return rk
        return [f"{r}|{p}" for r, p in zip(rk, ps, strict=False)]

    def _rank_split_labels(self, links):
        """Multi-node loss diagnosis: per-link 'same_rank' /
        'cross_rank' labels for selective_eval(link_groups=...), from the
        partition named by --eval_rank_split_dir: a pair is same_rank when
        the rank holding e1 in {r}.npy is the rank kg2_assignments.json
        gives e2. Pointing an N=1 run at an N>1 partition reports the same
        split for a like-for-like comparison. None when the flag is unset."""
        split_dir = getattr(self.cfg, "eval_rank_split_dir", "") or ""
        if not split_dir or not links:
            return None
        cache = getattr(self, "_rank_split_cache", None)
        if cache is None:
            cache = self._rank_split_cache = {}
        key = (id(links), len(links))
        if key in cache:
            return cache[key]
        owners = getattr(self, "_rank_split_owner", None)
        if owners is None:
            parts = []
            r = 0
            while os.path.exists(os.path.join(split_dir, f"{r}.npy")):
                parts.append(np.load(os.path.join(split_dir, f"{r}.npy")).astype(np.int64))
                r += 1
            if not parts:
                raise FileNotFoundError(f"--eval_rank_split_dir {split_dir!r}: no 0.npy")
            size1 = int(max(int(p.max()) for p in parts if p.size)) + 1
            own1 = np.full(size1, -1, dtype=np.int64)
            for rr, p in enumerate(parts):
                own1[p] = rr
            with open(os.path.join(split_dir, "kg2_assignments.json")) as fh:
                assign = json.load(fh)
            ids2 = [(int(e), int(rr)) for rr, es in assign.items() for e in es]
            size2 = max(e for e, _ in ids2) + 1
            own2 = np.full(size2, -2, dtype=np.int64)
            for e, rr in ids2:
                own2[e] = rr
            owners = self._rank_split_owner = (own1, own2)
        own1, own2 = owners
        arr = np.asarray(links, dtype=np.int64).reshape(-1, 2)
        o1 = np.where(arr[:, 0] < own1.size, own1[np.minimum(arr[:, 0], own1.size - 1)], -1)
        o2 = np.where(arr[:, 1] < own2.size, own2[np.minimum(arr[:, 1], own2.size - 1)], -2)
        labels = np.where(o1 == o2, "same_rank", "cross_rank").tolist()
        cache[key] = labels
        if self.rank == 0:
            n_same = sum(1 for x in labels if x == "same_rank")
            logging.info(
                f"[eval] rank split from {split_dir}: {n_same}/{len(labels)} pairs same_rank"
            )
        return labels

    def _map_links_to_local(self, global2local: dict, owned_kg1_n=None) -> list:
        """train_links -> local index pairs. owned_kg1_n (full-graph ablation
        only) keeps a pair only when its KG1 endpoint is one of this rank's
        OWNED rows (local index < owned_kg1_n), i.e. the partition's pair
        sharding, even though every KG1 row is now present locally."""
        pairs = []
        for kg1_gid, kg2_gid in self.train_links:
            li1 = global2local.get(int(kg1_gid))
            li2 = global2local.get(int(kg2_gid))
            if li1 is not None and li2 is not None:
                if owned_kg1_n is not None and li1 >= owned_kg1_n:
                    continue
                pairs.append((li1, li2))
        return pairs
