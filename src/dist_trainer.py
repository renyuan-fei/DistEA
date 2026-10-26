"""
dist_trainer.py — DistTrainer: single training epoch with all_reduce sync.

Parameter synchronisation uses dist.all_reduce(ReduceOp.AVG) after each epoch,
giving a true global average across all ranks (LeapGNN-style collective sync).
This covers the model's own (small) parameters only — h_init (the per-entity
table) is never part of it; --consistency_lambda adds an
optional gradient-level term pulling h_init's halo rows toward their
last-known remote value every epoch, on top of framework.py's existing
once-per-iteration hard overwrite.

--shared_h_init is a different, more direct answer to
the same underlying problem: instead of h_init being partitioned per-rank and
patched/regularized after the fact, every rank holds the FULL entity table and
`local_idx` (see train_epoch) selects this rank's local subgraph rows from it
via a differentiable gather — h_init itself then gets all_reduce'd every
epoch alongside the model's own parameters, so there's no separate rank-owned
copy to reconcile in the first place.

--shared_h_init_weighted_avg fixes a cost of the
naive version: dist.all_reduce(AVG) divides every row by world_size
regardless of how many ranks actually touched it this epoch. A row exclusive
to one rank's local subgraph gets averaged with (world_size - 1) OTHER
ranks' unchanged, stale copies of that same row — diluting its effective
learning rate by 1/world_size even though only one rank ever had a real
gradient for it. _masked_row_and_count + a SUM-then-divide-by-touch-count
scheme (see below) averages each row only over the ranks that actually
touched it, at the cost of one extra small all_reduce per epoch.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed import ReduceOp

from comm_stats import CommStats
from dist_utils import all_reduce_avg
from models.gcn import (
    alignment_loss,
    combined_align_loss,
    consistency_reg_loss,
    nhsm_loss,
)


def _masked_row_and_count(
    h_init: torch.Tensor, local_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a (N, D) tensor holding this rank's current h_init values only at
    `local_idx` rows (zero elsewhere), and a matching (N,) indicator counting
    1 at those rows. Summed across ranks via all_reduce(SUM) and divided
    (masked_sum / count.clamp_min(1)), this averages each row only over the
    ranks that actually touched it this epoch — instead of dist.all_reduce's
    plain AVG, which divides every row by world_size regardless of how many
    ranks touched it (--shared_h_init_weighted_avg).

    Pure function, no communication — the caller does the actual
    all_reduce(SUM) on both returned tensors.
    """
    masked = torch.zeros_like(h_init.data)
    count = torch.zeros(h_init.size(0), device=h_init.device, dtype=h_init.dtype)
    masked[local_idx] = h_init.data[local_idx]
    count[local_idx] = 1.0
    return masked, count


class DistTrainer:
    """
    Encapsulates one training epoch for a DistEA rank.

    Args:
        model:              GCN instance (on device)
        optimizer:          torch Optimizer
        rank:               this process's rank
        world_size:         total number of ranks
        neg_k:              number of negatives per side per positive pair
        margin:             contrastive loss margin
        use_infonce:        if True, use ELsEA-style InfoNCE+align combined loss
        infonce_temp:       temperature for InfoNCE
        loss_weight:        learnable (2,) parameter mixing InfoNCE vs align
        neg_resample_every: re-sample negatives every N epochs (ELsEA = 10)
        consistency_lambda: weight for the h_init cross-rank consistency
                             regularizer (--consistency_lambda);
                             0.0 (default) disables it entirely, zero overhead.
        weighted_shared_h_init: when True (and local_idx is given to
                             train_epoch — --shared_h_init), average each
                             h_init row only over the ranks that actually
                             touched it this epoch instead of dividing by
                             world_size unconditionally
                             (--shared_h_init_weighted_avg). No effect
                             without --shared_h_init.
        comm_stats:         optional CommStats to record all_reduce timing/volume
                             into (defaults to a private, unshared instance).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        rank: int,
        world_size: int,
        neg_k: int = 1,
        margin: float = 6.0,
        use_infonce: bool = True,
        infonce_temp: float = 0.1,
        loss_weight: torch.Tensor | None = None,
        neg_resample_every: int = 10,
        consistency_lambda: float = 0.0,
        weighted_shared_h_init: bool = False,
        comm_stats: CommStats | None = None,
        loss_fn: str = "margin",
        nhsm_gamma: float = 1.0,
        amp: str = "none",
    ):
        self.model = model
        self.optimizer = optimizer
        self.rank = rank
        self.world_size = world_size
        # CUDA events around each nccl parameter all_reduce. Under nccl the
        # collective is asynchronous: the host timer in _allreduce_sync only
        # sees the enqueue, and the time this rank's GPU waits for the
        # slowest rank lands in the NEXT epoch's wall time on every rank, so
        # all ranks' epoch times converge whatever their load (on 8-node
        # DBP1M, ranks whose graphs differ ~3x all report ~0.5 s/epoch).
        # The events time that wait on the GPU without blocking the host;
        # allreduce_gpu_wait_s() reads them once the epoch loop is done.
        self._ar_events: list = []
        self.neg_k = neg_k
        self.margin = margin
        self.use_infonce = use_infonce
        self.infonce_temp = infonce_temp
        self.loss_weight = loss_weight
        self.neg_resample_every = max(1, neg_resample_every)
        self.consistency_lambda = consistency_lambda
        self.weighted_shared_h_init = weighted_shared_h_init
        # "margin" = existing ELsEA-faithful path (align/combined);
        # "nhsm" = Dual-AMN Normalized Hard Sample Mining (models/gcn.py::
        # nhsm_loss). Under nhsm, neg_k/neg_resample_every/pair_weights/
        # use_infonce are inert -- the reference loss has no such knobs.
        self.loss_fn = loss_fn
        self.nhsm_gamma = nhsm_gamma
        # "bf16" runs the model forward under torch.autocast
        # (bfloat16); everything else -- loss math, backward, optimizer,
        # allreduce -- stays fp32. See train_epoch for why the boundary
        # sits exactly at the forward.
        self.amp = amp
        if rank == 0 and loss_fn == "nhsm":
            logging.info(
                "[DistTrainer] --loss nhsm active: neg_k/neg_resample_every/"
                "pair_weights/use_infonce are inert for this loss."
            )
        self.comm_stats = comm_stats or CommStats()
        self._epoch_count = 0
        self._cached_neg_kg1: torch.Tensor | None = None
        self._cached_neg_kg2: torch.Tensor | None = None

    def train_epoch(
        self,
        g,
        h_init: torch.Tensor,
        train_pairs: torch.Tensor,
        kg1_size: int,
        pair_weights: torch.Tensor | None = None,
        halo_kg2_emb: torch.Tensor | None = None,
        consistency_idx: torch.Tensor | None = None,
        consistency_target: torch.Tensor | None = None,
        local_idx: torch.Tensor | None = None,
    ) -> float:
        """
        Run one training epoch.

        Args:
            g:            local DGL graph or sparse adj
            h_init:       (N_local, hidden_dim) initial features — OR, when
                          `local_idx` is given (--shared_h_init), the FULL
                          (N_global, hidden_dim) table; `local_idx` selects
                          this rank's local subgraph rows from it.
            train_pairs:  (P, 2) local-index pairs (kg1_local, kg2_local)
            kg1_size:     number of KG1 nodes (local + halo) in the local graph
            pair_weights: optional (P,) per-pair weight from MM/VBIP (see
                          dist_vbip.py); None trains with the plain unweighted loss.
            halo_kg2_emb: optional (M, output_dim) extra KG2 candidate
                          embeddings (see dist_ctx_kg2.py's "KG2 Counterpart
                          Halo") — appended after the local graph's own nodes
                          so alignment_loss's negative sampling can draw from
                          them too. These entities are not part of `g`; they
                          contribute no graph edges/message-passing, only
                          extra negative-sampling targets.
            consistency_idx:    optional local indices into h_init (halo_kg1
                          + halo_kg2 rows with a known remote value this
                          iteration) for the consistency regularizer (see
                          consistency_reg_loss, models/gcn.py). None/empty
                          skips the term entirely regardless of
                          self.consistency_lambda. Not meaningful together
                          with `local_idx` (--shared_h_init) — the two
                          mechanisms solve the same problem differently and
                          aren't intended to combine.
            consistency_target: matching (M, output_dim) detached tensor of
                          each row's last-known remote value.
            local_idx:    optional (N_local,) global-id indices selecting
                          this rank's local subgraph rows out of a FULL
                          shared h_init table (--shared_h_init). `h_init[local_idx]` is a differentiable
                          gather — gradients accumulate into the full table
                          at exactly these rows via ordinary autograd, same
                          mechanism as an embedding lookup. None (default):
                          `h_init` is used as-is (per-rank-local
                          behaviour), zero overhead.

        Returns:
            loss value (float)
        """
        self.model.train()
        self.optimizer.zero_grad()
        self._epoch_count += 1

        h_local = h_init[local_idx] if local_idx is not None else h_init
        # bf16 autocast covers only the model forward (the
        # matmul-heavy part, where tensor cores pay off); `emb` is recast
        # to fp32 immediately so every loss below (margin z-score, nhsm's
        # per-row std + logsumexp(30x+10)) keeps full precision -- those
        # are exactly the ops where bf16's 8-bit mantissa loses accuracy.
        # Backward/optimizer/allreduce are outside autocast by
        # construction, so no GradScaler is needed (bf16 shares fp32's
        # exponent range).
        if self.amp == "bf16":
            with torch.autocast(
                device_type="cuda" if h_local.is_cuda else "cpu",
                dtype=torch.bfloat16,
            ):
                emb = self.model(g, h_local)
            emb = emb.float()
        else:
            emb = self.model(g, h_local)
        if halo_kg2_emb is not None and halo_kg2_emb.numel() > 0:
            emb_for_loss = torch.cat([emb, halo_kg2_emb.to(emb.device, emb.dtype)], dim=0)
        else:
            emb_for_loss = emb

        # Resample negatives every `neg_resample_every` epochs (ELsEA: 10),
        # not every epoch — GCN_Align_run.py's `if epoch % 10 == 0:
        # neg_left/neg_right = np.random.choice(...)`. Reusing the same
        # negatives for the epochs in between is the actual reference
        # behaviour, not an optimisation.
        if self.loss_fn == "nhsm":
            # whole-table negatives, no sampling/caching -- see
            # nhsm_loss's docstring for the exact reference semantics.
            if len(train_pairs) > 0:
                loss = nhsm_loss(emb_for_loss, train_pairs, gamma=self.nhsm_gamma)
            else:
                loss = emb_for_loss.sum() * 0.0
            if (
                self.consistency_lambda > 0
                and consistency_idx is not None
                and consistency_idx.numel() > 0
            ):
                loss = loss + self.consistency_lambda * consistency_reg_loss(
                    h_init, consistency_idx, consistency_target
                )
            loss.backward()
            self.optimizer.step()
            loss_val = float(loss.item())
            if self.world_size > 1:
                self._allreduce_sync(
                    h_init if local_idx is not None else None, local_idx,
                )
            return loss_val

        P = len(train_pairs)
        need_resample = (
            self._cached_neg_kg1 is None
            or (self._epoch_count - 1) % self.neg_resample_every == 0
        )
        if P > 0 and need_resample:
            N = emb_for_loss.size(0)
            # RREA/runner.py::get_train_set draws BOTH corrupted endpoints
            # from the full node range (np.random.randint(0, node_size, ...)),
            # not restricted to "corrupt-left from KG1 / corrupt-right from
            # KG2" — a same-side hard negative (e.g. a wrong KG1 entity
            # standing in for the true KG1 anchor) is exactly as valid a
            # negative as a cross-side one. kg1_size is unused here but
            # kept as a parameter — alignment_loss's signature still uses it
            # as the KG1/KG2 boundary for other bookkeeping.
            self._cached_neg_kg1 = torch.randint(
                0, N, (P, self.neg_k), device=emb_for_loss.device
            )
            self._cached_neg_kg2 = torch.randint(
                0, N, (P, self.neg_k), device=emb_for_loss.device
            )

        if self.use_infonce and self.loss_weight is not None:
            loss = combined_align_loss(
                emb_for_loss,
                train_pairs,
                kg1_size=kg1_size,
                weight=self.loss_weight,
                neg_k=self.neg_k,
                margin=self.margin,
                infonce_temp=self.infonce_temp,
                pair_weights=pair_weights,
                neg_kg1=self._cached_neg_kg1,
                neg_kg2=self._cached_neg_kg2,
            )
        else:
            loss = alignment_loss(
                emb_for_loss,
                train_pairs,
                kg1_size=kg1_size,
                neg_k=self.neg_k,
                margin=self.margin,
                pair_weights=pair_weights,
                neg_kg1=self._cached_neg_kg1,
                neg_kg2=self._cached_neg_kg2,
            )

        if (
            self.consistency_lambda > 0
            and consistency_idx is not None
            and consistency_idx.numel() > 0
        ):
            loss = loss + self.consistency_lambda * consistency_reg_loss(
                h_init, consistency_idx, consistency_target
            )

        loss.backward()
        self.optimizer.step()

        loss_val = float(loss.item())

        if self.world_size > 1:
            self._allreduce_sync(
                h_init if local_idx is not None else None, local_idx,
            )

        return loss_val

    def allreduce_gpu_wait_s(self) -> float | None:
        """GPU time from "this rank's work before the parameter all_reduce
        is done" to "the all_reduce is done", summed over the epochs since
        the last call: the wait for the slowest rank plus the transfer.
        None when no nccl all_reduce was timed (gloo, CPU, world_size 1) or
        the events could not be read. Synchronises on the last event, so
        call it after the epoch loop."""
        if not self._ar_events:
            return None
        events, self._ar_events = self._ar_events, []
        try:
            events[-1][1].synchronize()
            return sum(a.elapsed_time(b) for a, b in events) / 1000.0
        except Exception:  # noqa: BLE001 -- timing must never stop a run
            return None

    def _allreduce_sync(
        self,
        shared_h_init: torch.Tensor | None = None,
        local_idx: torch.Tensor | None = None,
    ):
        """
        Average model parameters across all ranks. When `shared_h_init` is
        given (--shared_h_init), also sync the FULL
        h_init table alongside the model's own parameters — this is what
        keeps every rank's copy identical without any halo-row patching.

        Two ways to combine ranks' updates:
          - Plain dist.all_reduce(AVG) (default): divides every row by
            world_size unconditionally, even rows only one rank actually
            touched this epoch, so the effective learning rate for
            exclusively-local rows is diluted by 1/world_size.
          - Masked SUM + touch-count divide (self.weighted_shared_h_init,
            --shared_h_init_weighted_avg): each row is averaged only over
            the ranks whose local_idx included it this epoch — one extra
            small all_reduce (the touch-count vector), same size all_reduce
            for the masked values.
        """
        # Collectives stage through CPU ONLY under gloo: gloo's CUDA path
        # allocates its own reduction buffer per call, and when ranks are
        # processes sharing one physical GPU (--gres=gpu:4,
        # --nproc_per_node=128, i.e. 32 ranks/GPU) those buffers pile up on
        # the same card 32 times over on top of each process's own CUDA
        # context, which runs out of GPU memory at world_size=128 (the same
        # failure class as eval's _all_gather_embeddings). These are plain value syncs on
        # `.data` after optimizer.step() already ran, not part of the
        # autograd graph, so staging through CPU changes nothing about
        # gradient semantics — only where the collective's temporary
        # buffers live.
        #
        # Under nccl (true multi-node, one real GPU per rank), the CPU
        # staging above is not just unnecessary but fatal: nccl has no CPU
        # backend and `dist.all_reduce` on a CPU tensor raises `RuntimeError:
        # No backend type associated with device type cpu`. So under nccl,
        # reduce in place on the param's own
        # (CUDA) device instead.
        use_cpu_staging = dist.get_backend() == "gloo"
        stage = torch.device("cpu") if use_cpu_staging else None
        ev = None
        if stage is None and torch.cuda.is_available():
            try:
                ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                ev[0].record()
            except Exception:  # noqa: BLE001 -- timing must never stop a run
                ev = None
        with self.comm_stats.timed("all_reduce"):
            for param in self.model.parameters():
                data = param.data.to(stage) if stage is not None else param.data
                all_reduce_avg(data)
                if stage is not None:
                    param.data.copy_(data)
            if shared_h_init is not None:
                if self.weighted_shared_h_init and local_idx is not None:
                    masked, count = _masked_row_and_count(shared_h_init, local_idx)
                    m = masked.to(stage) if stage is not None else masked
                    c = count.to(stage) if stage is not None else count
                    dist.all_reduce(m, op=ReduceOp.SUM)
                    dist.all_reduce(c, op=ReduceOp.SUM)
                    result = (m / c.clamp_min(1.0).unsqueeze(1))
                    shared_h_init.data.copy_(
                        result.to(shared_h_init.device) if stage is not None else result
                    )
                else:
                    h = shared_h_init.data.to(stage) if stage is not None else shared_h_init.data
                    all_reduce_avg(h)
                    if stage is not None:
                        shared_h_init.data.copy_(h.to(shared_h_init.device))
        if ev is not None:
            try:
                ev[1].record()
                self._ar_events.append(ev)
            except Exception:  # noqa: BLE001
                pass
        n_bytes = sum(p.numel() for p in self.model.parameters()) * 4
        if shared_h_init is not None:
            n_bytes += shared_h_init.numel() * 4
            if self.weighted_shared_h_init and local_idx is not None:
                n_bytes += shared_h_init.size(0) * 4
        self.comm_stats.record("all_reduce", num_bytes=n_bytes, calls=1)
