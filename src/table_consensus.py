"""
table_consensus.py -- periodic
consensus of the entity-table (h_init) rows that exist on more than one rank.

Why. With the partitioned table each rank holds its OWNED rows plus HALO
copies of neighbours owned elsewhere, and a seed pair is trained only on its
owner rank. For a test pair whose two entities are owned by different ranks
the alignment signal therefore lands on halo copies, while evaluation gathers
owner rows only. On EN_DE_100K_V1 / N=2 (--eval_rank_split_dir) cross-rank
test pairs reach Hit@1 0.017 vs same-rank 0.551 without consensus (a single
GPU scores 0.32 on such pairs). Overwriting halo
copies with the owner value (--halo_refresh_interval) discards exactly that
signal; this module merges all copies of a row instead:

  mode "mean"   every copy := mean over the row's copies (per-row model
                averaging, as rel_emb / attention kernels already get);
  mode "delta"  every copy := base + sum over copies of (copy - base), base
                being the common value left by the previous sync -- each
                rank's local change is kept in full, as summed gradients
                would be. AdamW's decoupled decay runs on every copy, so a
                row with k copies is decayed k times per interval;
  mode "delta_dc" the same with that decay counted once: adds back
                (k - 1) * (1 - f) * base, f = (1 - lr * wd) ** steps since the
                previous sync (adamw_decay_factor reads both from the
                optimizer). Exact when the parameter stays near base within
                the interval; f = 1 (no decay) makes it plain delta.

Membership is static within a run (the local graph does not change), so the
shared-row set is built once from one all_reduce of an (N_global,) counter;
every sync is one all_reduce of a (B, D) buffer, B = number of rows that
have two or more copies. All collective tensors live on `device`, so NCCL
and gloo both work. Rows held by one rank only are never touched.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.distributed as dist


class TableConsensus:
    def __init__(self, global_ids, device, mode: str = "mean"):
        if mode not in ("mean", "delta", "delta_dc"):
            raise ValueError(f"table consensus mode must be mean, delta or delta_dc, got {mode!r}")
        self.mode = mode
        self.device = torch.device(device)
        ids = torch.as_tensor(np.asarray(global_ids, dtype=np.int64)).to(self.device)
        n_max = torch.tensor([int(ids.max().item()) + 1 if ids.numel() else 0],
                             dtype=torch.long, device=self.device)
        dist.all_reduce(n_max, op=dist.ReduceOp.MAX)
        n_global = int(n_max.item())
        cnt = torch.zeros(n_global, dtype=torch.float32, device=self.device)
        cnt.index_add_(0, ids, torch.ones(ids.numel(), dtype=torch.float32, device=self.device))
        # A global id may sit in two rows of one rank's table (DBP1M: 107 KG2
        # triples touch KG1 ids, so DistGraphLoader can put some of a rank's
        # own KG1 entities into its halo_kg2 too). Such a row is just one
        # more copy: index_add_ counts and sums every row, so all copies of the id,
        # in-rank duplicates included, get merged.
        self.in_rank_duplicates = int((cnt > 1).sum().item())
        dist.all_reduce(cnt)  # SUM: number of ranks holding each id
        shared = torch.nonzero(cnt >= 2, as_tuple=False).squeeze(1)  # sorted, identical on all ranks
        pos = torch.full((n_global,), -1, dtype=torch.long, device=self.device)
        pos[shared] = torch.arange(shared.numel(), device=self.device)
        local_pos = pos[ids]
        keep = local_pos >= 0
        self.local_rows = torch.nonzero(keep, as_tuple=False).squeeze(1)
        self.local_pos = local_pos[keep]
        self.counts = cnt[shared]
        self._cnt_local = self.counts[self.local_pos]
        self.n_shared = int(shared.numel())
        self.n_local_shared = int(self.local_rows.numel())
        self._base: torch.Tensor | None = None
        self.syncs = 0
        self.time_s = 0.0
        self.bytes = 0

    @torch.no_grad()
    def sync(self, table: torch.Tensor, decay_factor: float = 1.0) -> None:
        """Merge every shared row of `table` (this rank's (N_local, D) h_init
        data, rows in the order of `global_ids`) in place. Collective: every
        rank must call it the same number of times."""
        t0 = time.perf_counter()
        width = table.size(1)
        rows = table[self.local_rows].float()
        buf = torch.zeros(self.n_shared, width, dtype=torch.float32, device=table.device)
        if self.mode == "mean" or self._base is None:
            buf.index_add_(0, self.local_pos, rows)
            dist.all_reduce(buf)
            new_local = (buf / self.counts.unsqueeze(1))[self.local_pos]
        else:
            buf.index_add_(0, self.local_pos, rows - self._base)
            dist.all_reduce(buf)
            new_local = self._base + buf[self.local_pos]
            if self.mode == "delta_dc" and decay_factor != 1.0:
                new_local = new_local + ((self._cnt_local - 1.0) * (1.0 - float(decay_factor))
                                         ).unsqueeze(1) * self._base
        table[self.local_rows] = new_local.to(table.dtype)
        self._base = new_local.clone() if self.mode in ("delta", "delta_dc") else None
        if table.is_cuda:
            torch.cuda.synchronize(table.device)  # NCCL collectives are async: time the real cost
        self.syncs += 1
        self.bytes += int(buf.numel()) * 4
        self.time_s += time.perf_counter() - t0

    def summary(self) -> dict:
        return {"mode": self.mode, "shared_rows": self.n_shared,
                "local_shared_rows": self.n_local_shared,
                "in_rank_duplicates": self.in_rank_duplicates, "syncs": self.syncs,
                "time_s": round(self.time_s, 3), "bytes": self.bytes}


def adamw_decay_factor(optimizer, param, last_step: float) -> tuple[float, float]:
    """(f, step_now) for --table_consensus_mode delta_dc: f = (1 - lr * wd)
    ** (optimizer steps taken on `param` since `last_step`), with lr / wd of
    the param group holding `param`. 1.0 unless the optimizer applies
    decoupled decay (torch.optim.AdamW) with wd > 0."""
    st = optimizer.state.get(param, {})
    step = st.get("step", 0)
    step_now = float(step.item() if torch.is_tensor(step) else step)
    if not isinstance(optimizer, torch.optim.AdamW):
        return 1.0, step_now
    for grp in optimizer.param_groups:
        if any(p is param for p in grp["params"]):
            wd = float(grp.get("weight_decay", 0.0) or 0.0)
            lr = float(grp["lr"])
            n = max(step_now - float(last_step), 0.0)
            return (1.0 - lr * wd) ** n, step_now
    return 1.0, step_now
