"""
sparse_table.py — SparseEntityTable for the mini-batch path.

Replaces "one dense h_init tensor covering the whole local+halo partition"
(which runs out of memory on DBP1M at ws=2 — framework.py Step 2, ~950K
rows held live for the whole run) with on-demand materialization of only the entities
a given mini-batch actually touches.

Two facts make this tractable instead of a full embedding-layer rewrite:
  - build_entity_features (models/gcn.py) is deterministic/ID-seeded: an
    entity's INITIAL value needs no storage at all, it's re-derivable from
    its global ID for free. Only entities that have actually received a
    gradient update need persistent storage.
  - GCN-Align's first layer is featureless (models/gcn.py module docstring)
    — h_init/this table already IS the model's entire learnable-entity
    capacity, not one piece of a larger parameter structure.

No torch.optim integration: there is no single nn.Parameter to hand an
optimizer, so this implements a small per-row Adam step directly (same
exp_avg/exp_avg_sq state shape framework.py's _reset_adam_momentum_for_rows
already manipulates for the full-graph path, just keyed by global id in a
dict instead of row-indexed into one big tensor).
"""

from __future__ import annotations

import numpy as np
import torch

from models.gcn import build_entity_features


class SparseEntityTable:
    def __init__(
        self,
        dim: int,
        device: torch.device,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.dim = dim
        self.device = device
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self._rows: dict[int, torch.Tensor] = {}
        self._exp_avg: dict[int, torch.Tensor] = {}
        self._exp_avg_sq: dict[int, torch.Tensor] = {}
        self._step_count: dict[int, int] = {}

    def _init_row(self, gid: int) -> torch.Tensor:
        return build_entity_features(np.array([gid]), self.dim, self.device)[0]

    def gather(self, global_ids: np.ndarray) -> torch.Tensor:
        """Return a (len(global_ids), dim) leaf tensor, requires_grad=True.
        Rows never touched before are the deterministic initial value;
        rows previously updated via apply_gradients() return that value.

        Rows missing from self._rows are batched into a single
        build_entity_features() call rather than one call per id -- that
        function already loops per-id internally to seed each row's RNG
        (models/gcn.py), so calling it once with all N missing ids instead
        of N times with one id each produces bit-identical values (each
        id's RNG is constructed independently either way) while paying the
        Python/array-construction overhead once instead of N times.

        Deliberately does NOT write the freshly-computed rows into
        self._rows -- only apply_gradients() persists. Persisting on a bare
        read would make gather()-without-apply_gradients calls (eval does
        exactly this) silently inflate __len__()'s "entities ever
        materialized" count with entities that were only ever read, never
        trained."""
        ids = [int(gid) for gid in global_ids]
        if not ids:
            # A rank padding with an empty batch (minibatch_eval /
            # _run_minibatch synchronise the batch COUNT across ranks, so a
            # rank with fewer local pairs still participates in every
            # collective round) must get a (0, dim) leaf, not a
            # torch.stack([]) crash (e.g. 23 vs 24 batches per rank).
            return torch.zeros((0, self.dim), device=self.device).requires_grad_(True)
        missing = [gid for gid in ids if gid not in self._rows]
        fresh_by_id: dict[int, torch.Tensor] = {}
        if missing:
            fresh = build_entity_features(np.array(missing), self.dim, self.device)
            fresh_by_id = dict(zip(missing, fresh, strict=False))
        rows = [self._rows[gid] if gid in self._rows else fresh_by_id[gid] for gid in ids]
        return torch.stack(rows, dim=0).clone().detach().requires_grad_(True)

    def apply_gradients(self, global_ids: np.ndarray, batch: torch.Tensor) -> None:  # noqa: D417
        """Call after loss.backward() has populated batch.grad (batch must
        be exactly the tensor returned by gather(global_ids) for the same
        global_ids, in the same order). Applies one per-row Adam step and
        persists both the updated value and this row's moment state, so a
        later gather() for the same id continues its own Adam trajectory
        rather than restarting it.

        Vectorised over all N touched rows in one batched tensor op instead
        of a Python loop doing N separate small-tensor Adam updates -- safe
        because each row's update is independent (no cross-row reduction
        anywhere in Adam), so batching is numerically equivalent to a
        per-row loop (within float32 tolerance -- the bias-correction power
        is computed via tensor pow rather than Python's float**int, which
        can differ at the ~1e-7 level, negligible for a stochastic
        optimizer), just without paying N-1 extra rounds
        of Python/dict overhead for the actual arithmetic. Per-row step
        counts can differ (some rows are on their 1st update, others their
        100th), so the bias-correction terms (1 - beta^t) are computed with
        elementwise tensor power, not a Python scalar power, to keep each
        row's own t.
        """
        if len(global_ids) == 0:
            return  # padding batch (see gather): nothing to update

        if batch.grad is None:
            raise ValueError(
                "apply_gradients called before backward() populated batch.grad "
                "-- batch must be the exact tensor gather() returned, with "
                "requires_grad=True intact through the forward/backward pass."
            )
        grad = batch.grad
        ids = [int(gid) for gid in global_ids]
        with torch.no_grad():
            zeros = torch.zeros(self.dim, device=self.device)
            m = torch.stack([self._exp_avg.get(gid, zeros) for gid in ids])
            v = torch.stack([self._exp_avg_sq.get(gid, zeros) for gid in ids])
            t = torch.tensor(
                [self._step_count.get(gid, 0) + 1 for gid in ids],
                dtype=torch.float32, device=self.device,
            )

            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)
            bias1 = (1 - self.beta1 ** t).unsqueeze(1)  # (N, 1), broadcasts over dim
            bias2 = (1 - self.beta2 ** t).unsqueeze(1)
            m_hat = m / bias1
            v_hat = v / bias2
            new_val = batch.detach() - self.lr * m_hat / (v_hat.sqrt() + self.eps)

            for i, gid in enumerate(ids):
                self._rows[gid] = new_val[i]
                self._exp_avg[gid] = m[i]
                self._exp_avg_sq[gid] = v[i]
                self._step_count[gid] = int(t[i].item())

    def __len__(self) -> int:
        return len(self._rows)
