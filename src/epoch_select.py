"""
epoch_select.py — pure-logic epoch selection for the
per-epoch evaluation loop in framework.py (`--keep_best_epoch`,
`--epoch_patience`, `--select_by`).

Why this exists: without it a single-iteration run reports the LAST epoch's
test numbers. A 100K 3000-epoch curve peaks at 1250 epochs (Hit@1 0.4229)
and collapses to 0.2944 by 3000, and DBP1M peaks at 200-300
epochs, so any longer training budget needs an epoch picked on the VALID
split, with the test numbers reported for that epoch. The existing
`--eval_on_valid --patience` machinery is per OUTER iteration and does
nothing with `--max_iterations 1`.

The class holds no tensors and does no I/O: framework.py feeds it one
observation per evaluated epoch (the valid or test Hit@1 of that epoch,
whichever `--select_by` names) and asks two questions back -- "is this a new
best, i.e. should I snapshot the model/h_init now?" and "has patience run
out?". Keeping the decision logic here makes it testable on CPU without a
graph (tests/test_epoch_selector.py).

Semantics (all deterministic, rank-0 decides and broadcasts in framework.py):
  * `observe(epoch, score)` returns True exactly when score > best so far
    (strict improvement; ties do NOT replace the earlier epoch, matching the
    existing per-iteration valid logic in framework.py Step 7d).
  * `should_stop()` is True once `patience` consecutive non-improving
    observations have been seen (patience <= 0 disables stopping).
  * `min_epochs` guards against stopping before a floor epoch, mirroring
    `--min_iterations` for the outer loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpochSelector:
    """Track the best evaluated epoch and a patience counter.

    Args:
        patience: consecutive non-improving evaluations before
            `should_stop()` turns True. 0 (default) never stops.
        min_epochs: never report `should_stop()` before this epoch
            (inclusive floor on the epoch number, 1-based like the log).
        metric: label only (e.g. "valid_Hit@1"); recorded in `summary()`.
    """

    patience: int = 0
    min_epochs: int = 0
    metric: str = "valid_Hit@1"
    best_score: float = float("-inf")
    best_epoch: int = 0
    last_epoch: int = 0
    bad_evals: int = 0
    history: list[tuple[int, float]] = field(default_factory=list)

    def observe(self, epoch: int, score: float) -> bool:
        """Record one evaluation; return True iff it is a new best."""
        if epoch <= self.last_epoch and self.history:
            raise ValueError(
                f"EpochSelector.observe: epochs must increase "
                f"(got {epoch} after {self.last_epoch})"
            )
        self.last_epoch = int(epoch)
        score = float(score)
        self.history.append((int(epoch), score))
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = int(epoch)
            self.bad_evals = 0
            return True
        self.bad_evals += 1
        return False

    def should_stop(self) -> bool:
        """True once patience is exhausted (and the min-epoch floor passed)."""
        if self.patience <= 0:
            return False
        if self.last_epoch < self.min_epochs:
            return False
        return self.bad_evals >= self.patience

    @property
    def has_best(self) -> bool:
        return self.best_epoch > 0

    def summary(self) -> dict:
        """Small JSON-able block for metrics.json."""
        return {
            "select_by": self.metric,
            "best_epoch": self.best_epoch,
            "best_score": (
                None if self.best_score == float("-inf") else round(self.best_score, 6)
            ),
            "evaluated_epochs": len(self.history),
            "stopped_early": self.should_stop(),
            "patience": self.patience,
        }


_SPLITS = ("test", "valid", "train")


def parse_eval_splits(spec: str | None, select_by: str = "test") -> list[str]:
    """Parse `--eval_every_splits` ("test,valid") into an ordered, deduped
    list; the `--select_by` split is appended if missing so the selector
    always has its score. Unknown names raise ValueError."""
    out: list[str] = []
    for tok in (spec or "test").split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t not in _SPLITS:
            raise ValueError(f"--eval_every_splits: unknown split '{t}' (use {_SPLITS})")
        if t not in out:
            out.append(t)
    if select_by in _SPLITS and select_by not in out:
        out.append(select_by)
    return out or ["test"]


def split_prefix(split: str) -> str:
    """Metric-key prefix per split: test keys stay bare (metrics.json schema)."""
    return "" if split == "test" else f"{split}_"


def pick_best_iteration(iter_metrics: list[dict], select_by: str = "test") -> dict:
    """Choose the `best` entry for metrics.json.

    `select_by="test"` is the default (max test Hit@1 over
    iterations). `select_by="valid"` picks the iteration with the highest
    recorded `valid_Hit@1` and falls back to test Hit@1 for iterations
    that carry no valid number (older runs / `--eval_on_valid` off), so
    the key always resolves. Returns {} for an empty list.
    """
    if not iter_metrics:
        return {}
    if select_by == "valid":
        def key(m: dict) -> tuple[int, float]:
            v = m.get("valid_Hit@1")
            if v is None:
                return (0, float(m.get("Hit@1", 0.0)))
            return (1, float(v))
        return max(iter_metrics, key=key)
    return max(iter_metrics, key=lambda m: float(m.get("Hit@1", 0.0)))
