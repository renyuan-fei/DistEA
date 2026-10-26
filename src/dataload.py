"""dataload.py — dataset loading for DistEA.

`OpenEAData` reads a dataset in the OpenEA layout (`rel_triples_1`, `rel_triples_2` and a split
directory with `train_links` / `valid_links` / `test_links`) and assigns integer ids: KG1
entities first, KG2 entities after them, relations of both KGs in one id space. Only relation
triples are read. To add another dataset format, subclass `Data`.
"""
import hashlib
import json
import os
import socket
import zipfile
from abc import ABC, abstractmethod

import numpy as np


def read_tab_lines(file_path):
    with open(file_path, encoding="utf-8") as f:
        return [line.strip().split("\t") for line in f if line.strip()]


_PARSED_CACHE_VERSION = 1


def _data_cache_enabled() -> bool:
    """DISTEA_DATA_CACHE=1: OpenEAData keeps the parsed
    triples -- entity / relation names in id order and the (T, 3) id arrays --
    in one .npz (.distea_parsed_v1_<hash of the data dir's real path>.npz,
    176 MB at DBP1M) in DISTEA_DATA_CACHE_DIR, or next to the data when that
    is unset, keyed on the two triple files' size and mtime. A hit replaces
    the per-line text parse + dict build (13-20 s on every rank at DBP1M)
    with a load (4.0 s for the whole constructor, links included) and
    pre-fills the arrays DistGraphLoader's
    vectorised path reads (dist_graph._triples_array), so the int-tuple lists
    are never built on that path (DistGraphLoader.load 9.6 -> 1.1 s at
    N=8). A miss parses the text files and writes the cache (atomic rename;
    an OSError only skips the write). Exact: the cache holds what the parse
    produced (tests/test_data_cache.py). Read at construction; default off."""
    return os.environ.get("DISTEA_DATA_CACHE", "0").lower() in ("1", "true", "yes")


def _names_blob(names: list[str]) -> np.ndarray:
    # Names never contain "\n" (lines are split on it, and text mode turns
    # "\r" into "\n"), so a newline-joined UTF-8 blob round-trips; the
    # caller checks the decoded count.
    return np.frombuffer("\n".join(names).encode("utf-8"), dtype=np.uint8)


def _blob_names(blob: np.ndarray, n: int) -> list[str]:
    return blob.tobytes().decode("utf-8").split("\n") if n else []


class Data(ABC):
    """
    Abstract base for entity alignment datasets.

    Subclasses must implement:
      kg1_triples, kg2_triples, load_train_alignment, load_test_alignment

    Subclass constructors must populate:
      ent2id, rel2id, ent_num_1, ent_num_2, rel_num
    """

    def __init__(self, data_dir: str, fold_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.fold_dir = fold_dir
        self.output_dir = output_dir

        self.ent2id: dict[str, int] = {}
        self.rel2id: dict[str, int] = {}
        self.ent_num: int = 0
        self.ent_num_1: int = 0
        self.ent_num_2: int = 0
        self.rel_num: int = 0

        # Populated by load_weights(); empty dicts → uniform 1.0 edge weights.
        # Each value is (r2f, r2if) — kept separate (not summed) because ELsEA
        # applies them asymmetrically: r2if to the head->tail direction, r2f
        # to tail->head (see dist_graph.py's add_edge, and ELsEA's
        # get_weighted_adj in DivEA/GCN_Align/utils.py).
        self.edge_weights1: dict[int, tuple[float, float]] = {}
        self.edge_weights2: dict[int, tuple[float, float]] = {}
        self.node_weights1: dict[int, float] = {}
        self.node_weights2: dict[int, float] = {}

    @property
    def kg1_entities(self) -> list[int]:
        return list(range(self.ent_num_1))

    @property
    def kg2_entities(self) -> list[int]:
        return list(range(self.ent_num_1, self.ent_num_1 + self.ent_num_2))

    @property
    @abstractmethod
    def kg1_triples(self) -> list[tuple[int, int, int]]:
        """KG1 triples as (head_id, rel_id, tail_id) integer tuples."""

    @property
    @abstractmethod
    def kg2_triples(self) -> list[tuple[int, int, int]]:
        """KG2 triples as (head_id, rel_id, tail_id) integer tuples."""

    @abstractmethod
    def load_train_alignment(self) -> list[tuple[int, int]]:
        """Return (kg1_entity_id, kg2_entity_id) pairs for training."""

    @abstractmethod
    def load_test_alignment(self) -> list[tuple[int, int]]:
        """Return (kg1_entity_id, kg2_entity_id) pairs for testing."""

    def load_valid_alignment(self) -> list[tuple[int, int]]:
        """Return (kg1_entity_id, kg2_entity_id) pairs for validation. Default: empty."""
        return []

    def load_weights(self) -> None:  # noqa: B027 -- intentionally optional override, not abstract
        """Load pre-computed edge/node weights. Override for dataset-specific formats."""


class OpenEAData(Data):
    """OpenEA benchmark dataset (rel_triples_1/2 + fold split files)."""

    def __init__(self, data_dir: str, fold_dir: str, output_dir: str):
        super().__init__(data_dir, fold_dir, output_dir)

        triple_files = [os.path.join(data_dir, "rel_triples_1"),
                        os.path.join(data_dir, "rel_triples_2")]
        use_cache = _data_cache_enabled()
        cached = use_cache and self._load_parsed_cache(triple_files)
        if not cached:
            self._rel_triples_1 = read_tab_lines(triple_files[0])
            self._rel_triples_2 = read_tab_lines(triple_files[1])
        self._train_links = read_tab_lines(os.path.join(fold_dir, "train_links"))
        self._valid_links = read_tab_lines(os.path.join(fold_dir, "valid_links"))
        self._test_links = read_tab_lines(os.path.join(fold_dir, "test_links"))

        if not cached:
            self._build_dict()
            if use_cache:
                self._write_parsed_cache(triple_files)

    def _get_ent_id(self, ent: str) -> int:
        if ent not in self.ent2id:
            self.ent2id[ent] = self.ent_num
            self.ent_num += 1
        return self.ent2id[ent]

    def _get_rel_id(self, rel: str) -> int:
        if rel not in self.rel2id:
            self.rel2id[rel] = self.rel_num
            self.rel_num += 1
        return self.rel2id[rel]

    def _build_dict(self) -> None:
        for h, r, t in self._rel_triples_1:
            self._get_ent_id(h)
            self._get_ent_id(t)
            self._get_rel_id(r)
        self.ent_num_1 = self.ent_num

        for h, r, t in self._rel_triples_2:
            self._get_ent_id(h)
            self._get_ent_id(t)
            self._get_rel_id(r)
        self.ent_num_2 = self.ent_num - self.ent_num_1

    @property
    def kg1_triples(self) -> list[tuple[int, int, int]]:
        if getattr(self, "_rel_triples_1", None) is None:  # loaded from the parsed cache
            return list(map(tuple, self._kg1_triples_np.tolist()))
        return [
            (self.ent2id[h], self.rel2id[r], self.ent2id[t])
            for h, r, t in self._rel_triples_1
        ]

    @property
    def kg2_triples(self) -> list[tuple[int, int, int]]:
        if getattr(self, "_rel_triples_2", None) is None:
            return list(map(tuple, self._kg2_triples_np.tolist()))
        return [
            (self.ent2id[h], self.rel2id[r], self.ent2id[t])
            for h, r, t in self._rel_triples_2
        ]

    # ── parsed-triples cache (DISTEA_DATA_CACHE=1) ───────────────────────────
    def _parsed_cache_path(self) -> str:
        real = os.path.realpath(self.data_dir)
        tag = hashlib.sha1(real.encode("utf-8")).hexdigest()[:10]
        folder = os.environ.get("DISTEA_DATA_CACHE_DIR") or self.data_dir
        return os.path.join(folder, f".distea_parsed_v{_PARSED_CACHE_VERSION}_{tag}.npz")

    @staticmethod
    def _triple_files_key(paths: list[str]) -> np.ndarray:
        st = [os.stat(p) for p in paths]
        return np.array([v for x in st for v in (x.st_size, x.st_mtime_ns)], dtype=np.int64)

    def _load_parsed_cache(self, paths: list[str]) -> bool:
        """Restore ent2id / rel2id / counts and the (T, 3) id arrays from the
        cache; False (state untouched) on a missing, stale or unreadable file."""
        path = self._parsed_cache_path()
        if not os.path.exists(path):
            return False
        try:
            key = self._triple_files_key(paths)
            with np.load(path, allow_pickle=False) as z:
                if int(z["version"]) != _PARSED_CACHE_VERSION or not np.array_equal(z["key"], key):
                    return False
                n_ent, n1, n2, n_rel = (int(x) for x in z["counts"])
                ent_names = _blob_names(z["ent_names"], n_ent)
                rel_names = _blob_names(z["rel_names"], n_rel)
                tri1 = z["tri1"].astype(np.int64)
                tri2 = z["tri2"].astype(np.int64)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            return False
        if len(ent_names) != n_ent or len(rel_names) != n_rel or n1 + n2 != n_ent:
            return False
        ent2id = dict(zip(ent_names, range(n_ent), strict=True))
        rel2id = dict(zip(rel_names, range(n_rel), strict=True))
        if len(ent2id) != n_ent or len(rel2id) != n_rel:
            return False
        self.ent2id, self.rel2id = ent2id, rel2id
        self.ent_num, self.ent_num_1, self.ent_num_2, self.rel_num = n_ent, n1, n2, n_rel
        self._rel_triples_1 = self._rel_triples_2 = None
        # The attribute names dist_graph._triples_array caches under.
        self._kg1_triples_np = tri1.reshape(-1, 3)
        self._kg2_triples_np = tri2.reshape(-1, 3)
        return True

    def _write_parsed_cache(self, paths: list[str]) -> None:
        tri1 = np.asarray(self.kg1_triples, dtype=np.int64).reshape(-1, 3)
        tri2 = np.asarray(self.kg2_triples, dtype=np.int64).reshape(-1, 3)
        # Pre-fill dist_graph._triples_array's cache: the same arrays it
        # would build from the lists.
        self._kg1_triples_np, self._kg2_triples_np = tri1, tri2
        ent_names: list = [None] * self.ent_num
        for name, i in self.ent2id.items():
            ent_names[i] = name
        rel_names: list = [None] * self.rel_num
        for name, i in self.rel2id.items():
            rel_names[i] = name
        if max(self.ent_num, self.rel_num) >= 2**31:
            return
        ent_blob, rel_blob = _names_blob(ent_names), _names_blob(rel_names)
        if (_blob_names(ent_blob, self.ent_num) != ent_names
                or _blob_names(rel_blob, self.rel_num) != rel_names):
            return  # a name that does not round-trip: keep parsing every time
        path = self._parsed_cache_path()
        tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"
        try:
            with open(tmp, "wb") as f:
                np.savez(
                    f, version=np.int64(_PARSED_CACHE_VERSION),
                    key=self._triple_files_key(paths),
                    counts=np.array([self.ent_num, self.ent_num_1, self.ent_num_2, self.rel_num],
                                    dtype=np.int64),
                    ent_names=ent_blob, rel_names=rel_blob,
                    tri1=tri1.astype(np.int32), tri2=tri2.astype(np.int32),
                )
            os.replace(tmp, path)  # atomic: concurrent ranks write the same content
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def load_train_alignment(self) -> list[tuple[int, int]]:
        return [(self.ent2id[e1], self.ent2id[e2]) for e1, e2 in self._train_links]

    def load_valid_alignment(self) -> list[tuple[int, int]]:
        return [(self.ent2id[e1], self.ent2id[e2]) for e1, e2 in self._valid_links]

    def load_test_alignment(self) -> list[tuple[int, int]]:
        return [(self.ent2id[e1], self.ent2id[e2]) for e1, e2 in self._test_links]

    def load_weights(self) -> None:
        edge_path = os.path.join(self.output_dir, "KGs_edge_info.json")
        node_path = os.path.join(self.output_dir, "KGs_node_info.json")
        if not os.path.exists(edge_path):
            return  # graph_metrics not yet run; use uniform weights
        with open(edge_path) as f:
            obj = json.loads(f.read())

        for r in range(self.rel_num):
            s = str(r)
            kg1 = obj["KG1"]
            self.edge_weights1[r] = (kg1["r2f"].get(s, 0.0), kg1["r2if"].get(s, 0.0))
            kg2 = obj["KG2"]
            self.edge_weights2[r] = (kg2["r2f"].get(s, 0.0), kg2["r2if"].get(s, 0.0))

        if not os.path.exists(node_path):
            return
        with open(node_path) as f:
            obj = json.loads(f.read())

        kg1n, kg2n = obj["KG1"], obj["KG2"]
        for k in self.kg1_entities:
            s = str(k)
            self.node_weights1[k] = kg1n["in_degree"][s] + kg1n["out_degree"][s] + 1
        for k in self.kg2_entities:
            s = str(k)
            self.node_weights2[k] = kg2n["in_degree"].get(s, 0) + kg2n["out_degree"].get(s, 0) + 1
