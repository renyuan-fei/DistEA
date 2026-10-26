"""
test_data_cache.py — DISTEA_DATA_CACHE=1: OpenEAData restored from
.distea_parsed_v1_<tag>.npz must equal a plain parse EXACTLY -- ent2id
and rel2id (values and insertion order), the four counts, kg1/kg2_triples
(lists of int tuples), the (T, 3) arrays DistGraphLoader's vectorised path
reads, and the train / valid / test links -- on a synthetic OpenEA directory
with KG2 lines naming KG1 entities, relations shared by both KGs, duplicate
triples, non-ASCII names, blank lines and CRLF line ends. Also: the first
cached construction writes the file, a changed triple file (size / mtime)
is a miss that re-parses the new content, and a corrupt cache file falls
back to the parse.

DGL-free, CPU, < 10 s.
Run:  .venv/bin/python tests/test_data_cache.py
"""
from __future__ import annotations

import glob
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dataload  # noqa: E402
from dataload import OpenEAData  # noqa: E402
from dist_graph import _triples_array  # noqa: E402


def _write(path, rows, crlf=False):
    end = "\r\n" if crlf else "\n"
    with open(path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(("\t".join(r) if isinstance(r, tuple) else r) + end)


def _make_dir(tmp, seed=0):
    rng = np.random.default_rng(seed)
    e1 = [f"http://kg1/E{i}_é" for i in range(400)]
    e2 = [f"http://kg2/F{i}_München" for i in range(300)] + e1[:5]   # KG2 names a few KG1 entities
    rels = [f"http://r/{i}" for i in range(12)]
    t1 = [(e1[a], rels[r], e1[b]) for a, r, b in zip(rng.integers(0, 400, 1500), rng.integers(0, 8, 1500),
                                                      rng.integers(0, 400, 1500), strict=True)]
    t1 += t1[:50]                                                   # duplicate triples
    t2 = [(e2[a], rels[r], e2[b]) for a, r, b in zip(rng.integers(0, 305, 900), rng.integers(4, 12, 900),
                                                     rng.integers(0, 305, 900), strict=True)]
    _write(os.path.join(tmp, "rel_triples_1"), t1[:700] + [""] + t1[700:])   # a blank line
    _write(os.path.join(tmp, "rel_triples_2"), t2, crlf=True)
    fold = os.path.join(tmp, "fold")
    os.makedirs(fold, exist_ok=True)
    seen1 = sorted({h for h, _, _ in t1} | {t for _, _, t in t1})
    seen2 = sorted(({h for h, _, _ in t2} | {t for _, _, t in t2}) - set(seen1))
    links = list(zip(seen1[:120], seen2[:120], strict=True))
    _write(os.path.join(fold, "train_links"), links[:60])
    _write(os.path.join(fold, "valid_links"), links[60:80])
    _write(os.path.join(fold, "test_links"), links[80:])
    return fold


def _cache_files(folder):
    return glob.glob(os.path.join(folder, ".distea_parsed_v1_*.npz"))


def _state(d):
    return {
        "ent2id": list(d.ent2id.items()), "rel2id": list(d.rel2id.items()),
        "counts": (d.ent_num, d.ent_num_1, d.ent_num_2, d.rel_num),
        "kg1": d.kg1_triples, "kg2": d.kg2_triples,
        "arr1": _triples_array(d, "kg1_triples").tolist(), "arr2": _triples_array(d, "kg2_triples").tolist(),
        "links": (d.load_train_alignment(), d.load_valid_alignment(), d.load_test_alignment()),
    }


def _build(tmp, fold, cache):
    os.environ["DISTEA_DATA_CACHE"] = "1" if cache else "0"
    try:
        return OpenEAData(tmp, fold, os.path.join(tmp, "out"))
    finally:
        os.environ.pop("DISTEA_DATA_CACHE", None)


def test_cache_roundtrip_exact():
    with tempfile.TemporaryDirectory() as tmp:
        fold = _make_dir(tmp)
        ref = _state(_build(tmp, fold, cache=False))
        assert not _cache_files(tmp), "cache off must not write"
        miss = _build(tmp, fold, cache=True)
        assert len(_cache_files(tmp)) == 1, "a miss must write the cache"
        assert miss._rel_triples_1 is not None, "a miss parses the text"
        hit = _build(tmp, fold, cache=True)
        assert hit._rel_triples_1 is None, "the second construction must be a cache hit"
        for name, d in (("miss", miss), ("hit", hit)):
            st = _state(d)
            for k in ref:
                assert st[k] == ref[k], f"{name}: {k} differs from the plain parse"
        assert hit._kg1_triples_np.dtype == np.int64 and hit._kg1_triples_np.shape[1] == 3
        assert ref["counts"][2] > 0 and len(ref["kg1"]) == 1550
    print("  test_cache_roundtrip_exact PASSED (ent2id/rel2id order, counts, triples, arrays, links)")


def test_stale_and_corrupt_cache():
    with tempfile.TemporaryDirectory() as tmp:
        fold = _make_dir(tmp, seed=1)
        _build(tmp, fold, cache=True)                      # writes the cache
        # Change KG2's content: a different size and mtime -> miss, re-parse.
        with open(os.path.join(tmp, "rel_triples_2"), "a", encoding="utf-8") as f:
            f.write("http://kg2/NEW\thttp://r/99\thttp://kg2/F0_München\n")
        os.utime(os.path.join(tmp, "rel_triples_2"), ns=(1, 1))
        ref = _state(_build(tmp, fold, cache=False))
        stale = _build(tmp, fold, cache=True)
        assert stale._rel_triples_2 is not None, "a changed triple file must be a miss"
        assert _state(stale) == ref
        assert ("http://kg2/NEW" in stale.ent2id) and ("http://r/99" in stale.rel2id)
        hit = _build(tmp, fold, cache=True)
        assert hit._rel_triples_2 is None and _state(hit) == ref, "the rewritten cache must hit"
        with open(_cache_files(tmp)[0], "wb") as f:
            f.write(b"not a zip file")
        bad = _build(tmp, fold, cache=True)
        assert bad._rel_triples_1 is not None and _state(bad) == ref, "a corrupt cache must fall back"
    print("  test_stale_and_corrupt_cache PASSED")


def test_cache_dir_env():
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as cdir:
        fold = _make_dir(tmp, seed=2)
        ref = _state(_build(tmp, fold, cache=False))
        os.environ["DISTEA_DATA_CACHE_DIR"] = cdir
        try:
            _build(tmp, fold, cache=True)
            assert len(_cache_files(cdir)) == 1 and not _cache_files(tmp), "the cache goes to DISTEA_DATA_CACHE_DIR"
            hit = _build(tmp, fold, cache=True)
        finally:
            os.environ.pop("DISTEA_DATA_CACHE_DIR", None)
        assert hit._rel_triples_1 is None and _state(hit) == ref
    print("  test_cache_dir_env PASSED")


def test_names_blob_roundtrip():
    for names in ([], [""], ["a", "", "b"], ["ü", "x y", "z\u00a0"]):
        assert dataload._blob_names(dataload._names_blob(names), len(names)) == names, names
    print("  test_names_blob_roundtrip PASSED")


if __name__ == "__main__":
    for fn in (test_cache_roundtrip_exact, test_stale_and_corrupt_cache, test_cache_dir_env,
               test_names_blob_roundtrip):
        print(f"[{fn.__name__}]")
        fn()
    print("test_data_cache: ALL PASSED")
