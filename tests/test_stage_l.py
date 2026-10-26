"""
test_stage_l.py — Stage L tests: DBP1M/FB-DBP2M -> OpenEAData layout
reshaping scripts (scripts/prepare_dbp1m.py, scripts/prepare_fb_dbp2m.py).

Regression focus: FB-DBP2M's dbp_triple_rel.txt/fb_triple_rel.txt use
independently 0-indexed entity/relation IDs (KG1 entity "0" and KG2 entity
"0" are different real-world entities) -- prepare_fb_dbp2m.py must prefix
them into globally-unique strings, or OpenEAData would silently merge
unrelated KG1/KG2 entities that happen to share a numeric ID.

DGL-free, no GPU required, <5 sec.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import prepare_dbp1m
import prepare_fb_dbp2m


def test_prepare_dbp1m_symlinks_point_to_correct_source_files():
    with tempfile.TemporaryDirectory() as tmp:
        mkdata_dir = os.path.join(tmp, "mkdata")
        os.makedirs(mkdata_dir)
        with open(os.path.join(mkdata_dir, "de_triples_1"), "w") as f:
            f.write("http://a\thttp://r\thttp://b\n")
        with open(os.path.join(mkdata_dir, "de_triples_2"), "w") as f:
            f.write("http://c\thttp://r2\thttp://d\n")
        with open(os.path.join(mkdata_dir, "de_ent_links"), "w") as f:
            f.write("http://a\thttp://c\n")

        out_dir = os.path.join(tmp, "DBP1M_DE")
        old_argv = sys.argv
        try:
            sys.argv = [
                "prepare_dbp1m.py",
                "--mkdata_dir", mkdata_dir,
                "--lang", "de",
                "--out_dir", out_dir,
            ]
            prepare_dbp1m.main()
        finally:
            sys.argv = old_argv

        assert os.path.islink(os.path.join(out_dir, "rel_triples_1"))
        assert os.path.islink(os.path.join(out_dir, "rel_triples_2"))
        with open(os.path.join(out_dir, "rel_triples_1")) as f:
            assert f.read() == "http://a\thttp://r\thttp://b\n"
        with open(os.path.join(out_dir, "ent_links")) as f:
            assert f.read() == "http://a\thttp://c\n"
        print("  test_prepare_dbp1m_symlinks_point_to_correct_source_files PASSED")


def test_prepare_dbp1m_filters_ent_links_dangling_references():
    # h in de_triples_1, k in de_triples_2 -> known entities are {h, i, j, k}.
    # ent_links has one valid pair (h, k) and one dangling pair (missing, k)
    # whose left side never appears in any triples file -- must be dropped,
    # or OpenEAData's ent2id[...] lookup would KeyError on it later.
    with tempfile.TemporaryDirectory() as tmp:
        mkdata_dir = os.path.join(tmp, "mkdata")
        os.makedirs(mkdata_dir)
        with open(os.path.join(mkdata_dir, "de_triples_1"), "w") as f:
            f.write("h\tr\ti\n")
        with open(os.path.join(mkdata_dir, "de_triples_2"), "w") as f:
            f.write("j\tr2\tk\n")
        with open(os.path.join(mkdata_dir, "de_ent_links"), "w") as f:
            f.write("h\tk\nmissing\tk\n")

        out_dir = os.path.join(tmp, "DBP1M_DE")
        old_argv = sys.argv
        try:
            sys.argv = [
                "prepare_dbp1m.py",
                "--mkdata_dir", mkdata_dir,
                "--lang", "de",
                "--out_dir", out_dir,
            ]
            prepare_dbp1m.main()
        finally:
            sys.argv = old_argv

        with open(os.path.join(out_dir, "ent_links")) as f:
            lines = f.read().splitlines()
        assert lines == ["h\tk"], lines
        print("  test_prepare_dbp1m_filters_ent_links_dangling_references PASSED")


def test_prepare_dbp1m_rerun_does_not_truncate_source_via_stale_symlink():
    # Re-running the script must not destroy the source de_ent_links file:
    # if out_dir/ent_links is still a symlink to the source from an earlier
    # run, opening it in "w" mode would follow the link and truncate the
    # source. Pre-create out_dir/ent_links as such a symlink, call main(),
    # and assert the source file is untouched afterward.
    with tempfile.TemporaryDirectory() as tmp:
        mkdata_dir = os.path.join(tmp, "mkdata")
        os.makedirs(mkdata_dir)
        with open(os.path.join(mkdata_dir, "de_triples_1"), "w") as f:
            f.write("h\tr\ti\n")
        with open(os.path.join(mkdata_dir, "de_triples_2"), "w") as f:
            f.write("j\tr2\tk\n")
        ent_links_src = os.path.join(mkdata_dir, "de_ent_links")
        with open(ent_links_src, "w") as f:
            f.write("h\tk\n")

        out_dir = os.path.join(tmp, "DBP1M_DE")
        os.makedirs(out_dir)
        os.symlink(os.path.abspath(ent_links_src), os.path.join(out_dir, "ent_links"))

        old_argv = sys.argv
        try:
            sys.argv = [
                "prepare_dbp1m.py",
                "--mkdata_dir", mkdata_dir,
                "--lang", "de",
                "--out_dir", out_dir,
            ]
            prepare_dbp1m.main()
        finally:
            sys.argv = old_argv

        with open(ent_links_src) as f:
            assert f.read() == "h\tk\n", "source de_ent_links was truncated!"
        print("  test_prepare_dbp1m_rerun_does_not_truncate_source_via_stale_symlink PASSED")


def test_prepare_fb_dbp2m_disambiguates_colliding_numeric_ids():
    with tempfile.TemporaryDirectory() as tmp:
        src_dir = os.path.join(tmp, "fb_dbp")
        os.makedirs(src_dir)
        # KG1 (dbp) triple uses entity/relation id "0"; KG2 (fb) triple ALSO
        # uses entity/relation id "0" -- these must NOT collide after prefixing.
        with open(os.path.join(src_dir, "dbp_triple_rel.txt"), "w") as f:
            f.write("0\t0\t1\n")
        with open(os.path.join(src_dir, "fb_triple_rel.txt"), "w") as f:
            f.write("0\t0\t1\n")
        # Alignment: KG1 entity 0 <-> KG2 entity 1 (deliberately NOT the same
        # numeral on both sides, to make the test unambiguous). The release
        # file lists the Freebase (KG2) id first: "<fb id>\t<dbp id>".
        with open(os.path.join(src_dir, "alignment_of_entity.txt"), "w") as f:
            f.write("1\t0\n")

        out_dir = os.path.join(tmp, "FB_DBP2M")
        old_argv = sys.argv
        try:
            sys.argv = [
                "prepare_fb_dbp2m.py",
                "--src_dir", src_dir,
                "--out_dir", out_dir,
            ]
            prepare_fb_dbp2m.main()
        finally:
            sys.argv = old_argv

        with open(os.path.join(out_dir, "rel_triples_1")) as f:
            kg1_line = f.read().strip()
        with open(os.path.join(out_dir, "rel_triples_2")) as f:
            kg2_line = f.read().strip()
        with open(os.path.join(out_dir, "ent_links")) as f:
            links_line = f.read().strip()

        h1, r1, t1 = kg1_line.split("\t")
        h2, r2, t2 = kg2_line.split("\t")

        # Same raw numeral "0"/"1" on both sides -- prefixed forms must differ.
        assert h1 != h2, "KG1 and KG2 entity '0' must not collide after prefixing"
        assert r1 != r2, "KG1 and KG2 relation '0' must not collide after prefixing"
        assert t1 != t2, "KG1 and KG2 entity '1' must not collide after prefixing"

        left, right = links_line.split("\t")
        assert left == h1, "alignment's KG1 side must use the same prefixed id as rel_triples_1"
        assert right == t2, "alignment's KG2 side must use the same prefixed id as rel_triples_2"
        print("  test_prepare_fb_dbp2m_disambiguates_colliding_numeric_ids PASSED")


def test_prepare_fb_dbp2m_filters_alignment_dangling_references():
    # KG2 entity "9" is referenced by the alignment file but never appears in
    # fb_triple_rel.txt -- must be dropped, or OpenEAData's ent2id[...]
    # lookup would KeyError on it later (the real FB-DBP2M release contains
    # such dangling alignment entries).
    with tempfile.TemporaryDirectory() as tmp:
        src_dir = os.path.join(tmp, "fb_dbp")
        os.makedirs(src_dir)
        with open(os.path.join(src_dir, "dbp_triple_rel.txt"), "w") as f:
            f.write("0\t0\t1\n")
        with open(os.path.join(src_dir, "fb_triple_rel.txt"), "w") as f:
            f.write("2\t0\t3\n")
        with open(os.path.join(src_dir, "alignment_of_entity.txt"), "w") as f:
            f.write("3\t0\n9\t0\n")  # "<fb id>\t<dbp id>"; the second pair's KG2 side ("9") is dangling

        out_dir = os.path.join(tmp, "FB_DBP2M")
        old_argv = sys.argv
        try:
            sys.argv = [
                "prepare_fb_dbp2m.py",
                "--src_dir", src_dir,
                "--out_dir", out_dir,
            ]
            prepare_fb_dbp2m.main()
        finally:
            sys.argv = old_argv

        with open(os.path.join(out_dir, "ent_links")) as f:
            lines = f.read().splitlines()
        assert lines == ["dbp_e0\tfb_e3"], lines
        print("  test_prepare_fb_dbp2m_filters_alignment_dangling_references PASSED")


if __name__ == "__main__":
    print("Stage L tests:")
    test_prepare_dbp1m_symlinks_point_to_correct_source_files()
    test_prepare_dbp1m_filters_ent_links_dangling_references()
    test_prepare_dbp1m_rerun_does_not_truncate_source_via_stale_symlink()
    test_prepare_fb_dbp2m_disambiguates_colliding_numeric_ids()
    test_prepare_fb_dbp2m_filters_alignment_dangling_references()
    print("All Stage L tests PASSED.")
