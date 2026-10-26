"""
test_stage_o.py — Stage O tests: scripts/export_to_divea.py.

Checks that {kgid}_triple_rel.txt keeps OpenEA's (head, relation, tail)
column order. DivEA's load_kg() names its loop variables as if the tail came
before the relation, but that naming is cosmetic: every consumer in divea/*.py
unpacks triples as (h, r, t), so the exporter must copy columns straight
through rather than reorder them.

DGL-free, no GPU required, <30 sec.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from export_to_divea import (
    convert_kg,
    write_triple_rel,
)
from export_to_divea import (
    main as export_to_divea_main,
)


def test_write_triple_rel_preserves_h_r_t_order():
    # (head=1, relation=99, tail=2) must round-trip unchanged -- NOT
    # reordered to (head, tail, relation).
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "en_triple_rel.txt")
        write_triple_rel(path, [(1, 99, 2), (3, 5, 4)])
        with open(path) as f:
            lines = f.read().splitlines()
        assert lines == ["1\t99\t2", "3\t5\t4"], lines


def test_convert_kg_end_to_end_preserves_column_order():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        dst = os.path.join(tmp, "dst")
        os.makedirs(src)
        os.makedirs(dst)
        with open(os.path.join(src, "ent_ids_1"), "w") as f:
            f.write("0 0\n1 1\n2 2\n")
        with open(os.path.join(src, "triples_1"), "w") as f:
            # OpenEA order: head rel tail
            f.write("0 10 1\n1 11 2\n")

        convert_kg(src, dst, "en", "ent_ids_1", "triples_1")

        with open(os.path.join(dst, "en_triple_rel.txt")) as f:
            lines = f.read().splitlines()
        # Must match the source file's column order exactly: h, r, t.
        assert lines == ["0\t10\t1", "1\t11\t2"], lines

        with open(os.path.join(dst, "en_relation_id2uri.txt")) as f:
            rel_lines = sorted(f.read().splitlines())
        # Relation ids are the middle column (10, 11), not the tail column.
        assert rel_lines == ["10\t10", "11\t11"], rel_lines


def test_main_cli_writes_expected_files_with_correct_order():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        dst = os.path.join(tmp, "dst")
        os.makedirs(src)
        with open(os.path.join(src, "ent_ids_1"), "w") as f:
            f.write("0 0\n1 1\n")
        with open(os.path.join(src, "ent_ids_2"), "w") as f:
            f.write("2 2\n3 3\n")
        with open(os.path.join(src, "triples_1"), "w") as f:
            f.write("0 7 1\n")
        with open(os.path.join(src, "triples_2"), "w") as f:
            f.write("2 8 3\n")
        with open(os.path.join(src, "ref_ent_ids_train"), "w") as f:
            f.write("0 2\n")
        with open(os.path.join(src, "ref_ent_ids_test"), "w") as f:
            f.write("1 3\n")

        old_argv = sys.argv
        sys.argv = ["export_to_divea.py", "--src_dir", src, "--dst_dir", dst,
                    "--kgid1", "en", "--kgid2", "de"]
        try:
            export_to_divea_main()
        finally:
            sys.argv = old_argv

        with open(os.path.join(dst, "en_triple_rel.txt")) as f:
            assert f.read().splitlines() == ["0\t7\t1"]
        with open(os.path.join(dst, "de_triple_rel.txt")) as f:
            assert f.read().splitlines() == ["2\t8\t3"]
        with open(os.path.join(dst, "train_alignment.txt")) as f:
            assert f.read().splitlines() == ["0\t2"]
        with open(os.path.join(dst, "alignment_of_entity.txt")) as f:
            assert sorted(f.read().splitlines()) == ["0\t2", "1\t3"]
    print("  test_main_cli_writes_expected_files_with_correct_order PASSED")


if __name__ == "__main__":
    print("Stage O tests:")
    test_write_triple_rel_preserves_h_r_t_order()
    print("  test_write_triple_rel_preserves_h_r_t_order PASSED")
    test_convert_kg_end_to_end_preserves_column_order()
    print("  test_convert_kg_end_to_end_preserves_column_order PASSED")
    test_main_cli_writes_expected_files_with_correct_order()
    print("All Stage O tests PASSED.")
