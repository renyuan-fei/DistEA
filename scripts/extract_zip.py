"""extract_zip.py -- extract a zip archive, skipping macOS metadata.

Usage:
    python scripts/extract_zip.py <zip_path> <dest_dir> [--nested]

--nested: the archive holds a single inner .zip with the real data; the inner archive is
extracted from memory. Leading '/' is stripped from entry names and __MACOSX entries are
skipped, so archives created on macOS extract cleanly.
"""
import io
import os
import sys
import zipfile


def _extract(zf: zipfile.ZipFile, dest: str) -> None:
    for member in zf.infolist():
        if "__MACOSX" in member.filename:
            continue
        name = member.filename.lstrip("/")
        if not name:
            continue
        member.filename = name
        zf.extract(member, dest)


def main() -> None:
    args = sys.argv[1:]
    nested = "--nested" in args
    args = [a for a in args if not a.startswith("--")]
    if len(args) != 2:
        sys.exit("usage: extract_zip.py <zip> <dest> [--nested]")
    src, dest = args
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(src) as outer:
        if not nested:
            _extract(outer, dest)
            return
        inner = [n for n in outer.namelist() if n.endswith(".zip") and "__MACOSX" not in n]
        if not inner:
            sys.exit(f"--nested: no inner .zip found in {src}")
        data = outer.read(inner[0])
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        _extract(zf, dest)


if __name__ == "__main__":
    main()
