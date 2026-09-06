#!/usr/bin/env python3
"""Check official GSSC checkpoint hashes without deserializing the files."""
import argparse
import hashlib
from pathlib import Path

EXPECTED = {
    "sig_net_v1.pt": "aa4bdcd7e29653138b096b3447d8db2d48508dc729e739b7ba9806dfd3c7c433",
    "gru_net_v1.pt": "124cfc858f49e3599b0798e494d7378d0d57a83d249faf79702da4cd8ee83304",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gssc-root", type=Path,
                        default=Path(__file__).resolve().parent / "core/.vendor/gssc")
    args = parser.parse_args()
    for name, expected in EXPECTED.items():
        path = args.gssc_root / "gssc/nets" / name
        if not path.is_file():
            parser.exit(1, f"Missing: {path}\nSee DATA_AND_MODELS.md.\n")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            parser.exit(1, f"Hash mismatch: {path}; do not load this checkpoint.\n")
        print(f"OK: {name}")


if __name__ == "__main__":
    main()
