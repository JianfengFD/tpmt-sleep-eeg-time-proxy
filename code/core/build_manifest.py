#!/usr/bin/env python3
"""Write a SHA-256 manifest for every regular file except the manifest itself."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role(relative: Path) -> str:
    first = relative.parts[0]
    return {
        "code": "source_code",
        "data": "derived_data",
        "evidence": "result_evidence",
        "figures": "figure",
    }.get(first, "documentation")


def provenance(relative: Path) -> str:
    text = relative.as_posix()
    if text.startswith("code/core/analysis/"):
        return "copied unchanged from project analysis/ tree"
    if text.startswith("data/core/physical_f/"):
        return "project outputs/dense_repeat_fawake_v2-v3"
    if text.startswith("data/core/gssc/") or text.startswith("data/core/staging_inputs/"):
        return "project GSSC outputs or compact filters of upstream tables"
    if text.startswith("data/core/global_g/") or text.startswith("data/core/correlations/"):
        return "project outputs/dense_repeat_fawake_v3/stage_weighted_g_literature_v1"
    if text.startswith("data/core/historical_baseline/") or text.startswith("evidence/historical_baseline/"):
        return "superseded project REM-only baseline; retained for audit only"
    if text.startswith("data/core/dream_duration/"):
        return "project six-LLM dream-duration consensus table"
    if text.startswith("figures/test_samples/") or text.startswith("data/test_samples/"):
        return "paper-candidate test-sample figure workflow"
    if text.startswith("evidence/metrics/"):
        return "primary stage-weighted result evidence or package verification"
    return "paper-candidate authored or packaging file"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    output = (args.output or root / "FILE_MANIFEST.csv").resolve()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.resolve() != output
        and path.name != ".DS_Store"
        and not any(part in {".git", ".venv", "__pycache__", "reproduced", "downloads"}
                    for part in path.relative_to(root).parts)
        and not path.relative_to(root).as_posix().startswith("code/core/.vendor/gssc/")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "bytes", "sha256", "role", "provenance"],
        )
        writer.writeheader()
        for path in files:
            relative = path.relative_to(root)
            writer.writerow(
                {
                    "relative_path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "role": role(relative),
                    "provenance": provenance(relative),
                }
            )
    print(f"Wrote {len(files)} entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
