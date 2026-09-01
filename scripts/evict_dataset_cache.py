#!/usr/bin/env python3
"""Evict clean TFRecord pages without touching dataset contents."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def cgroup_file_bytes() -> int:
    try:
        for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines():
            key, value = line.split()[:2]
            if key == "file":
                return int(value)
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0


def evict(root: Path, threshold_bytes: int) -> int:
    if cgroup_file_bytes() < threshold_bytes:
        return 0
    count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            count += 1
        except OSError:
            continue
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--threshold-gib", type=float, default=12.0)
    args = parser.parse_args()
    threshold = int(args.threshold_gib * 1024**3)
    before = cgroup_file_bytes()
    count = evict(args.root, threshold)
    after = cgroup_file_bytes()
    if count:
        print(
            f"[cache-evict] files={count} "
            f"file_cache_gib={before / 1024**3:.1f}->{after / 1024**3:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
