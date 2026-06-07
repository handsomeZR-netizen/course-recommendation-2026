from __future__ import annotations

import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


def read_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def copy_config(src: str | Path, dst_dir: str | Path) -> None:
    shutil.copy2(src, Path(dst_dir) / "config.yaml")


@contextmanager
def tee_stdout(log_path: str | Path) -> Iterator[None]:
    log_file = open(log_path, "w", encoding="utf-8")
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data: str) -> int:
            for stream in self.streams:
                stream.write(data)
                stream.flush()
            return len(data)

        def flush(self) -> None:
            for stream in self.streams:
                stream.flush()

    try:
        tee = Tee(old_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()
