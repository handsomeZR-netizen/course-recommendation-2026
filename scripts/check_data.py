from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from course_rec.data import build_dataset, dataset_stats
def count_tsv_edges(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="课程作业数据集")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    data = build_dataset(data_dir)
    print(json.dumps(dataset_stats(data), ensure_ascii=False, indent=2))

    rel_dir = data_dir / "MOOCCube (1)" / "MOOCCube" / "relations"
    if rel_dir.exists():
        print("\nKG relation edge counts:")
        for path in sorted(rel_dir.glob("*.json")):
            print(f"{path.name}: {count_tsv_edges(path)}")


if __name__ == "__main__":
    main()
