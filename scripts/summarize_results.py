from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--out", default="results/summary.csv")
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(Path(args.outputs).glob("*/metrics.json")):
        if not (metrics_path.parent / "config.yaml").exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["run_name"] = metrics_path.parent.name
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cols = ["run_name"] + [c for c in df.columns if c != "run_name"]
    df = df[cols] if len(df) else pd.DataFrame(columns=["run_name"])
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {out_path} rows={len(df)}")
    if len(df):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
