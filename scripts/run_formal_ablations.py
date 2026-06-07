from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from course_rec.data import build_dataset
from course_rec.io_utils import dump_json, ensure_dir, read_config
from course_rec.metrics import evaluate_full_ranking
from course_rec.run_experiment import build_scorer


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="课程作业数据集")
    parser.add_argument("--base-config", default="configs/adaptive_fusion_validated.yaml")
    parser.add_argument("--fixed-config", default="configs/fusion_validated.yaml")
    parser.add_argument("--kg-config", default="configs/kg_profile.yaml")
    parser.add_argument("--out", default="reports/tables/formal_ablation_table.csv")
    parser.add_argument("--config-dir", default="configs/ablations")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data = build_dataset(data_dir)
    base = read_config(args.base_config)
    fixed = read_config(args.fixed_config)
    kg_only = read_config(args.kg_config)
    validated_dir = Path(args.config_dir)

    variants = [
        ("RAKF-M full", set_run_name(base, "adaptive_fusion_validated")),
        (
            "w/o KG",
            read_or_fallback(
                validated_dir / "ablation_wo_kg_validated.yaml",
                remove_adaptive_component(base, "ablation_wo_kg", "kg_profile"),
            ),
        ),
        (
            "w/o RP3",
            read_or_fallback(
                validated_dir / "ablation_wo_rp3_validated.yaml",
                remove_adaptive_component(base, "ablation_wo_rp3", "rp3beta"),
            ),
        ),
        (
            "w/o EASE",
            read_or_fallback(
                validated_dir / "ablation_wo_ease_validated.yaml",
                remove_adaptive_component(base, "ablation_wo_ease", "ease"),
            ),
        ),
        ("w/o adaptive grouping", set_run_name(fixed, "fusion_validated")),
        ("KG profile only", set_run_name(kg_only, "kg_profile")),
    ]

    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, config in variants:
        run_name = str(config["run_name"])
        config_path = config_dir / f"{run_name}.yaml"
        write_yaml(config_path, config)
        out_dir = ensure_dir(Path(args.output_root) / run_name)
        write_yaml(out_dir / "config.yaml", config)
        print(f"running {label}: {run_name}", flush=True)
        scorer = build_scorer(config, data, data_dir)
        metrics = evaluate_full_ranking(
            data=data,
            score_fn=scorer.score,
            ks=tuple(int(k) for k in config.get("ks", [10, 20])),
            batch_size=int(config.get("batch_size", 2048)),
            sample_topk_users=int(config.get("sample_topk_users", 20)),
            sample_path=out_dir / "topk_sample.csv",
        )
        dump_json(metrics, out_dir / "metrics.json")
        with open(out_dir / "run.log", "w", encoding="utf-8") as f:
            f.write(f"run_name={run_name}\n")
            f.write(f"variant={label}\n")
            f.write("metrics=\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
        row = {"variant": label, "run_name": run_name, **metrics}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    df = pd.DataFrame(rows)
    full_ndcg = float(df.loc[df["variant"] == "RAKF-M full", "NDCG@20"].iloc[0])
    full_hr = float(df.loc[df["variant"] == "RAKF-M full", "HR@20"].iloc[0])
    df["Delta_NDCG@20"] = df["NDCG@20"] - full_ndcg
    df["Delta_HR@20"] = df["HR@20"] - full_hr
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {out_path}", flush=True)


def set_run_name(config: dict, run_name: str) -> dict:
    out = copy.deepcopy(config)
    out["run_name"] = run_name
    return out


def read_or_fallback(path: Path, fallback: dict) -> dict:
    if path.exists():
        return read_config(path)
    return fallback


def remove_adaptive_component(config: dict, run_name: str, component_name: str) -> dict:
    out = set_run_name(config, run_name)
    fusion = out["adaptive_fusion"]
    aliases = aliases_for(component_name)
    fusion["components"] = [c for c in fusion["components"] if c["name"] not in aliases]
    for segment in fusion["segments"]:
        weights = dict(segment["weights"])
        for alias in aliases:
            weights.pop(alias, None)
        segment["weights"] = weights
    return out


def aliases_for(component_name: str) -> set[str]:
    mapping = {
        "rp3beta": {"rp3beta", "rp3"},
        "popularity": {"popularity", "pop"},
        "kg_profile": {"kg_profile", "kg"},
    }
    return mapping.get(component_name, {component_name})


def write_yaml(path: str | Path, config: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    main()
