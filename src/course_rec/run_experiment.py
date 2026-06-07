from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np

from .baselines import (
    AdaptiveFusionScorer,
    EASEScorer,
    FusionScorer,
    ItemKNNScorer,
    PopularityScorer,
    RP3BetaScorer,
)
from .data import build_dataset, dataset_stats
from .io_utils import copy_config, dump_json, ensure_dir, read_config, tee_stdout
from .kg import KGProfileScorer, build_course_feature_matrix
from .metrics import evaluate_full_ranking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default="课程作业数据集")
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()

    config = read_config(args.config)
    run_name = config.get("run_name") or f"{config['model']}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = ensure_dir(Path(args.output_root) / run_name)
    copy_config(args.config, out_dir)

    with tee_stdout(out_dir / "run.log"):
        print(f"run_name={run_name}")
        print(f"python={platform.python_version()} platform={platform.platform()}")
        data = build_dataset(args.data_dir)
        stats = dataset_stats(data)
        print("data_stats=", stats)
        dump_json(stats, out_dir / "data_stats.json")

        scorer = build_scorer(config, data, Path(args.data_dir))
        ks = tuple(int(k) for k in config.get("ks", [10, 20]))
        metrics = evaluate_full_ranking(
            data=data,
            score_fn=scorer.score,
            ks=ks,
            batch_size=int(config.get("batch_size", 4096)),
            sample_topk_users=int(config.get("sample_topk_users", 20)),
            sample_path=out_dir / "topk_sample.csv",
        )
        dump_json(metrics, out_dir / "metrics.json")
        print("metrics=")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


def build_scorer(config: dict, data, data_dir: Path):
    model = config["model"]
    popularity = PopularityScorer.fit(data)
    if model == "popularity":
        return popularity
    if model == "ease":
        ease_cfg = config.get("ease", {})
        return EASEScorer.fit(
            data,
            reg=float(ease_cfg.get("reg", 500.0)),
            cold_fallback=ease_cfg.get("cold_fallback", "popularity") == "popularity",
            pop_tie_breaker=float(ease_cfg.get("pop_tie_breaker", 0.0)),
        )
    if model == "kg_profile":
        kg_cfg = kg_feature_config(config.get("kg", {}))
        item_features = build_course_feature_matrix(
            data, data_dir / "MOOCCube (1)" / "MOOCCube", **kg_cfg
        )
        print(f"kg_features shape={item_features.shape} nnz={item_features.nnz}")
        return KGProfileScorer(data, item_features, np.log1p(data.item_popularity))
    if model == "itemknn":
        cfg = config.get("itemknn", {})
        return ItemKNNScorer.fit(
            data,
            topk=int(cfg.get("topk", 100)),
            shrink=float(cfg.get("shrink", 100.0)),
            normalize=cfg.get("normalize", "cosine"),
        )
    if model == "rp3beta":
        cfg = config.get("rp3beta", {})
        return RP3BetaScorer.fit(
            data,
            alpha=float(cfg.get("alpha", 0.7)),
            beta=float(cfg.get("beta", 0.4)),
            topk=int(cfg.get("topk", 100)),
        )
    if model == "fusion":
        return build_fusion_scorer(config, data, data_dir, popularity)
    if model == "adaptive_fusion":
        return build_adaptive_fusion_scorer(config, data, data_dir, popularity)
    raise ValueError(f"Unknown model: {model}")


def build_fusion_scorer(config: dict, data, data_dir: Path, popularity: PopularityScorer):
    kg_features = None
    scorers = []
    weights = []
    for component in config.get("fusion", {}).get("components", []):
        name = component["name"]
        weights.append(float(component.get("weight", 1.0)))
        if name == "popularity":
            scorers.append(popularity)
        elif name == "ease":
            scorers.append(
                EASEScorer.fit(
                    data,
                    reg=float(component.get("reg", 500.0)),
                    cold_fallback=True,
                    pop_tie_breaker=float(component.get("pop_tie_breaker", 1.0e-8)),
                )
            )
        elif name == "kg_profile":
            if kg_features is None:
                kg_features = build_course_feature_matrix(
                    data,
                    data_dir / "MOOCCube (1)" / "MOOCCube",
                    **kg_feature_config(config.get("kg", {})),
                )
                print(f"kg_features shape={kg_features.shape} nnz={kg_features.nnz}")
            scorers.append(KGProfileScorer(data, kg_features, np.log1p(data.item_popularity)))
        elif name == "itemknn":
            scorers.append(
                ItemKNNScorer.fit(
                    data,
                    topk=int(component.get("topk", 100)),
                    shrink=float(component.get("shrink", 100.0)),
                    normalize=component.get("normalize", "cosine"),
                )
            )
        elif name == "rp3beta":
            scorers.append(
                RP3BetaScorer.fit(
                    data,
                    alpha=float(component.get("alpha", 0.7)),
                    beta=float(component.get("beta", 0.4)),
                    topk=int(component.get("topk", 100)),
                )
            )
        else:
            raise ValueError(f"Unknown fusion component: {name}")
    return FusionScorer(
        scorers=scorers,
        weights=weights,
        normalize=config.get("fusion", {}).get("normalize", "zscore"),
    )


def build_adaptive_fusion_scorer(config: dict, data, data_dir: Path, popularity: PopularityScorer):
    fusion_cfg = config.get("adaptive_fusion", {})
    names, scorers = build_component_scorers(fusion_cfg.get("components", []), config, data, data_dir, popularity)
    segments = []
    for segment in fusion_cfg.get("segments", []):
        weights_by_name = segment["weights"]
        weights = [float(weight_for_name(weights_by_name, name)) for name in names]
        segments.append(
            {
                "name": segment.get("name", ""),
                "min_count": segment.get("min_count", 0),
                "max_count": segment.get("max_count"),
                "weights": weights,
            }
        )
    return AdaptiveFusionScorer(
        data=data,
        scorers=scorers,
        segments=segments,
        normalize=fusion_cfg.get("normalize", "zscore"),
    )


def weight_for_name(weights_by_name: dict, name: str) -> float:
    if name in weights_by_name:
        return weights_by_name[name]
    aliases = {"rp3beta": "rp3", "popularity": "pop", "kg_profile": "kg"}
    alias = aliases.get(name)
    if alias and alias in weights_by_name:
        return weights_by_name[alias]
    raise KeyError(name)


def build_component_scorers(components: list[dict], config: dict, data, data_dir: Path, popularity: PopularityScorer):
    kg_features = None
    names = []
    scorers = []
    for component in components:
        name = component["name"]
        names.append(name)
        if name == "popularity":
            scorers.append(popularity)
        elif name == "ease":
            scorers.append(
                EASEScorer.fit(
                    data,
                    reg=float(component.get("reg", 500.0)),
                    cold_fallback=True,
                    pop_tie_breaker=float(component.get("pop_tie_breaker", 1.0e-8)),
                )
            )
        elif name == "kg_profile":
            if kg_features is None:
                kg_features = build_course_feature_matrix(
                    data,
                    data_dir / "MOOCCube (1)" / "MOOCCube",
                    **kg_feature_config(config.get("kg", {})),
                )
                print(f"kg_features shape={kg_features.shape} nnz={kg_features.nnz}")
            scorers.append(KGProfileScorer(data, kg_features, np.log1p(data.item_popularity)))
        elif name == "itemknn":
            scorers.append(
                ItemKNNScorer.fit(
                    data,
                    topk=int(component.get("topk", 100)),
                    shrink=float(component.get("shrink", 100.0)),
                    normalize=component.get("normalize", "cosine"),
                )
            )
        elif name == "rp3beta":
            scorers.append(
                RP3BetaScorer.fit(
                    data,
                    alpha=float(component.get("alpha", 0.7)),
                    beta=float(component.get("beta", 0.4)),
                    topk=int(component.get("topk", 100)),
                )
            )
        else:
            raise ValueError(f"Unknown fusion component: {name}")
    return names, scorers


def kg_feature_config(config: dict) -> dict:
    allowed = {
        "use_concept",
        "use_field",
        "use_parent",
        "use_teacher",
        "use_school",
        "use_prerequisite",
        "max_df_ratio",
        "min_df",
        "top_features_per_course",
    }
    return {key: value for key, value in config.items() if key in allowed}


if __name__ == "__main__":
    main()
