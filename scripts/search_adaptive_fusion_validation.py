from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from course_rec.baselines import EASEScorer, ItemKNNScorer, PopularityScorer, RP3BetaScorer
from course_rec.data import build_dataset, build_dataset_from_frames, read_interactions
from course_rec.kg import KGProfileScorer, build_course_feature_matrix
from course_rec.metrics import _topk
from search_fusion_weights import build_eval_cache


ROOT = Path(__file__).resolve().parents[1]

SEGMENTS = [
    {"name": "cold", "min_count": 0, "max_count": 0},
    {"name": "one", "min_count": 1, "max_count": 1},
    {"name": "two", "min_count": 2, "max_count": 2},
    {"name": "three_four", "min_count": 3, "max_count": 4},
    {"name": "five_plus", "min_count": 5, "max_count": None},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="课程作业数据集")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--out", default="results/adaptive_fusion_validation_search.csv")
    parser.add_argument("--fixed-out", default="results/fusion_validation_search.csv")
    parser.add_argument("--config-out", default="configs/adaptive_fusion_validated.yaml")
    parser.add_argument("--fixed-config-out", default="configs/fusion_validated.yaml")
    parser.add_argument("--segments-out", default="results/adaptive_fusion_validated_segments.csv")
    parser.add_argument("--metrics-out", default="results/adaptive_fusion_validated_metrics.json")
    parser.add_argument("--run-name", default="adaptive_fusion_validated")
    parser.add_argument("--fixed-run-name", default="fusion_validated")
    parser.add_argument("--ease-reg", type=float, default=100.0)
    parser.add_argument("--rp3-alpha", type=float, default=0.9)
    parser.add_argument("--rp3-beta", type=float, default=0.2)
    parser.add_argument("--rp3-topk", type=int, default=300)
    parser.add_argument("--max-valid-users", type=int, default=60000)
    parser.add_argument("--drop-component", choices=["kg", "rp3", "ease"], default=None)
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rng = np.random.default_rng(args.seed)
    train_df = read_interactions(data_dir / "train.csv")
    sub_train, valid = leave_one_out_split(train_df, seed=args.seed)
    valid_data = build_dataset_from_frames(sub_train, valid)
    eval_users = valid_data.test_user_ids
    print(
        f"validation split: sub_train={len(sub_train)} valid={len(valid)} "
        f"users={valid_data.n_users} items={valid_data.n_items}",
        flush=True,
    )

    all_eval_users = eval_users
    eval_users = sample_eval_users(eval_users, args.max_valid_users, args.seed)
    print(
        f"validation eval users: sampled={len(eval_users)} total={len(all_eval_users)} "
        f"max_valid_users={args.max_valid_users}",
        flush=True,
    )
    eval_cache = build_eval_cache(valid_data, eval_users)
    names, matrices = build_component_scores(
        valid_data,
        data_dir,
        eval_users,
        args.ease_reg,
        args.rp3_alpha,
        args.rp3_beta,
        args.rp3_topk,
        args.drop_component,
    )
    print("components=", names, flush=True)

    candidates = candidate_weights(rng, args.trials, names)
    fixed_rows, best_fixed = search_fixed(valid_data, eval_users, eval_cache, matrices, names, candidates)
    write_csv(args.fixed_out, fixed_rows)
    fixed_config = build_fixed_config(
        args.fixed_run_name,
        names,
        best_fixed["weights"],
        args.ease_reg,
        args.rp3_alpha,
        args.rp3_beta,
        args.rp3_topk,
    )
    write_yaml(args.fixed_config_out, fixed_config)

    counts = valid_data.train_matrix[eval_users].getnnz(axis=1)
    adaptive_rows, chosen = search_by_segment(
        valid_data,
        eval_users,
        eval_cache,
        matrices,
        names,
        candidates,
        counts,
        best_fixed["weights"],
    )
    write_csv(args.out, adaptive_rows)
    adaptive_config = build_adaptive_config(
        args.run_name,
        names,
        chosen,
        args.ease_reg,
        args.rp3_alpha,
        args.rp3_beta,
        args.rp3_topk,
    )
    write_yaml(args.config_out, adaptive_config)
    print(f"wrote {args.out}, {args.fixed_out}, {args.config_out}, {args.fixed_config_out}", flush=True)

    if args.skip_test:
        return

    del matrices, eval_cache
    gc.collect()

    test_segments, test_metrics = evaluate_test_segments(
        data_dir,
        adaptive_config,
        chosen,
        args.ease_reg,
        args.rp3_alpha,
        args.rp3_beta,
        args.rp3_topk,
        args.drop_component,
    )
    write_csv(args.segments_out, test_segments)
    write_json(args.metrics_out, test_metrics)
    print("test full", test_metrics, flush=True)
    print(f"wrote {args.segments_out}, {args.metrics_out}", flush=True)


def leave_one_out_split(train_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    valid_indices = []
    for _, group in train_df.groupby("user", sort=False):
        if len(group) >= 2:
            valid_indices.append(int(rng.choice(group.index.to_numpy())))
    valid_set = set(valid_indices)
    valid = train_df.loc[sorted(valid_set)].reset_index(drop=True)
    sub_train = train_df.drop(index=valid_set).reset_index(drop=True)
    return sub_train, valid


def sample_eval_users(eval_users: np.ndarray, max_users: int, seed: int) -> np.ndarray:
    if max_users <= 0 or len(eval_users) <= max_users:
        return eval_users
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eval_users), size=max_users, replace=False)
    return eval_users[np.sort(idx)]


def build_component_scores(
    data,
    data_dir: Path,
    eval_users: np.ndarray,
    ease_reg: float,
    rp3_alpha: float,
    rp3_beta: float,
    rp3_topk: int,
    drop_component: str | None = None,
    score_batch_size: int = 512,
) -> tuple[list[str], list[np.ndarray]]:
    drop_aliases = component_aliases(drop_component)
    scorers = [
        ("ease", EASEScorer.fit(data, reg=ease_reg, cold_fallback=True, pop_tie_breaker=1.0e-8)),
        ("itemknn", ItemKNNScorer.fit(data, topk=100, shrink=100.0, normalize="cosine")),
        ("rp3", RP3BetaScorer.fit(data, alpha=rp3_alpha, beta=rp3_beta, topk=rp3_topk)),
        ("pop", PopularityScorer.fit(data)),
    ]
    scorers = [(name, scorer) for name, scorer in scorers if name not in drop_aliases]
    if "kg" not in drop_aliases:
        kg_features = build_course_feature_matrix(data, data_dir / "MOOCCube (1)" / "MOOCCube")
        scorers.append(("kg", KGProfileScorer(data, kg_features, np.log1p(data.item_popularity))))

    names = []
    matrices = []
    for name, scorer in scorers:
        print("scoring", name, flush=True)
        scores = []
        for start in range(0, len(eval_users), score_batch_size):
            batch = eval_users[start : start + score_batch_size]
            raw = np.asarray(scorer.score(batch), dtype=np.float32)
            mean = raw.mean(axis=1, keepdims=True)
            std = raw.std(axis=1, keepdims=True)
            raw = (raw - mean) / np.maximum(std, 1.0e-6)
            scores.append(raw.astype(np.float32, copy=False))
        names.append(name)
        matrices.append(np.vstack(scores))
    return names, matrices


def candidate_weights(rng: np.random.Generator, trials: int, names: list[str]) -> list[np.ndarray]:
    base_names = ["ease", "itemknn", "rp3", "pop", "kg"]
    indexes = [base_names.index(name) for name in names]
    seeds = [
        [1.0, 0.23710607881543747, 0.32923220048201696, 0.15535817052667067, 0.04837418608001981],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.75, 0.05, 0.15, 0.05, 0.0],
        [0.70, 0.10, 0.10, 0.05, 0.05],
        [1.00, 0.05, 0.05, 0.00, -0.05],
        [1.00, -0.05, 0.10, 0.00, -0.05],
    ]
    weights = [np.array(w, dtype=np.float64)[indexes] for w in seeds]
    for _ in range(trials):
        base = np.array(
            [
                rng.uniform(0.4, 1.3),
                rng.uniform(-0.2, 0.4),
                rng.uniform(-0.1, 0.6),
                rng.uniform(-0.2, 0.25),
                rng.uniform(-0.25, 0.25),
            ],
            dtype=np.float64,
        )
        weights.append(base[indexes])
    return weights


def search_fixed(data, eval_users, eval_cache, matrices, names, candidates):
    rows = []
    best = None
    for idx, weights in enumerate(candidates):
        metrics = evaluate_weighted_matrix(data, eval_users, matrices, weights, eval_cache)
        row = {"candidate": idx, **weight_columns(names, weights), **metrics}
        rows.append(row)
        if best is None or metrics["NDCG@20"] > best["NDCG@20"]:
            best = {"weights": weights, **metrics}
            print("fixed best", idx, {k: v for k, v in row.items() if k != "candidate"}, flush=True)
    rows = sorted(rows, key=lambda x: x["NDCG@20"], reverse=True)
    return rows, best


def evaluate_weighted_matrix(
    data,
    eval_users: np.ndarray,
    matrices: list[np.ndarray],
    weights: np.ndarray,
    eval_cache: dict[str, np.ndarray],
    row_indices: np.ndarray | None = None,
    batch_size: int = 1024,
) -> dict[str, float | int]:
    if row_indices is None:
        row_indices = np.arange(len(eval_users), dtype=np.int64)
    totals = {f"HR@{k}": 0.0 for k in (10, 20)}
    totals.update({f"NDCG@{k}": 0.0 for k in (10, 20)})
    totals.update({f"Recall@{k}": 0.0 for k in (10, 20)})
    discounts = 1.0 / np.log2(np.arange(2, 22))
    for start in range(0, len(row_indices), batch_size):
        rows = row_indices[start : start + batch_size]
        block = np.zeros((len(rows), data.n_items), dtype=np.float32)
        scratch = np.empty_like(block)
        for matrix, weight in zip(matrices, weights, strict=True):
            if weight:
                matrix_block = matrix_rows(matrix, rows)
                np.multiply(matrix_block, np.float32(weight), out=scratch)
                np.add(block, scratch, out=block)
        block[eval_cache["train_mask"][rows]] = -np.inf
        topk = _topk(block, 20)
        test_bool = eval_cache["test_bool"][rows]
        rel = test_bool[np.arange(len(rows))[:, None], topk]
        test_counts = eval_cache["test_counts"][rows]
        for k in (10, 20):
            rel_k = rel[:, :k]
            hit_counts = rel_k.sum(axis=1)
            totals[f"HR@{k}"] += float((hit_counts > 0).sum())
            totals[f"Recall@{k}"] += float((hit_counts / np.maximum(test_counts, 1)).sum())
            dcg = (rel_k * discounts[:k]).sum(axis=1)
            idcg = eval_cache[f"idcg_{k}"][rows]
            totals[f"NDCG@{k}"] += float(
                np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0).sum()
            )
    n = max(1, len(row_indices))
    metrics = {name: value / n for name, value in totals.items()}
    metrics["evaluated_users"] = int(len(row_indices))
    metrics["candidate_courses"] = int(data.n_items)
    metrics["mask_train_positives"] = 1
    return metrics


def matrix_rows(matrix: np.ndarray, rows: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return matrix[:0]
    first = int(rows[0])
    last = int(rows[-1])
    if last - first + 1 == len(rows) and np.all(rows == np.arange(first, last + 1)):
        return matrix[first : last + 1]
    return matrix[rows]


def search_by_segment(data, eval_users, eval_cache, matrices, names, candidates, counts, fallback_weights):
    rows = []
    chosen = {}
    for segment in SEGMENTS:
        mask = segment_mask(counts, segment)
        row_indices = np.flatnonzero(mask)
        sub_users = eval_users[mask]
        if not len(sub_users):
            weights = np.asarray(fallback_weights, dtype=np.float64)
            chosen[segment["name"]] = weights
            rows.append({"segment": segment["name"], "users": 0, **weight_columns(names, weights)})
            print("segment no validation users", rows[-1], flush=True)
            continue
        best = None
        for weights in candidates:
            metrics = evaluate_weighted_matrix(data, eval_users, matrices, weights, eval_cache, row_indices)
            if best is None or metrics["NDCG@20"] > best["NDCG@20"]:
                best = {"weights": weights, **metrics}
        chosen[segment["name"]] = best["weights"]
        row = {
            "segment": segment["name"],
            "users": int(mask.sum()),
            **weight_columns(names, best["weights"]),
            **{k: v for k, v in best.items() if k != "weights"},
        }
        rows.append(row)
        print("segment best", row, flush=True)
        gc.collect()
    full_metrics = aggregate_segment_metrics(rows)
    rows.append({"segment": "assembled", "users": int(len(eval_users)), **full_metrics})
    return rows, chosen


def aggregate_segment_metrics(rows: list[dict]) -> dict[str, float | int]:
    metric_keys = ["HR@10", "NDCG@10", "Recall@10", "HR@20", "NDCG@20", "Recall@20"]
    total_users = sum(int(row.get("users", 0)) for row in rows)
    out = {}
    for key in metric_keys:
        num = sum(float(row.get(key, 0.0)) * int(row.get("users", 0)) for row in rows)
        out[key] = num / max(1, total_users)
    out["evaluated_users"] = int(total_users)
    out["candidate_courses"] = 698
    out["mask_train_positives"] = 1
    return out


def segment_mask(counts: np.ndarray, segment: dict) -> np.ndarray:
    mask = counts >= int(segment["min_count"])
    if segment["max_count"] is not None:
        mask &= counts <= int(segment["max_count"])
    return mask


def weight_columns(names: list[str], weights: np.ndarray) -> dict[str, float]:
    return {f"w_{name}": float(weight) for name, weight in zip(names, weights, strict=True)}


def component_configs(
    ease_reg: float,
    rp3_alpha: float,
    rp3_beta: float,
    rp3_topk: int,
    names: list[str] | None = None,
) -> list[dict]:
    configs = [
        {"name": "ease", "reg": float(ease_reg)},
        {"name": "itemknn", "normalize": "cosine", "topk": 100, "shrink": 100.0},
        {"name": "rp3beta", "alpha": float(rp3_alpha), "beta": float(rp3_beta), "topk": int(rp3_topk)},
        {"name": "popularity"},
        {"name": "kg_profile"},
    ]
    if names is None:
        return configs
    allowed = set(names)
    return [config for config in configs if component_alias(config["name"]) in allowed]


def component_alias(name: str) -> str:
    return {"rp3beta": "rp3", "popularity": "pop", "kg_profile": "kg"}.get(name, name)


def component_aliases(name: str | None) -> set[str]:
    if name is None:
        return set()
    mapping = {"rp3": {"rp3", "rp3beta"}, "kg": {"kg", "kg_profile"}, "ease": {"ease"}}
    return mapping[name]


def kg_config() -> dict:
    return {
        "use_concept": True,
        "use_field": True,
        "use_parent": True,
        "use_teacher": True,
        "use_school": True,
        "use_prerequisite": True,
        "max_df_ratio": 0.35,
        "min_df": 2,
        "top_features_per_course": 500,
    }


def build_fixed_config(run_name, names, weights, ease_reg, rp3_alpha, rp3_beta, rp3_topk) -> dict:
    components = []
    weights_by_name = dict(zip(names, weights, strict=True))
    for component in component_configs(ease_reg, rp3_alpha, rp3_beta, rp3_topk, names):
        name = component["name"]
        alias = component_alias(name)
        component = dict(component)
        component["weight"] = float(weights_by_name[alias])
        components.append(component)
    return {
        "run_name": run_name,
        "model": "fusion",
        "ks": [10, 20],
        "batch_size": 2048,
        "sample_topk_users": 20,
        "fusion": {"normalize": "zscore", "components": components},
        "kg": kg_config(),
    }


def build_adaptive_config(run_name, names, chosen, ease_reg, rp3_alpha, rp3_beta, rp3_topk) -> dict:
    by_name = {segment["name"]: segment for segment in SEGMENTS}
    segments = []
    for name, weights in chosen.items():
        segment = dict(by_name[name])
        segment["weights"] = {component: float(weight) for component, weight in zip(names, weights, strict=True)}
        segments.append(segment)
    return {
        "run_name": run_name,
        "model": "adaptive_fusion",
        "ks": [10, 20],
        "batch_size": 2048,
        "sample_topk_users": 20,
        "adaptive_fusion": {
            "normalize": "zscore",
            "components": component_configs(ease_reg, rp3_alpha, rp3_beta, rp3_topk, names),
            "segments": segments,
        },
        "kg": kg_config(),
    }


def evaluate_test_segments(
    data_dir: Path,
    config: dict,
    chosen: dict,
    ease_reg,
    rp3_alpha,
    rp3_beta,
    rp3_topk,
    drop_component: str | None = None,
):
    data = build_dataset(data_dir)
    eval_users = data.test_user_ids
    eval_cache = build_eval_cache(data, eval_users)
    names, matrices = build_component_scores(
        data, data_dir, eval_users, ease_reg, rp3_alpha, rp3_beta, rp3_topk, drop_component
    )
    counts = data.train_matrix[eval_users].getnnz(axis=1)
    rows = []
    for segment in SEGMENTS:
        mask = segment_mask(counts, segment)
        row_indices = np.flatnonzero(mask)
        weights = chosen[segment["name"]]
        metrics = evaluate_weighted_matrix(data, eval_users, matrices, weights, eval_cache, row_indices)
        rows.append(
            {
                "segment": segment["name"],
                "users": int(mask.sum()),
                **weight_columns(names, weights),
                **metrics,
            }
        )
    full_metrics = aggregate_segment_metrics(rows)
    rows.append({"segment": "assembled", "users": int(len(eval_users)), **full_metrics})
    return rows, full_metrics


def write_csv(path: str | Path, rows: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def write_yaml(path: str | Path, config: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def write_json(path: str | Path, value: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
