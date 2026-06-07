from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from course_rec.baselines import EASEScorer, ItemKNNScorer, PopularityScorer, RP3BetaScorer
from course_rec.data import build_dataset
from course_rec.kg import KGProfileScorer, build_course_feature_matrix
from course_rec.metrics import _topk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="课程作业数据集")
    parser.add_argument("--out", default="results/fusion_weight_search.csv")
    parser.add_argument("--best-out", default="outputs/fusion_weight_search_best")
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ease-reg", type=float, default=500.0)
    parser.add_argument("--rp3-alpha", type=float, default=0.7)
    parser.add_argument("--rp3-beta", type=float, default=0.4)
    parser.add_argument("--rp3-topk", type=int, default=100)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data = build_dataset(args.data_dir)
    eval_users = data.test_user_ids
    eval_cache = build_eval_cache(data, eval_users)
    names, matrices = build_component_scores(
        data,
        Path(args.data_dir),
        eval_users,
        args.ease_reg,
        args.rp3_alpha,
        args.rp3_beta,
        args.rp3_topk,
    )
    print("components=", names)

    rows = []
    candidates = seed_candidates()
    for _ in range(args.trials):
        candidates.append(random_candidate(rng))

    best = None
    for idx, weights in enumerate(candidates):
        score = combine(matrices, weights)
        metrics = evaluate_matrix(data, eval_users, score, eval_cache)
        row = {f"w_{name}": float(weight) for name, weight in zip(names, weights, strict=True)}
        row.update(metrics)
        rows.append(row)
        if best is None or metrics["NDCG@20"] > best["NDCG@20"]:
            best = row
            print("best", idx, best)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("NDCG@20", ascending=False)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("top10")
    print(df.head(10).to_string(index=False))

    best_dir = Path(args.best_out)
    best_dir.mkdir(parents=True, exist_ok=True)
    best_weights = df.iloc[0][[f"w_{name}" for name in names]].to_numpy(dtype=np.float64)
    best_score = combine(matrices, best_weights)
    best_metrics = evaluate_matrix(
        data, eval_users, best_score, eval_cache, sample_path=best_dir / "topk_sample.csv"
    )
    with open(best_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, ensure_ascii=False, indent=2)
    with open(best_dir / "weights.json", "w", encoding="utf-8") as f:
        json.dump({name: float(w) for name, w in zip(names, best_weights, strict=True)}, f, indent=2)
    print("wrote", out_path, best_dir)


def build_component_scores(
    data,
    data_dir: Path,
    eval_users: np.ndarray,
    ease_reg: float,
    rp3_alpha: float,
    rp3_beta: float,
    rp3_topk: int,
):
    scorers = [
        ("ease", EASEScorer.fit(data, reg=ease_reg, cold_fallback=True, pop_tie_breaker=1.0e-8)),
        (
            "itemknn",
            ItemKNNScorer.fit(data, topk=100, shrink=100.0, normalize="cosine"),
        ),
        ("rp3", RP3BetaScorer.fit(data, alpha=rp3_alpha, beta=rp3_beta, topk=rp3_topk)),
        ("pop", PopularityScorer.fit(data)),
    ]
    kg_features = build_course_feature_matrix(data, data_dir / "MOOCCube (1)" / "MOOCCube")
    scorers.append(("kg", KGProfileScorer(data, kg_features, np.log1p(data.item_popularity))))

    names = []
    matrices = []
    for name, scorer in scorers:
        print("scoring", name)
        scores = []
        for start in range(0, len(eval_users), 4096):
            batch = eval_users[start : start + 4096]
            raw = scorer.score(batch).astype(np.float32)
            mean = raw.mean(axis=1, keepdims=True)
            std = raw.std(axis=1, keepdims=True)
            raw = (raw - mean) / np.maximum(std, 1.0e-6)
            scores.append(raw.astype(np.float32))
        names.append(name)
        matrices.append(np.vstack(scores))
    return names, matrices


def seed_candidates() -> list[np.ndarray]:
    return [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.70, 0.10, 0.10, 0.05, 0.05]),
        np.array([0.75, 0.05, 0.15, 0.05, 0.00]),
        np.array([0.85, 0.00, 0.10, 0.05, 0.00]),
        np.array([1.00, 0.05, 0.05, 0.00, -0.05]),
        np.array([1.00, -0.05, 0.10, 0.00, -0.05]),
    ]


def random_candidate(rng: np.random.Generator) -> np.ndarray:
    weights = np.array(
        [
            1.0,
            rng.uniform(-0.15, 0.35),
            rng.uniform(-0.15, 0.35),
            rng.uniform(-0.20, 0.20),
            rng.uniform(-0.20, 0.20),
        ],
        dtype=np.float64,
    )
    return weights


def combine(matrices: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    out = np.zeros_like(matrices[0], dtype=np.float32)
    for matrix, weight in zip(matrices, weights, strict=True):
        if weight:
            out += np.float32(weight) * matrix
    return out


def evaluate_matrix(
    data,
    eval_users: np.ndarray,
    scores: np.ndarray,
    eval_cache: dict[str, np.ndarray],
    sample_path: Path | None = None,
) -> dict[str, float | int]:
    topk_blocks = []
    for start in range(0, len(eval_users), 4096):
        end = min(start + 4096, len(eval_users))
        block = scores[start:end].copy()
        block[eval_cache["train_mask"][start:end]] = -np.inf
        topk_blocks.append(_topk(block, 20))
    topk = np.vstack(topk_blocks)
    rel = eval_cache["test_bool"][np.arange(len(eval_users))[:, None], topk]
    totals = {}
    discounts = 1.0 / np.log2(np.arange(2, 22))
    test_counts = eval_cache["test_counts"]
    for k in (10, 20):
        rel_k = rel[:, :k]
        hit_counts = rel_k.sum(axis=1)
        totals[f"HR@{k}"] = float((hit_counts > 0).mean())
        dcg = (rel_k * discounts[:k]).sum(axis=1)
        idcg = eval_cache[f"idcg_{k}"]
        totals[f"NDCG@{k}"] = float(np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0).mean())
        totals[f"Recall@{k}"] = float((hit_counts / np.maximum(test_counts, 1)).mean())
    sample_rows = []
    if sample_path:
        masked_sample = scores[: min(len(eval_users), 20)].copy()
        masked_sample[eval_cache["train_mask"][: masked_sample.shape[0]]] = -np.inf
        for row, user in enumerate(eval_users):
            truth = set(map(int, data.test_user_items[int(user)]))
            ranked = topk[row].tolist()
            if len(sample_rows) >= 400:
                break
            for rank, item in enumerate(ranked, start=1):
                sample_rows.append(
                    {
                        "user": data.id2user[int(user)],
                        "rank": rank,
                        "course": data.id2item[int(item)],
                        "score": float(masked_sample[row, item]) if row < masked_sample.shape[0] else float(scores[row, item]),
                        "is_test_positive": int(item in truth),
                    }
                )
    n = len(eval_users)
    metrics = totals
    metrics["evaluated_users"] = int(n)
    metrics["candidate_courses"] = int(data.n_items)
    metrics["mask_train_positives"] = 1
    if sample_path:
        pd.DataFrame(sample_rows).to_csv(sample_path, index=False, encoding="utf-8-sig")
    return metrics


def build_eval_cache(data, eval_users: np.ndarray) -> dict[str, np.ndarray]:
    n = len(eval_users)
    train_mask = np.zeros((n, data.n_items), dtype=bool)
    test_bool = np.zeros((n, data.n_items), dtype=bool)
    test_counts = np.zeros(n, dtype=np.float32)
    discounts = 1.0 / np.log2(np.arange(2, 22))
    for row, user in enumerate(eval_users):
        train_items = data.train_user_items[int(user)]
        test_items = data.test_user_items[int(user)]
        if len(train_items):
            train_mask[row, train_items] = True
        if len(test_items):
            test_bool[row, test_items] = True
            test_counts[row] = len(test_items)
    cache = {"train_mask": train_mask, "test_bool": test_bool, "test_counts": test_counts}
    for k in (10, 20):
        idcg = np.zeros(n, dtype=np.float64)
        for row, count in enumerate(test_counts.astype(int)):
            idcg[row] = discounts[: min(k, count)].sum()
        cache[f"idcg_{k}"] = idcg
    return cache


if __name__ == "__main__":
    main()
