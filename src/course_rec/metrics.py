from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .data import InteractionData

ScoreFn = Callable[[np.ndarray], np.ndarray]


def evaluate_full_ranking(
    data: InteractionData,
    score_fn: ScoreFn,
    ks: tuple[int, ...] = (10, 20),
    batch_size: int = 4096,
    mask_train: bool = True,
    sample_topk_users: int = 20,
    sample_path: str | Path | None = None,
) -> dict[str, float | int]:
    if not ks:
        raise ValueError("ks must not be empty")
    max_k = min(max(ks), data.n_items)
    eval_users = data.test_user_ids
    totals = {f"HR@{k}": 0.0 for k in ks}
    totals.update({f"NDCG@{k}": 0.0 for k in ks})
    totals.update({f"Recall@{k}": 0.0 for k in ks})
    sample_rows: list[dict[str, str | int | float]] = []

    for start in range(0, len(eval_users), batch_size):
        user_batch = eval_users[start : start + batch_size]
        scores = np.asarray(score_fn(user_batch), dtype=np.float64)
        if scores.shape != (len(user_batch), data.n_items):
            raise ValueError(
                f"score_fn returned {scores.shape}, expected {(len(user_batch), data.n_items)}"
            )
        if mask_train:
            for row, u in enumerate(user_batch):
                train_items = data.train_user_items[int(u)]
                if len(train_items):
                    scores[row, train_items] = -np.inf

        topk = _topk(scores, max_k)
        for row, u in enumerate(user_batch):
            truth = set(map(int, data.test_user_items[int(u)]))
            ranked = topk[row].tolist()
            truth_size = len(truth)
            for k in ks:
                prefix = ranked[:k]
                hits = [1 if item in truth else 0 for item in prefix]
                hit_count = sum(hits)
                totals[f"HR@{k}"] += 1.0 if hit_count > 0 else 0.0
                totals[f"Recall@{k}"] += hit_count / truth_size
                dcg = sum(rel / np.log2(rank + 2) for rank, rel in enumerate(hits))
                idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, truth_size)))
                totals[f"NDCG@{k}"] += dcg / idcg if idcg > 0 else 0.0

            if sample_path and len(sample_rows) < sample_topk_users * max_k:
                user_name = data.id2user[int(u)]
                for rank, item in enumerate(ranked, start=1):
                    sample_rows.append(
                        {
                            "user": user_name,
                            "rank": rank,
                            "course": data.id2item[int(item)],
                            "score": float(scores[row, item]),
                            "is_test_positive": int(item in truth),
                        }
                    )

    n = max(1, len(eval_users))
    metrics = {name: value / n for name, value in totals.items()}
    metrics["evaluated_users"] = int(len(eval_users))
    metrics["candidate_courses"] = int(data.n_items)
    metrics["mask_train_positives"] = int(mask_train)

    if sample_path:
        pd.DataFrame(sample_rows).to_csv(sample_path, index=False, encoding="utf-8-sig")
    return metrics


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.shape[1]:
        part = np.tile(np.arange(scores.shape[1]), (scores.shape[0], 1))
    else:
        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="mergesort")
    return np.take_along_axis(part, order, axis=1)
