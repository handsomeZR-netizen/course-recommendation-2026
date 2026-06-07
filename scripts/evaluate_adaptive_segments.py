from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from course_rec.data import build_dataset
from course_rec.io_utils import read_config
from course_rec.metrics import _topk
from course_rec.run_experiment import build_scorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/adaptive_fusion_validated.yaml")
    parser.add_argument("--data-dir", default="课程作业数据集")
    parser.add_argument("--out", default="results/adaptive_fusion_validated_segments.csv")
    args = parser.parse_args()

    config = read_config(args.config)
    data = build_dataset(args.data_dir)
    scorer = build_scorer(config, data, Path(args.data_dir))
    eval_users = data.test_user_ids
    counts = data.train_matrix[eval_users].getnnz(axis=1)
    rows = []
    for segment in config["adaptive_fusion"]["segments"]:
        mask = segment_mask(counts, segment)
        users = eval_users[mask]
        metrics = evaluate_user_subset(
            data,
            scorer.score,
            users,
            batch_size=int(config.get("batch_size", 2048)),
        )
        rows.append(
            {
                "segment": segment["name"],
                "users": int(len(users)),
                **weight_columns(segment["weights"]),
                **metrics,
            }
        )
        print(rows[-1], flush=True)
    rows.append({"segment": "assembled", "users": int(len(eval_users)), **aggregate(rows)})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {out_path}", flush=True)


def segment_mask(counts: np.ndarray, segment: dict) -> np.ndarray:
    mask = counts >= int(segment.get("min_count", 0))
    max_count = segment.get("max_count")
    if max_count is not None:
        mask &= counts <= int(max_count)
    return mask


def weight_columns(weights: dict) -> dict[str, float]:
    return {
        "w_ease": float(weights.get("ease", 0.0)),
        "w_itemknn": float(weights.get("itemknn", 0.0)),
        "w_rp3": float(weights.get("rp3", weights.get("rp3beta", 0.0))),
        "w_pop": float(weights.get("pop", weights.get("popularity", 0.0))),
        "w_kg": float(weights.get("kg", weights.get("kg_profile", 0.0))),
    }


def evaluate_user_subset(data, score_fn, eval_users: np.ndarray, batch_size: int) -> dict[str, float | int]:
    totals = {f"HR@{k}": 0.0 for k in (10, 20)}
    totals.update({f"NDCG@{k}": 0.0 for k in (10, 20)})
    totals.update({f"Recall@{k}": 0.0 for k in (10, 20)})
    discounts = 1.0 / np.log2(np.arange(2, 22))
    for start in range(0, len(eval_users), batch_size):
        batch = eval_users[start : start + batch_size]
        scores = np.asarray(score_fn(batch), dtype=np.float64)
        for row, user in enumerate(batch):
            train_items = data.train_user_items[int(user)]
            if len(train_items):
                scores[row, train_items] = -np.inf
        topk = _topk(scores, 20)
        for row, user in enumerate(batch):
            truth = set(map(int, data.test_user_items[int(user)]))
            truth_size = len(truth)
            ranked = topk[row].tolist()
            for k in (10, 20):
                hits = np.array([1 if item in truth else 0 for item in ranked[:k]], dtype=np.float64)
                hit_count = float(hits.sum())
                totals[f"HR@{k}"] += 1.0 if hit_count > 0 else 0.0
                totals[f"Recall@{k}"] += hit_count / max(1, truth_size)
                dcg = float((hits * discounts[:k]).sum())
                idcg = float(discounts[: min(k, truth_size)].sum())
                totals[f"NDCG@{k}"] += dcg / idcg if idcg > 0 else 0.0
    n = max(1, len(eval_users))
    metrics = {key: value / n for key, value in totals.items()}
    metrics["evaluated_users"] = int(len(eval_users))
    metrics["candidate_courses"] = int(data.n_items)
    metrics["mask_train_positives"] = 1
    return metrics


def aggregate(rows: list[dict]) -> dict[str, float | int]:
    total = sum(int(row["users"]) for row in rows)
    metric_keys = ["HR@10", "NDCG@10", "Recall@10", "HR@20", "NDCG@20", "Recall@20"]
    out = {}
    for key in metric_keys:
        out[key] = sum(float(row[key]) * int(row["users"]) for row in rows) / max(1, total)
    out["evaluated_users"] = int(total)
    out["candidate_courses"] = 698
    out["mask_train_positives"] = 1
    return out


if __name__ == "__main__":
    main()
