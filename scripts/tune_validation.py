from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from course_rec.baselines import EASEScorer, FusionScorer, PopularityScorer
from course_rec.data import build_dataset_from_frames, read_interactions
from course_rec.kg import KGProfileScorer, build_course_feature_matrix
from course_rec.metrics import evaluate_full_ranking


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="课程作业数据集")
    parser.add_argument("--out", default="results/validation_tuning.csv")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--regs", default="50,100,200,500,1000,2000")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_df = read_interactions(data_dir / "train.csv")
    sub_train, valid = leave_one_out_split(train_df, seed=args.seed)
    data = build_dataset_from_frames(sub_train, valid)
    print(f"sub_train={len(sub_train)} valid={len(valid)} users={data.n_users} items={data.n_items}")

    rows = []
    popularity = PopularityScorer.fit(data)
    pop_metrics = evaluate_full_ranking(data, popularity.score, batch_size=4096)
    rows.append({"model": "popularity", "reg": None, "w_ease": 0, "w_kg": 0, "w_pop": 1, **pop_metrics})

    best_ease = None
    best_ndcg = -1.0
    for reg in [float(x) for x in args.regs.split(",") if x.strip()]:
        scorer = EASEScorer.fit(data, reg=reg, cold_fallback=True, pop_tie_breaker=1.0e-8)
        metrics = evaluate_full_ranking(data, scorer.score, batch_size=4096)
        row = {"model": "ease", "reg": reg, "w_ease": 1, "w_kg": 0, "w_pop": 0, **metrics}
        rows.append(row)
        if metrics["NDCG@20"] > best_ndcg:
            best_ndcg = float(metrics["NDCG@20"])
            best_ease = scorer
            best_reg = reg
        print(f"reg={reg} NDCG@20={metrics['NDCG@20']:.6f}")

    kg_features = build_course_feature_matrix(data, data_dir / "MOOCCube (1)" / "MOOCCube")
    kg_scorer = KGProfileScorer(data, kg_features, np.log1p(data.item_popularity))
    kg_metrics = evaluate_full_ranking(data, kg_scorer.score, batch_size=2048)
    rows.append({"model": "kg_profile", "reg": None, "w_ease": 0, "w_kg": 1, "w_pop": 0, **kg_metrics})

    for w_kg in [0.0, 0.03, 0.06, 0.10, 0.15, 0.20]:
        for w_pop in [0.0, 0.03, 0.06, 0.10]:
            w_ease = 1.0 - w_kg - w_pop
            if w_ease < 0:
                continue
            fusion = FusionScorer(
                scorers=[best_ease, kg_scorer, popularity],
                weights=[w_ease, w_kg, w_pop],
                normalize="zscore",
            )
            metrics = evaluate_full_ranking(data, fusion.score, batch_size=2048)
            rows.append(
                {
                    "model": "fusion",
                    "reg": best_reg,
                    "w_ease": w_ease,
                    "w_kg": w_kg,
                    "w_pop": w_pop,
                    **metrics,
                }
            )
            print(
                f"fusion w=({w_ease:.2f},{w_kg:.2f},{w_pop:.2f}) "
                f"NDCG@20={metrics['NDCG@20']:.6f}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("NDCG@20", ascending=False)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {out_path}")
    print(df.head(10).to_string(index=False))


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


if __name__ == "__main__":
    main()
