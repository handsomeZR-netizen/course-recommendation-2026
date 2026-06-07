from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class InteractionData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    user2id: dict[str, int]
    item2id: dict[str, int]
    id2user: list[str]
    id2item: list[str]
    train_matrix: sparse.csr_matrix
    train_user_items: list[np.ndarray]
    test_user_items: list[np.ndarray]
    train_pairs: np.ndarray
    item_popularity: np.ndarray

    @property
    def n_users(self) -> int:
        return len(self.id2user)

    @property
    def n_items(self) -> int:
        return len(self.id2item)

    @property
    def test_user_ids(self) -> np.ndarray:
        return np.array(
            [u for u, items in enumerate(self.test_user_items) if len(items) > 0],
            dtype=np.int64,
        )


def read_interactions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"user": "string", "course": "string"})
    expected = {"user", "course"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[["user", "course"]].dropna().drop_duplicates().reset_index(drop=True)
    return df.astype({"user": str, "course": str})


def build_dataset(data_dir: str | Path) -> InteractionData:
    data_dir = Path(data_dir)
    train_df = read_interactions(data_dir / "train.csv")
    test_df = read_interactions(data_dir / "test.csv")
    return build_dataset_from_frames(train_df, test_df)


def build_dataset_from_frames(train_df: pd.DataFrame, test_df: pd.DataFrame) -> InteractionData:
    users = sorted(set(train_df["user"]).union(test_df["user"]))
    items = sorted(set(train_df["course"]).union(test_df["course"]))
    user2id = {u: i for i, u in enumerate(users)}
    item2id = {c: i for i, c in enumerate(items)}

    train_users = train_df["user"].map(user2id).to_numpy(np.int64)
    train_items = train_df["course"].map(item2id).to_numpy(np.int64)
    test_users = test_df["user"].map(user2id).to_numpy(np.int64)
    test_items = test_df["course"].map(item2id).to_numpy(np.int64)

    train_matrix = sparse.csr_matrix(
        (np.ones(len(train_users), dtype=np.float32), (train_users, train_items)),
        shape=(len(users), len(items)),
        dtype=np.float32,
    )
    train_matrix.sum_duplicates()

    train_user_items = _group_items(train_users, train_items, len(users))
    test_user_items = _group_items(test_users, test_items, len(users))
    train_pairs = np.column_stack([train_users, train_items]).astype(np.int64)
    item_popularity = np.asarray(train_matrix.sum(axis=0)).ravel().astype(np.float64)

    return InteractionData(
        train_df=train_df,
        test_df=test_df,
        user2id=user2id,
        item2id=item2id,
        id2user=users,
        id2item=items,
        train_matrix=train_matrix,
        train_user_items=train_user_items,
        test_user_items=test_user_items,
        train_pairs=train_pairs,
        item_popularity=item_popularity,
    )


def _group_items(users: np.ndarray, items: np.ndarray, n_users: int) -> list[np.ndarray]:
    buckets: list[list[int]] = [[] for _ in range(n_users)]
    for u, i in zip(users, items, strict=True):
        buckets[int(u)].append(int(i))
    return [np.array(sorted(set(v)), dtype=np.int64) for v in buckets]


def dataset_stats(data: InteractionData) -> dict[str, int | float]:
    train_users = int(data.train_df["user"].nunique())
    test_users = int(data.test_df["user"].nunique())
    train_items = int(data.train_df["course"].nunique())
    test_items = int(data.test_df["course"].nunique())
    train_user_set = set(data.train_df["user"])
    test_user_set = set(data.test_df["user"])
    train_item_set = set(data.train_df["course"])
    test_item_set = set(data.test_df["course"])
    test_counts = data.test_df.groupby("user").size()
    train_counts = data.train_df.groupby("user").size()
    pair_overlap = pd.merge(data.train_df, data.test_df, on=["user", "course"]).shape[0]
    return {
        "train_interactions": int(len(data.train_df)),
        "test_interactions": int(len(data.test_df)),
        "all_users": data.n_users,
        "all_courses": data.n_items,
        "train_users": train_users,
        "test_users": test_users,
        "train_courses": train_items,
        "test_courses": test_items,
        "test_users_seen_in_train": len(train_user_set & test_user_set),
        "test_users_unseen_in_train": len(test_user_set - train_user_set),
        "test_courses_seen_in_train": len(train_item_set & test_item_set),
        "test_courses_unseen_in_train": len(test_item_set - train_item_set),
        "train_test_pair_overlap": int(pair_overlap),
        "train_interactions_per_user_mean": float(train_counts.mean()),
        "train_interactions_per_user_median": float(train_counts.median()),
        "test_interactions_per_user_mean": float(test_counts.mean()),
        "test_interactions_per_user_median": float(test_counts.median()),
        "test_single_positive_users": int((test_counts == 1).sum()),
        "test_multi_positive_users": int((test_counts > 1).sum()),
        "density_train": float(len(data.train_df) / (max(1, train_users * train_items))),
    }
