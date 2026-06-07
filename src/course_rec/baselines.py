from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .data import InteractionData


class Scorer:
    def score(self, user_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class PopularityScorer(Scorer):
    popularity: np.ndarray

    @classmethod
    def fit(cls, data: InteractionData) -> "PopularityScorer":
        return cls(np.log1p(data.item_popularity).astype(np.float64))

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        return np.tile(self.popularity, (len(user_ids), 1)).astype(np.float64, copy=True)


@dataclass
class EASEScorer(Scorer):
    data: InteractionData
    weights: np.ndarray
    popularity: np.ndarray
    cold_fallback: bool = True
    pop_tie_breaker: float = 0.0

    @classmethod
    def fit(
        cls,
        data: InteractionData,
        reg: float = 500.0,
        cold_fallback: bool = True,
        pop_tie_breaker: float = 0.0,
    ) -> "EASEScorer":
        x = data.train_matrix.astype(np.float64)
        gram = (x.T @ x).toarray()
        diag = np.diag_indices(gram.shape[0])
        gram[diag] += reg
        precision = np.linalg.inv(gram)
        weights = -precision / np.diag(precision)
        weights[diag] = 0.0
        popularity = np.log1p(data.item_popularity).astype(np.float64)
        return cls(data, weights, popularity, cold_fallback, pop_tie_breaker)

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        scores = self.data.train_matrix[user_ids].dot(self.weights)
        scores = np.asarray(scores, dtype=np.float64)
        if self.pop_tie_breaker:
            scores += self.pop_tie_breaker * self.popularity
        if self.cold_fallback:
            nnz = self.data.train_matrix[user_ids].getnnz(axis=1)
            cold_rows = np.flatnonzero(nnz == 0)
            if len(cold_rows):
                scores[cold_rows] = self.popularity
        return scores


@dataclass
class ItemKNNScorer(Scorer):
    data: InteractionData
    similarity: np.ndarray
    popularity: np.ndarray
    cold_fallback: bool = True

    @classmethod
    def fit(
        cls,
        data: InteractionData,
        topk: int = 100,
        shrink: float = 100.0,
        normalize: str = "cosine",
    ) -> "ItemKNNScorer":
        x = data.train_matrix.astype(np.float64)
        gram = (x.T @ x).toarray()
        diag = np.diag(gram).copy()
        if normalize == "cosine":
            denom = np.sqrt(np.outer(diag, diag)) + shrink
            sim = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
        elif normalize == "jaccard":
            denom = diag[:, None] + diag[None, :] - gram + shrink
            sim = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
        elif normalize == "conditional":
            denom = diag[:, None] + shrink
            sim = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
        else:
            raise ValueError(f"Unknown ItemKNN normalize mode: {normalize}")
        np.fill_diagonal(sim, 0.0)
        sim = keep_topk_columns(sim, topk)
        popularity = np.log1p(data.item_popularity).astype(np.float64)
        return cls(data, sim, popularity)

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        scores = self.data.train_matrix[user_ids].dot(self.similarity)
        scores = np.asarray(scores, dtype=np.float64)
        if self.cold_fallback:
            nnz = self.data.train_matrix[user_ids].getnnz(axis=1)
            cold_rows = np.flatnonzero(nnz == 0)
            if len(cold_rows):
                scores[cold_rows] = self.popularity
        return scores


@dataclass
class RP3BetaScorer(Scorer):
    data: InteractionData
    similarity: np.ndarray
    popularity: np.ndarray
    cold_fallback: bool = True

    @classmethod
    def fit(
        cls,
        data: InteractionData,
        alpha: float = 0.7,
        beta: float = 0.4,
        topk: int = 100,
    ) -> "RP3BetaScorer":
        x = data.train_matrix.astype(np.float64).tocsr()
        p_ui = row_normalize(x)
        p_iu = row_normalize(x.T.tocsr())
        if alpha != 1.0:
            p_ui = p_ui.power(alpha)
            p_iu = p_iu.power(alpha)
        sim = (p_iu @ p_ui).toarray()
        item_pop = np.asarray(x.sum(axis=0)).ravel().astype(np.float64)
        if beta:
            sim /= np.power(item_pop[None, :] + 1.0e-12, beta)
        np.fill_diagonal(sim, 0.0)
        sim = keep_topk_columns(sim, topk)
        popularity = np.log1p(data.item_popularity).astype(np.float64)
        return cls(data, sim, popularity)

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        user_profile = row_normalize(self.data.train_matrix[user_ids].astype(np.float64))
        scores = user_profile.dot(self.similarity)
        scores = np.asarray(scores, dtype=np.float64)
        if self.cold_fallback:
            nnz = self.data.train_matrix[user_ids].getnnz(axis=1)
            cold_rows = np.flatnonzero(nnz == 0)
            if len(cold_rows):
                scores[cold_rows] = self.popularity
        return scores


@dataclass
class FusionScorer(Scorer):
    scorers: list[Scorer]
    weights: list[float]
    normalize: str = "zscore"

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        output = None
        for scorer, weight in zip(self.scorers, self.weights, strict=True):
            scores = scorer.score(user_ids)
            scores = self._normalize(scores)
            weighted = weight * scores
            output = weighted if output is None else output + weighted
        if output is None:
            raise ValueError("FusionScorer needs at least one component")
        return output

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return scores
        if self.normalize == "minmax":
            lo = np.nanmin(scores, axis=1, keepdims=True)
            hi = np.nanmax(scores, axis=1, keepdims=True)
            return (scores - lo) / np.maximum(hi - lo, 1.0e-12)
        if self.normalize == "zscore":
            mean = np.nanmean(scores, axis=1, keepdims=True)
            std = np.nanstd(scores, axis=1, keepdims=True)
            return (scores - mean) / np.maximum(std, 1.0e-12)
        raise ValueError(f"Unknown normalize mode: {self.normalize}")


@dataclass
class AdaptiveFusionScorer(Scorer):
    data: InteractionData
    scorers: list[Scorer]
    segments: list[dict]
    normalize: str = "zscore"

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        component_scores = [self._normalize(scorer.score(user_ids)) for scorer in self.scorers]
        stacked = np.stack(component_scores, axis=0)
        counts = self.data.train_matrix[user_ids].getnnz(axis=1)
        output = np.zeros_like(component_scores[0], dtype=np.float64)
        for row, count in enumerate(counts):
            weights = self._weights_for_count(int(count))
            output[row] = np.tensordot(weights, stacked[:, row, :], axes=(0, 0))
        return output

    def _weights_for_count(self, count: int) -> np.ndarray:
        for segment in self.segments:
            min_count = int(segment.get("min_count", 0))
            max_count = segment.get("max_count")
            if count >= min_count and (max_count is None or count <= int(max_count)):
                return np.asarray(segment["weights"], dtype=np.float64)
        return np.asarray(self.segments[-1]["weights"], dtype=np.float64)

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        return FusionScorer([], [], self.normalize)._normalize(scores)


def keep_topk_columns(matrix: np.ndarray, topk: int) -> np.ndarray:
    if topk <= 0 or topk >= matrix.shape[0]:
        return matrix
    out = np.zeros_like(matrix)
    for col in range(matrix.shape[1]):
        column = matrix[:, col]
        idx = np.argpartition(-column, kth=topk - 1)[:topk]
        out[idx, col] = column[idx]
    return out


def row_normalize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sum, dtype=np.float64)
    nonzero = row_sum > 0
    inv[nonzero] = 1.0 / row_sum[nonzero]
    return sparse.diags(inv) @ matrix
