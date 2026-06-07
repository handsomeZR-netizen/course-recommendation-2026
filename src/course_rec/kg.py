from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from .baselines import Scorer
from .data import InteractionData


def read_tsv_edges(path: str | Path) -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                edges.append((parts[0], parts[1]))
    return edges


def build_course_feature_matrix(
    data: InteractionData,
    kg_dir: str | Path,
    *,
    use_concept: bool = True,
    use_field: bool = True,
    use_parent: bool = True,
    use_teacher: bool = True,
    use_school: bool = True,
    use_prerequisite: bool = True,
    max_df_ratio: float = 0.35,
    min_df: int = 2,
    top_features_per_course: int | None = 500,
) -> sparse.csr_matrix:
    kg_dir = Path(kg_dir)
    rel_dir = kg_dir / "relations"
    course_features: list[defaultdict[str, float]] = [
        defaultdict(float) for _ in range(data.n_items)
    ]
    item_set = set(data.item2id)
    course_concepts: dict[str, set[str]] = defaultdict(set)

    if use_concept or use_field or use_parent or use_prerequisite:
        for course, concept in read_tsv_edges(rel_dir / "course-concept.json"):
            if course in item_set:
                course_concepts[course].add(concept)

    allowed_concepts: set[str] = set()
    if course_concepts:
        df: defaultdict[str, int] = defaultdict(int)
        for concepts in course_concepts.values():
            for concept in concepts:
                df[concept] += 1
        max_df = max(1, int(max_df_ratio * data.n_items))
        allowed_concepts = {c for c, count in df.items() if min_df <= count <= max_df}

    if use_concept:
        for course, concepts in course_concepts.items():
            item_id = data.item2id[course]
            filtered = [c for c in concepts if c in allowed_concepts]
            if top_features_per_course:
                filtered = filtered[:top_features_per_course]
            for concept in filtered:
                course_features[item_id][f"concept:{concept}"] += 1.0

    if use_field:
        concept_field = dict(read_tsv_edges(rel_dir / "concept-field.json"))
        for course, concepts in course_concepts.items():
            item_id = data.item2id[course]
            for concept in concepts:
                field = concept_field.get(concept)
                if field:
                    course_features[item_id][f"field:{field}"] += 1.0

    if use_parent:
        child_to_parents: defaultdict[str, list[str]] = defaultdict(list)
        for parent, child in read_tsv_edges(rel_dir / "parent-son.json"):
            child_to_parents[child].append(parent)
        for course, concepts in course_concepts.items():
            item_id = data.item2id[course]
            for concept in concepts:
                for parent in child_to_parents.get(concept, [])[:3]:
                    course_features[item_id][f"parent:{parent}"] += 1.0

    if use_teacher:
        for teacher, course in read_tsv_edges(rel_dir / "teacher-course.json"):
            if course in item_set:
                course_features[data.item2id[course]][f"teacher:{teacher}"] += 1.5

    if use_school:
        for school, course in read_tsv_edges(rel_dir / "school-course.json"):
            if course in item_set:
                course_features[data.item2id[course]][f"school:{school}"] += 0.5

    if use_prerequisite:
        prereq_edges = read_tsv_edges(rel_dir / "prerequisite-dependency.json")
        prereq_next: defaultdict[str, list[str]] = defaultdict(list)
        prereq_prev: defaultdict[str, list[str]] = defaultdict(list)
        for source, target in prereq_edges:
            prereq_next[source].append(target)
            prereq_prev[target].append(source)
        for course, concepts in course_concepts.items():
            item_id = data.item2id[course]
            for concept in concepts:
                for target in prereq_next.get(concept, [])[:5]:
                    course_features[item_id][f"pre_next:{target}"] += 0.8
                for source in prereq_prev.get(concept, [])[:5]:
                    course_features[item_id][f"pre_prev:{source}"] += 0.8

    vectorizer = DictVectorizer(dtype=np.float32)
    counts = vectorizer.fit_transform(course_features)
    if counts.shape[1] == 0:
        return sparse.csr_matrix((data.n_items, 1), dtype=np.float32)
    tfidf = TfidfTransformer(norm="l2", smooth_idf=True, sublinear_tf=True)
    return tfidf.fit_transform(counts).tocsr().astype(np.float32)


@dataclass
class KGProfileScorer(Scorer):
    data: InteractionData
    item_features: sparse.csr_matrix
    popularity: np.ndarray
    cold_fallback: bool = True

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        user_course_counts = self.data.train_matrix[user_ids]
        counts = np.asarray(user_course_counts.sum(axis=1)).ravel()
        user_features = user_course_counts @ self.item_features
        nonzero = counts > 0
        if np.any(nonzero):
            inv_counts = np.zeros_like(counts, dtype=np.float32)
            inv_counts[nonzero] = 1.0 / counts[nonzero]
            user_features = sparse.diags(inv_counts, dtype=np.float32) @ user_features
        scores = (user_features @ self.item_features.T).toarray().astype(np.float64)
        if self.cold_fallback:
            cold_rows = np.flatnonzero(counts == 0)
            if len(cold_rows):
                scores[cold_rows] = self.popularity
        return scores
