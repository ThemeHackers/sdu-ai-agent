"""
RAG Retrieval Metrics — มาตรวัดคุณภาพการค้นหาเชิง Information Retrieval

Implements standard IR evaluation metrics:
  • Precision@K, Recall@K, F1@K
  • Mean Reciprocal Rank (MRR)
  • Normalized Discounted Cumulative Gain (nDCG@K)
  • Mean Average Precision (MAP)
  • Hit Rate@K

References:
  - Manning, Raghavan, Schütze — "Introduction to Information Retrieval"
  - TREC Evaluation Measures
"""

from __future__ import annotations

import math
from typing import List, Set, Union, Dict, Any


# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _to_set(ids: Union[List, Set]) -> Set:
    """Normalise an iterable to a set for O(1) lookups."""
    return set(ids) if not isinstance(ids, set) else ids


# ═══════════════════════════════════════════════════════════════════════════════
# Core Retrieval Metrics
# ═══════════════════════════════════════════════════════════════════════════════

class RetrievalMetrics:
    """
    Stateless metric helpers for Information Retrieval evaluation.
    All methods are @staticmethod so they can be called without instantiation.
    """

    # ── Precision@K ──────────────────────────────────────────────────────────
    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: Union[List[str], Set[str]],
        k: int | None = None,
    ) -> float:
        """
        Precision@K = |{retrieved ∩ relevant}| / K

        Measures the fraction of retrieved documents that are relevant.

        Args:
            retrieved_ids: Ordered list of retrieved document IDs.
            relevant_ids:  Ground-truth relevant document IDs.
            k:             Cut-off (defaults to len(retrieved_ids)).
        Returns:
            float in [0, 1].
        """
        if not retrieved_ids:
            return 0.0
        k = k or len(retrieved_ids)
        top_k = retrieved_ids[:k]
        relevant = _to_set(relevant_ids)
        hits = sum(1 for doc_id in top_k if doc_id in relevant)
        return hits / k

    # ── Recall@K ─────────────────────────────────────────────────────────────
    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: Union[List[str], Set[str]],
        k: int | None = None,
    ) -> float:
        """
        Recall@K = |{retrieved ∩ relevant}| / |relevant|

        Measures the fraction of relevant documents that are retrieved.
        """
        relevant = _to_set(relevant_ids)
        if not relevant:
            return 0.0
        k = k or len(retrieved_ids)
        top_k = retrieved_ids[:k]
        hits = sum(1 for doc_id in top_k if doc_id in relevant)
        return hits / len(relevant)

    # ── F1@K ─────────────────────────────────────────────────────────────────
    @staticmethod
    def f1_at_k(
        retrieved_ids: List[str],
        relevant_ids: Union[List[str], Set[str]],
        k: int | None = None,
    ) -> float:
        """
        F1@K = 2 · (Precision@K · Recall@K) / (Precision@K + Recall@K)

        Harmonic mean of Precision@K and Recall@K.
        """
        p = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
        r = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    # ── Hit Rate@K ───────────────────────────────────────────────────────────
    @staticmethod
    def hit_rate_at_k(
        retrieved_ids: List[str],
        relevant_ids: Union[List[str], Set[str]],
        k: int | None = None,
    ) -> float:
        """
        Hit Rate@K = 1 if at least one relevant document is in top-K, else 0.

        Binary indicator — useful for single-answer retrieval tasks.
        """
        k = k or len(retrieved_ids)
        relevant = _to_set(relevant_ids)
        return 1.0 if any(doc_id in relevant for doc_id in retrieved_ids[:k]) else 0.0

    # ── Mean Reciprocal Rank (MRR) ──────────────────────────────────────────
    @staticmethod
    def reciprocal_rank(
        retrieved_ids: List[str],
        relevant_ids: Union[List[str], Set[str]],
    ) -> float:
        """
        RR = 1 / rank_of_first_relevant_document

        Returns 0 if no relevant document is found.
        MRR is computed by averaging RR over multiple queries.
        """
        relevant = _to_set(relevant_ids)
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    # ── Discounted Cumulative Gain (DCG) & nDCG@K ───────────────────────────
    @staticmethod
    def _dcg(relevance_scores: List[float], k: int) -> float:
        """
        DCG@K = Σ_{i=1}^{K} (2^{rel_i} - 1) / log₂(i + 1)

        Uses the standard formula with gain = 2^rel - 1.
        """
        dcg = 0.0
        for i, rel in enumerate(relevance_scores[:k]):
            dcg += (2 ** rel - 1) / math.log2(i + 2)   # i+2 because i is 0-indexed
        return dcg

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: List[str],
        relevant_ids: Union[List[str], Set[str]],
        k: int | None = None,
        relevance_grades: Dict[str, float] | None = None,
    ) -> float:
        """
        nDCG@K = DCG@K / IDCG@K

        Normalized Discounted Cumulative Gain — accounts for the position
        of relevant results. Supports graded relevance via *relevance_grades*
        dict (doc_id -> grade). Falls back to binary relevance (0/1).

        Properties:
          • nDCG ∈ [0, 1]
          • nDCG = 1.0 when ranking is ideal
        """
        relevant = _to_set(relevant_ids)
        k = k or len(retrieved_ids)

        # Build relevance vector for retrieved list
        if relevance_grades:
            rel_vector = [relevance_grades.get(doc_id, 0.0) for doc_id in retrieved_ids[:k]]
            ideal = sorted(relevance_grades.values(), reverse=True)[:k]
        else:
            rel_vector = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved_ids[:k]]
            ideal = [1.0] * min(k, len(relevant))

        dcg = RetrievalMetrics._dcg(rel_vector, k)
        idcg = RetrievalMetrics._dcg(ideal, k)

        return dcg / idcg if idcg > 0 else 0.0

    # ── Average Precision (AP) & MAP ─────────────────────────────────────────
    @staticmethod
    def average_precision(
        retrieved_ids: List[str],
        relevant_ids: Union[List[str], Set[str]],
    ) -> float:
        """
        AP = (1 / |relevant|) · Σ_{k=1}^{n} Precision@k · rel(k)

        Average Precision — area under the Precision–Recall curve.
        MAP is the mean of AP over multiple queries.
        """
        relevant = _to_set(relevant_ids)
        if not relevant:
            return 0.0

        ap_sum = 0.0
        relevant_count = 0
        for i, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant:
                relevant_count += 1
                ap_sum += relevant_count / i
        return ap_sum / len(relevant)

    # ── Aggregate helpers ────────────────────────────────────────────────────
    @staticmethod
    def compute_all(
        retrieved_ids: List[str],
        relevant_ids: Union[List[str], Set[str]],
        k: int | None = None,
        relevance_grades: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """Return a dict containing all retrieval metrics for a single query."""
        k = k or len(retrieved_ids)
        return {
            "precision@k": RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k),
            "recall@k": RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k),
            "f1@k": RetrievalMetrics.f1_at_k(retrieved_ids, relevant_ids, k),
            "hit_rate@k": RetrievalMetrics.hit_rate_at_k(retrieved_ids, relevant_ids, k),
            "reciprocal_rank": RetrievalMetrics.reciprocal_rank(retrieved_ids, relevant_ids),
            "ndcg@k": RetrievalMetrics.ndcg_at_k(retrieved_ids, relevant_ids, k, relevance_grades),
            "average_precision": RetrievalMetrics.average_precision(retrieved_ids, relevant_ids),
        }

    @staticmethod
    def mean_over_queries(per_query_results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate per-query metric dicts into means (e.g. MRR = mean(RR), MAP = mean(AP)).
        """
        if not per_query_results:
            return {}

        keys = per_query_results[0].keys()
        n = len(per_query_results)
        return {
            f"mean_{key}": sum(r[key] for r in per_query_results) / n
            for key in keys
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Quick smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    retrieved = ["doc_1", "doc_3", "doc_5", "doc_2", "doc_4"]
    relevant  = {"doc_1", "doc_2", "doc_4"}

    print("── Single-query retrieval metrics (K=5) ──")
    results = RetrievalMetrics.compute_all(retrieved, relevant, k=5)
    for name, value in results.items():
        print(f"  {name:>20s}: {value:.4f}")

    print("\n── Multi-query aggregation ──")
    q2_retrieved = ["doc_10", "doc_11", "doc_12"]
    q2_relevant  = {"doc_11"}
    q2_results   = RetrievalMetrics.compute_all(q2_retrieved, q2_relevant, k=3)

    agg = RetrievalMetrics.mean_over_queries([results, q2_results])
    for name, value in agg.items():
        print(f"  {name:>30s}: {value:.4f}")
