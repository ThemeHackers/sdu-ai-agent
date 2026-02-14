"""
Semantic Evaluation Metrics for RAG Systems

Embedding-based metrics that capture meaning beyond lexical overlap:
  - Cosine Similarity between embeddings
  - Embedding-based Answer Relevancy  (query <-> answer)
  - Context Relevancy                  (query <-> context chunks)
  - Faithfulness Score                 (answer grounded in context)
  - Semantic Similarity                (reference <-> generated answer)

Uses Google GenAI Embedding API (gemini-embedding-001) -- the same model
as the project's retrieval pipeline for consistency.

Mathematical foundations:
  - cosine_sim(u, v) = (u . v) / (||u|| . ||v||)
  - Token-level NLI is approximated without a dedicated model by using
    embedding similarity of sentence-level decomposition.
"""

from __future__ import annotations

import math
import os
import re
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Vector primitives (pure-Python — no numpy dependency required)
# ═══════════════════════════════════════════════════════════════════════════════

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: List[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    cos(θ) = (a · b) / (‖a‖ · ‖b‖)

    Returns value in [-1, 1].  For normalised embeddings this equals the dot product.
    """
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


def pairwise_cosine_matrix(
    vecs_a: List[List[float]],
    vecs_b: List[List[float]],
) -> List[List[float]]:
    """Return an |A|×|B| matrix of cosine similarities."""
    return [[cosine_similarity(a, b) for b in vecs_b] for a in vecs_a]


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding provider  (thin wrapper around the project's existing API)
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingProvider:
    """Embeds texts via Google GenAI — reuses the same model as the RAG pipeline."""

    def __init__(self, api_key: str | None = None, model: str = "models/gemini-embedding-001"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model
        self._client = None
        self._dim = 3072  # gemini-embedding-001 produces 3072-d vectors

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def embed(self, texts: List[str], task_type: str = "SEMANTIC_SIMILARITY") -> List[List[float]]:
        """
        Embed a list of texts. Returns list of embedding vectors.
        
        task_type options:
          RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY, CLASSIFICATION
        """
        embeddings = []
        for text in texts:
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                    config={"task_type": task_type},
                )
                embeddings.append(response.embeddings[0].values)
            except Exception as e:
                logger.warning(f"Embedding failed for text ({len(text)} chars): {e}")
                embeddings.append([0.0] * self._dim)
        return embeddings

    def embed_single(self, text: str, task_type: str = "SEMANTIC_SIMILARITY") -> List[float]:
        return self.embed([text], task_type)[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Metrics Class
# ═══════════════════════════════════════════════════════════════════════════════

class SemanticMetrics:
    """
    Embedding-based evaluation metrics for a RAG pipeline.

    All methods return floats in [0, 1] (cosine similarity clamped to non-negative).
    """

    def __init__(self, embedding_provider: EmbeddingProvider | None = None):
        self.embedder = embedding_provider or EmbeddingProvider()

    # ── Answer Relevancy ─────────────────────────────────────────────────────
    def answer_relevancy(self, query: str, answer: str) -> float:
        """
        Semantic similarity between the query and the generated answer.

        answer_relevancy = max(0, cos(embed(query), embed(answer)))

        Interpretation:
          • 1.0 → answer perfectly addresses the query semantically
          • 0.0 → answer is orthogonal to the query
        """
        vecs = self.embedder.embed(
            [query, answer],
            task_type="SEMANTIC_SIMILARITY",
        )
        return max(0.0, cosine_similarity(vecs[0], vecs[1]))

    # ── Context Relevancy ────────────────────────────────────────────────────
    def context_relevancy(self, query: str, context_chunks: List[str]) -> Dict[str, float]:
        """
        Measure how relevant each retrieved context chunk is to the query.

        Returns:
          {
            "mean_relevancy":  average cosine similarity across chunks,
            "max_relevancy":   best chunk similarity,
            "min_relevancy":   worst chunk similarity,
            "per_chunk":       list of per-chunk scores,
          }
        """
        if not context_chunks:
            return {"mean_relevancy": 0.0, "max_relevancy": 0.0, "min_relevancy": 0.0, "per_chunk": []}

        q_vec = self.embedder.embed_single(query, task_type="RETRIEVAL_QUERY")
        c_vecs = self.embedder.embed(context_chunks, task_type="RETRIEVAL_DOCUMENT")

        scores = [max(0.0, cosine_similarity(q_vec, cv)) for cv in c_vecs]
        return {
            "mean_relevancy": sum(scores) / len(scores),
            "max_relevancy": max(scores),
            "min_relevancy": min(scores),
            "per_chunk": scores,
        }

    # ── Faithfulness (Answer Grounding) ──────────────────────────────────────
    def faithfulness(self, answer: str, context_chunks: List[str]) -> Dict[str, float]:
        """
        Measure how well the answer is grounded / faithful to the context.

        Process:
          1. Split the answer into sentences (claims).
          2. For each claim, find the maximum cosine similarity with any context chunk.
          3. A claim is "supported" if max_sim ≥ threshold.

        faithfulness = |supported_claims| / |total_claims|

        This follows the RAGAS-style faithfulness decomposition but uses
        embedding similarity instead of an NLI model.

        Returns:
          {
            "faithfulness_score":   ratio of supported claims,
            "mean_claim_score":     average max similarity,
            "num_claims":           number of claims extracted,
            "claim_details":        list of {claim, max_sim, supported},
          }
        """
        SUPPORT_THRESHOLD = 0.65

        claims = self._split_into_claims(answer)
        if not claims:
            return {"faithfulness_score": 0.0, "mean_claim_score": 0.0, "num_claims": 0, "claim_details": []}

        if not context_chunks:
            return {
                "faithfulness_score": 0.0,
                "mean_claim_score": 0.0,
                "num_claims": len(claims),
                "claim_details": [{"claim": c, "max_sim": 0.0, "supported": False} for c in claims],
            }

        claim_vecs = self.embedder.embed(claims, task_type="SEMANTIC_SIMILARITY")
        ctx_vecs = self.embedder.embed(context_chunks, task_type="SEMANTIC_SIMILARITY")

        details = []
        supported = 0
        total_sim = 0.0

        for claim, claim_vec in zip(claims, claim_vecs):
            sims = [max(0.0, cosine_similarity(claim_vec, cv)) for cv in ctx_vecs]
            max_sim = max(sims) if sims else 0.0
            is_supported = max_sim >= SUPPORT_THRESHOLD
            if is_supported:
                supported += 1
            total_sim += max_sim
            details.append({"claim": claim, "max_sim": round(max_sim, 4), "supported": is_supported})

        return {
            "faithfulness_score": supported / len(claims),
            "mean_claim_score": total_sim / len(claims),
            "num_claims": len(claims),
            "claim_details": details,
        }

    # ── Semantic Similarity (Reference-based) ────────────────────────────────
    def semantic_similarity(self, reference: str, generated: str) -> float:
        """
        Cosine similarity between a reference answer and the generated answer.

        Useful when a gold-standard answer exists.

        sim = max(0, cos(embed(reference), embed(generated)))
        """
        vecs = self.embedder.embed(
            [reference, generated],
            task_type="SEMANTIC_SIMILARITY",
        )
        return max(0.0, cosine_similarity(vecs[0], vecs[1]))

    # ── Answer Correctness (Composite) ───────────────────────────────────────
    def answer_correctness(
        self,
        query: str,
        answer: str,
        reference: str,
        context_chunks: List[str],
        weights: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """
        Composite correctness score combining multiple dimensions:

          correctness = w₁·relevancy + w₂·faithfulness + w₃·similarity

        Default weights: relevancy=0.3, faithfulness=0.4, similarity=0.3

        Returns all sub-scores plus the weighted composite.
        """
        w = weights or {"relevancy": 0.3, "faithfulness": 0.4, "similarity": 0.3}

        rel = self.answer_relevancy(query, answer)
        faith = self.faithfulness(answer, context_chunks)
        sim = self.semantic_similarity(reference, answer)

        composite = (
            w["relevancy"] * rel
            + w["faithfulness"] * faith["faithfulness_score"]
            + w["similarity"] * sim
        )

        return {
            "answer_relevancy": round(rel, 4),
            "faithfulness_score": round(faith["faithfulness_score"], 4),
            "semantic_similarity": round(sim, 4),
            "composite_correctness": round(composite, 4),
        }

    # ── Internal helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _split_into_claims(text: str) -> List[str]:
        """
        Heuristically split text into sentence-level "claims".
        Works well for Thai + English mixed text.
        """
        # Split on Thai/English sentence terminators
        parts = re.split(r'(?<=[.!?。\n])\s*', text.strip())
        # Filter out very short fragments
        return [p.strip() for p in parts if len(p.strip()) >= 10]


# ═══════════════════════════════════════════════════════════════════════════════
# Quick smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("── Semantic Metrics smoke test ──\n")

    # Pure-vector test (no API call)
    a = [1.0, 0.0, 0.0]
    b = [0.707, 0.707, 0.0]
    print(f"cosine_similarity([1,0,0], [0.707,0.707,0]): {cosine_similarity(a, b):.4f}")
    print(f"Expected ≈ 0.7071\n")

    # If API key available, run full test
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        sm = SemanticMetrics()
        query = "มหาวิทยาลัยสวนดุสิตอยู่ที่ไหน"
        answer = "มหาวิทยาลัยสวนดุสิตตั้งอยู่ที่เขตดุสิต กรุงเทพมหานคร"
        context = ["มหาวิทยาลัยสวนดุสิต ตั้งอยู่เลขที่ 295 ถนนนครราชสีมา เขตดุสิต กรุงเทพมหานคร"]

        print(f"Answer Relevancy: {sm.answer_relevancy(query, answer):.4f}")
        print(f"Context Relevancy: {sm.context_relevancy(query, context)}")
        print(f"Faithfulness: {sm.faithfulness(answer, context)}")
    else:
        print("⚠️  GEMINI_API_KEY not set — skipping API-dependent tests.")
