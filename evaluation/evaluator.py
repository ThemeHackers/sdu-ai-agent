"""
RAG Evaluator — Unified evaluation pipeline for the SDU AI Agent.

Orchestrates the full evaluation workflow:
  1. Retrieval quality   → metrics.py  (Precision, Recall, F1, MRR, nDCG, MAP)
  2. Semantic quality    → semantic_metrics.py  (Relevancy, Faithfulness, Similarity)
  3. Statistical rigour  → statistical_analysis.py  (CI, Significance, Correlation)

Usage:
    evaluator = RAGEvaluator()
    report = evaluator.evaluate(test_cases)
    evaluator.print_report(report)
"""

from __future__ import annotations

import json
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from evaluation.metrics import RetrievalMetrics
from evaluation.semantic_metrics import SemanticMetrics, EmbeddingProvider
from evaluation.statistical_analysis import (
    DescriptiveStats,
    BootstrapCI,
    PermutationTest,
    CorrelationAnalysis,
    DistributionAnalysis,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Test-case schema
# ═══════════════════════════════════════════════════════════════════════════════
# Each test case is a dict with the following keys:
#
#   {
#     "query":           str,        # The user question
#     "reference_answer": str,       # (optional) Gold-standard answer
#     "relevant_ids":    [str, ...], # Ground-truth relevant chunk IDs
#     "retrieved_ids":   [str, ...], # IDs returned by the retriever
#     "context_chunks":  [str, ...], # Text of retrieved context
#     "generated_answer": str,       # The model's generated answer
#   }


# ═══════════════════════════════════════════════════════════════════════════════
# RAG Evaluator
# ═══════════════════════════════════════════════════════════════════════════════

class RAGEvaluator:
    """End-to-end evaluator for a RAG pipeline."""

    def __init__(self, api_key: str | None = None):
        self.semantic = SemanticMetrics(EmbeddingProvider(api_key=api_key))

    # ── Main evaluation entry point ──────────────────────────────────────────
    def evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        k: int = 5,
        run_semantic: bool = True,
        bootstrap_ci: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline.

        Args:
            test_cases:    List of test-case dicts (see schema above).
            k:             Cut-off for retrieval metrics.
            run_semantic:  Whether to compute embedding-based metrics (requires API).
            bootstrap_ci:  Whether to compute bootstrap confidence intervals.

        Returns:
            Comprehensive evaluation report dict.
        """
        start = time.time()
        n = len(test_cases)

        # ── Phase 1: Per-query retrieval metrics ─────────────────────────────
        retrieval_results = []
        for tc in test_cases:
            retrieved = tc.get("retrieved_ids", [])
            relevant = tc.get("relevant_ids", [])
            retrieval_results.append(
                RetrievalMetrics.compute_all(retrieved, relevant, k=k)
            )

        retrieval_aggregated = RetrievalMetrics.mean_over_queries(retrieval_results)

        # ── Phase 2: Per-query semantic metrics ──────────────────────────────
        semantic_results = []
        if run_semantic:
            for tc in test_cases:
                query = tc.get("query", "")
                answer = tc.get("generated_answer", "")
                context = tc.get("context_chunks", [])
                reference = tc.get("reference_answer", "")

                sem = {}
                if query and answer:
                    sem["answer_relevancy"] = self.semantic.answer_relevancy(query, answer)

                if answer and context:
                    faith = self.semantic.faithfulness(answer, context)
                    sem["faithfulness_score"] = faith["faithfulness_score"]
                    sem["mean_claim_score"] = faith["mean_claim_score"]

                if query and context:
                    ctx_rel = self.semantic.context_relevancy(query, context)
                    sem["context_relevancy"] = ctx_rel["mean_relevancy"]

                if reference and answer:
                    sem["semantic_similarity"] = self.semantic.semantic_similarity(reference, answer)

                if reference and query and answer and context:
                    composite = self.semantic.answer_correctness(query, answer, reference, context)
                    sem["composite_correctness"] = composite["composite_correctness"]

                semantic_results.append(sem)

        # ── Phase 3: Statistical analysis ────────────────────────────────────
        stats_report = {}

        # 3a: Descriptive stats for each retrieval metric
        retrieval_metric_names = list(retrieval_results[0].keys()) if retrieval_results else []
        retrieval_score_vectors: Dict[str, List[float]] = {
            name: [r[name] for r in retrieval_results] for name in retrieval_metric_names
        }

        for name, scores in retrieval_score_vectors.items():
            stats_report[name] = {
                "descriptive": DescriptiveStats.summary(scores),
                "distribution": DistributionAnalysis.histogram(scores, bins=5),
                "grade": DistributionAnalysis.quality_grade(DescriptiveStats.mean(scores)),
            }
            if bootstrap_ci and len(scores) >= 5:
                stats_report[name]["bootstrap_95ci"] = BootstrapCI.confidence_interval(
                    scores, confidence=0.95, n_bootstrap=5000
                )

        # 3b: Descriptive stats for semantic metrics
        if semantic_results:
            sem_metric_names = set()
            for s in semantic_results:
                sem_metric_names.update(s.keys())
            sem_score_vectors: Dict[str, List[float]] = {}
            for name in sem_metric_names:
                sem_score_vectors[name] = [s.get(name, 0.0) for s in semantic_results]

            for name, scores in sem_score_vectors.items():
                stats_report[name] = {
                    "descriptive": DescriptiveStats.summary(scores),
                    "distribution": DistributionAnalysis.histogram(scores, bins=5),
                    "grade": DistributionAnalysis.quality_grade(DescriptiveStats.mean(scores)),
                }
                if bootstrap_ci and len(scores) >= 5:
                    stats_report[name]["bootstrap_95ci"] = BootstrapCI.confidence_interval(
                        scores, confidence=0.95, n_bootstrap=5000
                    )

        # 3c: Cross-metric correlation
        all_vectors = {}
        all_vectors.update(retrieval_score_vectors)
        if semantic_results:
            all_vectors.update(sem_score_vectors)

        # Only compute correlation if we have ≥ 2 metrics with ≥ 3 samples
        valid_vectors = {k: v for k, v in all_vectors.items() if len(v) >= 3}
        correlation_matrix = {}
        if len(valid_vectors) >= 2:
            correlation_matrix = CorrelationAnalysis.correlation_matrix(
                valid_vectors, method="spearman"
            )

        # ── Assemble final report ────────────────────────────────────────────
        elapsed = time.time() - start

        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_test_cases": n,
                "k": k,
                "run_semantic": run_semantic,
                "elapsed_seconds": round(elapsed, 2),
            },
            "retrieval_summary": retrieval_aggregated,
            "per_query_retrieval": retrieval_results,
            "per_query_semantic": semantic_results if run_semantic else [],
            "statistical_analysis": stats_report,
            "correlation_matrix": correlation_matrix,
        }

        return report

    # ── Quick evaluation (for a single query, useful for live monitoring) ────
    def evaluate_single(
        self,
        query: str,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        context_chunks: List[str],
        generated_answer: str,
        reference_answer: str = "",
        k: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate a single RAG query-response pair."""
        test_case = {
            "query": query,
            "reference_answer": reference_answer,
            "relevant_ids": relevant_ids,
            "retrieved_ids": retrieved_ids,
            "context_chunks": context_chunks,
            "generated_answer": generated_answer,
        }
        return self.evaluate([test_case], k=k, bootstrap_ci=False)

    # ── Compare two systems ──────────────────────────────────────────────────
    @staticmethod
    def compare_systems(
        report_a: Dict[str, Any],
        report_b: Dict[str, Any],
        metric_name: str = "ndcg@k",
    ) -> Dict[str, Any]:
        """
        Statistical comparison of two system evaluation reports using
        Paired Permutation Test.

        Args:
            report_a, report_b: Reports from evaluate().
            metric_name:        Which metric to compare.

        Returns:
            Permutation test result dict.
        """
        scores_a = [r[metric_name] for r in report_a["per_query_retrieval"]]
        scores_b = [r[metric_name] for r in report_b["per_query_retrieval"]]
        return PermutationTest.paired_test(scores_a, scores_b)

    # ── Pretty print ─────────────────────────────────────────────────────────
    @staticmethod
    def print_report(report: Dict[str, Any]) -> None:
        """Print a human-readable evaluation report."""
        meta = report["metadata"]
        print("\n" + "═" * 70)
        print("  📊  SDU AI Agent — RAG Evaluation Report")
        print("═" * 70)
        print(f"  📅 Timestamp   : {meta['timestamp']}")
        print(f"  📝 Test Cases  : {meta['num_test_cases']}")
        print(f"  🔢 K (cut-off) : {meta['k']}")
        print(f"  ⏱️  Elapsed     : {meta['elapsed_seconds']}s")

        # Retrieval summary
        print("\n" + "─" * 70)
        print("  🔍 RETRIEVAL METRICS (averaged over queries)")
        print("─" * 70)
        for name, value in report["retrieval_summary"].items():
            grade = DistributionAnalysis.quality_grade(value)
            bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            print(f"  {name:>25s}: {value:.4f}  [{bar}]  Grade: {grade}")

        # Statistical analysis
        print("\n" + "─" * 70)
        print("  📈 STATISTICAL ANALYSIS")
        print("─" * 70)
        for metric_name, analysis in report["statistical_analysis"].items():
            desc = analysis["descriptive"]
            print(f"\n  ▸ {metric_name}")
            print(f"    Mean={desc['mean']:.4f}  Std={desc['std']:.4f}  "
                  f"Median={desc['median']:.4f}  "
                  f"[{desc['min']:.4f}, {desc['max']:.4f}]  "
                  f"Grade={analysis['grade']}")
            if "bootstrap_95ci" in analysis:
                ci = analysis["bootstrap_95ci"]
                print(f"    95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]  "
                      f"SE={ci['std_error']:.4f}")

        # Correlation
        if report.get("correlation_matrix"):
            print("\n" + "─" * 70)
            print("  🔗 SPEARMAN CORRELATION MATRIX")
            print("─" * 70)
            matrix = report["correlation_matrix"]
            names = list(matrix.keys())
            header = "  " + " " * 22 + "".join(f"{n[:8]:>10s}" for n in names)
            print(header)
            for row_name in names:
                row = "  " + f"{row_name[:20]:>20s}  "
                for col_name in names:
                    val = matrix[row_name][col_name]
                    row += f"{val:>10.3f}"
                print(row)

        print("\n" + "═" * 70)
        print("  ✅ Evaluation complete.")
        print("═" * 70 + "\n")

    # ── Save report to JSON ──────────────────────────────────────────────────
    @staticmethod
    def save_report(report: Dict[str, Any], path: str = "evaluation/report.json") -> str:
        """Save the evaluation report as JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"📁 Report saved to: {path}")
        return path
