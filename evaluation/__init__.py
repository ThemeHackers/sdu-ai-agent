"""
evaluation — Comprehensive RAG Evaluation Framework for SDU AI Agent

Modules:
    metrics              Retrieval metrics (Precision, Recall, F1, MRR, nDCG, MAP)
    semantic_metrics     Embedding-based semantic metrics (Relevancy, Faithfulness)
    statistical_analysis Statistical tools (Bootstrap CI, Permutation Test, Correlation)
    evaluator            Unified RAG evaluation pipeline

Quick start:
    from evaluation.evaluator import RAGEvaluator
    evaluator = RAGEvaluator()
    report = evaluator.evaluate(test_cases)
    evaluator.print_report(report)
"""

from evaluation.metrics import RetrievalMetrics
from evaluation.semantic_metrics import SemanticMetrics
from evaluation.statistical_analysis import (
    DescriptiveStats,
    BootstrapCI,
    PermutationTest,
    CorrelationAnalysis,
    DistributionAnalysis,
)
from evaluation.evaluator import RAGEvaluator

__all__ = [
    "RetrievalMetrics",
    "SemanticMetrics",
    "DescriptiveStats",
    "BootstrapCI",
    "PermutationTest",
    "CorrelationAnalysis",
    "DistributionAnalysis",
    "RAGEvaluator",
]
