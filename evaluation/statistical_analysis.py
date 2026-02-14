"""
Statistical Analysis for RAG Evaluation Results

Provides rigorous statistical tools for interpreting evaluation metrics:
  • Descriptive statistics (mean, variance, std, skewness, kurtosis)
  • Bootstrap Confidence Intervals (BCa method)
  • Paired Permutation Test for significance between two systems
  • Effect Size (Cohen's d)
  • Score Distribution Analysis (histogram binning)
  • Correlation Analysis (Pearson, Spearman)

These tools let you answer questions like:
  "Is system A statistically significantly better than system B?"
  "What is the 95 % CI for my MRR score?"
  "Are retrieval precision and answer faithfulness correlated?"

All implementations are pure-Python (no scipy/statsmodels dependency).

References:
  - Efron & Tibshirani — "An Introduction to the Bootstrap" (1993)
  - Noreen — "Computer Intensive Methods for Testing Hypotheses" (1989)
"""

from __future__ import annotations

import math
import random
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Descriptive Statistics
# ═══════════════════════════════════════════════════════════════════════════════

class DescriptiveStats:
    """Standard descriptive statistics with no external dependencies."""

    @staticmethod
    def mean(data: List[float]) -> float:
        """Arithmetic mean: μ = (1/n) Σxᵢ"""
        if not data:
            return 0.0
        return sum(data) / len(data)

    @staticmethod
    def variance(data: List[float], ddof: int = 1) -> float:
        """
        Sample variance: s² = Σ(xᵢ - x̄)² / (n - ddof)

        ddof=1 → unbiased (Bessel's correction)
        ddof=0 → population variance
        """
        n = len(data)
        if n <= ddof:
            return 0.0
        mu = DescriptiveStats.mean(data)
        return sum((x - mu) ** 2 for x in data) / (n - ddof)

    @staticmethod
    def std(data: List[float], ddof: int = 1) -> float:
        """Standard deviation: s = √(s²)"""
        return math.sqrt(DescriptiveStats.variance(data, ddof))

    @staticmethod
    def median(data: List[float]) -> float:
        """Median value."""
        if not data:
            return 0.0
        s = sorted(data)
        n = len(s)
        mid = n // 2
        return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]

    @staticmethod
    def percentile(data: List[float], p: float) -> float:
        """
        p-th percentile using linear interpolation (p in [0, 100]).
        """
        if not data:
            return 0.0
        s = sorted(data)
        k = (p / 100) * (len(s) - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[f] * (c - k) + s[c] * (k - f)

    @staticmethod
    def iqr(data: List[float]) -> float:
        """Interquartile range = Q3 - Q1."""
        return DescriptiveStats.percentile(data, 75) - DescriptiveStats.percentile(data, 25)

    @staticmethod
    def skewness(data: List[float]) -> float:
        """
        Sample skewness (Fisher's definition):
          γ₁ = (m₃ / m₂^{3/2}) · √(n(n-1)) / (n-2)
        where mₖ = central moments.
        """
        n = len(data)
        if n < 3:
            return 0.0
        mu = DescriptiveStats.mean(data)
        m2 = sum((x - mu) ** 2 for x in data) / n
        m3 = sum((x - mu) ** 3 for x in data) / n
        if m2 == 0:
            return 0.0
        g1 = m3 / (m2 ** 1.5)
        # Adjust for sample
        return g1 * math.sqrt(n * (n - 1)) / (n - 2)

    @staticmethod
    def kurtosis(data: List[float]) -> float:
        """
        Excess kurtosis (Fisher):
          Kurt = m₄/m₂² - 3
        Normal distribution has excess kurtosis = 0.
        """
        n = len(data)
        if n < 4:
            return 0.0
        mu = DescriptiveStats.mean(data)
        m2 = sum((x - mu) ** 2 for x in data) / n
        m4 = sum((x - mu) ** 4 for x in data) / n
        if m2 == 0:
            return 0.0
        return (m4 / (m2 ** 2)) - 3.0

    @staticmethod
    def summary(data: List[float]) -> Dict[str, float]:
        """Complete descriptive statistics summary."""
        return {
            "n": len(data),
            "mean": round(DescriptiveStats.mean(data), 6),
            "std": round(DescriptiveStats.std(data), 6),
            "variance": round(DescriptiveStats.variance(data), 6),
            "median": round(DescriptiveStats.median(data), 6),
            "min": min(data) if data else 0.0,
            "max": max(data) if data else 0.0,
            "q1": round(DescriptiveStats.percentile(data, 25), 6),
            "q3": round(DescriptiveStats.percentile(data, 75), 6),
            "iqr": round(DescriptiveStats.iqr(data), 6),
            "skewness": round(DescriptiveStats.skewness(data), 6),
            "kurtosis": round(DescriptiveStats.kurtosis(data), 6),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap Confidence Interval
# ═══════════════════════════════════════════════════════════════════════════════

class BootstrapCI:
    """
    Non-parametric Bootstrap Confidence Intervals.

    The bootstrap resamples the observed data *with replacement* B times,
    computes the statistic for each resample, and uses the distribution
    of the bootstrap statistics to construct a confidence interval.

    Method: Percentile Bootstrap (with optional BCa).
    """

    @staticmethod
    def confidence_interval(
        data: List[float],
        confidence: float = 0.95,
        n_bootstrap: int = 10_000,
        statistic=None,
        seed: int | None = 42,
    ) -> Dict[str, float]:
        """
        Compute a bootstrap CI for a given statistic (default: mean).

        Args:
            data:         Observed sample.
            confidence:   Confidence level (e.g. 0.95 for 95 % CI).
            n_bootstrap:  Number of bootstrap resamples.
            statistic:    Callable that takes a list and returns a scalar.
                          Defaults to arithmetic mean.
            seed:         Random seed for reproducibility.

        Returns:
            {
                "point_estimate": ...,
                "ci_lower": ...,
                "ci_upper": ...,
                "confidence_level": ...,
                "n_bootstrap": ...,
                "std_error": ...,
            }
        """
        if not data:
            return {
                "point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "confidence_level": confidence, "n_bootstrap": n_bootstrap,
                "std_error": 0.0,
            }

        stat_fn = statistic or DescriptiveStats.mean
        rng = random.Random(seed)
        n = len(data)

        # Bootstrap
        boot_stats = []
        for _ in range(n_bootstrap):
            sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
            boot_stats.append(stat_fn(sample))

        boot_stats.sort()
        alpha = 1 - confidence
        lo_idx = int(math.floor((alpha / 2) * n_bootstrap))
        hi_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
        lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
        hi_idx = max(0, min(hi_idx, n_bootstrap - 1))

        return {
            "point_estimate": round(stat_fn(data), 6),
            "ci_lower": round(boot_stats[lo_idx], 6),
            "ci_upper": round(boot_stats[hi_idx], 6),
            "confidence_level": confidence,
            "n_bootstrap": n_bootstrap,
            "std_error": round(DescriptiveStats.std(boot_stats, ddof=1), 6),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Paired Permutation Test (non-parametric significance test)
# ═══════════════════════════════════════════════════════════════════════════════

class PermutationTest:
    """
    Paired Permutation Test (two-sided).

    H₀: the two systems have the same expected metric value.
    H₁: the systems differ.

    Procedure:
      1. Compute observed Δ = mean(A) - mean(B).
      2. For each of T permutations, randomly flip the sign of each paired
         difference dᵢ = Aᵢ - Bᵢ, then compute the mean of the permuted diffs.
      3. p-value = fraction of permuted |Δ*| ≥ |observed Δ|.
    """

    @staticmethod
    def paired_test(
        scores_a: List[float],
        scores_b: List[float],
        n_permutations: int = 10_000,
        seed: int | None = 42,
    ) -> Dict[str, float]:
        """
        Args:
            scores_a: Per-query metric scores for system A.
            scores_b: Per-query metric scores for system B (same length).
            n_permutations: Number of random permutations.

        Returns:
            {
                "mean_a", "mean_b", "observed_diff",
                "p_value", "significant_at_005", "significant_at_001",
                "effect_size_cohens_d",
            }
        """
        assert len(scores_a) == len(scores_b), "Both score lists must have the same length."

        diffs = [a - b for a, b in zip(scores_a, scores_b)]
        n = len(diffs)
        observed_diff = sum(diffs) / n

        rng = random.Random(seed)
        count_extreme = 0
        for _ in range(n_permutations):
            perm_mean = sum(d * (1 if rng.random() < 0.5 else -1) for d in diffs) / n
            if abs(perm_mean) >= abs(observed_diff):
                count_extreme += 1

        p_value = count_extreme / n_permutations

        # Cohen's d  =  mean(diff) / std(diff)
        std_diff = DescriptiveStats.std(diffs)
        cohens_d = observed_diff / std_diff if std_diff > 0 else 0.0

        return {
            "mean_a": round(DescriptiveStats.mean(scores_a), 6),
            "mean_b": round(DescriptiveStats.mean(scores_b), 6),
            "observed_diff": round(observed_diff, 6),
            "p_value": round(p_value, 6),
            "significant_at_005": p_value < 0.05,
            "significant_at_001": p_value < 0.01,
            "effect_size_cohens_d": round(cohens_d, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Correlation Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class CorrelationAnalysis:
    """Pearson and Spearman rank correlation."""

    @staticmethod
    def pearson(x: List[float], y: List[float]) -> float:
        """
        Pearson correlation coefficient:
          r = Σ(xᵢ - x̄)(yᵢ - ȳ) / √[Σ(xᵢ - x̄)² · Σ(yᵢ - ȳ)²]

        Measures linear relationship.  r ∈ [-1, 1].
        """
        n = len(x)
        assert n == len(y) and n > 1

        mx, my = DescriptiveStats.mean(x), DescriptiveStats.mean(y)
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        den_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
        den_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
        if den_x == 0 or den_y == 0:
            return 0.0
        return num / (den_x * den_y)

    @staticmethod
    def _rank(data: List[float]) -> List[float]:
        """Assign ranks with average-rank tie-breaking."""
        indexed = sorted(enumerate(data), key=lambda t: t[1])
        ranks = [0.0] * len(data)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j + 1) / 2  # average of 1-indexed ranks
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    @staticmethod
    def spearman(x: List[float], y: List[float]) -> float:
        """
        Spearman's rank correlation:
          ρ = pearson(rank(x), rank(y))

        Measures monotonic relationship.  ρ ∈ [-1, 1].
        """
        return CorrelationAnalysis.pearson(
            CorrelationAnalysis._rank(x),
            CorrelationAnalysis._rank(y),
        )

    @staticmethod
    def correlation_matrix(
        metrics_dict: Dict[str, List[float]],
        method: str = "pearson",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise correlation matrix for a dict of named metric
        score lists.

        Args:
            metrics_dict: {"metric_name": [score1, score2, ...], ...}
            method: "pearson" or "spearman"

        Returns:
            Nested dict — matrix[metric_a][metric_b] = correlation.
        """
        fn = CorrelationAnalysis.pearson if method == "pearson" else CorrelationAnalysis.spearman
        names = list(metrics_dict.keys())
        matrix = {}
        for a in names:
            matrix[a] = {}
            for b in names:
                matrix[a][b] = round(fn(metrics_dict[a], metrics_dict[b]), 4)
        return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# Score Distribution Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class DistributionAnalysis:
    """Analyse the distribution shape of evaluation scores."""

    @staticmethod
    def histogram(data: List[float], bins: int = 10, range_: Tuple[float, float] = (0.0, 1.0)) -> Dict:
        """
        Create a histogram for score distribution.

        Returns:
            {
                "bin_edges": [...],
                "counts": [...],
                "frequencies": [...],   # counts / total
            }
        """
        lo, hi = range_
        bin_width = (hi - lo) / bins
        edges = [lo + i * bin_width for i in range(bins + 1)]
        counts = [0] * bins

        for x in data:
            idx = int((x - lo) / bin_width)
            idx = min(idx, bins - 1)
            idx = max(idx, 0)
            counts[idx] += 1

        n = len(data) if data else 1
        freqs = [c / n for c in counts]

        return {
            "bin_edges": [round(e, 4) for e in edges],
            "counts": counts,
            "frequencies": [round(f, 4) for f in freqs],
        }

    @staticmethod
    def quality_grade(score: float) -> str:
        """
        Map a [0, 1] score to a human-readable quality grade.

        Grade boundaries:
          A+: ≥0.95 | A: ≥0.90 | B+: ≥0.85 | B: ≥0.80
          C+: ≥0.75 | C: ≥0.70 | D: ≥0.60 | F: <0.60
        """
        if score >= 0.95:
            return "A+"
        elif score >= 0.90:
            return "A"
        elif score >= 0.85:
            return "B+"
        elif score >= 0.80:
            return "B"
        elif score >= 0.75:
            return "C+"
        elif score >= 0.70:
            return "C"
        elif score >= 0.60:
            return "D"
        else:
            return "F"


# ═══════════════════════════════════════════════════════════════════════════════
# Quick smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_a = [0.8, 0.75, 0.9, 0.85, 0.7, 0.88, 0.92, 0.83, 0.79, 0.87]
    data_b = [0.6, 0.65, 0.7, 0.68, 0.55, 0.72, 0.78, 0.63, 0.60, 0.71]

    print("═══ Descriptive Statistics ═══")
    print(f"System A: {DescriptiveStats.summary(data_a)}")
    print(f"System B: {DescriptiveStats.summary(data_b)}")

    print("\n═══ Bootstrap 95% CI for mean (System A) ═══")
    ci = BootstrapCI.confidence_interval(data_a, confidence=0.95)
    print(f"  {ci}")

    print("\n═══ Paired Permutation Test (A vs B) ═══")
    result = PermutationTest.paired_test(data_a, data_b)
    print(f"  {result}")

    print("\n═══ Correlation ═══")
    print(f"  Pearson(A, B):  {CorrelationAnalysis.pearson(data_a, data_b):.4f}")
    print(f"  Spearman(A, B): {CorrelationAnalysis.spearman(data_a, data_b):.4f}")

    print("\n═══ Distribution ═══")
    hist = DistributionAnalysis.histogram(data_a, bins=5)
    print(f"  Histogram: {hist}")
    print(f"  Grade for mean={DescriptiveStats.mean(data_a):.2f}: "
          f"{DistributionAnalysis.quality_grade(DescriptiveStats.mean(data_a))}")
