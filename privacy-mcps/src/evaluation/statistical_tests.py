"""
Statistical Significance Analysis.

Implements paired t-tests for comparing model performance
across multiple experimental runs (Table XV in the paper).
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
) -> Dict:
    """
    Perform paired t-test between two sets of accuracy scores.

    H0: No significant difference between methods A and B.
    H1: Significant difference exists.

    Args:
        scores_a: Accuracy scores from method A (proposed) across N runs.
        scores_b: Accuracy scores from method B (baseline) across N runs.
        alpha: Significance level.

    Returns:
        Dictionary with test results.
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    assert len(scores_a) == len(scores_b), "Score arrays must have equal length"

    mean_diff = np.mean(scores_a - scores_b)
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    significant = p_value < alpha

    result = {
        "mean_a": np.mean(scores_a),
        "mean_b": np.mean(scores_b),
        "mean_diff": mean_diff,
        "std_diff": np.std(scores_a - scores_b, ddof=1),
        "t_statistic": t_stat,
        "p_value": p_value,
        "alpha": alpha,
        "significant": significant,
        "n_runs": len(scores_a),
    }

    logger.info(
        f"Paired t-test: mean_diff={mean_diff:.4f}%, "
        f"t={t_stat:.4f}, p={p_value:.2e}, "
        f"{'SIGNIFICANT' if significant else 'NOT significant'}"
    )
    return result


def run_significance_analysis(
    all_results: Dict[str, List[float]],
    proposed_key: str = "phe_proposed",
    alpha: float = 0.05,
) -> List[Dict]:
    """
    Run paired t-tests comparing proposed method against all baselines.

    Args:
        all_results: Dict mapping method names to lists of accuracy scores.
        proposed_key: Key for the proposed method in all_results.
        alpha: Significance level.

    Returns:
        List of test result dictionaries.
    """
    proposed_scores = all_results[proposed_key]
    results = []

    for method_name, scores in all_results.items():
        if method_name == proposed_key:
            continue
        logger.info(f"\nComparing: {proposed_key} vs {method_name}")
        test_result = paired_t_test(proposed_scores, scores, alpha)
        test_result["comparison"] = f"{proposed_key} vs {method_name}"
        results.append(test_result)

    return results


def print_significance_table(results: List[Dict]):
    """Print formatted significance analysis table."""
    print("\n" + "=" * 90)
    print("  STATISTICAL SIGNIFICANCE ANALYSIS (Paired T-Test)")
    print("=" * 90)
    print(f"  {'Comparison':<40} {'Mean Diff':>10} {'t-stat':>10} {'p-value':>12} {'Sig?':>6}")
    print("-" * 90)
    for r in results:
        sig = "Yes" if r["significant"] else "No"
        print(f"  {r['comparison']:<40} {r['mean_diff']:>9.2f}% {r['t_statistic']:>10.2f} {r['p_value']:>12.2e} {sig:>6}")
    print("=" * 90 + "\n")
