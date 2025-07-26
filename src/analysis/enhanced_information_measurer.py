"""
Enhanced Information Measurer Module with Statistical Tests

This module provides functionality to measure information-theoretic properties
with statistical validation including bootstrap confidence intervals and t-tests.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from scipy import stats
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedInformationMeasurer:
    """Enhanced measurer with statistical tests and confidence intervals."""

    @staticmethod
    def estimate_mutual_information_with_bootstrap(
            X: np.ndarray,
            Y: np.ndarray,
            n_bootstrap: int = 100,
            confidence_level: float = 0.95,
            n_bins: int = 20
    ) -> Dict[str, float]:
        """
        Estimate mutual information with bootstrap confidence intervals.

        Args:
            X: First variable samples
            Y: Second variable samples
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            n_bins: Number of bins for histogram estimation

        Returns:
            Dictionary with mean, std, confidence intervals
        """
        n_samples = len(X)
        bootstrap_mis = []

        for _ in range(n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices] if X.ndim == 1 else X[indices, :]
            Y_boot = Y[indices] if Y.ndim == 1 else Y[indices, :]

            # Estimate MI for this bootstrap sample
            mi_boot = EnhancedInformationMeasurer._estimate_single_mi(X_boot, Y_boot, n_bins)
            bootstrap_mis.append(mi_boot)

        # Calculate statistics
        bootstrap_mis = np.array(bootstrap_mis)
        alpha = 1 - confidence_level

        return {
            'mean': np.mean(bootstrap_mis),
            'std': np.std(bootstrap_mis, ddof=1),
            'ci_lower': np.percentile(bootstrap_mis, 100 * alpha / 2),
            'ci_upper': np.percentile(bootstrap_mis, 100 * (1 - alpha / 2)),
            'all_samples': bootstrap_mis.tolist()
        }

    @staticmethod
    def _estimate_single_mi(X: np.ndarray, Y: np.ndarray, n_bins: int = 20) -> float:
        """Single MI estimation (helper function)."""
        # Convert high-dimensional data to 1D if needed
        if X.ndim > 1:
            if X.shape[1] > 1:
                # Use weighted PCA approach
                n_components = min(5, X.shape[1], X.shape[0] // 10)
                if n_components > 1:
                    pca = PCA(n_components=n_components)
                    X_reduced = pca.fit_transform(X)
                    weights = np.array([0.5, 0.3, 0.15, 0.04, 0.01])[:n_components]
                    weights = weights / weights.sum()
                    X_1d = X_reduced @ weights
                else:
                    X_1d = PCA(n_components=1).fit_transform(X).flatten()
            else:
                X_1d = X.flatten()
        else:
            X_1d = X

        if Y.ndim > 1:
            if Y.shape[1] > 1:
                n_components = min(5, Y.shape[1], Y.shape[0] // 10)
                if n_components > 1:
                    pca = PCA(n_components=n_components)
                    Y_reduced = pca.fit_transform(Y)
                    weights = np.array([0.5, 0.3, 0.15, 0.04, 0.01])[:n_components]
                    weights = weights / weights.sum()
                    Y_1d = Y_reduced @ weights
                else:
                    Y_1d = PCA(n_components=1).fit_transform(Y).flatten()
            else:
                Y_1d = Y.flatten()
        else:
            Y_1d = Y

        # Create joint histogram
        try:
            hist_2d, _, _ = np.histogram2d(X_1d, Y_1d, bins=n_bins)

            # Normalize to get joint probability
            p_xy = hist_2d / float(np.sum(hist_2d))

            # Get marginal probabilities
            p_x = np.sum(p_xy, axis=1)
            p_y = np.sum(p_xy, axis=0)

            # Compute mutual information
            mi = 0.0
            for i in range(len(p_x)):
                for j in range(len(p_y)):
                    if p_xy[i, j] > 1e-10:  # Avoid log(0)
                        mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j] + 1e-10))

            return max(0, mi)  # MI should be non-negative

        except Exception as e:
            logger.warning(f"MI estimation failed: {e}")
            return 0.0

    @staticmethod
    def compare_performance_with_ttest(
            baseline_results: List[float],
            improved_results: List[float],
            paired: bool = True
    ) -> Dict[str, float]:
        """
        Compare performance between two conditions using t-test.

        Args:
            baseline_results: Baseline performance values
            improved_results: Improved performance values
            paired: Whether to use paired t-test

        Returns:
            Dictionary with statistical test results
        """
        baseline_arr = np.array(baseline_results)
        improved_arr = np.array(improved_results)

        if paired and len(baseline_arr) == len(improved_arr):
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(improved_arr, baseline_arr)
            df = len(baseline_arr) - 1
        else:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(improved_arr, baseline_arr)
            df = len(baseline_arr) + len(improved_arr) - 2

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(baseline_arr) - 1) * np.var(baseline_arr, ddof=1) +
             (len(improved_arr) - 1) * np.var(improved_arr, ddof=1)) / df
        )

        mean_diff = np.mean(improved_arr) - np.mean(baseline_arr)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        # Confidence interval for mean difference
        se_diff = pooled_std * np.sqrt(1 / len(baseline_arr) + 1 / len(improved_arr))
        t_critical = stats.t.ppf(0.975, df)  # 95% CI
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'effect_size_cohens_d': cohens_d,
            'mean_difference': mean_diff,
            'mean_baseline': np.mean(baseline_arr),
            'mean_improved': np.mean(improved_arr),
            'std_baseline': np.std(baseline_arr, ddof=1),
            'std_improved': np.std(improved_arr, ddof=1),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant_at_05': p_value < 0.05,
            'significant_at_01': p_value < 0.01
        }

    @staticmethod
    def run_multiple_experiments(
            experiment_func,
            n_runs: int = 10,
            base_seed: int = 42,
            **kwargs
    ) -> Dict[str, List]:
        """
        Run experiment multiple times with different seeds.

        Args:
            experiment_func: Function to run experiment
            n_runs: Number of runs
            base_seed: Base seed value
            **kwargs: Additional arguments for experiment_func

        Returns:
            Dictionary with aggregated results
        """
        all_results = []

        for run_id in range(n_runs):
            seed = base_seed + run_id
            logger.info(f"Running experiment {run_id + 1}/{n_runs} with seed={seed}")

            # Run single experiment
            result = experiment_func(seed=seed, **kwargs)
            result['run_id'] = run_id
            result['seed'] = seed
            all_results.append(result)

        return {
            'individual_runs': all_results,
            'n_runs': n_runs,
            'seeds_used': [base_seed + i for i in range(n_runs)]
        }

    @staticmethod
    def calculate_experiment_statistics(results_list: List[Dict]) -> Dict:
        """
        Calculate statistics across multiple experimental runs.

        Args:
            results_list: List of individual experiment results

        Returns:
            Dictionary with statistical summaries
        """
        # Extract metrics that are common across results
        metrics = set()
        for result in results_list:
            metrics.update(k for k, v in result.items()
                           if isinstance(v, (int, float)) and k not in ['run_id', 'seed'])

        statistics = {}

        for metric in metrics:
            values = [result[metric] for result in results_list if metric in result]

            if len(values) > 1:
                statistics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values, ddof=1),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'n_samples': len(values),
                    'ci_95_lower': np.percentile(values, 2.5),
                    'ci_95_upper': np.percentile(values, 97.5),
                    'all_values': values
                }

                # Add normality test for larger samples
                if len(values) >= 8:
                    _, p_value_normality = stats.shapiro(values)
                    statistics[metric]['normality_p_value'] = p_value_normality
                    statistics[metric]['appears_normal'] = p_value_normality > 0.05

        return statistics

    @staticmethod
    def format_statistical_summary(stats_dict: Dict) -> str:
        """Format statistical summary for reporting."""
        summary_lines = []
        summary_lines.append("Statistical Summary:")
        summary_lines.append("=" * 50)

        for metric, stats in stats_dict.items():
            if isinstance(stats, dict) and 'mean' in stats:
                mean = stats['mean']
                std = stats['std']
                ci_lower = stats['ci_95_lower']
                ci_upper = stats['ci_95_upper']
                n = stats['n_samples']

                summary_lines.append(f"{metric}:")
                summary_lines.append(f"  Mean ± SD: {mean:.4f} ± {std:.4f} (n={n})")
                summary_lines.append(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

                if 'appears_normal' in stats:
                    normal_status = "✓" if stats['appears_normal'] else "✗"
                    summary_lines.append(f"  Normal distribution: {normal_status}")

                summary_lines.append("")

        return "\n".join(summary_lines)


# Usage example for integration
if __name__ == "__main__":
    # Example of how to integrate with existing code
    measurer = EnhancedInformationMeasurer()

    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 64)
    Y = 0.8 * X + 0.2 * np.random.randn(100, 64)

    # Test bootstrap MI estimation
    mi_results = measurer.estimate_mutual_information_with_bootstrap(X, Y, n_bootstrap=50)
    print(f"MI with bootstrap: {mi_results['mean']:.4f} ± {mi_results['std']:.4f}")
    print(f"95% CI: [{mi_results['ci_lower']:.4f}, {mi_results['ci_upper']:.4f}]")

    # Test statistical comparison
    baseline = np.random.normal(0.5, 0.1, 20)
    improved = np.random.normal(0.7, 0.1, 20)

    comparison = measurer.compare_performance_with_ttest(baseline, improved)
    print(f"\nPerformance comparison:")
    print(f"Mean difference: {comparison['mean_difference']:.4f}")
    print(f"p-value: {comparison['p_value']:.4f}")
    print(f"Significant: {comparison['significant_at_05']}")
    print(f"Effect size (Cohen's d): {comparison['effect_size_cohens_d']:.4f}")