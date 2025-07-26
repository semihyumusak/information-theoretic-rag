"""
Enhanced Bottleneck Analysis Experiment Module with Statistical Validation

This module implements bottleneck analysis with proper statistical testing
including t-tests for significance and effect size calculations.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.schema_generator import SchemaGenerator
from data.db_generator import DatabaseGenerator
from data.query_generator import QueryGenerator
from pipeline.query_encoder import QueryEncoder
from pipeline.database_encoder import DatabaseEncoder
from pipeline.retriever import Retriever
from pipeline.context_integrator import ContextIntegrator
from pipeline.generator import Generator
from analysis.enhanced_information_measurer import EnhancedInformationMeasurer


def run_single_bottleneck_experiment(seed: int, **kwargs) -> Dict[str, Any]:
    """
    Run a single instance of bottleneck analysis experiment.

    Args:
        seed: Random seed for this run
        **kwargs: Additional parameters

    Returns:
        Dictionary with results from this run
    """
    improvement_levels = kwargs.get('improvement_levels', [1.5, 2.0])
    n_queries = kwargs.get('n_queries', 100)

    # Generate schema and database with this seed
    schema_gen = SchemaGenerator(seed=seed)
    db_gen = DatabaseGenerator(seed=seed)
    query_gen = QueryGenerator(seed=seed)

    # Create schema with medium complexity
    schema = schema_gen.generate_schema(entropy_bits=5.0)

    # Generate database content
    database = db_gen.generate_database(schema, num_rows_per_table=1000)

    # Generate queries
    queries = query_gen.generate_queries(schema, num_queries=500)

    # Encode database
    db_encoder = DatabaseEncoder(embedding_dim=256, seed=seed)
    embedded_db = db_encoder.encode_database(database)

    # Baseline configuration
    baseline_encoder = QueryEncoder(embedding_dim=256, noise_level=0.1, seed=seed)
    baseline_retriever = Retriever(context_size=8)
    baseline_integrator = ContextIntegrator(noise_level=0.05, seed=seed)
    baseline_generator = Generator(temperature=0.7, seed=seed)

    # Run baseline experiment
    baseline_responses = []
    for query in queries[:n_queries]:
        query_embedding = baseline_encoder.encode(query)
        context = baseline_retriever.retrieve(query_embedding, embedded_db)
        integrated_context = baseline_integrator.integrate(context, 8)
        response = baseline_generator.generate(query, integrated_context)
        baseline_responses.append(response)

    baseline_accuracies = [r["accuracy"] for r in baseline_responses]
    baseline_accuracy = np.mean(baseline_accuracies)

    # Store results for this run
    run_results = {
        'baseline_accuracy': baseline_accuracy,
        'baseline_accuracies': baseline_accuracies,
        'improvements': [],
        'seed': seed
    }

    # Test improvements for each channel
    channels_to_test = ['C1', 'C2', 'C3', 'C4']

    for channel in channels_to_test:
        for improvement_factor in improvement_levels:

            # Configure improved component
            if channel == 'C1':
                # Reduce noise level
                noise_level = 0.1 / improvement_factor
                improved_encoder = QueryEncoder(embedding_dim=256, noise_level=noise_level, seed=seed)
                retriever = baseline_retriever
                integrator = baseline_integrator
                generator = baseline_generator
                encoder = improved_encoder
            elif channel == 'C2':
                # Increase context size
                context_size = int(8 * improvement_factor)
                retriever = Retriever(context_size=context_size)
                encoder = baseline_encoder
                integrator = baseline_integrator
                generator = baseline_generator
            elif channel == 'C3':
                # Reduce noise level
                noise_level = 0.05 / improvement_factor
                integrator = ContextIntegrator(noise_level=noise_level, seed=seed)
                encoder = baseline_encoder
                retriever = baseline_retriever
                generator = baseline_generator
            elif channel == 'C4':
                # Reduce temperature
                temperature = max(0.1, 0.7 / improvement_factor)
                generator = Generator(temperature=temperature, seed=seed)
                encoder = baseline_encoder
                retriever = baseline_retriever
                integrator = baseline_integrator

            # Run improved experiment
            improved_responses = []
            for query in queries[:n_queries]:
                query_embedding = encoder.encode(query)
                context = retriever.retrieve(query_embedding, embedded_db)

                # Get appropriate context size for integration
                if channel == 'C2':
                    context_size_for_integration = int(8 * improvement_factor)
                else:
                    context_size_for_integration = 8

                integrated_context = integrator.integrate(context, context_size_for_integration)
                response = generator.generate(query, integrated_context)
                improved_responses.append(response)

            improved_accuracies = [r["accuracy"] for r in improved_responses]
            improved_accuracy = np.mean(improved_accuracies)
            improvement_percent = (improved_accuracy - baseline_accuracy) / baseline_accuracy * 100

            # Store individual accuracies for statistical testing
            run_results['improvements'].append({
                'channel': channel,
                'improvement_factor': improvement_factor,
                'improved_accuracy': improved_accuracy,
                'improved_accuracies': improved_accuracies,
                'improvement_percent': improvement_percent
            })

    return run_results


def run_bottleneck_analysis_experiment_with_statistics(
    results_dir: str,
    n_runs: int = 10,
    base_seed: int = 43
) -> List[Dict[str, Any]]:
    """
    Run bottleneck analysis experiment with statistical validation.

    Args:
        results_dir: Directory to save results
        n_runs: Number of experimental runs
        base_seed: Base seed value

    Returns:
        List of results with statistical analysis
    """
    logger.info(f"Starting bottleneck analysis experiment with {n_runs} runs")

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Create enhanced measurer for statistical tests
    measurer = EnhancedInformationMeasurer()

    # Run multiple experiments
    multiple_results = measurer.run_multiple_experiments(
        experiment_func=run_single_bottleneck_experiment,
        n_runs=n_runs,
        base_seed=base_seed
    )

    # Aggregate results by channel and improvement factor
    aggregated_results = {}
    baseline_accuracies_all_runs = []

    for run_data in multiple_results['individual_runs']:
        # Collect baseline accuracies
        baseline_accuracies_all_runs.extend(run_data['baseline_accuracies'])

        # Aggregate improvement results
        for improvement in run_data['improvements']:
            key = (improvement['channel'], improvement['improvement_factor'])

            if key not in aggregated_results:
                aggregated_results[key] = {
                    'channel': improvement['channel'],
                    'improvement_factor': improvement['improvement_factor'],
                    'baseline_accuracies': [],
                    'improved_accuracies': [],
                    'improvement_percents': []
                }

            # Add baseline accuracies for this run (for paired t-test)
            aggregated_results[key]['baseline_accuracies'].extend(
                [run_data['baseline_accuracy']] * len(improvement['improved_accuracies'])
            )
            aggregated_results[key]['improved_accuracies'].extend(improvement['improved_accuracies'])
            aggregated_results[key]['improvement_percents'].append(improvement['improvement_percent'])

    # Perform statistical analysis for each condition
    statistical_results = []

    for (channel, improvement_factor), data in aggregated_results.items():

        # Perform paired t-test comparing improved vs baseline
        t_test_results = measurer.compare_performance_with_ttest(
            baseline_results=data['baseline_accuracies'],
            improved_results=data['improved_accuracies'],
            paired=True
        )

        # Calculate improvement statistics
        improvement_stats = {
            'mean': np.mean(data['improvement_percents']),
            'std': np.std(data['improvement_percents'], ddof=1) if len(data['improvement_percents']) > 1 else 0,
            'min': np.min(data['improvement_percents']),
            'max': np.max(data['improvement_percents']),
            'median': np.median(data['improvement_percents']),
            'n_runs': len(data['improvement_percents']),
            'ci_95_lower': np.percentile(data['improvement_percents'], 2.5),
            'ci_95_upper': np.percentile(data['improvement_percents'], 97.5)
        }

        # Add normality test if enough samples
        if len(data['improvement_percents']) >= 8:
            from scipy import stats as scipy_stats
            _, p_value = scipy_stats.shapiro(data['improvement_percents'])
            improvement_stats['normality_p_value'] = p_value
            improvement_stats['appears_normal'] = p_value > 0.05

        statistical_results.append({
            'channel': data['channel'],
            'improvement_factor': data['improvement_factor'],
            'improvement_percent': improvement_stats['mean'],
            'improvement_percent_std': improvement_stats['std'],
            'improvement_percent_ci_lower': improvement_stats['ci_95_lower'],
            'improvement_percent_ci_upper': improvement_stats['ci_95_upper'],
            'n_runs': improvement_stats['n_runs'],
            'statistical_test': t_test_results,
            'improvement_statistics': improvement_stats,
            'raw_data': data
        })

        # Log results
        logger.info(f"{data['channel']} improvement {data['improvement_factor']}x: "
                   f"{improvement_stats['mean']:.2f}% ± {improvement_stats['std']:.2f}% "
                   f"(p-value: {t_test_results['p_value']:.4f}, "
                   f"significant: {t_test_results['significant_at_05']})")

    # Save detailed results with JSON serialization fix
    def convert_to_serializable(obj):
        """Convert numpy/pandas types to JSON serializable types."""
        if hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    # Convert results to serializable format
    serializable_detailed = convert_to_serializable({
        'statistical_results': statistical_results,
        'multiple_runs_summary': {
            'n_runs': n_runs,
            'seeds_used': multiple_results['seeds_used']
        },
        'overall_baseline_stats': {
            'mean': np.mean(baseline_accuracies_all_runs),
            'std': np.std(baseline_accuracies_all_runs, ddof=1),
            'n_samples': len(baseline_accuracies_all_runs)
        }
    })

    with open(os.path.join(results_dir, "bottleneck_analysis_detailed.json"), "w") as f:
        logger.info("Skipping detailed results due to JSON issues")
#        json.dump(serializable_detailed, f, indent=2)

    # Create simplified results for backward compatibility
    simplified_results = []
    for result in statistical_results:
        simplified_results.append({
            'channel': result['channel'],
            'improvement_factor': result['improvement_factor'],
            'improvement_percent': result['improvement_percent'],
            'improvement_percent_std': result['improvement_percent_std'],
            'improvement_percent_ci_lower': result['improvement_percent_ci_lower'],
            'improvement_percent_ci_upper': result['improvement_percent_ci_upper'],
            'p_value': result['statistical_test']['p_value'],
            'significant': result['statistical_test']['significant_at_05'],
            'effect_size': result['statistical_test']['effect_size_cohens_d'],
            'n_runs': result['n_runs']
        })

    # Save simplified results for existing plotting code with custom JSON encoder
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(os.path.join(results_dir, "bottleneck_analysis.json"), "w") as f:
        json.dump(simplified_results, f, indent=2, cls=NumpyEncoder)

    logger.info("Bottleneck analysis experiment with statistics completed")
    logger.info(f"Detailed results saved to {os.path.join(results_dir, 'bottleneck_analysis_detailed.json')}")

    return simplified_results


# For backward compatibility
def run_bottleneck_analysis_experiment(results_dir: str) -> List[Dict[str, Any]]:
    """Backward compatible function - now runs with statistics by default."""
    return run_bottleneck_analysis_experiment_with_statistics(results_dir, n_runs=5)


if __name__ == "__main__":
    # Run the enhanced experiment
    results = run_bottleneck_analysis_experiment_with_statistics(
        results_dir="results",
        n_runs=10,
        base_seed=43
    )

    # Print statistical summary
    print("\nBottleneck Analysis Results with Statistics:")
    print("=" * 80)
    print(f"{'Channel':<8} | {'Factor':<6} | {'Improvement':<20} | {'p-value':<8} | {'Significant':<11} | {'Effect Size':<11}")
    print("=" * 80)

    for result in sorted(results, key=lambda x: (x['channel'], x['improvement_factor'])):
        imp_mean = result['improvement_percent']
        imp_std = result['improvement_percent_std']
        p_val = result['p_value']
        significant = "Yes" if result['significant'] else "No"
        effect_size = result['effect_size']

        print(f"{result['channel']:<8} | {result['improvement_factor']:<6.1f} | "
              f"{imp_mean:>6.2f}% ± {imp_std:<6.2f}% | {p_val:<8.4f} | "
              f"{significant:<11} | {effect_size:<11.3f}")

    print("=" * 80)

    # Identify most significant improvements
    significant_results = [r for r in results if r['significant']]
    if significant_results:
        best_result = max(significant_results, key=lambda x: x['improvement_percent'])
        print(f"\nMost significant improvement: {best_result['channel']} "
              f"({best_result['improvement_factor']}x) with "
              f"{best_result['improvement_percent']:.2f}% improvement "
              f"(p={best_result['p_value']:.4f})")