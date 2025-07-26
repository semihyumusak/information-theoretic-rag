"""
Enhanced Schema Entropy Experiment Module with Statistical Validation

This module implements the experiment to measure the effect of schema entropy
on RAG performance with proper statistical testing and validation of theoretical bounds.
"""

import os
import json
import logging
import numpy as np
import math
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from scipy import stats as scipy_stats

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
from analysis.information_measurer import InformationMeasurer
from analysis.enhanced_information_measurer import EnhancedInformationMeasurer


def run_single_schema_entropy_experiment(seed: int, **kwargs) -> Dict[str, Any]:
    """
    Run a single instance of schema entropy experiment.

    Args:
        seed: Random seed for this run
        **kwargs: Additional parameters

    Returns:
        Dictionary with results from this run
    """
    schema_entropy_levels = kwargs.get('schema_entropy_levels', [2.0, 4.0, 6.0, 8.0, 10.0])
    embedding_dims = kwargs.get('embedding_dims', [64, 128, 256, 512, 1024])
    n_queries = kwargs.get('n_queries', 50)
    n_rows_per_table = kwargs.get('n_rows_per_table', 1000)

    # Fixed components with this seed
    query_gen = QueryGenerator(seed=seed)
    db_gen = DatabaseGenerator(seed=seed)

    # Results storage for this run
    run_results = []

    for entropy_level in tqdm(schema_entropy_levels, desc=f"Schema entropy levels (seed={seed})", leave=False):
        # Generate schema with specified entropy
        schema_gen = SchemaGenerator(seed=seed)
        schema = schema_gen.generate_schema(entropy_bits=entropy_level)

        # Generate database content
        database = db_gen.generate_database(schema, num_rows_per_table=n_rows_per_table)

        # Generate queries
        queries = query_gen.generate_queries(schema, num_queries=200)

        # Test with different embedding dimensions
        for dim in embedding_dims:
            # Initialize components
            query_encoder = QueryEncoder(embedding_dim=dim, noise_level=0.1, seed=seed)
            db_encoder = DatabaseEncoder(embedding_dim=dim, seed=seed)
            retriever = Retriever(context_size=8)
            integrator = ContextIntegrator(noise_level=0.05, seed=seed)
            generator = Generator(temperature=0.7, seed=seed)

            # Encode database
            embedded_db = db_encoder.encode_database(database)

            # Process queries
            responses = []
            precisions = []
            recalls = []

            for query in queries[:n_queries]:
                query_embedding = query_encoder.encode(query)
                context = retriever.retrieve(query_embedding, embedded_db)

                # Measure retrieval quality
                precision = InformationMeasurer.measure_retrieval_precision(
                    context, query.get("ground_truth", {})
                )
                recall = InformationMeasurer.measure_retrieval_recall(
                    context, query.get("ground_truth", {})
                )

                integrated_context = integrator.integrate(context, 8)
                response = generator.generate(query, integrated_context)

                responses.append(response)
                precisions.append(precision)
                recalls.append(recall)

            # Calculate metrics
            accuracy = np.mean([r["accuracy"] for r in responses])
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)

            # Calculate theoretical capacity bound
            # min(log₂(d), H(S))
            theory_bound = min(np.log2(dim), entropy_level)

            # Store individual query results for statistical analysis
            individual_accuracies = [r["accuracy"] for r in responses]
            individual_precisions = precisions
            individual_recalls = recalls

            result = {
                "schema_entropy": entropy_level,
                "embedding_dim": dim,
                "accuracy": accuracy,
                "precision": avg_precision,
                "recall": avg_recall,
                "theoretical_bound": theory_bound,
                "individual_accuracies": individual_accuracies,
                "individual_precisions": individual_precisions,
                "individual_recalls": individual_recalls,
                "n_queries": n_queries,
                "seed": seed
            }

            run_results.append(result)

    return {'results': run_results, 'seed': seed}


def run_schema_entropy_experiment_with_statistics(
    results_dir: str,
    n_runs: int = 10,
    base_seed: int = 45
) -> List[Dict[str, Any]]:
    """
    Run schema entropy experiment with statistical validation.

    Args:
        results_dir: Directory to save results
        n_runs: Number of experimental runs
        base_seed: Base seed value

    Returns:
        List of results with statistical analysis
    """
    logger.info(f"Starting schema entropy experiment with {n_runs} runs")

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Create enhanced measurer for statistical tests
    measurer = EnhancedInformationMeasurer()

    # Run multiple experiments
    multiple_results = measurer.run_multiple_experiments(
        experiment_func=run_single_schema_entropy_experiment,
        n_runs=n_runs,
        base_seed=base_seed
    )

    # Aggregate results by schema entropy and embedding dimension
    aggregated_results = {}

    for run_data in multiple_results['individual_runs']:
        for result in run_data['results']:
            key = (result['schema_entropy'], result['embedding_dim'])
            
            if key not in aggregated_results:
                aggregated_results[key] = {
                    'schema_entropy': result['schema_entropy'],
                    'embedding_dim': result['embedding_dim'],
                    'theoretical_bound': result['theoretical_bound'],
                    'accuracies': [],
                    'precisions': [],
                    'recalls': [],
                    'individual_accuracies': [],
                    'individual_precisions': [],
                    'individual_recalls': []
                }
            
            aggregated_results[key]['accuracies'].append(result['accuracy'])
            aggregated_results[key]['precisions'].append(result['precision'])
            aggregated_results[key]['recalls'].append(result['recall'])
            aggregated_results[key]['individual_accuracies'].extend(result['individual_accuracies'])
            aggregated_results[key]['individual_precisions'].extend(result['individual_precisions'])
            aggregated_results[key]['individual_recalls'].extend(result['individual_recalls'])

    # Perform statistical analysis
    statistical_results = []

    for key, data in aggregated_results.items():
        schema_entropy, embedding_dim = key
        
        # Calculate statistics for each metric
        metrics = ['accuracies', 'precisions', 'recalls']
        individual_metrics = ['individual_accuracies', 'individual_precisions', 'individual_recalls']
        
        result = {
            'schema_entropy': schema_entropy,
            'embedding_dim': embedding_dim,
            'theoretical_bound': data['theoretical_bound'],
            'n_runs': len(data['accuracies'])
        }

        for metric, individual_metric in zip(metrics, individual_metrics):
            values = data[metric]
            individual_values = data[individual_metric]
            
            # Basic statistics across runs
            stats = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1) if len(values) > 1 else 0,
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'ci_95_lower': np.percentile(values, 2.5),
                'ci_95_upper': np.percentile(values, 97.5)
            }
            
            # Statistical tests
            if len(values) > 1:
                # Test if performance is significantly different from zero
                t_stat, p_value = scipy_stats.ttest_1samp(values, 0)
                stats['t_statistic'] = t_stat
                stats['p_value_vs_zero'] = p_value
                stats['significant_performance'] = p_value < 0.05 and stats['mean'] > 0
            
            # Within-run consistency (using individual query results)
            if len(individual_values) > 1:
                stats['within_run_std'] = np.std(individual_values, ddof=1)
                stats['within_run_coefficient_of_variation'] = stats['within_run_std'] / np.mean(individual_values) if np.mean(individual_values) > 0 else np.inf
            
            metric_name = metric.replace('accuracies', 'accuracy').replace('precisions', 'precision').replace('recalls', 'recall')
            result[metric_name] = stats['mean']
            result[f'{metric_name}_std'] = stats['std']
            result[f'{metric_name}_ci_lower'] = stats['ci_95_lower']
            result[f'{metric_name}_ci_upper'] = stats['ci_95_upper']
            result[f'{metric_name}_statistics'] = stats

        # Test theoretical bound validation
        recall_values = data['recalls']
        theoretical_bound = data['theoretical_bound']
        
        # Check if recall consistently respects theoretical bound (with 5% tolerance)
        bound_violations = [r for r in recall_values if r > theoretical_bound * 1.05]
        result['bound_violations'] = len(bound_violations)
        result['bound_violation_rate'] = len(bound_violations) / len(recall_values)
        result['bound_respected'] = len(bound_violations) == 0
        
        # Test if performance is significantly below theoretical bound
        if len(recall_values) > 1:
            t_stat, p_value = scipy_stats.ttest_1samp(recall_values, theoretical_bound)
            result['bound_test_t_statistic'] = t_stat
            result['bound_test_p_value'] = p_value
            result['significantly_below_bound'] = p_value < 0.05 and np.mean(recall_values) < theoretical_bound
        
        # Calculate efficiency ratio (actual performance / theoretical bound)
        if theoretical_bound > 0:
            efficiency_ratios = [r / theoretical_bound for r in recall_values]
            result['efficiency_ratio'] = np.mean(efficiency_ratios)
            result['efficiency_ratio_std'] = np.std(efficiency_ratios, ddof=1) if len(efficiency_ratios) > 1 else 0
            result['efficiency_ratio_ci_lower'] = np.percentile(efficiency_ratios, 2.5)
            result['efficiency_ratio_ci_upper'] = np.percentile(efficiency_ratios, 97.5)

        statistical_results.append(result)

    # Analyze dimension vs entropy limiting factors
    dimension_limited_results = []
    entropy_limited_results = []

    for result in statistical_results:
        log_dim = np.log2(result['embedding_dim'])
        entropy = result['schema_entropy']
        
        if log_dim < entropy:
            dimension_limited_results.append(result)
        else:
            entropy_limited_results.append(result)

    # Statistical comparison between limiting factors
    limiting_factor_analysis = {}
    
    if dimension_limited_results and entropy_limited_results:
        dim_limited_ratios = [r['efficiency_ratio'] for r in dimension_limited_results if 'efficiency_ratio' in r]
        entropy_limited_ratios = [r['efficiency_ratio'] for r in entropy_limited_results if 'efficiency_ratio' in r]
        
        if dim_limited_ratios and entropy_limited_ratios:
            # Test if there's a significant difference between dimension-limited and entropy-limited cases
            t_stat, p_value = scipy_stats.ttest_ind(dim_limited_ratios, entropy_limited_ratios)
            
            limiting_factor_analysis = {
                'dimension_limited_count': len(dimension_limited_results),
                'entropy_limited_count': len(entropy_limited_results),
                'dimension_limited_efficiency_mean': np.mean(dim_limited_ratios),
                'entropy_limited_efficiency_mean': np.mean(entropy_limited_ratios),
                'dimension_limited_efficiency_std': np.std(dim_limited_ratios, ddof=1),
                'entropy_limited_efficiency_std': np.std(entropy_limited_ratios, ddof=1),
                'comparison_t_statistic': t_stat,
                'comparison_p_value': p_value,
                'significant_difference': p_value < 0.05
            }

    # Convert results to serializable format
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

    # Save detailed results with statistical analysis
    detailed_results = convert_to_serializable({
        'statistical_results': statistical_results,
        'limiting_factor_analysis': limiting_factor_analysis,
        'multiple_runs_summary': {
            'n_runs': n_runs,
            'seeds_used': multiple_results['seeds_used']
        },
        'experimental_parameters': {
            'schema_entropy_levels': [2.0, 4.0, 6.0, 8.0, 10.0],
            'embedding_dims': [64, 128, 256, 512, 1024],
            'n_queries_per_run': 50,
            'n_rows_per_table': 1000
        }
    })

    # Custom JSON encoder for numpy types
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

    # Save detailed results
    with open(os.path.join(results_dir, "schema_entropy_detailed.json"), "w") as f:
        json.dump(detailed_results, f, indent=2, cls=NumpyEncoder)

    # Create simplified results for backward compatibility
    simplified_results = []
    
    for result in statistical_results:
        simplified_result = {
            'schema_entropy': result['schema_entropy'],
            'embedding_dim': result['embedding_dim'],
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'theoretical_bound': result['theoretical_bound'],
            'accuracy_std': result['accuracy_std'],
            'precision_std': result['precision_std'],
            'recall_std': result['recall_std'],
            'accuracy_ci_lower': result['accuracy_ci_lower'],
            'accuracy_ci_upper': result['accuracy_ci_upper'],
            'precision_ci_lower': result['precision_ci_lower'],
            'precision_ci_upper': result['precision_ci_upper'],
            'recall_ci_lower': result['recall_ci_lower'],
            'recall_ci_upper': result['recall_ci_upper'],
            'bound_respected': result['bound_respected'],
            'bound_violation_rate': result['bound_violation_rate'],
            'efficiency_ratio': result.get('efficiency_ratio', 0),
            'efficiency_ratio_std': result.get('efficiency_ratio_std', 0),
            'significantly_below_bound': result.get('significantly_below_bound', False),
            'n_runs': result['n_runs']
        }
        simplified_results.append(simplified_result)

    # Save simplified results
    with open(os.path.join(results_dir, "schema_entropy.json"), "w") as f:
        json.dump(simplified_results, f, indent=2, cls=NumpyEncoder)

    logger.info("Schema entropy experiment with statistics completed")
    logger.info(f"Results saved to {os.path.join(results_dir, 'schema_entropy.json')}")
    logger.info(f"Detailed results saved to {os.path.join(results_dir, 'schema_entropy_detailed.json')}")

    return simplified_results


# For backward compatibility
def run_schema_entropy_experiment(results_dir: str) -> List[Dict[str, Any]]:
    """Backward compatible function - now runs with statistics by default."""
    return run_schema_entropy_experiment_with_statistics(results_dir, n_runs=5)


if __name__ == "__main__":
    # Run the enhanced experiment
    results = run_schema_entropy_experiment_with_statistics(
        results_dir="results",
        n_runs=10,
        base_seed=45
    )

    # Print statistical summary
    print("\nSchema Entropy Results with Statistics:")
    print("=" * 120)
    print(f"{'Schema Entropy':<15} | {'Embedding Dim':<15} | {'Recall':<15} | {'Theoretical Bound':<17} | {'Efficiency Ratio':<17} | {'Bound Respected':<15}")
    print("=" * 120)

    # Sort results by schema entropy and embedding dimension
    sorted_results = sorted(results, key=lambda x: (x['schema_entropy'], x['embedding_dim']))

    for result in sorted_results:
        recall_mean = result['recall']
        recall_std = result['recall_std']
        bound = result['theoretical_bound']
        efficiency = result['efficiency_ratio']
        efficiency_std = result['efficiency_ratio_std']
        bound_respected = "Yes" if result['bound_respected'] else "No"
        
        print(f"{result['schema_entropy']:<15.1f} | {result['embedding_dim']:<15d} | "
              f"{recall_mean:>6.4f}±{recall_std:<6.4f} | {bound:<17.4f} | "
              f"{efficiency:>6.4f}±{efficiency_std:<6.4f} | {bound_respected:<15}")

    print("=" * 120)

    # Summary statistics
    bounds_respected = sum(1 for r in results if r['bound_respected'])
    total_conditions = len(results)
    
    print(f"\nTheoretical Bound Validation:")
    print(f"  Bounds respected: {bounds_respected}/{total_conditions} ({100*bounds_respected/total_conditions:.1f}%)")
    
    # Check if any results significantly exceed bounds
    significant_violations = [r for r in results if r['bound_violation_rate'] > 0.1]  # More than 10% violations
    if significant_violations:
        print(f"  Conditions with >10% bound violations: {len(significant_violations)}")
        for r in significant_violations:
            print(f"    Schema entropy {r['schema_entropy']}, dim {r['embedding_dim']}: {r['bound_violation_rate']*100:.1f}% violations")
    else:
        print(f"  No conditions with significant bound violations detected")

    # Limiting factor analysis
    dimension_limited = [r for r in results if np.log2(r['embedding_dim']) < r['schema_entropy']]
    entropy_limited = [r for r in results if np.log2(r['embedding_dim']) >= r['schema_entropy']]

    if dimension_limited and entropy_limited:
        dim_limited_efficiency = np.mean([r['efficiency_ratio'] for r in dimension_limited])
        entropy_limited_efficiency = np.mean([r['efficiency_ratio'] for r in entropy_limited])
        
        print(f"\nLimiting Factor Analysis:")
        print(f"  Dimension-limited cases: {len(dimension_limited)} (avg efficiency: {dim_limited_efficiency:.4f})")
        print(f"  Entropy-limited cases: {len(entropy_limited)} (avg efficiency: {entropy_limited_efficiency:.4f})")
        
        # Statistical test
        from scipy import stats
        dim_ratios = [r['efficiency_ratio'] for r in dimension_limited]
        entropy_ratios = [r['efficiency_ratio'] for r in entropy_limited]
        t_stat, p_value = stats.ttest_ind(dim_ratios, entropy_ratios)
        
        if p_value < 0.05:
            print(f"  Significant difference between limiting factors (p={p_value:.4f})")
        else:
            print(f"  No significant difference between limiting factors (p={p_value:.4f})")

    # Performance consistency analysis
    high_variance_conditions = [r for r in results if r['recall_std'] / r['recall'] > 0.2]  # CV > 20%
    if high_variance_conditions:
        print(f"\nHigh Variance Conditions (CV > 20%):")
        for r in high_variance_conditions:
            cv = r['recall_std'] / r['recall']
            print(f"  Schema entropy {r['schema_entropy']}, dim {r['embedding_dim']}: CV = {cv:.3f}")
    else:
        print(f"\nAll conditions show good consistency (CV ≤ 20%)")

    print("=" * 120)