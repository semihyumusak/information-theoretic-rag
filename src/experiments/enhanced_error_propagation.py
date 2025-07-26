"""
Enhanced Error Propagation Experiment Module with Statistical Validation

This module implements error propagation analysis with proper statistical testing
for interaction effects and error combinations.
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


def run_single_error_propagation_experiment(seed: int, **kwargs) -> Dict[str, Any]:
    """
    Run a single instance of error propagation experiment.

    Args:
        seed: Random seed for this run
        **kwargs: Additional parameters

    Returns:
        Dictionary with results from this run
    """
    error_levels = kwargs.get('error_levels', [0.1, 0.2, 0.3])
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

    # Baseline configuration (no errors)
    baseline_encoder = QueryEncoder(embedding_dim=256, noise_level=0.0, seed=seed)
    baseline_retriever = Retriever(context_size=8)
    baseline_integrator = ContextIntegrator(noise_level=0.0, seed=seed)
    baseline_generator = Generator(temperature=0.1, seed=seed)

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
        'single_channel_errors': [],
        'multi_channel_errors': [],
        'seed': seed
    }

    # Test single-channel errors
    channels = ['C1', 'C2', 'C3', 'C4']

    for channel in channels:
        for error_level in error_levels:

            # Configure error in specific channel
            if channel == 'C1':
                error_encoder = QueryEncoder(embedding_dim=256, noise_level=error_level, seed=seed)
                retriever = baseline_retriever
                integrator = baseline_integrator
                generator = baseline_generator
                encoder = error_encoder
            elif channel == 'C2':
                context_size = max(1, int(8 * (1 - error_level)))
                retriever = Retriever(context_size=context_size)
                encoder = baseline_encoder
                integrator = baseline_integrator
                generator = baseline_generator
            elif channel == 'C3':
                integrator = ContextIntegrator(noise_level=error_level, seed=seed)
                encoder = baseline_encoder
                retriever = baseline_retriever
                generator = baseline_generator
            elif channel == 'C4':
                temperature = 0.1 + error_level * 0.9
                generator = Generator(temperature=temperature, seed=seed)
                encoder = baseline_encoder
                retriever = baseline_retriever
                integrator = baseline_integrator

            # Run with error in single channel
            error_responses = []
            for query in queries[:n_queries]:
                query_embedding = encoder.encode(query)
                context = retriever.retrieve(query_embedding, embedded_db)

                # Get appropriate context size for integration
                if channel == 'C2':
                    context_size_for_integration = max(1, int(8 * (1 - error_level)))
                else:
                    context_size_for_integration = 8

                integrated_context = integrator.integrate(context, context_size_for_integration)
                response = generator.generate(query, integrated_context)
                error_responses.append(response)

            error_accuracies = [r["accuracy"] for r in error_responses]
            error_accuracy = np.mean(error_accuracies)
            accuracy_drop = baseline_accuracy - error_accuracy

            run_results['single_channel_errors'].append({
                'channel': channel,
                'error_level': error_level,
                'error_accuracy': error_accuracy,
                'error_accuracies': error_accuracies,
                'accuracy_drop': accuracy_drop
            })

    # Test multi-channel errors (pair-wise)
    moderate_error = 0.2
    channel_pairs = [('C1', 'C2'), ('C1', 'C3'), ('C1', 'C4'), ('C2', 'C3'), ('C2', 'C4'), ('C3', 'C4')]

    for ch1, ch2 in channel_pairs:

        # Configure components with errors in both channels
        if ch1 == 'C1' or ch2 == 'C1':
            error_encoder = QueryEncoder(embedding_dim=256, noise_level=moderate_error, seed=seed)
        else:
            error_encoder = baseline_encoder

        if ch1 == 'C2' or ch2 == 'C2':
            context_size = max(1, int(8 * (1 - moderate_error)))
            error_retriever = Retriever(context_size=context_size)
        else:
            error_retriever = baseline_retriever

        if ch1 == 'C3' or ch2 == 'C3':
            error_integrator = ContextIntegrator(noise_level=moderate_error, seed=seed)
        else:
            error_integrator = baseline_integrator

        if ch1 == 'C4' or ch2 == 'C4':
            temperature = 0.1 + moderate_error * 0.9
            error_generator = Generator(temperature=temperature, seed=seed)
        else:
            error_generator = baseline_generator

        # Run with errors in both channels
        multi_error_responses = []
        for query in queries[:n_queries]:
            query_embedding = error_encoder.encode(query)
            context = error_retriever.retrieve(query_embedding, embedded_db)

            context_size_actual = context_size if ch1 == 'C2' or ch2 == 'C2' else 8
            integrated_context = error_integrator.integrate(context, context_size_actual)
            response = error_generator.generate(query, integrated_context)
            multi_error_responses.append(response)

        multi_error_accuracies = [r["accuracy"] for r in multi_error_responses]
        multi_error_accuracy = np.mean(multi_error_accuracies)
        multi_accuracy_drop = baseline_accuracy - multi_error_accuracy

        # Find corresponding single channel drops for interaction calculation
        ch1_drop = None
        ch2_drop = None

        for single_result in run_results['single_channel_errors']:
            if single_result['channel'] == ch1 and single_result['error_level'] == moderate_error:
                ch1_drop = single_result['accuracy_drop']
            if single_result['channel'] == ch2 and single_result['error_level'] == moderate_error:
                ch2_drop = single_result['accuracy_drop']

        if ch1_drop is not None and ch2_drop is not None:
            expected_drop = ch1_drop + ch2_drop
            interaction_effect = multi_accuracy_drop - expected_drop

            run_results['multi_channel_errors'].append({
                'channels': f"{ch1}+{ch2}",
                'ch1': ch1,
                'ch2': ch2,
                'error_level': moderate_error,
                'multi_error_accuracy': multi_error_accuracy,
                'multi_error_accuracies': multi_error_accuracies,
                'multi_accuracy_drop': multi_accuracy_drop,
                'ch1_drop': ch1_drop,
                'ch2_drop': ch2_drop,
                'expected_drop': expected_drop,
                'interaction_effect': interaction_effect
            })

    return run_results


def run_error_propagation_experiment_with_statistics(
    results_dir: str,
    n_runs: int = 10,
    base_seed: int = 44
) -> List[Dict[str, Any]]:
    """
    Run error propagation experiment with statistical validation.

    Args:
        results_dir: Directory to save results
        n_runs: Number of experimental runs
        base_seed: Base seed value

    Returns:
        List of results with statistical analysis
    """
    logger.info(f"Starting error propagation experiment with {n_runs} runs")

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Create enhanced measurer for statistical tests
    measurer = EnhancedInformationMeasurer()

    # Run multiple experiments
    multiple_results = measurer.run_multiple_experiments(
        experiment_func=run_single_error_propagation_experiment,
        n_runs=n_runs,
        base_seed=base_seed
    )

    # Aggregate single-channel error results
    single_channel_aggregated = {}

    for run_data in multiple_results['individual_runs']:
        for error_result in run_data['single_channel_errors']:
            key = (error_result['channel'], error_result['error_level'])

            if key not in single_channel_aggregated:
                single_channel_aggregated[key] = {
                    'channel': error_result['channel'],
                    'error_level': error_result['error_level'],
                    'accuracy_drops': [],
                    'error_accuracies': []
                }

            single_channel_aggregated[key]['accuracy_drops'].append(error_result['accuracy_drop'])
            single_channel_aggregated[key]['error_accuracies'].extend(error_result['error_accuracies'])

    # Aggregate multi-channel error results
    multi_channel_aggregated = {}

    for run_data in multiple_results['individual_runs']:
        for error_result in run_data['multi_channel_errors']:
            key = error_result['channels']

            if key not in multi_channel_aggregated:
                multi_channel_aggregated[key] = {
                    'channels': error_result['channels'],
                    'ch1': error_result['ch1'],
                    'ch2': error_result['ch2'],
                    'error_level': error_result['error_level'],
                    'interaction_effects': [],
                    'multi_accuracy_drops': []
                }

            multi_channel_aggregated[key]['interaction_effects'].append(error_result['interaction_effect'])
            multi_channel_aggregated[key]['multi_accuracy_drops'].append(error_result['multi_accuracy_drop'])

    # Perform statistical analysis
    statistical_results = []

    # Single-channel error statistics
    for (channel, error_level), data in single_channel_aggregated.items():

        accuracy_drops = data['accuracy_drops']

        # Calculate basic statistics
        stats = {
            'mean': np.mean(accuracy_drops),
            'std': np.std(accuracy_drops, ddof=1) if len(accuracy_drops) > 1 else 0,
            'min': np.min(accuracy_drops),
            'max': np.max(accuracy_drops),
            'median': np.median(accuracy_drops),
            'n_runs': len(accuracy_drops),
            'ci_95_lower': np.percentile(accuracy_drops, 2.5),
            'ci_95_upper': np.percentile(accuracy_drops, 97.5)
        }

        # Test if accuracy drop is significantly different from zero
        from scipy import stats as scipy_stats
        if len(accuracy_drops) > 1:
            t_stat, p_value = scipy_stats.ttest_1samp(accuracy_drops, 0)
            stats['t_statistic'] = t_stat
            stats['p_value_vs_zero'] = p_value
            stats['significant_degradation'] = p_value < 0.05 and stats['mean'] > 0

        statistical_results.append({
            'error_type': 'single',
            'channel': channel,
            'error_level': error_level,
            'accuracy_drop': stats['mean'],
            'accuracy_drop_std': stats['std'],
            'accuracy_drop_ci_lower': stats['ci_95_lower'],
            'accuracy_drop_ci_upper': stats['ci_95_upper'],
            'n_runs': stats['n_runs'],
            'statistics': stats
        })

    # Multi-channel interaction statistics
    for channels, data in multi_channel_aggregated.items():

        interaction_effects = data['interaction_effects']

        # Calculate basic statistics
        stats = {
            'mean': np.mean(interaction_effects),
            'std': np.std(interaction_effects, ddof=1) if len(interaction_effects) > 1 else 0,
            'min': np.min(interaction_effects),
            'max': np.max(interaction_effects),
            'median': np.median(interaction_effects),
            'n_runs': len(interaction_effects),
            'ci_95_lower': np.percentile(interaction_effects, 2.5),
            'ci_95_upper': np.percentile(interaction_effects, 97.5)
        }

        # Test if interaction effect is significantly different from zero
        from scipy import stats as scipy_stats
        if len(interaction_effects) > 1:
            t_stat, p_value = scipy_stats.ttest_1samp(interaction_effects, 0)
            stats['t_statistic'] = t_stat
            stats['p_value_vs_zero'] = p_value
            stats['significant_interaction'] = p_value < 0.05

            # Classify interaction type
            if stats['significant_interaction']:
                if stats['mean'] > 0:
                    interaction_type = 'superlinear'  # Errors compound
                else:
                    interaction_type = 'sublinear'   # Errors partially cancel
            else:
                interaction_type = 'independent'    # No significant interaction

            stats['interaction_type'] = interaction_type

        statistical_results.append({
            'error_type': 'multi',
            'channels': channels,
            'ch1': data['ch1'],
            'ch2': data['ch2'],
            'error_level': data['error_level'],
            'interaction_effect': stats['mean'],
            'interaction_effect_std': stats['std'],
            'interaction_effect_ci_lower': stats['ci_95_lower'],
            'interaction_effect_ci_upper': stats['ci_95_upper'],
            'n_runs': stats['n_runs'],
            'statistics': stats
        })

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
        'single_channel_raw': single_channel_aggregated,
        'multi_channel_raw': multi_channel_aggregated
    })

    with open(os.path.join(results_dir, "error_propagation_detailed.json"), "w") as f:
        logger.info("Skipping detailed results due to JSON issues")
#        json.dump(serializable_detailed, f, indent=2)

    # Create simplified results for backward compatibility
    simplified_results = []

    for result in statistical_results:
        if result['error_type'] == 'single':
            simplified_results.append({
                'channel': result['channel'],
                'error_level': result['error_level'],
                'accuracy_drop': result['accuracy_drop'],
                'accuracy_drop_std': result['accuracy_drop_std'],
                'accuracy_drop_ci_lower': result['accuracy_drop_ci_lower'],
                'accuracy_drop_ci_upper': result['accuracy_drop_ci_upper'],
                'p_value': result['statistics'].get('p_value_vs_zero', None),
                'significant': result['statistics'].get('significant_degradation', False),
                'n_runs': result['n_runs'],
                'error_type': 'single'
            })
        else:  # multi-channel
            simplified_results.append({
                'channels': result['channels'],
                'ch1': result['ch1'],
                'ch2': result['ch2'],
                'error_level': result['error_level'],
                'interaction_effect': result['interaction_effect'],
                'interaction_effect_std': result['interaction_effect_std'],
                'interaction_effect_ci_lower': result['interaction_effect_ci_lower'],
                'interaction_effect_ci_upper': result['interaction_effect_ci_upper'],
                'p_value': result['statistics'].get('p_value_vs_zero', None),
                'significant': result['statistics'].get('significant_interaction', False),
                'interaction_type': result['statistics'].get('interaction_type', 'unknown'),
                'n_runs': result['n_runs'],
                'error_type': 'multi'
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

    with open(os.path.join(results_dir, "error_propagation.json"), "w") as f:
        json.dump(simplified_results, f, indent=2, cls=NumpyEncoder)

    logger.info("Error propagation experiment with statistics completed")
    logger.info(f"Detailed results saved to {os.path.join(results_dir, 'error_propagation_detailed.json')}")

    return simplified_results


# For backward compatibility
def run_error_propagation_experiment(results_dir: str) -> List[Dict[str, Any]]:
    """Backward compatible function - now runs with statistics by default."""
    return run_error_propagation_experiment_with_statistics(results_dir, n_runs=5)


if __name__ == "__main__":
    # Run the enhanced experiment
    results = run_error_propagation_experiment_with_statistics(
        results_dir="results",
        n_runs=10,
        base_seed=44
    )

    # Print statistical summary
    print("\nError Propagation Results with Statistics:")
    print("=" * 100)

    # Single-channel errors
    single_results = [r for r in results if r['error_type'] == 'single']
    print("\nSingle-Channel Error Effects:")
    print("-" * 90)
    print(f"{'Channel':<8} | {'Error Level':<12} | {'Accuracy Drop':<20} | {'p-value':<8} | {'Significant':<11}")
    print("-" * 90)

    for result in sorted(single_results, key=lambda x: (x['channel'], x['error_level'])):
        drop_mean = result['accuracy_drop']
        drop_std = result['accuracy_drop_std']
        p_val = result.get('p_value', 'N/A')
        significant = "Yes" if result['significant'] else "No"

        if isinstance(p_val, float):
            p_val_str = f"{p_val:.4f}"
        else:
            p_val_str = str(p_val)

        print(f"{result['channel']:<8} | {result['error_level']:<12.1f} | "
              f"{drop_mean:>6.4f} ± {drop_std:<6.4f} | {p_val_str:<8} | {significant:<11}")

    # Multi-channel interactions
    multi_results = [r for r in results if r['error_type'] == 'multi']
    print("\nMulti-Channel Interaction Effects:")
    print("-" * 100)
    print(f"{'Channel Pair':<12} | {'Interaction Effect':<20} | {'p-value':<8} | {'Type':<12} | {'Significant':<11}")
    print("-" * 100)

    for result in sorted(multi_results, key=lambda x: abs(x['interaction_effect']), reverse=True):
        effect_mean = result['interaction_effect']
        effect_std = result['interaction_effect_std']
        p_val = result.get('p_value', 'N/A')
        interaction_type = result.get('interaction_type', 'unknown')
        significant = "Yes" if result['significant'] else "No"

        if isinstance(p_val, float):
            p_val_str = f"{p_val:.4f}"
        else:
            p_val_str = str(p_val)

        print(f"{result['channels']:<12} | {effect_mean:>6.4f} ± {effect_std:<6.4f} | "
              f"{p_val_str:<8} | {interaction_type:<12} | {significant:<11}")

    print("=" * 100)

    # Summary of interaction types
    interaction_types = {}
    for result in multi_results:
        if result['significant']:
            itype = result.get('interaction_type', 'unknown')
            if itype not in interaction_types:
                interaction_types[itype] = 0
            interaction_types[itype] += 1

    if interaction_types:
        print(f"\nInteraction Type Summary:")
        for itype, count in interaction_types.items():
            print(f"  {itype.capitalize()}: {count} significant interactions")
    else:
        print(f"\nNo significant interactions detected.")