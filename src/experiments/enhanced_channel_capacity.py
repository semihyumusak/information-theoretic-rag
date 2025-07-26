"""
Enhanced Channel Capacity Experiment Module with Statistical Validation

This module implements the experiment to measure the capacity of each channel
with proper statistical analysis including multiple runs and confidence intervals.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

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


def run_single_channel_capacity_experiment(seed: int, **kwargs) -> Dict[str, Any]:
    """
    Run a single instance of channel capacity experiment.

    Args:
        seed: Random seed for this run
        **kwargs: Additional parameters

    Returns:
        Dictionary with results from this run
    """
    # Experimental parameters
    embedding_dims = kwargs.get('embedding_dims', [64, 128, 256, 512, 1024])
    context_sizes = kwargs.get('context_sizes', [2, 4, 8, 16, 32])
    temperatures = kwargs.get('temperatures', [0.1, 0.3, 0.5, 0.7, 0.9])
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

    # Create enhanced information measurer
    measurer = EnhancedInformationMeasurer()

    # Initialize results storage for this run
    run_results = []

    # 1. Measure C1: Query Encoding Channel capacity
    logger.info(f"Measuring C1 capacity (seed={seed})")

    for dim in embedding_dims:
        # Initialize encoder with this dimension and seed
        encoder = QueryEncoder(embedding_dim=dim, noise_level=0.1, seed=seed)

        # Process queries
        query_embeddings = []
        query_texts = []

        for query in queries[:n_queries]:
            embedding = encoder.encode(query)
            query_embeddings.append(embedding)
            query_texts.append(query["text"])

        # Convert text to numerical representation
        vectorizer = CountVectorizer(max_features=100)
        text_features = vectorizer.fit_transform(query_texts).toarray()

        # Measure mutual information with bootstrap
        mi_stats = measurer.estimate_mutual_information_with_bootstrap(
            text_features, np.array(query_embeddings), n_bootstrap=50
        )

        run_results.append({
            "channel": "C1",
            "parameter_name": "embedding_dim",
            "parameter_value": dim,
            "mi_mean": mi_stats['mean'],
            "mi_std": mi_stats['std'],
            "mi_ci_lower": mi_stats['ci_lower'],
            "mi_ci_upper": mi_stats['ci_upper'],
            "seed": seed
        })

    # 2. Measure C2: Retrieval Channel capacity
    logger.info(f"Measuring C2 capacity (seed={seed})")

    # Encode database
    db_encoder = DatabaseEncoder(embedding_dim=256, seed=seed)
    embedded_db = db_encoder.encode_database(database)

    # Fix query encoder
    query_encoder = QueryEncoder(embedding_dim=256, noise_level=0.1, seed=seed)

    for context_size in context_sizes:
        # Initialize retriever
        retriever = Retriever(context_size=context_size)

        # Process queries
        query_embeddings = []
        retrieved_contexts = []

        for query in queries[:n_queries]:
            query_embedding = query_encoder.encode(query)
            context = retriever.retrieve(query_embedding, embedded_db)

            query_embeddings.append(query_embedding)

            # Create context vector
            context_vector = np.zeros(256)
            for item in context:
                context_vector += item["embedding"]
            if len(context) > 0:
                context_vector /= len(context)

            retrieved_contexts.append(context_vector)

        # Measure mutual information with bootstrap
        mi_stats = measurer.estimate_mutual_information_with_bootstrap(
            np.array(query_embeddings), np.array(retrieved_contexts), n_bootstrap=50
        )

        run_results.append({
            "channel": "C2",
            "parameter_name": "context_size",
            "parameter_value": context_size,
            "mi_mean": mi_stats['mean'],
            "mi_std": mi_stats['std'],
            "mi_ci_lower": mi_stats['ci_lower'],
            "mi_ci_upper": mi_stats['ci_upper'],
            "seed": seed
        })

    # 3. Measure C3: Context Integration Channel capacity
    logger.info(f"Measuring C3 capacity (seed={seed})")

    integrator = ContextIntegrator(noise_level=0.05, seed=seed)

    for context_size in context_sizes:
        # Initialize retriever with large context
        retriever = Retriever(context_size=32)

        # Process queries
        retrieved_contexts = []
        integrated_contexts = []

        for query in queries[:n_queries]:
            query_embedding = query_encoder.encode(query)
            context = retriever.retrieve(query_embedding, embedded_db)

            # Integrate context
            integrated_context = integrator.integrate(context, context_size)

            # Create feature vectors
            context_vector = np.zeros(256)
            for item in context:
                context_vector += item["embedding"]
            if len(context) > 0:
                context_vector /= len(context)

            integrated_vector = np.zeros(256)
            for item in integrated_context["elements"]:
                integrated_vector += item["embedding"]
            if len(integrated_context["elements"]) > 0:
                integrated_vector /= len(integrated_context["elements"])

            retrieved_contexts.append(context_vector)
            integrated_contexts.append(integrated_vector)

        # Measure mutual information with bootstrap
        mi_stats = measurer.estimate_mutual_information_with_bootstrap(
            np.array(retrieved_contexts), np.array(integrated_contexts), n_bootstrap=50
        )

        run_results.append({
            "channel": "C3",
            "parameter_name": "context_size",
            "parameter_value": context_size,
            "mi_mean": mi_stats['mean'],
            "mi_std": mi_stats['std'],
            "mi_ci_lower": mi_stats['ci_lower'],
            "mi_ci_upper": mi_stats['ci_upper'],
            "seed": seed
        })

    # 4. Measure C4: Generation Channel capacity
    logger.info(f"Measuring C4 capacity (seed={seed})")

    # Fix retriever and integrator
    retriever = Retriever(context_size=8)
    integrator = ContextIntegrator(noise_level=0.05, seed=seed)

    for temp in temperatures:
        # Initialize generator
        generator = Generator(temperature=temp, seed=seed)

        # Process queries
        integrated_contexts = []
        response_accuracies = []

        for query in queries[:n_queries]:
            query_embedding = query_encoder.encode(query)
            context = retriever.retrieve(query_embedding, embedded_db)
            integrated_context = integrator.integrate(context, 8)
            response = generator.generate(query, integrated_context)

            # Extract features
            integrated_vector = np.zeros(256)
            for item in integrated_context["elements"]:
                integrated_vector += item["embedding"]
            if len(integrated_context["elements"]) > 0:
                integrated_vector /= len(integrated_context["elements"])

            integrated_contexts.append(integrated_vector)
            response_accuracies.append([response["accuracy"]])

        # Measure mutual information with bootstrap
        mi_stats = measurer.estimate_mutual_information_with_bootstrap(
            np.array(integrated_contexts), np.array(response_accuracies), n_bootstrap=50
        )

        run_results.append({
            "channel": "C4",
            "parameter_name": "temperature",
            "parameter_value": temp,
            "mi_mean": mi_stats['mean'],
            "mi_std": mi_stats['std'],
            "mi_ci_lower": mi_stats['ci_lower'],
            "mi_ci_upper": mi_stats['ci_upper'],
            "seed": seed
        })

    return {"results": run_results, "seed": seed}


def run_channel_capacity_experiment_with_statistics(
    results_dir: str,
    n_runs: int = 10,
    base_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Run channel capacity experiment with multiple runs for statistical analysis.

    Args:
        results_dir: Directory to save results
        n_runs: Number of experimental runs
        base_seed: Base seed value

    Returns:
        List of aggregated results with statistics
    """
    logger.info(f"Starting channel capacity experiment with {n_runs} runs")

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Create enhanced measurer
    measurer = EnhancedInformationMeasurer()

    # Run multiple experiments
    multiple_results = measurer.run_multiple_experiments(
        experiment_func=run_single_channel_capacity_experiment,
        n_runs=n_runs,
        base_seed=base_seed
    )

    # Aggregate all individual results
    all_results = []
    for run_data in multiple_results['individual_runs']:
        all_results.extend(run_data['results'])

    # Calculate statistics for each parameter combination
    statistical_results = []

    # Group by channel and parameter combination
    unique_conditions = set()
    for result in all_results:
        condition = f"{result['channel']}_{result['parameter_name']}_{result['parameter_value']}"
        unique_conditions.add((result['channel'], result['parameter_name'], result['parameter_value']))

    for channel, param_name, param_value in unique_conditions:
        # Get all results for this condition
        condition_results = [
            r for r in all_results
            if (r['channel'] == channel and
                r['parameter_name'] == param_name and
                r['parameter_value'] == param_value)
        ]

        if len(condition_results) > 1:
            # Extract MI means for statistical analysis
            mi_means = [r['mi_mean'] for r in condition_results]

            # Calculate statistics
            stats = {
                'mean': np.mean(mi_means),
                'std': np.std(mi_means, ddof=1),
                'min': np.min(mi_means),
                'max': np.max(mi_means),
                'median': np.median(mi_means),
                'n_samples': len(mi_means),
                'ci_95_lower': np.percentile(mi_means, 2.5),
                'ci_95_upper': np.percentile(mi_means, 97.5)
            }

            # Add normality test if enough samples
            if len(mi_means) >= 8:
                from scipy import stats as scipy_stats
                _, p_value = scipy_stats.shapiro(mi_means)
                stats['normality_p_value'] = p_value
                stats['appears_normal'] = p_value > 0.05

            statistical_results.append({
                'channel': channel,
                'parameter_name': param_name,
                'parameter_value': param_value,
                'statistics': stats,
                'individual_results': condition_results
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
    serializable_results = convert_to_serializable({
        'statistical_results': statistical_results,
        'multiple_runs_summary': {
            'n_runs': n_runs,
            'seeds_used': multiple_results['seeds_used']
        },
        'all_individual_results': all_results
    })

    with open(os.path.join(results_dir, "channel_capacity_detailed.json"), "w") as f:
        json.dump(serializable_results, f, indent=2)

    # Create simplified results for backward compatibility
    simplified_results = []
    for result in statistical_results:
        simplified_results.append({
            'channel': result['channel'],
            result['parameter_name']: result['parameter_value'],
            'mutual_information': result['statistics']['mean'],
            'mutual_information_std': result['statistics']['std'],
            'mutual_information_ci_lower': result['statistics']['ci_95_lower'],
            'mutual_information_ci_upper': result['statistics']['ci_95_upper'],
            'n_runs': result['statistics']['n_samples']
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

    with open(os.path.join(results_dir, "channel_capacity.json"), "w") as f:
        json.dump(simplified_results, f, indent=2, cls=NumpyEncoder)

    logger.info("Channel capacity experiment with statistics completed")
    logger.info(f"Detailed results saved to {os.path.join(results_dir, 'channel_capacity_detailed.json')}")
    logger.info(f"Simplified results saved to {os.path.join(results_dir, 'channel_capacity.json')}")

    return simplified_results


# For backward compatibility
def run_channel_capacity_experiment(results_dir: str) -> List[Dict[str, Any]]:
    """Backward compatible function - now runs with statistics by default."""
    return run_channel_capacity_experiment_with_statistics(results_dir, n_runs=5)


if __name__ == "__main__":
    # Run the enhanced experiment
    results = run_channel_capacity_experiment_with_statistics(
        results_dir="results",
        n_runs=10,
        base_seed=42
    )

    # Print statistical summary
    print("\nChannel Capacity Results with Statistics:")
    print("=" * 60)

    for channel in ["C1", "C2", "C3", "C4"]:
        channel_results = [r for r in results if r["channel"] == channel]
        if channel_results:
            print(f"\n{channel} Channel:")
            param_name = next(k for k in channel_results[0].keys()
                            if k not in ["channel", "mutual_information", "mutual_information_std",
                                       "mutual_information_ci_lower", "mutual_information_ci_upper", "n_runs"])

            for result in channel_results:
                mi_mean = result['mutual_information']
                mi_std = result['mutual_information_std']
                ci_lower = result['mutual_information_ci_lower']
                ci_upper = result['mutual_information_ci_upper']
                n_runs = result['n_runs']

                print(f"  {param_name}={result[param_name]}: "
                      f"{mi_mean:.4f} ± {mi_std:.4f} "
                      f"(95% CI: [{ci_lower:.4f}, {ci_upper:.4f}], n={n_runs})")