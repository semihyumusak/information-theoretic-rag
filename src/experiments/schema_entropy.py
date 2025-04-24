"""
Schema Entropy Experiment Module

This module implements the experiment to measure the effect of schema entropy
on RAG performance, including validation of theoretical bounds.
"""

import os
import json
import logging
import numpy as np
import math
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
from analysis.information_measurer import InformationMeasurer


def run_schema_entropy_experiment(results_dir: str) -> List[Dict[str, Any]]:
    """
    Run experiment to measure the effect of schema entropy on RAG performance.

    Args:
        results_dir: Directory to save results

    Returns:
        List of result dictionaries
    """
    logger.info("Starting schema entropy experiment")

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Experimental parameters
    schema_entropy_levels = [2.0, 4.0, 6.0, 8.0, 10.0]
    embedding_dims = [64, 128, 256, 512, 1024]

    # Fixed components
    query_gen = QueryGenerator(seed=45)
    db_gen = DatabaseGenerator(seed=45)

    # Results storage
    schema_entropy_results = []

    for entropy_level in tqdm(schema_entropy_levels, desc="Schema entropy levels"):
        logger.info(f"Testing schema with entropy={entropy_level} bits")

        # Generate schema with specified entropy
        schema_gen = SchemaGenerator(seed=45)
        schema = schema_gen.generate_schema(entropy_bits=entropy_level)

        # Generate database content
        database = db_gen.generate_database(schema, num_rows_per_table=1000)

        # Generate queries
        queries = query_gen.generate_queries(schema, num_queries=200)

        # Test with different embedding dimensions
        for dim in tqdm(embedding_dims, desc=f"Embedding dims (entropy={entropy_level})", leave=False):
            logger.info(f"  Testing with embedding_dim={dim}")

            # Initialize components
            query_encoder = QueryEncoder(embedding_dim=dim, noise_level=0.1, seed=45)
            db_encoder = DatabaseEncoder(embedding_dim=dim, seed=45)
            retriever = Retriever(context_size=8)
            integrator = ContextIntegrator(noise_level=0.05, seed=45)
            generator = Generator(temperature=0.7, seed=45)

            # Encode database
            embedded_db = db_encoder.encode_database(database)

            # Process queries
            responses = []
            precisions = []
            recalls = []

            for query in tqdm(queries[:50], desc=f"Processing queries (dim={dim})", leave=False):
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

            result = {
                "schema_entropy": entropy_level,
                "embedding_dim": dim,
                "accuracy": accuracy,
                "precision": avg_precision,
                "recall": avg_recall,
                "theoretical_bound": theory_bound
            }

            schema_entropy_results.append(result)

            logger.info(f"  Results: accuracy={accuracy:.4f}, precision={avg_precision:.4f}, "
                        f"recall={avg_recall:.4f}, bound={theory_bound:.4f}")

    # Save results
    with open(os.path.join(results_dir, "schema_entropy.json"), "w") as f:
        json.dump(schema_entropy_results, f, indent=2)

    logger.info("Schema entropy experiment completed")
    logger.info(f"Results saved to {os.path.join(results_dir, 'schema_entropy.json')}")

    return schema_entropy_results


if __name__ == "__main__":
    # Run the experiment standalone
    results = run_schema_entropy_experiment(results_dir="results")

    # Print summary table
    print("\nSchema Entropy Results Summary:")
    print("-" * 80)
    print(f"{'Schema Entropy':<15} | {'Embedding Dim':<15} | {'Recall':<10} | {'Bound':<10} | {'Ratio':<10}")
    print("-" * 80)

    # Group by schema entropy and embedding dimension
    for entropy in sorted(set(r["schema_entropy"] for r in results)):
        for dim in sorted(set(r["embedding_dim"] for r in results)):
            result = next((r for r in results
                           if r["schema_entropy"] == entropy and r["embedding_dim"] == dim), None)
            if result:
                # Calculate ratio of actual performance to theoretical bound
                ratio = result["recall"] / result["theoretical_bound"] if result["theoretical_bound"] > 0 else 0
                print(
                    f"{result['schema_entropy']:<15.1f} | {result['embedding_dim']:<15d} | {result['recall']:<10.4f} | {result['theoretical_bound']:<10.4f} | {ratio:<10.4f}")

    # Check if theoretical bound is validated
    bounds_respected = all(r["recall"] * np.log2(r["embedding_dim"]) <= r["theoretical_bound"] * 1.05  # Allow 5% margin
                           for r in results)

    if bounds_respected:
        print("\nTheoretical bound validated: All measured performance levels are within the theoretical limits.")
    else:
        print(
            "\nWarning: Some measured performance levels exceed the theoretical bounds (accounting for measurement error).")

    # Analyze embedding dimension vs schema entropy effects
    dim_limited = []
    entropy_limited = []

    for r in results:
        if np.log2(r["embedding_dim"]) < r["schema_entropy"]:
            dim_limited.append(r)
        else:
            entropy_limited.append(r)

    if dim_limited:
        avg_ratio_dim = np.mean([r["recall"] / (np.log2(r["embedding_dim"]) / r["theoretical_bound"])
                                 for r in dim_limited])
        print(f"\nDimension-limited cases ({len(dim_limited)}): Average performance ratio = {avg_ratio_dim:.4f}")

    if entropy_limited:
        avg_ratio_entropy = np.mean([r["recall"] / (r["schema_entropy"] / r["theoretical_bound"])
                                     for r in entropy_limited])
        print(f"Entropy-limited cases ({len(entropy_limited)}): Average performance ratio = {avg_ratio_entropy:.4f}")