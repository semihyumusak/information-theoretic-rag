"""
Bottleneck Analysis Experiment Module

This module implements the experiment to identify bottlenecks in the RAG pipeline
by measuring the improvement in end-to-end performance when enhancing each channel.
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


def run_bottleneck_analysis_experiment(results_dir: str) -> List[Dict[str, Any]]:
    """
    Run experiment to identify bottlenecks in the RAG pipeline.

    Args:
        results_dir: Directory to save results

    Returns:
        List of result dictionaries
    """
    logger.info("Starting bottleneck analysis experiment")


    os.makedirs(results_dir, exist_ok=True)

    # Experimental parameters
    improvement_levels = [1.0, 1.5, 2.0]  # Baseline, 50% improvement, 100% improvement

    # Generate a schema and database for this experiment
    schema_gen = SchemaGenerator(seed=43)
    db_gen = DatabaseGenerator(seed=43)
    query_gen = QueryGenerator(seed=43)

    # Create schema with medium complexity
    schema = schema_gen.generate_schema(entropy_bits=5.0)

    # Generate database content
    logger.info("Generating database content")
    database = db_gen.generate_database(schema, num_rows_per_table=1000)

    # Generate queries
    logger.info("Generating queries")
    queries = query_gen.generate_queries(schema, num_queries=500)

    # Encode database
    logger.info("Encoding database")
    db_encoder = DatabaseEncoder(embedding_dim=256, seed=43)
    embedded_db = db_encoder.encode_database(database)

    # Baseline configuration
    baseline_encoder = QueryEncoder(embedding_dim=256, noise_level=0.1, seed=43)
    baseline_retriever = Retriever(context_size=8)
    baseline_integrator = ContextIntegrator(noise_level=0.05, seed=43)
    baseline_generator = Generator(temperature=0.7, seed=43)

    # Run baseline experiment to get reference performance
    logger.info("Running baseline experiment")
    baseline_responses = []

    for query in tqdm(queries[:100], desc="Baseline"):
        query_embedding = baseline_encoder.encode(query)
        context = baseline_retriever.retrieve(query_embedding, embedded_db)
        integrated_context = baseline_integrator.integrate(context, 8)
        response = baseline_generator.generate(query, integrated_context)
        baseline_responses.append(response)

    baseline_accuracy = np.mean([r["accuracy"] for r in baseline_responses])
    logger.info(f"Baseline end-to-end accuracy: {baseline_accuracy:.4f}")

    #
    # Test improving each channel independently
    bottleneck_results = []

    # 1. Improve C1: Query Encoding (reduce noise)
    logger.info("Testing improvements to C1: Query Encoding Channel")
    for improvement in tqdm(improvement_levels, desc="C₁ improvement"):
        if improvement == 1.0:
            continue  # Skip baseline

        # Reduce noise level by improvement factor
        noise_level = 0.1 / improvement
        improved_encoder = QueryEncoder(embedding_dim=256, noise_level=noise_level, seed=43)

        # Run with improved encoder
        responses = []

        for query in tqdm(queries[:100], desc=f"C₁ improvement {improvement}x", leave=False):
            query_embedding = improved_encoder.encode(query)
            context = baseline_retriever.retrieve(query_embedding, embedded_db)
            integrated_context = baseline_integrator.integrate(context, 8)
            response = baseline_generator.generate(query, integrated_context)
            responses.append(response)

        accuracy = np.mean([r["accuracy"] for r in responses])
        improvement_percent = (accuracy - baseline_accuracy) / baseline_accuracy * 100

        bottleneck_results.append({
            "channel": "C1",
            "improvement_factor": improvement,
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "improvement_percent": improvement_percent
        })

        logger.info(f"C1 improved by {improvement}x: accuracy={accuracy:.4f}, improvement={improvement_percent:.2f}%")

    # 2. Improve C2: Retrieval (increase context size)
    logger.info("Testing improvements to C2: Retrieval Channel")
    for improvement in tqdm(improvement_levels, desc="C₂ improvement"):
        if improvement == 1.0:
            continue  # Skip baseline

        # Increase context size by improvement factor
        context_size = int(8 * improvement)
        improved_retriever = Retriever(context_size=context_size)

        # Run with improved retriever
        responses = []

        for query in tqdm(queries[:100], desc=f"C₂ improvement {improvement}x", leave=False):
            query_embedding = baseline_encoder.encode(query)
            context = improved_retriever.retrieve(query_embedding, embedded_db)
            integrated_context = baseline_integrator.integrate(context, context_size)
            response = baseline_generator.generate(query, integrated_context)
            responses.append(response)

        accuracy = np.mean([r["accuracy"] for r in responses])
        improvement_percent = (accuracy - baseline_accuracy) / baseline_accuracy * 100

        bottleneck_results.append({
            "channel": "C2",
            "improvement_factor": improvement,
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "improvement_percent": improvement_percent
        })

        logger.info(f"C2 improved by {improvement}x: accuracy={accuracy:.4f}, improvement={improvement_percent:.2f}%")

    # 3. Improve C3: Context Integration (reduce noise)
    logger.info("Testing improvements to C3: Context Integration Channel")
    for improvement in tqdm(improvement_levels, desc="C₃ improvement"):
        if improvement == 1.0:
            continue  # Skip baseline

        # Reduce noise level by improvement factor
        noise_level = 0.05 / improvement
        improved_integrator = ContextIntegrator(noise_level=noise_level, seed=43)

        # Run with improved integrator
        responses = []

        for query in tqdm(queries[:100], desc=f"C₃ improvement {improvement}x", leave=False):
            query_embedding = baseline_encoder.encode(query)
            context = baseline_retriever.retrieve(query_embedding, embedded_db)
            integrated_context = improved_integrator.integrate(context, 8)
            response = baseline_generator.generate(query, integrated_context)
            responses.append(response)

        accuracy = np.mean([r["accuracy"] for r in responses])
        improvement_percent = (accuracy - baseline_accuracy) / baseline_accuracy * 100

        bottleneck_results.append({
            "channel": "C3",
            "improvement_factor": improvement,
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "improvement_percent": improvement_percent
        })

        logger.info(f"C3 improved by {improvement}x: accuracy={accuracy:.4f}, improvement={improvement_percent:.2f}%")

    # 4. Improve C4: Generation (reduce temperature)
    logger.info("Testing improvements to C4: Generation Channel")
    for improvement in tqdm(improvement_levels, desc="C₄ improvement"):
        if improvement == 1.0:
            continue  # Skip baseline

        # Reduce temperature by improvement factor (lower = more deterministic)
        temperature = max(0.1, 0.7 / improvement)
        improved_generator = Generator(temperature=temperature, seed=43)

        # Run with improved generator
        responses = []

        for query in tqdm(queries[:100], desc=f"C₄ improvement {improvement}x", leave=False):
            query_embedding = baseline_encoder.encode(query)
            context = baseline_retriever.retrieve(query_embedding, embedded_db)
            integrated_context = baseline_integrator.integrate(context, 8)
            response = improved_generator.generate(query, integrated_context)
            responses.append(response)

        accuracy = np.mean([r["accuracy"] for r in responses])
        improvement_percent = (accuracy - baseline_accuracy) / baseline_accuracy * 100

        bottleneck_results.append({
            "channel": "C4",
            "improvement_factor": improvement,
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "improvement_percent": improvement_percent
        })

        logger.info(f"C4 improved by {improvement}x: accuracy={accuracy:.4f}, improvement={improvement_percent:.2f}%")

    # Save results
    with open(os.path.join(results_dir, "bottleneck_analysis.json"), "w") as f:
        json.dump(bottleneck_results, f, indent=2)

    logger.info("Bottleneck analysis experiment completed")
    logger.info(f"Results saved to {os.path.join(results_dir, 'bottleneck_analysis.json')}")

    return bottleneck_results


if __name__ == "__main__":
    # Run the experiment standalone
    results = run_bottleneck_analysis_experiment(results_dir="results")

    # Print summary by channel
    channels = ["C1", "C2", "C3", "C4"]
    improvement_factors = [1.5, 2.0]  # Skip baseline

    print("\nBottleneck Analysis Summary:")
    print("-" * 60)
    print(f"{'Channel':<8} | {'Improvement Factor':<20} | {'Accuracy Improvement':<20}")
    print("-" * 60)

    for channel in channels:
        for factor in improvement_factors:
            result = next((r for r in results
                           if r["channel"] == channel and r["improvement_factor"] == factor), None)
            if result:
                print(f"{channel:<8} | {factor:<20.1f} | {result['improvement_percent']:<20.2f}%")

    print("-" * 60)

    # Identify the primary bottleneck
    best_improvements = {}
    for channel in channels:
        channel_results = [r for r in results if r["channel"] == channel]
        if channel_results:
            best_result = max(channel_results, key=lambda r: r["improvement_percent"])
            best_improvements[channel] = best_result["improvement_percent"]

    if best_improvements:
        bottleneck_channel = max(best_improvements, key=best_improvements.get)
        print(
            f"\nPrimary bottleneck identified: {bottleneck_channel} with {best_improvements[bottleneck_channel]:.2f}% improvement potential")
