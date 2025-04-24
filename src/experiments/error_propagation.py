"""
Error Propagation Experiment Module

This module implements the experiment to measure error propagation effects
in the RAG pipeline, including interaction effects between channels.
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
from analysis.information_measurer import InformationMeasurer


def run_error_propagation_experiment(results_dir: str) -> List[Dict[str, Any]]:
    """
    Run experiment to measure error propagation effects.

    Args:
        results_dir: Directory to save results

    Returns:
        List of result dictionaries
    """
    logger.info("Starting error propagation experiment")

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Experimental parameters
    error_levels = [0.0, 0.1, 0.2, 0.3]  # Error levels to inject

    # Generate a schema and database for this experiment
    schema_gen = SchemaGenerator(seed=44)
    db_gen = DatabaseGenerator(seed=44)
    query_gen = QueryGenerator(seed=44)

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
    db_encoder = DatabaseEncoder(embedding_dim=256, seed=44)
    embedded_db = db_encoder.encode_database(database)

    # Baseline configuration (no errors)
    baseline_encoder = QueryEncoder(embedding_dim=256, noise_level=0.0, seed=44)
    baseline_retriever = Retriever(context_size=8)
    baseline_integrator = ContextIntegrator(noise_level=0.0, seed=44)
    baseline_generator = Generator(temperature=0.1, seed=44)  # Low temperature = deterministic

    # Run baseline experiment to get reference performance
    logger.info("Running baseline experiment (no errors)")
    baseline_responses = []

    for query in tqdm(queries[:100], desc="Baseline"):
        query_embedding = baseline_encoder.encode(query)
        context = baseline_retriever.retrieve(query_embedding, embedded_db)
        integrated_context = baseline_integrator.integrate(context, 8)
        response = baseline_generator.generate(query, integrated_context)
        baseline_responses.append(response)

    baseline_accuracy = np.mean([r["accuracy"] for r in baseline_responses])
    logger.info(f"Baseline end-to-end accuracy (no errors): {baseline_accuracy:.4f}")

    # Test single-channel errors
    logger.info("Testing single-channel errors")
    single_channel_results = []

    # 1. C₁ Errors: Query Encoding
    logger.info("Testing C₁ errors: Query Encoding")
    for error_level in tqdm(error_levels, desc="C₁ error levels"):
        if error_level == 0.0:
            continue  # Skip baseline

        # Add noise to query encoder
        error_encoder = QueryEncoder(embedding_dim=256, noise_level=error_level, seed=44)

        # Run with error in encoder
        responses = []

        for query in tqdm(queries[:100], desc=f"C₁ error {error_level}", leave=False):
            query_embedding = error_encoder.encode(query)
            context = baseline_retriever.retrieve(query_embedding, embedded_db)
            integrated_context = baseline_integrator.integrate(context, 8)
            response = baseline_generator.generate(query, integrated_context)
            responses.append(response)

        accuracy = np.mean([r["accuracy"] for r in responses])
        accuracy_drop = baseline_accuracy - accuracy

        single_channel_results.append({
            "channel": "C1",
            "error_level": error_level,
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_drop": accuracy_drop,
            "error_type": "single"
        })

        logger.info(f"C1 error={error_level}: accuracy={accuracy:.4f}, drop={accuracy_drop:.4f}")

    # 2. C₂ Errors: Retrieval (reduce context size)
    logger.info("Testing C₂ errors: Retrieval")
    for error_level in tqdm(error_levels, desc="C₂ error levels"):
        if error_level == 0.0:
            continue  # Skip baseline

        # Reduce context size based on error level
        context_size = max(1, int(8 * (1 - error_level)))
        error_retriever = Retriever(context_size=context_size)

        # Run with error in retriever
        responses = []

        for query in tqdm(queries[:100], desc=f"C₂ error {error_level}", leave=False):
            query_embedding = baseline_encoder.encode(query)
            context = error_retriever.retrieve(query_embedding, embedded_db)
            integrated_context = baseline_integrator.integrate(context, context_size)
            response = baseline_generator.generate(query, integrated_context)
            responses.append(response)

        accuracy = np.mean([r["accuracy"] for r in responses])
        accuracy_drop = baseline_accuracy - accuracy

        single_channel_results.append({
            "channel": "C2",
            "error_level": error_level,
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_drop": accuracy_drop,
            "error_type": "single"
        })

        logger.info(f"C2 error={error_level}: accuracy={accuracy:.4f}, drop={accuracy_drop:.4f}")

    # 3. C₃ Errors: Context Integration
    logger.info("Testing C₃ errors: Context Integration")
    for error_level in tqdm(error_levels, desc="C₃ error levels"):
        if error_level == 0.0:
            continue  # Skip baseline

        # Add noise to context integrator
        error_integrator = ContextIntegrator(noise_level=error_level, seed=44)

        # Run with error in integrator
        responses = []

        for query in tqdm(queries[:100], desc=f"C₃ error {error_level}", leave=False):
            query_embedding = baseline_encoder.encode(query)
            context = baseline_retriever.retrieve(query_embedding, embedded_db)
            integrated_context = error_integrator.integrate(context, 8)
            response = baseline_generator.generate(query, integrated_context)
            responses.append(response)

        accuracy = np.mean([r["accuracy"] for r in responses])
        accuracy_drop = baseline_accuracy - accuracy

        single_channel_results.append({
            "channel": "C3",
            "error_level": error_level,
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_drop": accuracy_drop,
            "error_type": "single"
        })

        logger.info(f"C3 error={error_level}: accuracy={accuracy:.4f}, drop={accuracy_drop:.4f}")

    # 4. C₄ Errors: Generation
    logger.info("Testing C₄ errors: Generation")
    for error_level in tqdm(error_levels, desc="C₄ error levels"):
        if error_level == 0.0:
            continue  # Skip baseline

        # Increase temperature based on error level (higher = more random)
        temperature = 0.1 + error_level * 0.9
        error_generator = Generator(temperature=temperature, seed=44)

        # Run with error in generator
        responses = []

        for query in tqdm(queries[:100], desc=f"C₄ error {error_level}", leave=False):
            query_embedding = baseline_encoder.encode(query)
            context = baseline_retriever.retrieve(query_embedding, embedded_db)
            integrated_context = baseline_integrator.integrate(context, 8)
            response = error_generator.generate(query, integrated_context)
            responses.append(response)

        accuracy = np.mean([r["accuracy"] for r in responses])
        accuracy_drop = baseline_accuracy - accuracy

        single_channel_results.append({
            "channel": "C4",
            "error_level": error_level,
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_drop": accuracy_drop,
            "error_type": "single"
        })

        logger.info(f"C4 error={error_level}: accuracy={accuracy:.4f}, drop={accuracy_drop:.4f}")

    # Test multi-channel errors (pair-wise)
    logger.info("Testing multi-channel errors (pair-wise)")
    multi_channel_results = []

    # Test all pairs of channels with moderate error
    moderate_error = 0.2
    channel_pairs = [("C1", "C2"), ("C1", "C3"), ("C1", "C4"), ("C2", "C3"), ("C2", "C4"), ("C3", "C4")]

    for ch1, ch2 in tqdm(channel_pairs, desc="Channel pairs"):
        # Configure components with errors
        if ch1 == "C1" or ch2 == "C1":
            error_encoder = QueryEncoder(embedding_dim=256, noise_level=moderate_error, seed=44)
        else:
            error_encoder = baseline_encoder

        if ch1 == "C2" or ch2 == "C2":
            context_size = max(1, int(8 * (1 - moderate_error)))
            error_retriever = Retriever(context_size=context_size)
        else:
            error_retriever = baseline_retriever

        if ch1 == "C3" or ch2 == "C3":
            error_integrator = ContextIntegrator(noise_level=moderate_error, seed=44)
        else:
            error_integrator = baseline_integrator

        if ch1 == "C4" or ch2 == "C4":
            temperature = 0.1 + moderate_error * 0.9
            error_generator = Generator(temperature=temperature, seed=44)
        else:
            error_generator = baseline_generator

        # Run with errors in both channels
        responses = []

        for query in tqdm(queries[:100], desc=f"{ch1}+{ch2}", leave=False):
            query_embedding = error_encoder.encode(query)
            context = error_retriever.retrieve(query_embedding, embedded_db)
            context_size_actual = context_size if ch1 == "C2" or ch2 == "C2" else 8
            integrated_context = error_integrator.integrate(context, context_size_actual)
            response = error_generator.generate(query, integrated_context)
            responses.append(response)

        accuracy = np.mean([r["accuracy"] for r in responses])
        accuracy_drop = baseline_accuracy - accuracy

        # Find single channel drops for comparison
        ch1_drop = next(r["accuracy_drop"] for r in single_channel_results
                        if r["channel"] == ch1 and r["error_level"] == moderate_error)
        ch2_drop = next(r["accuracy_drop"] for r in single_channel_results
                        if r["channel"] == ch2 and r["error_level"] == moderate_error)

        # Calculate interaction effect
        expected_drop = ch1_drop + ch2_drop
        actual_drop = accuracy_drop
        interaction = actual_drop - expected_drop

        multi_channel_results.append({
            "channels": f"{ch1}+{ch2}",
            "error_level": moderate_error,
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_drop": accuracy_drop,
            "ch1_drop": ch1_drop,
            "ch2_drop": ch2_drop,
            "expected_drop": expected_drop,
            "interaction_effect": interaction,
            "error_type": "multi"
        })

        logger.info(f"{ch1}+{ch2} error={moderate_error}: accuracy={accuracy:.4f}, "
                    f"drop={accuracy_drop:.4f}, interaction={interaction:.4f}")

    # Combine results
    error_propagation_results = single_channel_results + multi_channel_results

    # Save results
    with open(os.path.join(results_dir, "error_propagation.json"), "w") as f:
        json.dump(error_propagation_results, f, indent=2)

    logger.info("Error propagation experiment completed")
    logger.info(f"Results saved to {os.path.join(results_dir, 'error_propagation.json')}")

    return error_propagation_results


if __name__ == "__main__":
    # Run the experiment standalone
    results = run_error_propagation_experiment(results_dir="results")

    # Print summary of single-channel errors
    single_results = [r for r in results if r["error_type"] == "single"]

    print("\nSingle-Channel Error Effects:")
    print("-" * 70)
    print(f"{'Channel':<8} | {'Error Level':<12} | {'Accuracy':<10} | {'Accuracy Drop':<15}")
    print("-" * 70)

    for result in sorted(single_results, key=lambda x: (x["channel"], x["error_level"])):
        print(
            f"{result['channel']:<8} | {result['error_level']:<12.1f} | {result['accuracy']:<10.4f} | {result['accuracy_drop']:<15.4f}")

    print("\nMulti-Channel Interaction Effects:")
    print("-" * 100)
    print(f"{'Channel Pair':<12} | {'Actual Drop':<12} | {'Expected Drop':<14} | {'Interaction':<12} | {'Ratio':<10}")
    print("-" * 100)

    # Print summary of multi-channel interactions
    multi_results = [r for r in results if r["error_type"] == "multi"]

    for result in sorted(multi_results, key=lambda x: abs(x["interaction_effect"]), reverse=True):
        interaction_ratio = result["interaction_effect"] / result["expected_drop"]
        print(
            f"{result['channels']:<12} | {result['accuracy_drop']:<12.4f} | {result['expected_drop']:<14.4f} | {result['interaction_effect']:<12.4f} | {interaction_ratio:<10.2f}")

    # Check for superlinear growth
    has_superlinear = any(r["interaction_effect"] > 0 for r in multi_results)
    if has_superlinear:
        print("\nSuperlinear error propagation confirmed.")
    else:
        print("\nNo significant superlinear error propagation detected.")