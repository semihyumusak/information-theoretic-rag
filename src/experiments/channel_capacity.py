"""
Channel Capacity Experiment Module

This module implements the experiment to measure the capacity of each channel
in the RAG pipeline under varying conditions.
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
from analysis.information_measurer import InformationMeasurer

def run_channel_capacity_experiment(results_dir: str) -> List[Dict[str, Any]]:
    """
    Run experiment to measure capacity of each channel under varying conditions.

    Args:
        results_dir: Directory to save results

    Returns:
        List of result dictionaries
    """
    logger.info("Starting channel capacity experiment")

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Experimental parameters
    embedding_dims = [64, 128, 256, 512, 1024]
    context_sizes = [2, 4, 8, 16, 32]
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Generate a schema and database for this experiment
    schema_gen = SchemaGenerator(seed=42)
    db_gen = DatabaseGenerator(seed=42)
    query_gen = QueryGenerator(seed=42)

    # Create schema with medium complexity
    schema = schema_gen.generate_schema(entropy_bits=5.0)

    # Generate database content
    database = db_gen.generate_database(schema, num_rows_per_table=1000)

    # Generate queries
    queries = query_gen.generate_queries(schema, num_queries=500)

    # Create information measurer
    measurer = InformationMeasurer()

    # Initialize results storage
    channel_capacity_results = []

    # 1. Measure C1: Query Encoding Channel capacity with varying embedding dimensions
    logger.info("Measuring C1: Query Encoding Channel capacity")
    c1_results = []

    for dim in tqdm(embedding_dims, desc="C1: Embedding dimensions"):
        # Initialize encoder with this dimension
        encoder = QueryEncoder(embedding_dim=dim, noise_level=0.1, seed=42)

        # Process subset of queries
        query_embeddings = []
        query_texts = []

        for query in queries[:100]:  # Use a subset for efficiency
            embedding = encoder.encode(query)
            query_embeddings.append(embedding)
            query_texts.append(query["text"])

        # Measure mutual information between queries and embeddings
        # For text, we need to convert to a numerical representation
        # Here we use a simple bag-of-words approach
        vectorizer = CountVectorizer(max_features=100)
        text_features = vectorizer.fit_transform(query_texts).toarray()

        # Measure mutual information
        mi = measurer.estimate_mutual_information(text_features, np.array(query_embeddings))

        c1_results.append({
            "embedding_dim": dim,
            "mutual_information": mi,
            "channel": "C1"
        })

        logger.info(f"C1 capacity with dim={dim}: {mi:.4f} bits")

    # 2. Measure C2: Retrieval Channel capacity with varying context sizes
    logger.info("Measuring C2: Retrieval Channel capacity")
    c2_results = []

    # Encode database
    db_encoder = DatabaseEncoder(embedding_dim=256, seed=42)
    embedded_db = db_encoder.encode_database(database)

    # Fix query encoder
    query_encoder = QueryEncoder(embedding_dim=256, noise_level=0.1, seed=42)

    for context_size in tqdm(context_sizes, desc="C2: Context sizes"):
        # Initialize retriever with this context size
        retriever = Retriever(context_size=context_size)

        # Process subset of queries
        query_embeddings = []
        retrieved_contexts = []

        for query in queries[:100]:
            query_embedding = query_encoder.encode(query)
            context = retriever.retrieve(query_embedding, embedded_db)

            query_embeddings.append(query_embedding)

            # Create a simple feature vector from the retrieved context
            context_vector = np.zeros(256)
            for item in context:
                context_vector += item["embedding"]

            if len(context) > 0:
                context_vector /= len(context)

            retrieved_contexts.append(context_vector)

        # Measure mutual information
        mi = measurer.estimate_mutual_information(
            np.array(query_embeddings),
            np.array(retrieved_contexts)
        )

        c2_results.append({
            "context_size": context_size,
            "mutual_information": mi,
            "channel": "C2"
        })

        logger.info(f"C2 capacity with context_size={context_size}: {mi:.4f} bits")

    # 3. Measure C3: Context Integration Channel capacity with varying context sizes
    logger.info("Measuring C3: Context Integration Channel capacity")
    c3_results = []

    # Fix context integrator
    integrator = ContextIntegrator(noise_level=0.05, seed=42)

    for context_size in tqdm(context_sizes, desc="C3: Context sizes"):
        # Initialize retriever with a large context size
        retriever = Retriever(context_size=32)  # Large initial retrieval

        # Process subset of queries
        retrieved_contexts = []
        integrated_contexts = []

        for query in queries[:100]:
            query_embedding = query_encoder.encode(query)
            context = retriever.retrieve(query_embedding, embedded_db)

            # Integrate context with current context size
            integrated_context = integrator.integrate(context, context_size)

            # Create feature vectors for original and integrated contexts
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

        # Measure mutual information
        mi = measurer.estimate_mutual_information(
            np.array(retrieved_contexts),
            np.array(integrated_contexts)
        )

        c3_results.append({
            "context_size": context_size,
            "mutual_information": mi,
            "channel": "C3"
        })

        logger.info(f"C3 capacity with context_size={context_size}: {mi:.4f} bits")

    # 4. Measure C4: Generation Channel capacity with varying temperatures
    logger.info("Measuring C4: Generation Channel capacity")
    c4_results = []

    # Fix retriever and integrator
    retriever = Retriever(context_size=8)
    integrator = ContextIntegrator(noise_level=0.05, seed=42)

    for temp in tqdm(temperatures, desc="C4: Temperatures"):
        # Initialize generator with this temperature
        generator = Generator(temperature=temp, seed=42)

        # Process subset of queries
        integrated_contexts = []
        response_accuracies = []

        for query in queries[:100]:
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

        # Measure mutual information
        mi = measurer.estimate_mutual_information(
            np.array(integrated_contexts),
            np.array(response_accuracies)
        )

        c4_results.append({
            "temperature": temp,
            "mutual_information": mi,
            "channel": "C4"
        })

        logger.info(f"C4 capacity with temperature={temp}: {mi:.4f} bits")

    # Combine all results
    channel_capacity_results = c1_results + c2_results + c3_results + c4_results

    # Save results
    with open(os.path.join(results_dir, "channel_capacity.json"), "w") as f:
        json.dump(channel_capacity_results, f, indent=2)

    logger.info("Channel capacity experiment completed")
    logger.info(f"Results saved to {os.path.join(results_dir, 'channel_capacity.json')}")

    return channel_capacity_results


if __name__ == "__main__":
    # Run the experiment standalone
    results = run_channel_capacity_experiment(results_dir="results")

    # Print summary
    for channel in ["C1", "C2", "C3", "C4"]:
        channel_results = [r for r in results if r["channel"] == channel]
        if channel_results:
            param_name = next(k for k in channel_results[0].keys() if k not in ["channel", "mutual_information"])
            print(f"\n{channel} Channel Capacity:")
            for result in channel_results:
                print(f"  {param_name}={result[param_name]}: {result['mutual_information']:.4f} bits")