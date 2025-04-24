"""
Information Measurer Module

This module provides functionality to measure information-theoretic properties
of the RAG pipeline components.
"""

import numpy as np
import logging
from typing import Dict, List
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InformationMeasurer:
    """Measure information-theoretic properties of the RAG pipeline."""

    @staticmethod
    def estimate_mutual_information(X: np.ndarray, Y: np.ndarray, n_bins: int = 20) -> float:
        """
        Estimate mutual information between two continuous variables.

        Args:
            X: First variable samples
            Y: Second variable samples
            n_bins: Number of bins for histogram estimation

        Returns:
            Estimated mutual information in bits
        """
        # Convert high-dimensional data to 1D if needed
        if X.ndim > 1:
            X_1d = PCA(n_components=1).fit_transform(X).flatten()
        else:
            X_1d = X

        if Y.ndim > 1:
            Y_1d = PCA(n_components=1).fit_transform(Y).flatten()
        else:
            Y_1d = Y

        # Create joint histogram
        hist_2d, x_edges, y_edges = np.histogram2d(X_1d, Y_1d, bins=n_bins)

        # Normalize to get joint probability
        p_xy = hist_2d / float(np.sum(hist_2d))

        # Get marginal probabilities
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        # Compute mutual information
        mi = 0.0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))

        return max(0, mi)  # MI should be non-negative

    @staticmethod
    def calculate_channel_capacity(input_samples: np.ndarray, output_samples: np.ndarray,
                                   n_bins: int = 20) -> float:
        """
        Estimate the channel capacity from input-output samples.
        This is a simplified estimation based on mutual information.

        Args:
            input_samples: Input variable samples
            output_samples: Output variable samples
            n_bins: Number of bins for histogram estimation

        Returns:
            Estimated channel capacity in bits
        """
        # For a more accurate estimation, we would need to optimize over input distributions
        # Here we use a simplified approach based on the provided samples
        return InformationMeasurer.estimate_mutual_information(input_samples, output_samples, n_bins)

    @staticmethod
    def measure_retrieval_precision(context: List[Dict], ground_truth: Dict) -> float:
        """
        Measure precision of retrieved context.

        Args:
            context: Retrieved context elements
            ground_truth: Ground truth for the query

        Returns:
            Precision score (0.0-1.0)
        """
        # Extract target tables and columns from ground truth
        target_tables = []
        target_columns = []

        if ground_truth.get("operation") == "join":
            target_tables = ground_truth.get("tables", [])
            target_columns = ground_truth.get("columns", [])
        elif ground_truth.get("operation") in ["select", "filter", "aggregate"]:
            target_tables = [ground_truth.get("table")]
            if ground_truth.get("column"):
                target_columns = [ground_truth.get("column")]
            elif "columns" in ground_truth:
                target_columns = ground_truth.get("columns", [])

        if not target_tables and not target_columns:
            return 1.0  # No targets to evaluate against

        # Count relevant retrieved elements
        relevant_count = 0

        for elem in context:
            if elem["type"] == "table" and elem["name"] in target_tables:
                relevant_count += 1
            elif elem["type"] == "column":
                if elem["table"] in target_tables and elem["name"] in target_columns:
                    relevant_count += 1

        # Calculate precision
        if len(context) > 0:
            precision = relevant_count / len(context)
        else:
            precision = 0.0

        return precision

    @staticmethod
    def measure_retrieval_recall(context: List[Dict], ground_truth: Dict) -> float:
        """
        Measure recall of retrieved context.

        Args:
            context: Retrieved context elements
            ground_truth: Ground truth for the query

        Returns:
            Recall score (0.0-1.0)
        """
        # Extract target tables and columns from ground truth
        target_tables = []
        target_columns = []

        if ground_truth.get("operation") == "join":
            target_tables = ground_truth.get("tables", [])
            target_columns = ground_truth.get("columns", [])
        elif ground_truth.get("operation") in ["select", "filter", "aggregate"]:
            target_tables = [ground_truth.get("table")]
            if ground_truth.get("column"):
                target_columns = [ground_truth.get("column")]
            elif "columns" in ground_truth:
                target_columns = ground_truth.get("columns", [])

        total_targets = len(target_tables) + len(target_columns)
        if total_targets == 0:
            return 1.0  # No targets to evaluate against

        # Count retrieved target elements
        retrieved_tables = set()
        retrieved_columns = set()

        for elem in context:
            if elem["type"] == "table" and elem["name"] in target_tables:
                retrieved_tables.add(elem["name"])
            elif elem["type"] == "column":
                if elem["table"] in target_tables and elem["name"] in target_columns:
                    retrieved_columns.add(elem["name"])

        # Calculate recall
        retrieved_count = len(retrieved_tables) + len(retrieved_columns)
        recall = retrieved_count / total_targets

        return recall

    @staticmethod
    def measure_context_coherence(context: List[Dict]) -> float:
        """
        Measure the semantic coherence of context elements.

        Args:
            context: Retrieved context elements

        Returns:
            Coherence score (0.0-1.0)
        """
        if len(context) <= 1:
            return 1.0  # Single element is perfectly coherent

        # Extract embeddings
        embeddings = [elem["embedding"] for elem in context]

        # Calculate average pairwise cosine similarity
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)

        if similarities:
            coherence = np.mean(similarities)
            # Scale to 0-1 (similarities could be negative)
            coherence = (coherence + 1) / 2
        else:
            coherence = 0.0

        return coherence

    @staticmethod
    def measure_response_fidelity(response: Dict, ground_truth: Dict) -> float:
        """
        Measure how faithfully the response reflects the ground truth.

        Args:
            response: Generated response
            ground_truth: Ground truth for the query

        Returns:
            Fidelity score (0.0-1.0)
        """
        # In a real implementation, this would compare semantic content
        # Here we use the pre-computed accuracy from the generator
        return response.get("accuracy", 0.0)

    @staticmethod
    def calculate_error_propagation(query_encoding_error: float,
                                    retrieval_error: float,
                                    context_error: float,
                                    generation_error: float) -> float:
        """
        Calculate error propagation based on individual channel errors.

        Args:
            query_encoding_error: Error in query encoding channel
            retrieval_error: Error in retrieval channel
            context_error: Error in context integration channel
            generation_error: Error in generation channel

        Returns:
            Total error propagation effect
        """
        # Linear component (sum of individual errors)
        linear_component = sum([
            query_encoding_error,
            retrieval_error,
            context_error,
            generation_error
        ])

        # Interaction terms (pairwise products)
        interaction_terms = [
            query_encoding_error * retrieval_error * 0.5,
            query_encoding_error * context_error * 0.3,
            query_encoding_error * generation_error * 0.2,
            retrieval_error * context_error * 0.7,
            retrieval_error * generation_error * 0.5,
            context_error * generation_error * 0.6
        ]

        interaction_component = sum(interaction_terms)

        # Combine linear and quadratic effects
        total_error = linear_component + interaction_component

        return total_error


if __name__ == "__main__":
    # Example usage
    import sys
    import os

    # Add the parent directory to sys.path to import from other modules
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from data.schema_generator import SchemaGenerator
    from data.db_generator import DatabaseGenerator
    from data.query_generator import QueryGenerator
    from pipeline.query_encoder import QueryEncoder
    from pipeline.database_encoder import DatabaseEncoder
    from pipeline.retriever import Retriever
    from pipeline.context_integrator import ContextIntegrator
    from pipeline.generator import Generator

    # Generate a schema
    schema_gen = SchemaGenerator(seed=42)
    schema = schema_gen.generate_schema(entropy_bits=3.0)

    # Generate database content
    db_gen = DatabaseGenerator(seed=42)
    database = db_gen.generate_database(schema, num_rows_per_table=50)

    # Generate queries
    query_gen = QueryGenerator(seed=42)
    queries = query_gen.generate_queries(schema, num_queries=10, ambiguity_levels=[0.1])

    # Set up the RAG pipeline
    query_encoder = QueryEncoder(embedding_dim=64)
    db_encoder = DatabaseEncoder(embedding_dim=64)
    retriever = Retriever(context_size=5)
    integrator = ContextIntegrator(noise_level=0.1)
    generator = Generator(temperature=0.5)

    # Encode the database
    embedded_db = db_encoder.encode_database(database)

    # Run the pipeline for a test query
    test_query = queries[0]
    query_embedding = query_encoder.encode(test_query)
    context = retriever.retrieve(query_embedding, embedded_db)
    integrated_context = integrator.integrate(context, context_size=3)
    response = generator.generate(test_query, integrated_context)

    # Measure information-theoretic properties
    measurer = InformationMeasurer()

    # Create sample data for mutual information estimation
    X = np.random.randn(100, 64)
    Y = 0.8 * X + 0.2 * np.random.randn(100, 64)
    mi = measurer.estimate_mutual_information(X, Y)
    print(f"Estimated mutual information: {mi:.4f} bits")

    # Measure retrieval quality
    precision = measurer.measure_retrieval_precision(context, test_query["ground_truth"])
    recall = measurer.measure_retrieval_recall(context, test_query["ground_truth"])
    print(f"Retrieval precision: {precision:.4f}")
    print(f"Retrieval recall: {recall:.4f}")

    # Measure context coherence
    coherence = measurer.measure_context_coherence(context)
    print(f"Context coherence: {coherence:.4f}")

    # Measure response fidelity
    fidelity = measurer.measure_response_fidelity(response, test_query["ground_truth"])
    print(f"Response fidelity: {fidelity:.4f}")

    # Calculate error propagation
    total_error = measurer.calculate_error_propagation(0.2, 0.3, 0.1, 0.2)
    print(f"Total error propagation: {total_error:.4f}")