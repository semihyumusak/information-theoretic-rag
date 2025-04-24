"""
Context Integrator Module

This module implements the context integration channel (C₃) of the RAG pipeline,
combining retrieved chunks into a coherent context.
"""

import numpy as np
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextIntegrator:
    """Simulates the context integration channel (C₃)."""

    def __init__(self, noise_level: float = 0.05, seed: int = 42):
        """
        Initialize the context integrator.

        Args:
            noise_level: Amount of noise to add during integration (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)

    def integrate(self, context: List[Dict], context_size: int) -> Dict:
        """
        Integrate retrieved context elements.

        Args:
            context: List of retrieved context elements
            context_size: Size of the context window

        Returns:
            Integrated context
        """
        # Sort context by similarity score
        context = sorted(context, key=lambda x: x["similarity"], reverse=True)

        # Take only up to context_size elements
        if len(context) > context_size:
            context = context[:context_size]

        # Calculate coherence score based on vector similarity
        coherence = 0.0
        if len(context) > 1:
            # Calculate average pairwise cosine similarity
            similarities = []
            for i in range(len(context)):
                for j in range(i + 1, len(context)):
                    sim = np.dot(context[i]["embedding"], context[j]["embedding"])
                    similarities.append(sim)

            if similarities:
                coherence = np.mean(similarities)

        # Add integration noise based on context size and coherence
        # More noise for larger contexts with lower coherence
        context_with_noise = []
        effective_noise = self.noise_level * (1 + context_size / 10) * (1 - coherence)

        for item in context:
            # Clone the item
            noisy_item = item.copy()

            # Add noise to embedding
            embedding = item["embedding"]
            noise = effective_noise * self.rng.randn(*embedding.shape)
            noisy_embedding = embedding + noise
            noisy_embedding = noisy_embedding / (np.linalg.norm(noisy_embedding) + 1e-8)

            noisy_item["embedding"] = noisy_embedding
            context_with_noise.append(noisy_item)

        # Create the integrated context object
        integrated_context = {
            "elements": context_with_noise,
            "coherence": coherence,
            "size": len(context)
        }

        logger.debug(f"Integrated {len(context)} context elements with coherence {coherence:.4f}")
        return integrated_context


if __name__ == "__main__":
    # Example usage (requires retriever.py and related modules)
    import sys
    import os

    # Add the parent directory to sys.path to import from data module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from data.schema_generator import SchemaGenerator
    from data.db_generator import DatabaseGenerator
    from data.query_generator import QueryGenerator
    from pipeline.query_encoder import QueryEncoder
    from pipeline.database_encoder import DatabaseEncoder
    from pipeline.retriever import Retriever

    # Generate a schema
    schema_gen = SchemaGenerator(seed=42)
    schema = schema_gen.generate_schema(entropy_bits=3.0)

    # Generate database content
    db_gen = DatabaseGenerator(seed=42)
    database = db_gen.generate_database(schema, num_rows_per_table=50)

    # Generate a query
    query_gen = QueryGenerator(seed=42)
    queries = query_gen.generate_queries(schema, num_queries=5, ambiguity_levels=[0.1])
    test_query = queries[0]

    # Encode the database
    db_encoder = DatabaseEncoder(embedding_dim=64)
    embedded_db = db_encoder.encode_database(database)

    # Encode the query
    query_encoder = QueryEncoder(embedding_dim=64)
    query_embedding = query_encoder.encode(test_query)

    # Retrieve relevant context
    retriever = Retriever(context_size=5)
    context = retriever.retrieve(query_embedding, embedded_db)

    # Integrate the context
    integrator = ContextIntegrator(noise_level=0.1)
    integrated_context = integrator.integrate(context, context_size=3)

    # Print the results
    print(f"Query: {test_query['text']}")
    print(f"Ground truth: {test_query['ground_truth']}")
    print(f"\nIntegrated context:")
    print(f"  Size: {integrated_context['size']}")
    print(f"  Coherence: {integrated_context['coherence']:.4f}")
    print(f"  Elements:")

    for i, item in enumerate(integrated_context["elements"]):
        if item["type"] == "table":
            print(f"    {i + 1}. Table: {item['name']} (similarity: {item['similarity']:.4f})")
        else:
            print(f"    {i + 1}. Column: {item['table']}.{item['name']} (similarity: {item['similarity']:.4f})")