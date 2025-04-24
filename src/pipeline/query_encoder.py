"""
Query Encoder Module

This module implements the query encoding channel (C₁) of the RAG pipeline,
transforming natural language queries into vector embeddings.
"""

import numpy as np
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryEncoder:
    """Simulates the query encoding channel (C₁)."""

    def __init__(self, embedding_dim: int = 128, noise_level: float = 0.1, seed: int = 42):
        """
        Initialize the query encoder.

        Args:
            embedding_dim: Dimension of the query embedding
            noise_level: Amount of noise to add to embeddings (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)

        # Pre-compute word embeddings for a simple vocabulary
        vocab_size = 1000
        self.word_embeddings = {}
        for i in range(vocab_size):
            word = f"word_{i}"
            self.word_embeddings[word] = self.rng.randn(embedding_dim)

        # Add embeddings for common words in our query templates
        for word in ["get", "show", "what", "find", "list", "retrieve", "table", "column",
                     "where", "average", "sum", "count", "maximum", "minimum", "from", "with"]:
            self.word_embeddings[word] = self.rng.randn(embedding_dim)

    def encode(self, query: Dict) -> np.ndarray:
        """
        Encode a query into an embedding vector.

        Args:
            query: Query dictionary with text field

        Returns:
            Embedding vector
        """
        query_text = query["text"].lower()
        words = query_text.split()

        # Get embeddings for each word
        word_vectors = []
        for word in words:
            # Clean the word (remove punctuation)
            word = ''.join(c for c in word if c.isalnum())

            if word in self.word_embeddings:
                vector = self.word_embeddings[word]
            else:
                # For unknown words, generate a random embedding
                vector = self.rng.randn(self.embedding_dim)
                self.word_embeddings[word] = vector

            word_vectors.append(vector)

        # Average word vectors to get query embedding
        if word_vectors:
            query_vector = np.mean(word_vectors, axis=0)
        else:
            query_vector = np.zeros(self.embedding_dim)

        # Normalize
        query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-8)

        # Add noise proportional to the ambiguity level and noise setting
        ambiguity_level = query.get("ambiguity_level", 0.0)
        effective_noise = self.noise_level * (1 + ambiguity_level)

        noise = effective_noise * self.rng.randn(self.embedding_dim)
        noisy_vector = query_vector + noise

        # Normalize again
        noisy_vector = noisy_vector / (np.linalg.norm(noisy_vector) + 1e-8)

        return noisy_vector


if __name__ == "__main__":
    # Example usage
    encoder = QueryEncoder(embedding_dim=64, noise_level=0.1)

    # Create a test query
    test_query = {
        "id": 1,
        "text": "Get the customer_name from customers table",
        "ambiguity_level": 0.5
    }

    # Encode the query
    embedding = encoder.encode(test_query)

    print(f"Query: {test_query['text']}")
    print(f"Ambiguity level: {test_query['ambiguity_level']}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.6f}")
    print(f"First 5 values: {embedding[:5]}")