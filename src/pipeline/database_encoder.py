"""
Database Encoder Module

This module provides functionality to encode database content into vector space,
preparing it for retrieval in the RAG pipeline.
"""

import numpy as np
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseEncoder:
    """Encodes database content into vector space."""

    def __init__(self, embedding_dim: int = 128, seed: int = 42):
        """
        Initialize the database encoder.

        Args:
            embedding_dim: Dimension of the embeddings
            seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.rng = np.random.RandomState(seed)

    def encode_database(self, database: Dict) -> Dict:
        """
        Encode database tables and columns into vectors.

        Args:
            database: Database dictionary with tables

        Returns:
            Dictionary with embedded database elements
        """
        embedded_db = {
            "metadata": database["metadata"],
            "embeddings": {}
        }

        # Encode each table
        for table_name, table_data in database["tables"].items():
            table_embedding = self.rng.randn(self.embedding_dim)
            table_embedding = table_embedding / np.linalg.norm(table_embedding)

            embedded_db["embeddings"][table_name] = {
                "table_embedding": table_embedding,
                "columns": {}
            }

            # Encode column names
            if table_data and isinstance(table_data[0], dict):
                column_names = list(table_data[0].keys())
                for column in column_names:
                    column_embedding = self.rng.randn(self.embedding_dim)
                    column_embedding = column_embedding / np.linalg.norm(column_embedding)

                    embedded_db["embeddings"][table_name]["columns"][column] = {
                        "embedding": column_embedding
                    }

        logger.info(f"Encoded database with {len(embedded_db['embeddings'])} tables")
        return embedded_db


if __name__ == "__main__":
    # Example usage (requires schema_generator.py and db_generator.py)
    import sys
    import os

    # Add the parent directory to sys.path to import from data module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from data.schema_generator import SchemaGenerator
    from data.db_generator import DatabaseGenerator

    # Generate a schema
    schema_gen = SchemaGenerator(seed=42)
    schema = schema_gen.generate_schema(entropy_bits=3.0)

    # Generate database content
    db_gen = DatabaseGenerator(seed=42)
    database = db_gen.generate_database(schema, num_rows_per_table=50)

    # Encode the database
    encoder = DatabaseEncoder(embedding_dim=64)
    embedded_db = encoder.encode_database(database)

    # Print some statistics
    print(f"Database encoding summary:")
    for table_name, table_data in embedded_db["embeddings"].items():
        print(f"\nTable: {table_name}")
        print(f"  Table embedding shape: {table_data['table_embedding'].shape}")
        print(f"  Table embedding norm: {np.linalg.norm(table_data['table_embedding']):.6f}")
        print(f"  Columns encoded: {len(table_data['columns'])}")

        # Print the first column embedding as an example
        if table_data["columns"]:
            first_col = next(iter(table_data["columns"]))
            first_emb = table_data["columns"][first_col]["embedding"]
            print(f"  Sample column '{first_col}' embedding first 5 values: {first_emb[:5]}")