"""
Retriever Module

This module implements the retrieval channel (C₂) of the RAG pipeline,
retrieving relevant database elements based on query embeddings.
"""

import numpy as np
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """Simulates the retrieval channel (C₂)."""

    def __init__(self, context_size: int = 5):
        """
        Initialize the retriever.

        Args:
            context_size: Number of chunks to retrieve
        """
        self.context_size = context_size

    def retrieve(self, query_embedding: np.ndarray, embedded_db: Dict) -> List[Dict]:
        """
        Retrieve relevant database elements based on query embedding.

        Args:
            query_embedding: Query embedding vector
            embedded_db: Embedded database

        Returns:
            List of retrieved context elements
        """
        context = []

        # Calculate similarity with all tables and columns
        similarities = []

        for table_name, table_data in embedded_db["embeddings"].items():
            table_embedding = table_data["table_embedding"]
            table_sim = np.dot(query_embedding, table_embedding)

            similarities.append({
                "type": "table",
                "name": table_name,
                "similarity": table_sim,
                "embedding": table_embedding
            })

            # Add columns
            for column_name, column_data in table_data["columns"].items():
                column_embedding = column_data["embedding"]
                column_sim = np.dot(query_embedding, column_embedding)

                similarities.append({
                    "type": "column",
                    "table": table_name,
                    "name": column_name,
                    "similarity": column_sim,
                    "embedding": column_embedding
                })

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Take top-k items
        top_k = similarities[:self.context_size]

        # Create context objects
        for item in top_k:
            if item["type"] == "table":
                context.append({
                    "type": "table",
                    "name": item["name"],
                    "similarity": item["similarity"],
                    "embedding": item["embedding"]
                })
            else:  # column
                context.append({
                    "type": "column",
                    "table": item["table"],
                    "name": item["name"],
                    "similarity": item["similarity"],
                    "embedding": item["embedding"]
                })

        logger.debug(f"Retrieved {len(context)} context elements")
        return context


if __name__ == "__main__":
    # Example usage (requires query_encoder.py and database_encoder.py)
    import sys
    import os

    # Add the parent directory to sys.path to import from data module
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from data.schema_generator import SchemaGenerator
    from data.db_generator import DatabaseGenerator
    from data.query_generator import QueryGenerator
    from pipeline.query_encoder import QueryEncoder
    from pipeline.database_encoder import DatabaseEncoder

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
    retriever = Retriever(context_size=3)
    context = retriever.retrieve(query_embedding, embedded_db)

    # Print the results
    print(f"Query: {test_query['text']}")
    print(f"Ground truth: {test_query['ground_truth']}")
    print("\nRetrieved context:")
    for i, item in enumerate(context):
        if item["type"] == "table":
            print(f"  {i + 1}. Table: {item['name']} (similarity: {item['similarity']:.4f})")
        else:
            print(f"  {i + 1}. Column: {item['table']}.{item['name']} (similarity: {item['similarity']:.4f})")