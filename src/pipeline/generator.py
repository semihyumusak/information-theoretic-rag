"""
Generator Module

This module implements the generation channel (C₄) of the RAG pipeline,
generating responses based on the query and integrated context.
"""

import numpy as np
from typing import Dict, List, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Generator:
    """Simulates the generation channel (C₄)."""

    def __init__(self, temperature: float = 0.7, seed: int = 42):
        """
        Initialize the generator.

        Args:
            temperature: Temperature parameter for generation (0.1-1.0)
            seed: Random seed for reproducibility
        """
        self.temperature = temperature
        self.rng = np.random.RandomState(seed)

    def generate(self, query: Dict, context: Dict) -> Dict:
        """
        Generate a response based on query and context.

        Args:
            query: Original query
            context: Integrated context

        Returns:
            Generated response with metadata
        """
        # Extract ground truth from query
        ground_truth = query.get("ground_truth", {})

        # Calculate response accuracy based on retrieved context relevance
        # and controlled by temperature

        # Check if the required tables and columns are in the context
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

        # Count how many required elements are in the context
        tables_found = 0
        columns_found = 0

        context_tables = set()
        context_columns = set()

        for elem in context["elements"]:
            if elem["type"] == "table" and elem["name"] in target_tables:
                tables_found += 1
                context_tables.add(elem["name"])
            elif elem["type"] == "column":
                if elem["table"] in target_tables and elem["name"] in target_columns:
                    columns_found += 1
                    context_columns.add(elem["name"])

        # Calculate base accuracy
        if target_tables:
            table_accuracy = len(context_tables) / len(target_tables)
        else:
            table_accuracy = 1.0

        if target_columns:
            column_accuracy = len(context_columns) / len(target_columns)
        else:
            column_accuracy = 1.0

        # Combine accuracies, with more weight on tables
        base_accuracy = 0.7 * table_accuracy + 0.3 * column_accuracy

        # Apply temperature effect: higher temp = more randomness = lower accuracy
        effective_accuracy = base_accuracy * (1.1 - self.temperature)

        # Add a small random component
        final_accuracy = min(1.0, max(0.0,
                                      effective_accuracy +
                                      self.rng.normal(0, 0.1 * self.temperature)))

        # Simulate response generation
        # Higher accuracy = response closer to ground truth
        if final_accuracy > 0.8:
            # Very accurate - closely follows ground truth
            quality = "high"
            response_text = self._generate_high_quality_response(query, ground_truth)
        elif final_accuracy > 0.5:
            # Moderately accurate - partially correct
            quality = "medium"
            response_text = self._generate_medium_quality_response(query, ground_truth)
        else:
            # Inaccurate - mostly incorrect
            quality = "low"
            response_text = self._generate_low_quality_response(query)

        # Create response object
        response = {
            "query_id": query["id"],
            "text": response_text,
            "accuracy": final_accuracy,
            "context_size": context["size"],
            "context_coherence": context["coherence"],
            "tables_retrieved": list(context_tables),
            "columns_retrieved": list(context_columns),
            "target_tables": target_tables,
            "target_columns": target_columns,
            "quality": quality
        }

        logger.debug(f"Generated response with accuracy {final_accuracy:.4f}")
        return response

    def _generate_high_quality_response(self, query: Dict, ground_truth: Dict) -> str:
        """Generate a high-quality response based on ground truth."""
        operation = ground_truth.get("operation", "")

        if operation == "select":
            table = ground_truth.get("table", "unknown")
            column = ground_truth.get("column", "unknown")
            return f"Here are the {column} values from the {table} table."

        elif operation == "filter":
            table = ground_truth.get("table", "unknown")
            columns = ground_truth.get("columns", [])
            value = ground_truth.get("filter_value", "unknown")
            if columns:
                return f"I found the {columns[0]} values in {table} where {columns[1]} equals {value}."
            else:
                return f"I found the matching records in {table} with the specified filter."

        elif operation == "join":
            tables = ground_truth.get("tables", [])
            columns = ground_truth.get("columns", [])
            if tables and len(tables) >= 2:
                return f"Here are the joined results from {tables[0]} and {tables[1]}, showing {columns[0]} and {columns[1]}."
            else:
                return f"Here are the joined results from the related tables."

        elif operation == "aggregate":
            table = ground_truth.get("table", "unknown")
            column = ground_truth.get("column", "unknown")
            agg = ground_truth.get("aggregation", "unknown")
            if agg == "avg":
                return f"The average {column} in {table} is 42.5."
            elif agg == "sum":
                return f"The sum of {column} in {table} is 1250."
            elif agg == "count":
                return f"There are 30 {column} entries in {table}."
            elif agg == "max":
                return f"The maximum {column} in {table} is 100."
            elif agg == "min":
                return f"The minimum {column} in {table} is 1."
            else:
                return f"The {agg} of {column} in {table} has been calculated."

        else:
            return f"Here are the results for your query about {query['text'].split()[-1]}."

    def _generate_medium_quality_response(self, query: Dict, ground_truth: Dict) -> str:
        """Generate a medium-quality response with partial correctness."""
        operation = ground_truth.get("operation", "")

        if operation == "select":
            table = ground_truth.get("table", "unknown")
            return f"I found some data from the {table} table that might be relevant."

        elif operation == "filter":
            table = ground_truth.get("table", "unknown")
            return f"Here are some filtered results from {table}, though I might not have caught all constraints."

        elif operation == "join":
            return f"I have some joined results that might answer your question, but I might be missing some relationships."

        elif operation == "aggregate":
            agg = ground_truth.get("aggregation", "unknown")
            if agg == "avg":
                return f"The average value is approximately 43, but this might not be exact."
            elif agg == "sum":
                return f"The total is around 1200, give or take."
            elif agg == "count":
                return f"There are roughly 30 items, though I might be missing some."
            elif agg == "max":
                return f"The maximum value appears to be 98, but there could be larger values."
            elif agg == "min":
                return f"The minimum value seems to be around 2."
            else:
                return f"I've calculated an approximate {agg}, but it might not be complete."

        else:
            return f"I found some information related to {query['text'].split()[-1]}, but it might be incomplete."

    def _generate_low_quality_response(self, query: Dict) -> str:
        """Generate a low-quality response with minimal correctness."""
        # Generic responses that don't correctly address the query
        generic_responses = [
            f"I'm not sure I understand what you're looking for regarding {query['text'].split()[-1]}.",
            f"I found some data that might be related, but I'm not confident it answers your question.",
            "The database doesn't seem to contain the information you're looking for.",
            "I couldn't find clear matches for your query in the available tables.",
            "Your question is ambiguous. Could you please clarify what you're looking for?"
        ]

        return self.rng.choice(generic_responses)


if __name__ == "__main__":
    # Example usage (requires the full pipeline)
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

    # Generate a schema
    schema_gen = SchemaGenerator(seed=42)
    schema = schema_gen.generate_schema(entropy_bits=3.0)

    # Generate database content
    db_gen = DatabaseGenerator(seed=42)
    database = db_gen.generate_database(schema, num_rows_per_table=50)

    # Generate a query
    query_gen = QueryGenerator(seed=42)
    queries = query_gen.generate_queries(schema, num_queries=5, ambiguity_levels=[0.1, 0.5, 0.9])

    # Run the pipeline for each query
    db_encoder = DatabaseEncoder(embedding_dim=64)
    embedded_db = db_encoder.encode_database(database)

    query_encoder = QueryEncoder(embedding_dim=64)
    retriever = Retriever(context_size=5)
    integrator = ContextIntegrator(noise_level=0.1)

    # Test with different temperatures
    for temp in [0.1, 0.5, 0.9]:
        generator = Generator(temperature=temp)
        print(f"\n=== Testing with temperature = {temp} ===")

        # Test with different ambiguity levels
        for query in queries:
            query_embedding = query_encoder.encode(query)
            context = retriever.retrieve(query_embedding, embedded_db)
            integrated_context = integrator.integrate(context, context_size=3)
            response = generator.generate(query, integrated_context)

            print(f"\nQuery (ambiguity={query['ambiguity_level']}): {query['text']}")
            print(f"Response quality: {response['quality']}")
            print(f"Response accuracy: {response['accuracy']:.4f}")
            print(f"Response: {response['text']}")