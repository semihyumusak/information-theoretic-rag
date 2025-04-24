"""
Query Generator Module

This module provides functionality to generate synthetic natural language queries
with controlled ambiguity levels for information-theoretic experiments.
"""

import numpy as np
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryGenerator:
    """Generate natural language queries with controlled ambiguity levels."""

    def __init__(self, seed: int = 42):
        """
        Initialize the query generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)

    def generate_queries(self, schema: Dict, num_queries: int = 1000,
                         ambiguity_levels: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]) -> List[Dict]:
        """
        Generate natural language queries with varying ambiguity levels.

        Args:
            schema: Database schema
            num_queries: Total number of queries to generate
            ambiguity_levels: List of ambiguity levels (0.0-1.0)

        Returns:
            List of query dictionaries with ground truth
        """
        queries = []

        # Define query templates based on common database operations
        select_templates = [
            "Get {column} from {table}",
            "Show me the {column} in {table}",
            "What is the {column} from {table}",
            "I need the {column} data from {table}",
            "Retrieve {column} values from {table}"
        ]

        filter_templates = [
            "Find {column1} from {table} where {column2} is {value}",
            "Show {column1} in {table} with {column2} equal to {value}",
            "What is the {column1} in {table} when {column2} is {value}",
            "Get {column1} from {table} filtered by {column2} = {value}",
            "List {column1} from {table} where {column2} matches {value}"
        ]

        join_templates = [
            "Get {column1} from {table1} and {column2} from {table2} where they are related",
            "Show me {column1} in {table1} along with the corresponding {column2} from {table2}",
            "What are the {column1} values in {table1} and their related {column2} in {table2}",
            "Join {table1} and {table2} to show {column1} and {column2}",
            "Combine {column1} from {table1} with {column2} from {table2}"
        ]

        aggregate_templates = [
            "What is the average {column} in {table}",
            "Count the number of {column} in {table}",
            "Find the maximum {column} value in {table}",
            "Calculate the sum of {column} in {table}",
            "Show the minimum {column} from {table}"
        ]

        # Combine all templates
        all_templates = select_templates + filter_templates + join_templates + aggregate_templates

        # Generate queries for each ambiguity level
        queries_per_level = num_queries // len(ambiguity_levels)

        for ambiguity_level in ambiguity_levels:
            for _ in range(queries_per_level):
                # Select a random template
                template = self.rng.choice(all_templates)

                # Select random tables and columns
                table_names = list(schema["tables"].keys())

                if "table1" in template and "table2" in template:
                    # Join query - need two tables
                    if len(table_names) >= 2:
                        table1, table2 = self.rng.choice(table_names, size=2, replace=False)
                    else:
                        # Not enough tables, use the same table twice
                        table1 = table2 = table_names[0]

                    column1 = self.rng.choice(list(schema["tables"][table1]["columns"].keys()))
                    column2 = self.rng.choice(list(schema["tables"][table2]["columns"].keys()))

                    query_text = template.format(
                        table1=table1,
                        table2=table2,
                        column1=column1,
                        column2=column2
                    )

                    ground_truth = {
                        "operation": "join",
                        "tables": [table1, table2],
                        "columns": [column1, column2]
                    }

                elif "column1" in template and "column2" in template:
                    # Filter query
                    table = self.rng.choice(table_names)
                    column_names = list(schema["tables"][table]["columns"].keys())

                    if len(column_names) >= 2:
                        column1, column2 = self.rng.choice(column_names, size=2, replace=False)
                    else:
                        column1 = column2 = column_names[0]

                    # Generate a random value based on column type
                    col_type = schema["tables"][table]["columns"][column2]["type"]
                    if col_type == "integer":
                        value = str(self.rng.randint(0, 100))
                    elif col_type == "float":
                        value = f"{self.rng.random() * 100:.2f}"
                    elif col_type == "string":
                        value = f"'{chr(self.rng.randint(97, 122))}'"
                    elif col_type == "date":
                        value = f"'2023-{self.rng.randint(1, 13):02d}-{self.rng.randint(1, 28):02d}'"
                    else:  # boolean
                        value = "true" if self.rng.random() > 0.5 else "false"

                    query_text = template.format(
                        table=table,
                        column1=column1,
                        column2=column2,
                        value=value
                    )

                    ground_truth = {
                        "operation": "filter",
                        "table": table,
                        "columns": [column1, column2],
                        "filter_value": value
                    }

                else:
                    # Simple select or aggregate
                    table = self.rng.choice(table_names)
                    column = self.rng.choice(list(schema["tables"][table]["columns"].keys()))

                    query_text = template.format(table=table, column=column)

                    if any(agg in template.lower() for agg in ["average", "count", "maximum", "minimum", "sum"]):
                        operation = "aggregate"
                        # Extract the aggregation function
                        if "average" in template.lower():
                            agg_function = "avg"
                        elif "count" in template.lower():
                            agg_function = "count"
                        elif "maximum" in template.lower():
                            agg_function = "max"
                        elif "minimum" in template.lower():
                            agg_function = "min"
                        else:
                            agg_function = "sum"
                    else:
                        operation = "select"
                        agg_function = None

                    ground_truth = {
                        "operation": operation,
                        "table": table,
                        "column": column,
                        "aggregation": agg_function
                    }

                # Add ambiguity based on the level
                if ambiguity_level > 0:
                    ambiguous_query = self._add_ambiguity(query_text, ambiguity_level)
                else:
                    ambiguous_query = query_text

                # Create the query object
                query = {
                    "id": len(queries),
                    "text": ambiguous_query,
                    "original_text": query_text,
                    "ambiguity_level": ambiguity_level,
                    "ground_truth": ground_truth
                }

                queries.append(query)

        logger.info(f"Generated {len(queries)} queries with ambiguity levels {ambiguity_levels}")
        return queries

    def _add_ambiguity(self, query_text: str, ambiguity_level: float) -> str:
        """
        Add controlled ambiguity to a query.

        Args:
            query_text: Original query text
            ambiguity_level: Level of ambiguity to add (0.0-1.0)

        Returns:
            Query with added ambiguity
        """
        # Implement different ambiguity transformations based on level

        if ambiguity_level < 0.3:
            # Low ambiguity: just add some filler words
            fillers = ["please", "I would like to", "could you", "I need to", "help me"]
            filler = self.rng.choice(fillers)
            return f"{filler} {query_text}"

        elif ambiguity_level < 0.6:
            # Medium ambiguity: make references less specific
            # Replace specific column names with more general terms
            words = query_text.split()
            for i in range(len(words)):
                if words[i].startswith("col_"):
                    if self.rng.random() < 0.5:
                        words[i] = "the data"

                if words[i].startswith("table_"):
                    if self.rng.random() < 0.5:
                        words[i] = "the table"

            return " ".join(words)

        else:
            # High ambiguity: introduce vague requirements and implied joins
            vague_queries = [
                f"What are the related {query_text.split()[-1]} values",
                f"Tell me about {query_text.split()[-1]}",
                f"I need information related to {query_text.split()[-1]}",
                f"Show me the data for {query_text.split()[-1]}",
                f"Can you find {query_text.split()[-1]} information"
            ]
            return self.rng.choice(vague_queries)


if __name__ == "__main__":
    # Example usage (requires schema_generator.py)
    from schema_generator import SchemaGenerator

    # Generate a schema
    schema_gen = SchemaGenerator(seed=42)
    schema = schema_gen.generate_schema(entropy_bits=5.0)

    # Generate queries
    query_gen = QueryGenerator(seed=42)
    queries = query_gen.generate_queries(
        schema,
        num_queries=50,
        ambiguity_levels=[0.0, 0.5, 0.9]
    )

    # Print some examples
    for i, level in enumerate([0.0, 0.5, 0.9]):
        print(f"\nAmbiguity Level: {level}")
        examples = [q for q in queries if q["ambiguity_level"] == level][:3]
        for j, query in enumerate(examples):
            print(f"Example {j + 1}:")
            print(f"  Original: {query['original_text']}")
            print(f"  Ambiguous: {query['text']}")
            print(f"  Ground Truth: {query['ground_truth']}")