"""
Schema Generator Module

This module provides functionality to generate synthetic database schemas
with controlled entropy levels for information-theoretic experiments.
"""

import numpy as np
import math
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchemaGenerator:
    """Generate synthetic database schemas with controlled entropy levels."""

    def __init__(self, seed: int = 42):
        """
        Initialize the schema generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)

    def generate_schema(self, entropy_bits: float) -> Dict:
        """
        Generate a synthetic schema with specified entropy in bits.

        Args:
            entropy_bits: Target entropy for the schema (1-10 bits)

        Returns:
            Dictionary representing the schema
        """
        # Calculate number of tables needed for target entropy
        num_tables = max(1, int(entropy_bits / 2))

        schema = {
            "metadata": {
                "entropy_bits": entropy_bits,
                "num_tables": num_tables
            },
            "tables": {}
        }

        # Generate tables with relationships to achieve target entropy
        remaining_entropy = entropy_bits
        for i in range(num_tables):
            table_name = f"table_{i}"

            # Allocate portion of entropy to this table
            if i < num_tables - 1:
                table_entropy = min(remaining_entropy / 2, 3)  # Max 3 bits per table
            else:
                table_entropy = remaining_entropy  # Last table gets remaining entropy

            # Calculate number of columns based on target entropy
            num_columns = max(2, int(table_entropy * 2))

            # Generate columns
            columns = {}
            for j in range(num_columns):
                col_name = f"col_{j}"
                # Assign random data type
                data_type = self.rng.choice(["integer", "float", "string", "date", "boolean"])
                columns[col_name] = {"type": data_type}

                # Mark some columns as primary or foreign keys
                if j == 0:
                    columns[col_name]["primary_key"] = True
                elif i > 0 and j == 1:
                    # Add foreign key to previous table
                    ref_table = f"table_{i - 1}"
                    columns[col_name]["foreign_key"] = {
                        "references": ref_table,
                        "column": "col_0"  # Reference primary key
                    }

            schema["tables"][table_name] = {
                "columns": columns,
                "entropy_contribution": table_entropy
            }

            remaining_entropy -= table_entropy

        # Verify that schema entropy matches target
        calculated_entropy = self._calculate_schema_entropy(schema)
        logger.info(f"Generated schema with target entropy {entropy_bits} bits, achieved {calculated_entropy:.2f} bits")

        return schema

    def _calculate_schema_entropy(self, schema: Dict) -> float:
        """
        Calculate the actual entropy of a generated schema.

        Args:
            schema: The database schema

        Returns:
            Calculated entropy in bits
        """
        # This is a simplified calculation - in a real implementation,
        # this would account for all schema properties

        num_tables = len(schema["tables"])
        total_columns = sum(len(table["columns"]) for table in schema["tables"].values())
        relationships = sum(1 for table in schema["tables"].values()
                            for col in table["columns"].values()
                            if "foreign_key" in col)

        # Approximate entropy calculation
        table_entropy = math.log2(num_tables + 1)
        column_entropy = math.log2(total_columns + 1)
        relationship_entropy = math.log2(relationships + 1)

        return table_entropy + column_entropy + relationship_entropy


if __name__ == "__main__":
    # Example usage
    generator = SchemaGenerator(seed=42)

    # Generate schemas with different entropy levels
    for entropy in [2.0, 5.0, 8.0]:
        schema = generator.generate_schema(entropy_bits=entropy)
        print(f"\nSchema with target entropy {entropy} bits:")
        print(f"Number of tables: {schema['metadata']['num_tables']}")
        for table_name, table_data in schema["tables"].items():
            print(f"  Table: {table_name}")
            print(f"  Columns: {len(table_data['columns'])}")
            print(f"  Entropy contribution: {table_data['entropy_contribution']:.2f} bits")