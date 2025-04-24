#!/usr/bin/env python3
"""
Main Script for RAG Information Theory Experiments

This script provides a command-line interface to run the experiments
for the research paper "Information-Theoretic Analysis of Retrieval-Augmented
Generation in Database Systems".
"""

import os
import argparse
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the experiment runner
from experiments.experiment_runner import ExperimentRunner


def main() -> None:
    """Main function to run the experiments."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run RAG information theory experiments.")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "channel_capacity", "bottleneck", "error_propagation", "schema_entropy"],
                        help="Which experiment to run")
    parser.add_argument("--figures_only", action="store_true",
                        help="Only generate figures from saved results")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save results")

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize experiment runner
    runner = ExperimentRunner(output_dir=args.output_dir)

    # Load existing results if only generating figures
    if args.figures_only:
        logger.info("Loading existing results to generate figures")
        runner.load_results()
        runner.generate_figures()
        runner.generate_tables()
        return

    # Run experiments based on command-line arguments
    runner.run_experiment(args.experiment)


if __name__ == "__main__":
    main()