"""
Experiments Module for RAG Information Theory Research

This module provides implementations of the four main experiments:
- Channel Capacity Experiment
- Bottleneck Analysis Experiment
- Error Propagation Experiment
- Schema Entropy Experiment

These experiments validate the theoretical framework presented in the paper
"Information-Theoretic Analysis of Retrieval-Augmented Generation in Database Systems".
"""

from .experiment_runner import ExperimentRunner
from .channel_capacity import run_channel_capacity_experiment
from .bottleneck_analysis import run_bottleneck_analysis_experiment
from .error_propagation import run_error_propagation_experiment
from .schema_entropy import run_schema_entropy_experiment

__all__ = [
    'ExperimentRunner',
    'run_channel_capacity_experiment',
    'run_bottleneck_analysis_experiment',
    'run_error_propagation_experiment',
    'run_schema_entropy_experiment'
]