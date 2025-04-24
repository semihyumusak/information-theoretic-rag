"""
RAG Pipeline Module for Information Theory Experiments

This module provides components that implement the four channels of the RAG pipeline:
- Query Encoding (C₁)
- Retrieval (C₂)
- Context Integration (C₃)
- Generation (C₄)
"""

from .query_encoder import QueryEncoder
from .database_encoder import DatabaseEncoder
from .retriever import Retriever
from .context_integrator import ContextIntegrator
from .generator import Generator

__all__ = ['QueryEncoder', 'DatabaseEncoder', 'Retriever', 'ContextIntegrator', 'Generator']