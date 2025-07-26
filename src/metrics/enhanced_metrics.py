"""
Enhanced Metrics Module

This module provides comprehensive evaluation metrics for RAG systems including
BLEU, ROUGE scores, response time, memory usage, and other performance indicators
requested by reviewers.
"""

import time
import psutil
import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedMetrics:
    """Comprehensive metrics calculator for RAG systems."""

    def __init__(self):
        """Initialize the enhanced metrics calculator."""
        self.process = psutil.Process(os.getpid())
        logger.info("Initialized enhanced metrics calculator")

    def calculate_bleu_score(
        self,
        reference: str,
        candidate: str,
        max_n: int = 4,
        weights: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Calculate BLEU score for response evaluation.

        Args:
            reference: Reference/ground truth text
            candidate: Generated candidate text
            max_n: Maximum n-gram order
            weights: Weights for different n-gram orders

        Returns:
            Dictionary with BLEU scores
        """
        if weights is None:
            weights = [0.25, 0.25, 0.25, 0.25]  # Uniform weights for BLEU-4

        # Tokenize and normalize
        ref_tokens = self._tokenize(reference.lower())
        cand_tokens = self._tokenize(candidate.lower())

        if not cand_tokens:
            return {f"bleu_{i+1}": 0.0 for i in range(max_n)} | {"bleu_overall": 0.0}

        # Calculate n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            cand_ngrams = self._get_ngrams(cand_tokens, n)

            if not cand_ngrams:
                precisions.append(0.0)
                continue

            # Count matches
            matches = 0
            ref_counts = Counter(ref_ngrams)
            cand_counts = Counter(cand_ngrams)

            for ngram in cand_counts:
                matches += min(cand_counts[ngram], ref_counts.get(ngram, 0))

            precision = matches / len(cand_ngrams)
            precisions.append(precision)

        # Calculate brevity penalty
        ref_length = len(ref_tokens)
        cand_length = len(cand_tokens)

        if cand_length > ref_length:
            brevity_penalty = 1.0
        else:
            brevity_penalty = np.exp(1 - ref_length / cand_length) if cand_length > 0 else 0.0

        # Calculate final BLEU score
        if all(p > 0 for p in precisions):
            log_precision_sum = sum(w * np.log(p) for w, p in zip(weights[:len(precisions)], precisions))
            bleu_overall = brevity_penalty * np.exp(log_precision_sum)
        else:
            bleu_overall = 0.0

        # Return detailed scores
        result = {f"bleu_{i+1}": precisions[i] for i in range(len(precisions))}
        result["bleu_overall"] = bleu_overall
        result["brevity_penalty"] = brevity_penalty
        result["reference_length"] = ref_length
        result["candidate_length"] = cand_length

        return result

    def calculate_rouge_score(
        self,
        reference: str,
        candidate: str,
        rouge_types: List[str] = ["rouge-1", "rouge-2", "rouge-l"]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE scores for response evaluation.

        Args:
            reference: Reference/ground truth text
            candidate: Generated candidate text
            rouge_types: Types of ROUGE to calculate

        Returns:
            Dictionary with ROUGE scores
        """
        ref_tokens = self._tokenize(reference.lower())
        cand_tokens = self._tokenize(candidate.lower())

        results = {}

        for rouge_type in rouge_types:
            if rouge_type == "rouge-1":
                results["rouge-1"] = self._calculate_rouge_n(ref_tokens, cand_tokens, 1)
            elif rouge_type == "rouge-2":
                results["rouge-2"] = self._calculate_rouge_n(ref_tokens, cand_tokens, 2)
            elif rouge_type == "rouge-l":
                results["rouge-l"] = self._calculate_rouge_l(ref_tokens, cand_tokens)

        return results

    def _calculate_rouge_n(self, ref_tokens: List[str], cand_tokens: List[str], n: int) -> Dict[str, float]:
        """Calculate ROUGE-N score."""
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        cand_ngrams = self._get_ngrams(cand_tokens, n)

        if not ref_ngrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        ref_counts = Counter(ref_ngrams)
        cand_counts = Counter(cand_ngrams)

        # Calculate overlap
        overlap = 0
        for ngram in cand_counts:
            overlap += min(cand_counts[ngram], ref_counts.get(ngram, 0))

        # Calculate precision, recall, F1
        precision = overlap / len(cand_ngrams) if cand_ngrams else 0.0
        recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def _calculate_rouge_l(self, ref_tokens: List[str], cand_tokens: List[str]) -> Dict[str, float]:
        """Calculate ROUGE-L score using Longest Common Subsequence."""
        lcs_length = self._lcs_length(ref_tokens, cand_tokens)

        ref_length = len(ref_tokens)
        cand_length = len(cand_tokens)

        if ref_length == 0 or cand_length == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        precision = lcs_length / cand_length
        recall = lcs_length / ref_length
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of Longest Common Subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return [token for token in text.split() if token]

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Get n-grams from tokens."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def measure_response_time(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Measure execution time of a function.

        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (result, execution_time_ms)
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        return result, execution_time_ms

    def measure_memory_usage(self) -> Dict[str, float]:
        """
        Measure current memory usage of the process.

        Returns:
            Dictionary with memory usage statistics
        """
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        return {
            "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
            "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
            "memory_percent": memory_percent,  # Percentage of total system memory
            "available_memory_mb": psutil.virtual_memory().available / (1024 * 1024)
        }

    def calculate_semantic_similarity(
        self,
        text1: str,
        text2: str,
        method: str = "jaccard"
    ) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            method: Similarity method ("jaccard", "cosine", "overlap")

        Returns:
            Similarity score (0.0 to 1.0)
        """
        tokens1 = set(self._tokenize(text1.lower()))
        tokens2 = set(self._tokenize(text2.lower()))

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        if method == "jaccard":
            intersection = tokens1 & tokens2
            union = tokens1 | tokens2
            return len(intersection) / len(union) if union else 0.0

        elif method == "overlap":
            intersection = tokens1 & tokens2
            return len(intersection) / min(len(tokens1), len(tokens2))

        elif method == "cosine":
            # Simple cosine similarity using word counts
            all_tokens = tokens1 | tokens2
            vec1 = np.array([1 if token in tokens1 else 0 for token in all_tokens])
            vec2 = np.array([1 if token in tokens2 else 0 for token in all_tokens])
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def calculate_fluency_score(self, text: str) -> Dict[str, float]:
        """
        Calculate fluency score for generated text.

        Args:
            text: Text to evaluate

        Returns:
            Dictionary with fluency metrics
        """
        tokens = self._tokenize(text)
        sentences = self._split_sentences(text)

        if not tokens:
            return {
                "average_sentence_length": 0.0,
                "vocabulary_richness": 0.0,
                "repetition_penalty": 1.0,
                "fluency_score": 0.0
            }

        # Average sentence length
        avg_sentence_length = len(tokens) / len(sentences) if sentences else 0

        # Vocabulary richness (unique words / total words)
        unique_tokens = set(tokens)
        vocabulary_richness = len(unique_tokens) / len(tokens)

        # Repetition penalty (penalize repeated phrases)
        bigrams = self._get_ngrams(tokens, 2)
        unique_bigrams = set(bigrams)
        repetition_penalty = len(unique_bigrams) / len(bigrams) if bigrams else 1.0

        # Combined fluency score
        fluency_score = (
            min(1.0, avg_sentence_length / 20) * 0.3 +  # Prefer moderate sentence length
            vocabulary_richness * 0.4 +  # Reward vocabulary diversity
            repetition_penalty * 0.3  # Penalize repetition
        )

        return {
            "average_sentence_length": avg_sentence_length,
            "vocabulary_richness": vocabulary_richness,
            "repetition_penalty": repetition_penalty,
            "fluency_score": fluency_score
        }

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def calculate_comprehensive_metrics(
        self,
        query: Dict,
        response: Dict,
        ground_truth: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        memory_usage: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for a query-response pair.

        Args:
            query: Query dictionary
            response: Response dictionary
            ground_truth: Ground truth answer (if available)
            execution_time_ms: Execution time in milliseconds
            memory_usage: Memory usage statistics

        Returns:
            Comprehensive metrics dictionary
        """
        metrics = {
            "query_id": query.get("id", "unknown"),
            "basic_metrics": {
                "response_length": len(response.get("text", "")),
                "response_accuracy": response.get("accuracy", 0.0),
                "context_coherence": response.get("context_coherence", 0.0)
            },
            "timing_metrics": {
                "execution_time_ms": execution_time_ms or 0.0
            },
            "memory_metrics": memory_usage or {},
            "quality_metrics": {}
        }

        response_text = response.get("text", "")

        # Calculate fluency metrics
        fluency = self.calculate_fluency_score(response_text)
        metrics["quality_metrics"]["fluency"] = fluency

        # Calculate semantic similarity with query
        query_similarity = self.calculate_semantic_similarity(
            query.get("text", ""), response_text, method="jaccard"
        )
        metrics["quality_metrics"]["query_similarity"] = query_similarity

        # If ground truth is available, calculate BLEU and ROUGE
        if ground_truth:
            bleu_scores = self.calculate_bleu_score(ground_truth, response_text)
            rouge_scores = self.calculate_rouge_score(ground_truth, response_text)
            
            metrics["quality_metrics"]["bleu"] = bleu_scores
            metrics["quality_metrics"]["rouge"] = rouge_scores

            # Ground truth similarity
            gt_similarity = self.calculate_semantic_similarity(
                ground_truth, response_text, method="cosine"
            )
            metrics["quality_metrics"]["ground_truth_similarity"] = gt_similarity

        return metrics

    def aggregate_metrics(self, metrics_list: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple query-response pairs.

        Args:
            metrics_list: List of individual metrics dictionaries

        Returns:
            Aggregated metrics
        """
        if not metrics_list:
            return {}

        aggregated = {
            "num_samples": len(metrics_list),
            "basic_metrics": {},
            "timing_metrics": {},
            "memory_metrics": {},
            "quality_metrics": {}
        }

        # Aggregate basic metrics
        response_lengths = [m["basic_metrics"]["response_length"] for m in metrics_list]
        accuracies = [m["basic_metrics"]["response_accuracy"] for m in metrics_list]
        coherences = [m["basic_metrics"]["context_coherence"] for m in metrics_list]

        aggregated["basic_metrics"] = {
            "avg_response_length": np.mean(response_lengths),
            "std_response_length": np.std(response_lengths),
            "avg_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "avg_coherence": np.mean(coherences),
            "std_coherence": np.std(coherences)
        }

        # Aggregate timing metrics
        execution_times = [m["timing_metrics"]["execution_time_ms"] for m in metrics_list]
        aggregated["timing_metrics"] = {
            "avg_execution_time_ms": np.mean(execution_times),
            "std_execution_time_ms": np.std(execution_times),
            "min_execution_time_ms": np.min(execution_times),
            "max_execution_time_ms": np.max(execution_times),
            "total_execution_time_ms": np.sum(execution_times)
        }

        # Aggregate memory metrics (if available)
        memory_metrics = [m["memory_metrics"] for m in metrics_list if m["memory_metrics"]]
        if memory_metrics:
            rss_values = [m.get("rss_mb", 0) for m in memory_metrics]
            aggregated["memory_metrics"] = {
                "avg_memory_usage_mb": np.mean(rss_values),
                "max_memory_usage_mb": np.max(rss_values),
                "memory_efficiency": 1.0 / (np.mean(rss_values) + 1)  # Simple efficiency metric
            }

        # Aggregate quality metrics
        fluency_scores = [m["quality_metrics"]["fluency"]["fluency_score"] for m in metrics_list]
        query_similarities = [m["quality_metrics"]["query_similarity"] for m in metrics_list]

        aggregated["quality_metrics"] = {
            "avg_fluency": np.mean(fluency_scores),
            "std_fluency": np.std(fluency_scores),
            "avg_query_similarity": np.mean(query_similarities),
            "std_query_similarity": np.std(query_similarities)
        }

        # Aggregate BLEU and ROUGE if available
        bleu_metrics = [m["quality_metrics"].get("bleu") for m in metrics_list if "bleu" in m["quality_metrics"]]
        if bleu_metrics:
            bleu_overall = [b["bleu_overall"] for b in bleu_metrics]
            aggregated["quality_metrics"]["avg_bleu"] = np.mean(bleu_overall)
            aggregated["quality_metrics"]["std_bleu"] = np.std(bleu_overall)

        rouge_metrics = [m["quality_metrics"].get("rouge") for m in metrics_list if "rouge" in m["quality_metrics"]]
        if rouge_metrics:
            rouge_1_f1 = [r["rouge-1"]["f1"] for r in rouge_metrics if "rouge-1" in r]
            rouge_2_f1 = [r["rouge-2"]["f1"] for r in rouge_metrics if "rouge-2" in r]
            rouge_l_f1 = [r["rouge-l"]["f1"] for r in rouge_metrics if "rouge-l" in r]

            if rouge_1_f1:
                aggregated["quality_metrics"]["avg_rouge_1_f1"] = np.mean(rouge_1_f1)
            if rouge_2_f1:
                aggregated["quality_metrics"]["avg_rouge_2_f1"] = np.mean(rouge_2_f1)
            if rouge_l_f1:
                aggregated["quality_metrics"]["avg_rouge_l_f1"] = np.mean(rouge_l_f1)

        return aggregated

    def generate_metrics_report(self, aggregated_metrics: Dict) -> str:
        """
        Generate a human-readable metrics report.

        Args:
            aggregated_metrics: Aggregated metrics dictionary

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("ENHANCED METRICS REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Samples analyzed: {aggregated_metrics.get('num_samples', 0)}")
        report_lines.append("")

        # Basic metrics
        basic = aggregated_metrics.get("basic_metrics", {})
        if basic:
            report_lines.append("BASIC PERFORMANCE METRICS:")
            report_lines.append("-" * 30)
            report_lines.append(f"Average Accuracy: {basic.get('avg_accuracy', 0):.4f} ± {basic.get('std_accuracy', 0):.4f}")
            report_lines.append(f"Average Response Length: {basic.get('avg_response_length', 0):.1f} chars")
            report_lines.append(f"Average Coherence: {basic.get('avg_coherence', 0):.4f} ± {basic.get('std_coherence', 0):.4f}")
            report_lines.append("")

        # Timing metrics
        timing = aggregated_metrics.get("timing_metrics", {})
        if timing:
            report_lines.append("PERFORMANCE TIMING:")
            report_lines.append("-" * 30)
            report_lines.append(f"Average Response Time: {timing.get('avg_execution_time_ms', 0):.2f} ms")
            report_lines.append(f"Min/Max Response Time: {timing.get('min_execution_time_ms', 0):.2f} / {timing.get('max_execution_time_ms', 0):.2f} ms")
            report_lines.append(f"Total Processing Time: {timing.get('total_execution_time_ms', 0):.2f} ms")
            report_lines.append("")

        # Memory metrics
        memory = aggregated_metrics.get("memory_metrics", {})
        if memory:
            report_lines.append("MEMORY USAGE:")
            report_lines.append("-" * 30)
            report_lines.append(f"Average Memory Usage: {memory.get('avg_memory_usage_mb', 0):.2f} MB")
            report_lines.append(f"Peak Memory Usage: {memory.get('max_memory_usage_mb', 0):.2f} MB")
            report_lines.append(f"Memory Efficiency: {memory.get('memory_efficiency', 0):.4f}")
            report_lines.append("")

        # Quality metrics
        quality = aggregated_metrics.get("quality_metrics", {})
        if quality:
            report_lines.append("QUALITY METRICS:")
            report_lines.append("-" * 30)
            report_lines.append(f"Average Fluency: {quality.get('avg_fluency', 0):.4f} ± {quality.get('std_fluency', 0):.4f}")
            report_lines.append(f"Query Similarity: {quality.get('avg_query_similarity', 0):.4f} ± {quality.get('std_query_similarity', 0):.4f}")
            
            if "avg_bleu" in quality:
                report_lines.append(f"BLEU Score: {quality['avg_bleu']:.4f} ± {quality.get('std_bleu', 0):.4f}")
            
            if "avg_rouge_1_f1" in quality:
                report_lines.append(f"ROUGE-1 F1: {quality['avg_rouge_1_f1']:.4f}")
            if "avg_rouge_2_f1" in quality:
                report_lines.append(f"ROUGE-2 F1: {quality['avg_rouge_2_f1']:.4f}")
            if "avg_rouge_l_f1" in quality:
                report_lines.append(f"ROUGE-L F1: {quality['avg_rouge_l_f1']:.4f}")

        return "\n".join(report_lines)


if __name__ == "__main__":
    # Example usage
    metrics_calculator = EnhancedMetrics()

    # Test BLEU calculation
    reference = "The quick brown fox jumps over the lazy dog"
    candidate = "A quick brown fox jumps over a lazy dog"
    
    bleu_scores = metrics_calculator.calculate_bleu_score(reference, candidate)
    print("BLEU Scores:")
    for metric, score in bleu_scores.items():
        print(f"  {metric}: {score:.4f}")

    # Test ROUGE calculation
    rouge_scores = metrics_calculator.calculate_rouge_score(reference, candidate)
    print("\nROUGE Scores:")
    for rouge_type, scores in rouge_scores.items():
        print(f"  {rouge_type}:")
        for metric, score in scores.items():
            print(f"    {metric}: {score:.4f}")

    # Test timing measurement
    def dummy_function(x):
        time.sleep(0.1)  # Simulate processing
        return x * 2

    result, exec_time = metrics_calculator.measure_response_time(dummy_function, 5)
    print(f"\nTiming Test:")
    print(f"  Result: {result}")
    print(f"  Execution time: {exec_time:.2f} ms")

    # Test memory measurement
    memory_usage = metrics_calculator.measure_memory_usage()
    print(f"\nMemory Usage:")
    for metric, value in memory_usage.items():
        print(f"  {metric}: {value:.2f}")

    # Test comprehensive metrics
    sample_query = {"id": 1, "text": "What is machine learning?"}
    sample_response = {"text": "Machine learning is a method of artificial intelligence.", "accuracy": 0.85}
    
    comprehensive = metrics_calculator.calculate_comprehensive_metrics(
        sample_query, sample_response, ground_truth=reference, 
        execution_time_ms=exec_time, memory_usage=memory_usage
    )
    
    print(f"\nComprehensive Metrics:")
    print(f"  Query similarity: {comprehensive['quality_metrics']['query_similarity']:.4f}")
    print(f"  Fluency score: {comprehensive['quality_metrics']['fluency']['fluency_score']:.4f}")
