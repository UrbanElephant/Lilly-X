"""
RAG Evaluation Module for LLIX.

Provides automated metrics for evaluating RAG query quality including
faithfulness, answer relevancy, and context precision.

Supports both simple placeholder metrics (always available) and 
advanced ragas metrics (when ragas package is installed).
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for RAG evaluation metrics."""
    
    query: str
    response: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    
    def __str__(self) -> str:
        """Format evaluation result as human-readable string."""
        lines = [
            f"Query: {self.query[:100]}...",
            f"Response: {self.response[:100]}...",
            "Metrics:"
        ]
        for metric, value in self.metrics.items():
            lines.append(f"  {metric}: {value:.3f}")
        return "\n".join(lines)


class RAGEvaluator:
    """
    Evaluator for RAG system quality metrics.
    
    Supports two modes:
    1. Simple metrics (always available): Basic keyword-based metrics
    2. Ragas metrics (optional): Advanced LLM-based evaluation
    
    Usage:
        # Simple metrics
        evaluator = RAGEvaluator(use_ragas=False)
        result = evaluator.evaluate_query(
            query="What is Python?",
            response="Python is a programming language",
            context=["Python is a high-level programming language"]
        )
        print(result.metrics)
        
        # With ragas (requires: pip install ragas)
        evaluator = RAGEvaluator(use_ragas=True)
        result = evaluator.evaluate_query(...)
    """
    
    def __init__(self, use_ragas: bool = False):
        """
        Initialize RAG evaluator.
        
        Args:
            use_ragas: If True, attempt to use ragas for advanced metrics.
                      Falls back to simple metrics if ragas not installed.
        """
        self.use_ragas = use_ragas
        self._ragas_available = False
        
        if use_ragas:
            try:
                # Attempt to import ragas
                from ragas import evaluate
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall
                )
                self._ragas_evaluate = evaluate
                self._ragas_metrics = {
                    'faithfulness': faithfulness,
                    'answer_relevancy': answer_relevancy,
                    'context_precision': context_precision,
                    'context_recall': context_recall
                }
                self._ragas_available = True
                logger.info("Ragas metrics enabled")
            except ImportError:
                logger.warning(
                    "Ragas not installed. Install with: pip install ragas\n"
                    "Falling back to simple metrics."
                )
                self.use_ragas = False
    
    def evaluate_query(
        self,
        query: str,
        response: str,
        context: List[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a single RAG query-response pair.
        
        Args:
            query: User query string
            response: RAG system response
            context: List of context chunks retrieved for this query
            ground_truth: Optional ground truth answer (for context_recall metric)
            
        Returns:
            EvaluationResult with metrics dictionary
        """
        if self._ragas_available and self.use_ragas:
            metrics = self._evaluate_with_ragas(query, response, context, ground_truth)
        else:
            metrics = self._evaluate_simple(query, response, context)
        
        return EvaluationResult(
            query=query,
            response=response,
            metrics=metrics,
            metadata={
                'num_context_chunks': len(context),
                'response_length': len(response),
                'evaluator_type': 'ragas' if self._ragas_available else 'simple'
            }
        )
    
    def evaluate_batch(
        self,
        queries: List[str],
        responses: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple query-response pairs efficiently.
        
        Args:
            queries: List of query strings
            responses: List of response strings
            contexts: List of context chunk lists
            ground_truths: Optional list of ground truth answers
            
        Returns:
            List of EvaluationResult objects
        """
        if ground_truths is None:
            ground_truths = [None] * len(queries)
        
        results = []
        for query, response, context, gt in zip(queries, responses, contexts, ground_truths):
            result = self.evaluate_query(query, response, context, gt)
            results.append(result)
        
        return results
    
    def _evaluate_simple(
        self,
        query: str,
        response: str,
        context: List[str]
    ) -> Dict[str, float]:
        """
        Simple keyword-based metrics (always available).
        
        Returns:
            Dictionary of metric name -> score (0.0 to 1.0)
        """
        metrics = {}
        
        # Extract keywords from query (simple tokenization)
        query_keywords = set(query.lower().split())
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where'}
        query_keywords = query_keywords - stop_words
        
        # Context Precision: % of context chunks containing query keywords
        if context and query_keywords:
            relevant_chunks = 0
            for chunk in context:
                chunk_lower = chunk.lower()
                if any(keyword in chunk_lower for keyword in query_keywords):
                    relevant_chunks += 1
            metrics['context_precision'] = relevant_chunks / len(context)
        else:
            metrics['context_precision'] = 0.0
        
        # Context Utilization: Check if response references context
        if context:
            context_text = ' '.join(context).lower()
            response_lower = response.lower()
            
            # Count overlapping words between context and response
            context_words = set(context_text.split())
            response_words = set(response_lower.split())
            overlap = context_words & response_words
            
            if response_words:
                metrics['context_utilization'] = len(overlap) / len(response_words)
            else:
                metrics['context_utilization'] = 0.0
        else:
            metrics['context_utilization'] = 0.0
        
        # Response Length (normalized to 0-1, assuming 500 chars is optimal)
        optimal_length = 500
        actual_length = len(response)
        if actual_length == 0:
            metrics['response_length_score'] = 0.0
        else:
            # Penalty for too short or too long
            ratio = min(actual_length, optimal_length) / optimal_length
            metrics['response_length_score'] = ratio
        
        # Query Coverage: % of query keywords present in response
        if query_keywords:
            response_lower = response.lower()
            covered_keywords = sum(1 for kw in query_keywords if kw in response_lower)
            metrics['query_coverage'] = covered_keywords / len(query_keywords)
        else:
            metrics['query_coverage'] = 0.0
        
        return metrics
    
    def _evaluate_with_ragas(
        self,
        query: str,
        response: str,
        context: List[str],
        ground_truth: Optional[str]
    ) -> Dict[str, float]:
        """
        Evaluate using ragas library (advanced LLM-based metrics).
        
        NOTE: This is structured for ragas integration but requires
        the ragas package to be installed and properly configured.
        
        Args:
            query: User query
            response: RAG response
            context: Retrieved context chunks
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary of metric name -> score
        """
        try:
            # Prepare dataset in ragas format
            from datasets import Dataset
            
            dataset_dict = {
                'question': [query],
                'answer': [response],
                'contexts': [context],
            }
            
            # Add ground truth if available
            if ground_truth:
                dataset_dict['ground_truth'] = [ground_truth]
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Select metrics based on available data
            metrics_to_use = [
                self._ragas_metrics['faithfulness'],
                self._ragas_metrics['answer_relevancy'],
                self._ragas_metrics['context_precision']
            ]
            
            # Only use context_recall if we have ground truth
            if ground_truth:
                metrics_to_use.append(self._ragas_metrics['context_recall'])
            
            # Run evaluation
            result = self._ragas_evaluate(
                dataset,
                metrics=metrics_to_use
            )
            
            # Extract scores from ragas result
            return {
                metric_name: float(result[metric_name])
                for metric_name in result.keys()
                if metric_name != 'question' and metric_name != 'answer'
            }
            
        except Exception as e:
            logger.error(f"Ragas evaluation failed: {e}. Falling back to simple metrics.")
            return self._evaluate_simple(query, response, context)


def demo_evaluation():
    """
    Demonstration of RAG evaluation usage.
    
    Run with: python -m src.evaluation
    """
    # Sample data
    query = "What is machine learning?"
    response = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
    context = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "It is a branch of artificial intelligence based on the idea that systems can learn from data.",
        "Machine learning algorithms build a model based on sample data, known as training data."
    ]
    
    # Test simple metrics
    print("=" * 60)
    print("RAG Evaluation Demo - Simple Metrics")
    print("=" * 60)
    
    evaluator = RAGEvaluator(use_ragas=False)
    result = evaluator.evaluate_query(query, response, context)
    print(result)
    print()
    
    # Test ragas metrics (if available)
    print("=" * 60)
    print("RAG Evaluation Demo - Ragas Metrics (if installed)")
    print("=" * 60)
    
    evaluator_ragas = RAGEvaluator(use_ragas=True)
    result_ragas = evaluator_ragas.evaluate_query(query, response, context)
    print(result_ragas)
    print()
    
    # Batch evaluation demo
    print("=" * 60)
    print("Batch Evaluation Demo")
    print("=" * 60)
    
    queries = [query, "What is deep learning?"]
    responses = [response, "Deep learning uses neural networks with multiple layers."]
    contexts = [context, ["Deep learning is a subset of machine learning using neural networks."]]
    
    batch_results = evaluator.evaluate_batch(queries, responses, contexts)
    for i, res in enumerate(batch_results):
        print(f"\nQuery {i+1}: {res.query[:50]}...")
        print(f"Metrics: {res.metrics}")


if __name__ == "__main__":
    demo_evaluation()
