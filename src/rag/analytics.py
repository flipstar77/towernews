"""RAG Analytics - Monitoring and evaluation for the RAG system.

Features:
- Query logging and analysis
- Performance metrics (latency, throughput)
- Quality metrics (zero results, low similarity)
- Popular queries tracking
- Evaluation against ground truth
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import Counter
import statistics


@dataclass
class QueryLog:
    """Log entry for a single query."""
    query: str
    timestamp: str
    latency_ms: float
    num_results: int
    top_similarity: float
    filters: Dict[str, Any] = field(default_factory=dict)
    expanded_query: str = ""
    user_feedback: Optional[str] = None  # "relevant", "not_relevant", None


@dataclass
class PerformanceMetrics:
    """Performance metrics for a time period."""
    total_queries: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    zero_result_rate: float
    low_similarity_rate: float  # < 0.5 similarity
    queries_per_minute: float


@dataclass
class QualityMetrics:
    """Quality metrics from evaluation."""
    mrr: float  # Mean Reciprocal Rank
    recall_at_5: float
    recall_at_10: float
    ndcg: float  # Normalized Discounted Cumulative Gain
    precision_at_5: float
    num_evaluated: int


class RAGAnalytics:
    """
    Analytics and monitoring for the RAG system.

    Tracks queries, performance, and quality metrics.
    """

    def __init__(self, data_dir: str = "data/rag_analytics"):
        """
        Initialize analytics.

        Args:
            data_dir: Directory for storing analytics data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.logs_file = self.data_dir / "query_logs.jsonl"
        self.metrics_file = self.data_dir / "metrics.json"
        self.popular_file = self.data_dir / "popular_queries.json"

        # In-memory buffer for batch writes
        self._log_buffer: List[QueryLog] = []
        self._buffer_size = 100

    def log_query(
        self,
        query: str,
        results: List[Any],
        latency_ms: float,
        filters: Dict[str, Any] = None,
        expanded_query: str = "",
        user_feedback: str = None
    ):
        """
        Log a query for analytics.

        Args:
            query: Original query string
            results: Search results
            latency_ms: Query latency in milliseconds
            filters: Applied filters
            expanded_query: Query after preprocessing
            user_feedback: Optional user feedback
        """
        # Calculate metrics
        num_results = len(results)
        top_similarity = 0.0

        if results:
            if hasattr(results[0], 'similarity'):
                top_similarity = results[0].similarity
            elif isinstance(results[0], dict):
                top_similarity = results[0].get('similarity', 0.0)

        log_entry = QueryLog(
            query=query,
            timestamp=datetime.now().isoformat(),
            latency_ms=latency_ms,
            num_results=num_results,
            top_similarity=top_similarity,
            filters=filters or {},
            expanded_query=expanded_query,
            user_feedback=user_feedback
        )

        self._log_buffer.append(log_entry)

        # Flush if buffer is full
        if len(self._log_buffer) >= self._buffer_size:
            self.flush_logs()

    def flush_logs(self):
        """Write buffered logs to disk."""
        if not self._log_buffer:
            return

        with open(self.logs_file, 'a', encoding='utf-8') as f:
            for log in self._log_buffer:
                f.write(json.dumps(asdict(log)) + '\n')

        self._log_buffer.clear()

    def get_recent_logs(self, hours: int = 24, limit: int = 1000) -> List[QueryLog]:
        """Get recent query logs."""
        if not self.logs_file.exists():
            return []

        cutoff = datetime.now() - timedelta(hours=hours)
        logs = []

        with open(self.logs_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if timestamp >= cutoff:
                        logs.append(QueryLog(**data))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        return logs[-limit:]

    def get_performance_metrics(self, hours: int = 24) -> PerformanceMetrics:
        """Calculate performance metrics for recent queries."""
        logs = self.get_recent_logs(hours=hours)

        if not logs:
            return PerformanceMetrics(
                total_queries=0,
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                zero_result_rate=0,
                low_similarity_rate=0,
                queries_per_minute=0
            )

        latencies = [log.latency_ms for log in logs]
        latencies_sorted = sorted(latencies)

        zero_results = sum(1 for log in logs if log.num_results == 0)
        low_similarity = sum(1 for log in logs if log.top_similarity < 0.5 and log.num_results > 0)

        # Calculate time span
        timestamps = [datetime.fromisoformat(log.timestamp) for log in logs]
        time_span_minutes = (max(timestamps) - min(timestamps)).total_seconds() / 60
        qpm = len(logs) / max(time_span_minutes, 1)

        return PerformanceMetrics(
            total_queries=len(logs),
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=self._percentile(latencies_sorted, 50),
            p95_latency_ms=self._percentile(latencies_sorted, 95),
            p99_latency_ms=self._percentile(latencies_sorted, 99),
            zero_result_rate=zero_results / len(logs),
            low_similarity_rate=low_similarity / len(logs),
            queries_per_minute=qpm
        )

    def _percentile(self, sorted_data: List[float], p: int) -> float:
        """Calculate percentile from sorted data."""
        if not sorted_data:
            return 0
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    def get_popular_queries(self, hours: int = 168, top_k: int = 20) -> List[Dict[str, Any]]:
        """Get most popular queries."""
        logs = self.get_recent_logs(hours=hours, limit=10000)

        # Normalize queries (lowercase, strip)
        query_counter = Counter(log.query.lower().strip() for log in logs)

        return [
            {"query": query, "count": count}
            for query, count in query_counter.most_common(top_k)
        ]

    def get_zero_result_queries(self, hours: int = 168) -> List[str]:
        """Get queries that returned no results."""
        logs = self.get_recent_logs(hours=hours, limit=10000)

        zero_results = [log.query for log in logs if log.num_results == 0]

        # Return unique queries
        return list(set(zero_results))

    def get_slow_queries(self, threshold_ms: float = 500, hours: int = 24) -> List[QueryLog]:
        """Get queries slower than threshold."""
        logs = self.get_recent_logs(hours=hours)
        return [log for log in logs if log.latency_ms > threshold_ms]


class RAGEvaluator:
    """
    Evaluates RAG quality against a ground truth dataset.

    Eval set format:
    [
        {
            "query": "best death wave build",
            "relevant_ids": ["doc1", "doc2", "doc3"],
            "relevant_content_keywords": ["death wave", "build", "damage"]
        },
        ...
    ]
    """

    def __init__(self, eval_set_path: str = "data/eval/rag_eval_set.json"):
        """
        Initialize evaluator.

        Args:
            eval_set_path: Path to evaluation dataset
        """
        self.eval_set_path = Path(eval_set_path)
        self.eval_set = self._load_eval_set()

    def _load_eval_set(self) -> List[Dict[str, Any]]:
        """Load evaluation dataset."""
        if not self.eval_set_path.exists():
            return []

        try:
            with open(self.eval_set_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def evaluate(
        self,
        search_fn: Callable[[str, int], List[Any]],
        limit: int = 10
    ) -> QualityMetrics:
        """
        Evaluate search quality against ground truth.

        Args:
            search_fn: Function that takes (query, limit) and returns results
            limit: Number of results to retrieve per query

        Returns:
            QualityMetrics with evaluation results
        """
        if not self.eval_set:
            print("[Evaluator] No evaluation set found")
            return QualityMetrics(
                mrr=0, recall_at_5=0, recall_at_10=0,
                ndcg=0, precision_at_5=0, num_evaluated=0
            )

        mrr_scores = []
        recall_5_scores = []
        recall_10_scores = []
        ndcg_scores = []
        precision_5_scores = []

        for item in self.eval_set:
            query = item["query"]
            relevant_ids = set(item.get("relevant_ids", []))
            relevant_keywords = item.get("relevant_content_keywords", [])

            # Get results
            results = search_fn(query, limit)

            # Extract result IDs
            result_ids = []
            for r in results:
                if hasattr(r, 'id'):
                    result_ids.append(r.id)
                elif isinstance(r, dict):
                    result_ids.append(r.get('id', ''))

            # If no explicit IDs, match by keywords
            if not relevant_ids and relevant_keywords:
                relevant_ids = self._match_by_keywords(results, relevant_keywords)

            # Calculate metrics
            mrr_scores.append(self._calculate_mrr(result_ids, relevant_ids))
            recall_5_scores.append(self._calculate_recall(result_ids[:5], relevant_ids))
            recall_10_scores.append(self._calculate_recall(result_ids[:10], relevant_ids))
            ndcg_scores.append(self._calculate_ndcg(result_ids, relevant_ids, limit))
            precision_5_scores.append(self._calculate_precision(result_ids[:5], relevant_ids))

        return QualityMetrics(
            mrr=statistics.mean(mrr_scores) if mrr_scores else 0,
            recall_at_5=statistics.mean(recall_5_scores) if recall_5_scores else 0,
            recall_at_10=statistics.mean(recall_10_scores) if recall_10_scores else 0,
            ndcg=statistics.mean(ndcg_scores) if ndcg_scores else 0,
            precision_at_5=statistics.mean(precision_5_scores) if precision_5_scores else 0,
            num_evaluated=len(self.eval_set)
        )

    def _match_by_keywords(
        self,
        results: List[Any],
        keywords: List[str]
    ) -> set:
        """Match results by keyword presence."""
        matched = set()

        for r in results:
            content = ""
            if hasattr(r, 'content'):
                content = r.content.lower()
            elif isinstance(r, dict):
                content = r.get('content', '').lower()

            # Check if enough keywords present
            matches = sum(1 for kw in keywords if kw.lower() in content)
            if matches >= len(keywords) * 0.5:  # At least 50% keywords
                rid = r.id if hasattr(r, 'id') else r.get('id', '')
                matched.add(rid)

        return matched

    def _calculate_mrr(self, result_ids: List[str], relevant_ids: set) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, rid in enumerate(result_ids):
            if rid in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    def _calculate_recall(self, result_ids: List[str], relevant_ids: set) -> float:
        """Calculate recall."""
        if not relevant_ids:
            return 1.0  # No relevant docs = perfect recall

        found = sum(1 for rid in result_ids if rid in relevant_ids)
        return found / len(relevant_ids)

    def _calculate_precision(self, result_ids: List[str], relevant_ids: set) -> float:
        """Calculate precision."""
        if not result_ids:
            return 0.0

        found = sum(1 for rid in result_ids if rid in relevant_ids)
        return found / len(result_ids)

    def _calculate_ndcg(
        self,
        result_ids: List[str],
        relevant_ids: set,
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        import math

        # DCG
        dcg = 0.0
        for i, rid in enumerate(result_ids[:k]):
            rel = 1 if rid in relevant_ids else 0
            dcg += rel / math.log2(i + 2)

        # Ideal DCG
        ideal_rels = [1] * min(len(relevant_ids), k) + [0] * max(0, k - len(relevant_ids))
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    def create_eval_template(self, output_path: str = None) -> str:
        """Create a template evaluation set."""
        template = [
            {
                "query": "best death wave build for tier 15",
                "relevant_ids": [],
                "relevant_content_keywords": ["death wave", "tier 15", "build", "damage"]
            },
            {
                "query": "how to farm coins efficiently",
                "relevant_ids": [],
                "relevant_content_keywords": ["coins", "farming", "efficient", "cph"]
            },
            {
                "query": "inner land mines vs black hole",
                "relevant_ids": [],
                "relevant_content_keywords": ["inner land mines", "black hole", "comparison"]
            },
            {
                "query": "tournament strategy guide",
                "relevant_ids": [],
                "relevant_content_keywords": ["tournament", "strategy", "guide"]
            },
            {
                "query": "what is ehp",
                "relevant_ids": [],
                "relevant_content_keywords": ["ehp", "effective health", "health points"]
            }
        ]

        output = output_path or str(self.eval_set_path)
        Path(output).parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2)

        return output


class SearchTimer:
    """Context manager for timing search operations."""

    def __init__(self, analytics: RAGAnalytics = None):
        self.analytics = analytics
        self.start_time = None
        self.elapsed_ms = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000


# Convenience functions
def create_analytics(data_dir: str = "data/rag_analytics") -> RAGAnalytics:
    """Create analytics instance."""
    return RAGAnalytics(data_dir=data_dir)


def create_evaluator(eval_set_path: str = "data/eval/rag_eval_set.json") -> RAGEvaluator:
    """Create evaluator instance."""
    return RAGEvaluator(eval_set_path=eval_set_path)


if __name__ == "__main__":
    # Test analytics
    print("RAG Analytics Test")
    print("=" * 60)

    analytics = RAGAnalytics()

    # Simulate some queries
    for i in range(10):
        analytics.log_query(
            query=f"test query {i % 3}",
            results=[{"id": "1", "similarity": 0.8}] if i % 2 == 0 else [],
            latency_ms=100 + i * 20,
            filters={"type": "post"}
        )

    analytics.flush_logs()

    # Get metrics
    metrics = analytics.get_performance_metrics(hours=1)
    print(f"\nPerformance Metrics:")
    print(f"  Total queries: {metrics.total_queries}")
    print(f"  Avg latency: {metrics.avg_latency_ms:.1f}ms")
    print(f"  P95 latency: {metrics.p95_latency_ms:.1f}ms")
    print(f"  Zero result rate: {metrics.zero_result_rate:.1%}")

    # Get popular queries
    popular = analytics.get_popular_queries(hours=1)
    print(f"\nPopular queries: {popular}")

    # Create eval template
    evaluator = RAGEvaluator()
    template_path = evaluator.create_eval_template("data/eval/rag_eval_set.json")
    print(f"\nCreated eval template: {template_path}")
