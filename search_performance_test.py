#!/usr/bin/env python3
"""
Search Performance Test
Demonstrates the performance improvements from the optimizations.
Run with: python3 search_performance_test.py
"""

import numpy as np
import time
import heapq
from dataclasses import dataclass
from typing import List, Tuple
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


# MARK: - Optimized Implementation

class EmbeddingStore:
    """Optimized embedding storage with contiguous memory and pre-normalization."""

    EMBEDDING_DIM = 512

    def __init__(self):
        self.ids: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)
        self.id_to_index: dict = {}

    @property
    def count(self) -> int:
        return len(self.ids)

    @property
    def is_empty(self) -> bool:
        return len(self.ids) == 0

    def add(self, id: str, embedding: np.ndarray):
        """Add embedding (will be normalized)."""
        normalized = self._normalize(embedding)

        if id in self.id_to_index:
            idx = self.id_to_index[id]
            self.embeddings[idx] = normalized
            return

        new_index = len(self.ids)
        self.ids.append(id)
        self.embeddings = np.vstack([self.embeddings, normalized.reshape(1, -1)]) if self.embeddings.size else normalized.reshape(1, -1)
        self.id_to_index[id] = new_index

    def add_batch(self, ids: List[str], embeddings: np.ndarray):
        """Add multiple embeddings at once (more efficient)."""
        # Normalize all at once
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms

        start_idx = len(self.ids)
        self.ids.extend(ids)

        if self.embeddings.size == 0:
            self.embeddings = normalized.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, normalized.astype(np.float32)])

        for i, id in enumerate(ids):
            self.id_to_index[id] = start_idx + i

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


@dataclass
class SimilarityResult:
    id: str
    score: float

    def __lt__(self, other):
        # For min-heap: smaller score = higher priority to remove
        return self.score < other.score


class TopKHeap:
    """Min-heap that maintains top-K highest scoring items."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.heap: List[SimilarityResult] = []

    def insert(self, result: SimilarityResult):
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, result)
        elif result.score > self.heap[0].score:
            heapq.heapreplace(self.heap, result)

    def sorted_results(self) -> List[SimilarityResult]:
        return sorted(self.heap, key=lambda x: x.score, reverse=True)


class SimilarityComputer:
    """High-performance similarity computer using pre-normalized embeddings."""

    def __init__(self, worker_count: int = None):
        self.worker_count = worker_count or mp.cpu_count()

    def find_top_k(self, query: np.ndarray, store: EmbeddingStore, k: int) -> List[SimilarityResult]:
        if store.is_empty:
            return []

        # Normalize query
        query_norm = np.linalg.norm(query)
        normalized_query = query / query_norm if query_norm > 0 else query

        actual_k = min(k, store.count)

        # For pre-normalized vectors, cosine similarity = dot product
        # Use numpy's optimized matrix-vector multiplication
        all_scores = store.embeddings @ normalized_query

        # Use numpy's argpartition for O(n) partial sort
        if actual_k < len(all_scores):
            top_indices = np.argpartition(all_scores, -actual_k)[-actual_k:]
            top_indices = top_indices[np.argsort(all_scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(all_scores)[::-1]

        return [
            SimilarityResult(id=store.ids[i], score=float(all_scores[i]))
            for i in top_indices[:actual_k]
        ]

    def find_top_k_heap(self, query: np.ndarray, store: EmbeddingStore, k: int) -> List[SimilarityResult]:
        """Alternative implementation using heap (matches Swift implementation)."""
        if store.is_empty:
            return []

        query_norm = np.linalg.norm(query)
        normalized_query = query / query_norm if query_norm > 0 else query

        actual_k = min(k, store.count)
        heap = TopKHeap(capacity=actual_k)

        # Compute dot products
        all_scores = store.embeddings @ normalized_query

        for i, score in enumerate(all_scores):
            heap.insert(SimilarityResult(id=store.ids[i], score=float(score)))

        return heap.sorted_results()


# MARK: - Legacy Implementation (simulates old code)

class LegacySearch:
    """Simulates the old search implementation."""

    @staticmethod
    def search(query: np.ndarray, embeddings: List[np.ndarray], ids: List[str], top_k: int) -> List[Tuple[str, float]]:
        results = []

        query_norm = np.linalg.norm(query)

        for i, embedding in enumerate(embeddings):
            # Simulate MLShapedArray conversion (copying data)
            emb_copy = np.array(embedding, copy=True)

            # Cosine similarity with magnitude calculation each time
            emb_norm = np.linalg.norm(emb_copy)
            dot_product = np.dot(query, emb_copy)
            similarity = dot_product / (query_norm * emb_norm) if query_norm * emb_norm > 0 else 0

            results.append((ids[i], similarity))

        # Full sort O(n log n)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


# MARK: - Test Runner

def create_random_embedding(dim=512) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


def run_performance_tests():
    print("=" * 60)
    print("Search Performance Test (Python)")
    print("=" * 60)
    print()

    embedding_counts = [1000, 5000, 10000, 35000]
    top_k = 120
    num_searches = 5

    for count in embedding_counts:
        print(f"Testing with {count} embeddings (k={top_k})...")
        print("-" * 50)

        # Create test data
        print(f"  Creating {count} random embeddings...")
        raw_embeddings = [create_random_embedding() for _ in range(count)]
        ids = [f"photo_{i}" for i in range(count)]

        # Create optimized store
        store = EmbeddingStore()
        all_emb = np.vstack(raw_embeddings)
        store.add_batch(ids, all_emb)

        query_embedding = create_random_embedding()

        # Test Legacy Search
        print(f"  Running legacy search ({num_searches} iterations)...")
        legacy_times = []
        for _ in range(num_searches):
            start = time.perf_counter()
            _ = LegacySearch.search(query_embedding, raw_embeddings, ids, top_k)
            elapsed = time.perf_counter() - start
            legacy_times.append(elapsed)
        avg_legacy = sum(legacy_times) / num_searches

        # Test Optimized Search (numpy vectorized)
        print(f"  Running optimized search ({num_searches} iterations)...")
        computer = SimilarityComputer()
        optimized_times = []
        for _ in range(num_searches):
            start = time.perf_counter()
            _ = computer.find_top_k(query_embedding, store, top_k)
            elapsed = time.perf_counter() - start
            optimized_times.append(elapsed)
        avg_optimized = sum(optimized_times) / num_searches

        # Test Heap-based Search (matches Swift implementation)
        print(f"  Running heap-based search ({num_searches} iterations)...")
        heap_times = []
        for _ in range(num_searches):
            start = time.perf_counter()
            _ = computer.find_top_k_heap(query_embedding, store, top_k)
            elapsed = time.perf_counter() - start
            heap_times.append(elapsed)
        avg_heap = sum(heap_times) / num_searches

        # Results
        speedup_vectorized = avg_legacy / avg_optimized
        speedup_heap = avg_legacy / avg_heap

        print()
        print("  Results:")
        print(f"    Legacy avg:         {avg_legacy:.4f}s")
        print(f"    Optimized (numpy):  {avg_optimized:.4f}s  ({speedup_vectorized:.1f}x speedup)")
        print(f"    Optimized (heap):   {avg_heap:.4f}s  ({speedup_heap:.1f}x speedup)")
        print()

    # Correctness tests
    print("=" * 60)
    print("Correctness Tests")
    print("=" * 60)

    print("\nVerifying result correctness...")
    test_count = 1000
    raw_embeddings = [create_random_embedding() for _ in range(test_count)]
    ids = [f"photo_{i}" for i in range(test_count)]

    store = EmbeddingStore()
    store.add_batch(ids, np.vstack(raw_embeddings))

    query = create_random_embedding()
    computer = SimilarityComputer()

    legacy_results = LegacySearch.search(query, raw_embeddings, ids, 10)
    optimized_results = computer.find_top_k(query, store, 10)

    print("  Top 10 comparison:")
    print("  Rank | Legacy ID          | Optimized ID       | Score Diff")
    print("  " + "-" * 60)

    for i in range(min(10, len(legacy_results))):
        legacy_id, legacy_score = legacy_results[i]
        opt_result = optimized_results[i]
        score_diff = abs(legacy_score - opt_result.score)
        print(f"  {i+1:4d} | {legacy_id:18s} | {opt_result.id:18s} | {score_diff:.6f}")

    # Check if top results match (allowing for small floating point differences)
    legacy_top_ids = set([r[0] for r in legacy_results[:5]])
    optimized_top_ids = set([r.id for r in optimized_results[:5]])
    overlap = len(legacy_top_ids & optimized_top_ids)

    print()
    print(f"  Top 5 overlap: {overlap}/5 (some variation expected due to floating point)")

    # TopKHeap test
    print("\nTesting TopKHeap...")
    heap = TopKHeap(capacity=5)
    scores = [0.1, 0.9, 0.5, 0.3, 0.8, 0.2, 0.95, 0.4]
    for i, score in enumerate(scores):
        heap.insert(SimilarityResult(id=f"item_{i}", score=score))

    heap_sorted = heap.sorted_results()
    result_scores = [r.score for r in heap_sorted]
    expected = [0.95, 0.9, 0.8, 0.5, 0.4]

    print(f"  Input scores: {scores}")
    print(f"  Top 5 (should be {expected}): {result_scores}")
    print(f"  ✓ Heap correct: {result_scores == expected}")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
The optimizations provide significant speedups through:

1. Pre-normalized embeddings
   - Cosine similarity becomes simple dot product
   - No per-query magnitude calculations

2. Contiguous memory layout
   - Enables SIMD/vectorized operations
   - Better cache locality

3. Heap-based Top-K selection
   - O(n log k) instead of O(n log n) full sort
   - Significant for small k relative to n

4. Removed async overhead
   - Synchronous computation for CPU-bound work
   - No context switching costs

Expected improvement on iOS with vDSP:
- Legacy: ~2.8s for 35k embeddings
- Optimized: ~0.3-0.5s (5-10x speedup)
""")


if __name__ == "__main__":
    run_performance_tests()
