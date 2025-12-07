//
//  SimilarityComputer.swift
//  Queryable
//
//  High-performance similarity computation using vDSP and parallel processing.
//  Uses pre-normalized embeddings for O(1) cosine similarity via dot product.
//

import Foundation
import Accelerate

/// Result of a similarity search
struct SimilarityResult: Comparable {
    let id: String
    let score: Float32

    static func < (lhs: SimilarityResult, rhs: SimilarityResult) -> Bool {
        lhs.score < rhs.score
    }
}

/// High-performance similarity computer for embedding search.
/// Designed to work with pre-normalized embeddings from EmbeddingStore.
final class SimilarityComputer {

    // MARK: - Configuration

    /// Number of parallel workers (defaults to CPU core count)
    let workerCount: Int

    // MARK: - Initialization

    init(workerCount: Int? = nil) {
        self.workerCount = workerCount ?? ProcessInfo.processInfo.activeProcessorCount
    }

    // MARK: - Search Methods

    /// Find top-K most similar embeddings to query.
    /// Uses heap-based selection for O(n log k) complexity instead of O(n log n) sort.
    ///
    /// - Parameters:
    ///   - query: Query embedding (will be normalized if not already)
    ///   - store: EmbeddingStore containing pre-normalized embeddings
    ///   - k: Number of top results to return
    /// - Returns: Array of top-K results sorted by descending similarity
    func findTopK(query: [Float32], in store: EmbeddingStore, k: Int) -> [SimilarityResult] {
        guard !store.isEmpty else { return [] }

        let normalizedQuery = EmbeddingStore.normalize(query)
        let actualK = min(k, store.count)

        // For small k, use heap-based selection
        if actualK <= store.count / 10 {
            return findTopKWithHeap(query: normalizedQuery, store: store, k: actualK)
        } else {
            // For large k, parallel sort may be faster
            return findTopKWithSort(query: normalizedQuery, store: store, k: actualK)
        }
    }

    /// Parallel similarity computation with heap-based top-K selection.
    private func findTopKWithHeap(query: [Float32], store: EmbeddingStore, k: Int) -> [SimilarityResult] {
        let chunkSize = (store.count + workerCount - 1) / workerCount

        // Each worker maintains its own top-K heap
        let results = DispatchQueue.global(qos: .userInitiated).sync {
            return (0..<workerCount).map { workerIndex -> [SimilarityResult] in
                let start = workerIndex * chunkSize
                let end = min(start + chunkSize, store.count)
                guard start < end else { return [] }

                var heap = TopKHeap(capacity: k)

                for i in start..<end {
                    let score = dotProduct(query, store.embeddingSlice(at: i))
                    heap.insert(SimilarityResult(id: store.ids[i], score: score))
                }

                return heap.sorted()
            }
        }

        // Merge results from all workers
        return mergeTopK(results: results, k: k)
    }

    /// Parallel similarity computation with full sort.
    private func findTopKWithSort(query: [Float32], store: EmbeddingStore, k: Int) -> [SimilarityResult] {
        var allScores = [SimilarityResult](repeating: SimilarityResult(id: "", score: 0), count: store.count)

        // Parallel computation of all similarities
        DispatchQueue.concurrentPerform(iterations: store.count) { i in
            let score = dotProduct(query, store.embeddingSlice(at: i))
            allScores[i] = SimilarityResult(id: store.ids[i], score: score)
        }

        // Partial sort to get top-K
        return Array(allScores.sorted(by: >).prefix(k))
    }

    /// Merge top-K results from multiple workers
    private func mergeTopK(results: [[SimilarityResult]], k: Int) -> [SimilarityResult] {
        var merged = TopKHeap(capacity: k)

        for workerResults in results {
            for result in workerResults {
                merged.insert(result)
            }
        }

        return merged.sorted()
    }

    // MARK: - Dot Product

    /// Fast dot product using vDSP for contiguous arrays
    @inline(__always)
    private func dotProduct(_ a: [Float32], _ b: ArraySlice<Float32>) -> Float32 {
        var result: Float32 = 0
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                vDSP_dotpr(aPtr.baseAddress!, 1,
                          bPtr.baseAddress!, 1,
                          &result,
                          vDSP_Length(a.count))
            }
        }
        return result
    }

    // MARK: - Batch Operations

    /// Compute all similarity scores (useful for debugging/testing)
    func computeAllScores(query: [Float32], store: EmbeddingStore) -> [String: Float32] {
        let normalizedQuery = EmbeddingStore.normalize(query)
        var results = [String: Float32]()
        results.reserveCapacity(store.count)

        for i in 0..<store.count {
            let score = dotProduct(normalizedQuery, store.embeddingSlice(at: i))
            results[store.ids[i]] = score
        }

        return results
    }
}

// MARK: - Min-Heap for Top-K Selection

/// Min-heap that maintains the top-K highest scoring items.
/// Insert is O(log k), extraction is O(k log k).
struct TopKHeap {

    private var elements: [SimilarityResult]
    private let capacity: Int

    init(capacity: Int) {
        self.capacity = capacity
        self.elements = []
        self.elements.reserveCapacity(capacity)
    }

    var count: Int { elements.count }

    /// Insert a new element. O(log k) amortized.
    mutating func insert(_ element: SimilarityResult) {
        if elements.count < capacity {
            elements.append(element)
            siftUp(elements.count - 1)
        } else if element.score > elements[0].score {
            // New element is larger than min, replace min
            elements[0] = element
            siftDown(0)
        }
        // Otherwise, element is smaller than everything in heap, discard
    }

    /// Get sorted results (highest first). O(k log k).
    func sorted() -> [SimilarityResult] {
        return elements.sorted(by: >)
    }

    // MARK: - Heap Operations

    private mutating func siftUp(_ index: Int) {
        var child = index
        var parent = (child - 1) / 2

        while child > 0 && elements[child] < elements[parent] {
            elements.swapAt(child, parent)
            child = parent
            parent = (child - 1) / 2
        }
    }

    private mutating func siftDown(_ index: Int) {
        var parent = index
        let count = elements.count

        while true {
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var smallest = parent

            if left < count && elements[left] < elements[smallest] {
                smallest = left
            }
            if right < count && elements[right] < elements[smallest] {
                smallest = right
            }

            if smallest == parent { break }

            elements.swapAt(parent, smallest)
            parent = smallest
        }
    }
}

// MARK: - Concurrent Top-K (Alternative Implementation)

extension SimilarityComputer {

    /// Alternative implementation using DispatchQueue for more control
    func findTopKConcurrent(query: [Float32], store: EmbeddingStore, k: Int) -> [SimilarityResult] {
        guard !store.isEmpty else { return [] }

        let normalizedQuery = EmbeddingStore.normalize(query)
        let actualK = min(k, store.count)
        let chunkSize = (store.count + workerCount - 1) / workerCount

        let queue = DispatchQueue(label: "similarity.compute", attributes: .concurrent)
        let group = DispatchGroup()

        var workerHeaps = [[SimilarityResult]](repeating: [], count: workerCount)
        let lock = NSLock()

        for workerIndex in 0..<workerCount {
            let start = workerIndex * chunkSize
            let end = min(start + chunkSize, store.count)
            guard start < end else { continue }

            group.enter()
            queue.async {
                var heap = TopKHeap(capacity: actualK)

                for i in start..<end {
                    let score = self.dotProduct(normalizedQuery, store.embeddingSlice(at: i))
                    heap.insert(SimilarityResult(id: store.ids[i], score: score))
                }

                let results = heap.sorted()
                lock.lock()
                workerHeaps[workerIndex] = results
                lock.unlock()
                group.leave()
            }
        }

        group.wait()

        return mergeTopK(results: workerHeaps, k: actualK)
    }
}
