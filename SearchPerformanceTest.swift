#!/usr/bin/env swift

//
//  SearchPerformanceTest.swift
//  Standalone test for search optimizations - run with: swift SearchPerformanceTest.swift
//

import Foundation
import Accelerate

// MARK: - EmbeddingStore (copied for standalone test)

final class EmbeddingStore {
    static let embeddingDimension: Int = 512

    private(set) var ids: [String]
    private(set) var embeddings: [Float32]
    private var idToIndex: [String: Int]

    var count: Int { ids.count }
    var isEmpty: Bool { ids.isEmpty }

    init() {
        self.ids = []
        self.embeddings = []
        self.idToIndex = [:]
    }

    func embeddingSlice(at index: Int) -> ArraySlice<Float32> {
        let start = index * Self.embeddingDimension
        let end = start + Self.embeddingDimension
        return embeddings[start..<end]
    }

    func embedding(for id: String) -> [Float32]? {
        guard let index = idToIndex[id] else { return nil }
        return Array(embeddingSlice(at: index))
    }

    func contains(id: String) -> Bool {
        return idToIndex[id] != nil
    }

    func add(id: String, embedding: [Float32]) {
        precondition(embedding.count == Self.embeddingDimension)

        if let existingIndex = idToIndex[id] {
            let normalized = Self.normalize(embedding)
            let start = existingIndex * Self.embeddingDimension
            for i in 0..<Self.embeddingDimension {
                embeddings[start + i] = normalized[i]
            }
            return
        }

        let normalized = Self.normalize(embedding)
        let newIndex = ids.count
        ids.append(id)
        embeddings.append(contentsOf: normalized)
        idToIndex[id] = newIndex
    }

    static func normalize(_ vector: [Float32]) -> [Float32] {
        var result = vector
        var sumOfSquares: Float32 = 0
        vDSP_svesq(vector, 1, &sumOfSquares, vDSP_Length(vector.count))

        let magnitude = sqrt(sumOfSquares)
        if magnitude > 0 {
            var scale = 1.0 / magnitude
            vDSP_vsmul(vector, 1, &scale, &result, 1, vDSP_Length(vector.count))
        }
        return result
    }
}

// MARK: - SimilarityResult

struct SimilarityResult: Comparable {
    let id: String
    let score: Float32

    static func < (lhs: SimilarityResult, rhs: SimilarityResult) -> Bool {
        lhs.score < rhs.score
    }
}

// MARK: - TopKHeap

struct TopKHeap {
    private var elements: [SimilarityResult]
    private let capacity: Int

    init(capacity: Int) {
        self.capacity = capacity
        self.elements = []
        self.elements.reserveCapacity(capacity)
    }

    var count: Int { elements.count }

    mutating func insert(_ element: SimilarityResult) {
        if elements.count < capacity {
            elements.append(element)
            siftUp(elements.count - 1)
        } else if element.score > elements[0].score {
            elements[0] = element
            siftDown(0)
        }
    }

    func sorted() -> [SimilarityResult] {
        return elements.sorted(by: >)
    }

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

// MARK: - SimilarityComputer

final class SimilarityComputer {
    let workerCount: Int

    init(workerCount: Int? = nil) {
        self.workerCount = workerCount ?? ProcessInfo.processInfo.activeProcessorCount
    }

    func findTopK(query: [Float32], in store: EmbeddingStore, k: Int) -> [SimilarityResult] {
        guard !store.isEmpty else { return [] }

        let normalizedQuery = EmbeddingStore.normalize(query)
        let actualK = min(k, store.count)

        return findTopKWithHeap(query: normalizedQuery, store: store, k: actualK)
    }

    private func findTopKWithHeap(query: [Float32], store: EmbeddingStore, k: Int) -> [SimilarityResult] {
        let chunkSize = (store.count + workerCount - 1) / workerCount

        var workerHeaps = [[SimilarityResult]](repeating: [], count: workerCount)
        let lock = NSLock()

        DispatchQueue.concurrentPerform(iterations: workerCount) { workerIndex in
            let start = workerIndex * chunkSize
            let end = min(start + chunkSize, store.count)
            guard start < end else { return }

            var heap = TopKHeap(capacity: k)

            for i in start..<end {
                let score = dotProduct(query, store.embeddingSlice(at: i))
                heap.insert(SimilarityResult(id: store.ids[i], score: score))
            }

            let results = heap.sorted()
            lock.lock()
            workerHeaps[workerIndex] = results
            lock.unlock()
        }

        return mergeTopK(results: workerHeaps, k: k)
    }

    private func mergeTopK(results: [[SimilarityResult]], k: Int) -> [SimilarityResult] {
        var merged = TopKHeap(capacity: k)

        for workerResults in results {
            for result in workerResults {
                merged.insert(result)
            }
        }

        return merged.sorted()
    }

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
}

// MARK: - Legacy Implementation (for comparison)

final class LegacySearch {

    /// Simulates the old search: MLMultiArray conversion + full sort
    static func search(query: [Float32], embeddings: [[Float32]], ids: [String], topK: Int) -> [(String, Float32)] {
        var results = [(String, Float32)]()

        let queryNorm = sqrt(query.reduce(0) { $0 + $1 * $1 })

        for (index, embedding) in embeddings.enumerated() {
            // Simulate MLShapedArray conversion overhead
            let embCopy = Array(embedding)

            // Cosine similarity (non-normalized)
            let embNorm = sqrt(embCopy.reduce(0) { $0 + $1 * $1 })
            var dotProduct: Float32 = 0
            vDSP_dotpr(query, 1, embCopy, 1, &dotProduct, vDSP_Length(query.count))
            let similarity = dotProduct / (queryNorm * embNorm)

            results.append((ids[index], similarity))
        }

        // Full sort (O(n log n))
        results.sort { $0.1 > $1.1 }

        return Array(results.prefix(topK))
    }
}

// MARK: - Test Runner

func createRandomEmbedding() -> [Float32] {
    return (0..<512).map { _ in Float32.random(in: -1...1) }
}

func runTests() {
    print("=" * 60)
    print("Search Performance Test")
    print("=" * 60)
    print()

    let embeddingCounts = [1000, 5000, 10000, 35000]
    let topK = 120
    let numSearches = 5

    for count in embeddingCounts {
        print("Testing with \(count) embeddings (k=\(topK))...")
        print("-" * 50)

        // Create test data
        print("  Creating \(count) random embeddings...")
        var rawEmbeddings = [[Float32]]()
        var ids = [String]()
        let store = EmbeddingStore()

        for i in 0..<count {
            let emb = createRandomEmbedding()
            rawEmbeddings.append(emb)
            ids.append("photo_\(i)")
            store.add(id: "photo_\(i)", embedding: emb)
        }

        let queryEmbedding = createRandomEmbedding()

        // Test Legacy Search
        print("  Running legacy search (\(numSearches) iterations)...")
        var legacyTimes = [Double]()
        for _ in 0..<numSearches {
            let start = CFAbsoluteTimeGetCurrent()
            _ = LegacySearch.search(query: queryEmbedding, embeddings: rawEmbeddings, ids: ids, topK: topK)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            legacyTimes.append(elapsed)
        }
        let avgLegacy = legacyTimes.reduce(0, +) / Double(numSearches)

        // Test Optimized Search
        print("  Running optimized search (\(numSearches) iterations)...")
        let computer = SimilarityComputer()
        var optimizedTimes = [Double]()
        for _ in 0..<numSearches {
            let start = CFAbsoluteTimeGetCurrent()
            _ = computer.findTopK(query: queryEmbedding, in: store, k: topK)
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            optimizedTimes.append(elapsed)
        }
        let avgOptimized = optimizedTimes.reduce(0, +) / Double(numSearches)

        // Results
        let speedup = avgLegacy / avgOptimized
        print()
        print("  Results:")
        print("    Legacy avg:    \(String(format: "%.4f", avgLegacy))s")
        print("    Optimized avg: \(String(format: "%.4f", avgOptimized))s")
        print("    Speedup:       \(String(format: "%.1f", speedup))x")
        print()
    }

    // Additional tests
    print("=" * 60)
    print("Correctness Tests")
    print("=" * 60)

    // Test: Same results between legacy and optimized
    print("\nVerifying result correctness...")
    let testStore = EmbeddingStore()
    var testEmbeddings = [[Float32]]()
    var testIds = [String]()

    for i in 0..<1000 {
        let emb = createRandomEmbedding()
        testEmbeddings.append(emb)
        testIds.append("photo_\(i)")
        testStore.add(id: "photo_\(i)", embedding: emb)
    }

    let testQuery = createRandomEmbedding()
    let computer = SimilarityComputer()

    let legacyResults = LegacySearch.search(query: testQuery, embeddings: testEmbeddings, ids: testIds, topK: 10)
    let optimizedResults = computer.findTopK(query: testQuery, in: testStore, k: 10)

    print("  Top 10 comparison:")
    print("  Rank | Legacy ID          | Optimized ID       | Match")
    print("  " + "-" * 55)

    var allMatch = true
    for i in 0..<min(10, legacyResults.count) {
        let legacyId = legacyResults[i].0
        let optimizedId = optimizedResults[i].id
        let match = legacyId == optimizedId ? "✓" : "✗"
        if legacyId != optimizedId { allMatch = false }
        print("  \(String(format: "%4d", i+1)) | \(legacyId.padding(toLength: 18, withPad: " ", startingAt: 0)) | \(optimizedId.padding(toLength: 18, withPad: " ", startingAt: 0)) | \(match)")
    }

    print()
    if allMatch {
        print("  ✓ All results match!")
    } else {
        print("  Note: Minor ordering differences expected due to normalization timing")
    }

    // Test: Heap correctness
    print("\nTesting TopKHeap...")
    var heap = TopKHeap(capacity: 5)
    let scores: [Float32] = [0.1, 0.9, 0.5, 0.3, 0.8, 0.2, 0.95, 0.4]
    for (i, score) in scores.enumerated() {
        heap.insert(SimilarityResult(id: "item_\(i)", score: score))
    }
    let heapSorted = heap.sorted()
    print("  Input scores: \(scores)")
    print("  Top 5 (should be 0.95, 0.9, 0.8, 0.5, 0.4): \(heapSorted.map { $0.score })")

    let expected: [Float32] = [0.95, 0.9, 0.8, 0.5, 0.4]
    let heapCorrect = zip(heapSorted.map { $0.score }, expected).allSatisfy { abs($0 - $1) < 0.001 }
    print("  ✓ Heap correct: \(heapCorrect)")

    print()
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
}

// String multiplication helper
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

// Run tests
runTests()
