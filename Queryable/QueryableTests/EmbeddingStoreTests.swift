//
//  EmbeddingStoreTests.swift
//  QueryableTests
//
//  Unit tests for EmbeddingStore and SimilarityComputer.
//

import XCTest
@testable import Queryable

final class EmbeddingStoreTests: XCTestCase {

    // MARK: - EmbeddingStore Tests

    func testEmptyStore() {
        let store = EmbeddingStore()
        XCTAssertTrue(store.isEmpty)
        XCTAssertEqual(store.count, 0)
    }

    func testAddAndRetrieveEmbedding() {
        let store = EmbeddingStore()
        let embedding = createRandomEmbedding()

        store.add(id: "photo1", embedding: embedding)

        XCTAssertEqual(store.count, 1)
        XCTAssertTrue(store.contains(id: "photo1"))
        XCTAssertFalse(store.contains(id: "photo2"))

        let retrieved = store.embedding(for: "photo1")
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(retrieved?.count, 512)
    }

    func testEmbeddingNormalization() {
        let embedding: [Float32] = Array(repeating: 2.0, count: 512)
        let normalized = EmbeddingStore.normalize(embedding)

        // Check that normalized vector has unit length
        let magnitude = sqrt(normalized.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(magnitude, 1.0, accuracy: 0.0001)
    }

    func testAddDuplicateId() {
        let store = EmbeddingStore()
        let embedding1 = createRandomEmbedding()
        let embedding2 = createRandomEmbedding()

        store.add(id: "photo1", embedding: embedding1)
        store.add(id: "photo1", embedding: embedding2)

        // Should update, not add duplicate
        XCTAssertEqual(store.count, 1)
    }

    func testRemoveEmbedding() {
        let store = EmbeddingStore()
        store.add(id: "photo1", embedding: createRandomEmbedding())
        store.add(id: "photo2", embedding: createRandomEmbedding())
        store.add(id: "photo3", embedding: createRandomEmbedding())

        XCTAssertEqual(store.count, 3)

        let removed = store.remove(id: "photo2")
        XCTAssertTrue(removed)
        XCTAssertEqual(store.count, 2)
        XCTAssertFalse(store.contains(id: "photo2"))
        XCTAssertTrue(store.contains(id: "photo1"))
        XCTAssertTrue(store.contains(id: "photo3"))
    }

    func testRetainOnly() {
        let store = EmbeddingStore()
        store.add(id: "photo1", embedding: createRandomEmbedding())
        store.add(id: "photo2", embedding: createRandomEmbedding())
        store.add(id: "photo3", embedding: createRandomEmbedding())
        store.add(id: "photo4", embedding: createRandomEmbedding())

        let validIds: Set<String> = ["photo1", "photo3"]
        let removedCount = store.retainOnly(validIds: validIds)

        XCTAssertEqual(removedCount, 2)
        XCTAssertEqual(store.count, 2)
        XCTAssertTrue(store.contains(id: "photo1"))
        XCTAssertTrue(store.contains(id: "photo3"))
        XCTAssertFalse(store.contains(id: "photo2"))
        XCTAssertFalse(store.contains(id: "photo4"))
    }

    func testSaveAndLoad() throws {
        let store = EmbeddingStore()
        store.add(id: "photo1", embedding: createRandomEmbedding())
        store.add(id: "photo2", embedding: createRandomEmbedding())
        store.add(id: "photo3", embedding: createRandomEmbedding())

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_embeddings")

        try store.save(to: tempURL)

        let loadedStore = try EmbeddingStore.load(from: tempURL)

        XCTAssertEqual(loadedStore.count, 3)
        XCTAssertTrue(loadedStore.contains(id: "photo1"))
        XCTAssertTrue(loadedStore.contains(id: "photo2"))
        XCTAssertTrue(loadedStore.contains(id: "photo3"))

        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }

    // MARK: - SimilarityComputer Tests

    func testFindTopK() {
        let store = EmbeddingStore()

        // Add some embeddings
        for i in 0..<100 {
            store.add(id: "photo\(i)", embedding: createRandomEmbedding())
        }

        // Create a query that's similar to photo50
        let queryEmbedding = store.embedding(for: "photo50")!

        let computer = SimilarityComputer()
        let results = computer.findTopK(query: queryEmbedding, in: store, k: 10)

        XCTAssertEqual(results.count, 10)

        // The most similar should be the same embedding (similarity ~= 1.0)
        XCTAssertEqual(results.first?.id, "photo50")
        XCTAssertEqual(results.first?.score ?? 0, 1.0, accuracy: 0.001)

        // Results should be sorted by descending score
        for i in 1..<results.count {
            XCTAssertGreaterThanOrEqual(results[i-1].score, results[i].score)
        }
    }

    func testFindTopKWithSmallStore() {
        let store = EmbeddingStore()
        store.add(id: "photo1", embedding: createRandomEmbedding())
        store.add(id: "photo2", embedding: createRandomEmbedding())

        let computer = SimilarityComputer()
        let results = computer.findTopK(query: createRandomEmbedding(), in: store, k: 10)

        // Should only return 2 results even though we asked for 10
        XCTAssertEqual(results.count, 2)
    }

    func testFindTopKEmptyStore() {
        let store = EmbeddingStore()
        let computer = SimilarityComputer()
        let results = computer.findTopK(query: createRandomEmbedding(), in: store, k: 10)

        XCTAssertTrue(results.isEmpty)
    }

    // MARK: - TopKHeap Tests

    func testTopKHeapBasic() {
        var heap = TopKHeap(capacity: 3)

        heap.insert(SimilarityResult(id: "a", score: 0.5))
        heap.insert(SimilarityResult(id: "b", score: 0.8))
        heap.insert(SimilarityResult(id: "c", score: 0.3))
        heap.insert(SimilarityResult(id: "d", score: 0.9))
        heap.insert(SimilarityResult(id: "e", score: 0.7))

        let sorted = heap.sorted()

        XCTAssertEqual(sorted.count, 3)
        XCTAssertEqual(sorted[0].id, "d") // 0.9
        XCTAssertEqual(sorted[1].id, "b") // 0.8
        XCTAssertEqual(sorted[2].id, "e") // 0.7
    }

    func testTopKHeapMaintainsCapacity() {
        var heap = TopKHeap(capacity: 5)

        for i in 0..<1000 {
            heap.insert(SimilarityResult(id: "item\(i)", score: Float32.random(in: 0...1)))
        }

        XCTAssertEqual(heap.count, 5)
    }

    // MARK: - Performance Tests

    func testSearchPerformance() {
        let store = EmbeddingStore()

        // Create a store with 10,000 embeddings
        for i in 0..<10000 {
            store.add(id: "photo\(i)", embedding: createRandomEmbedding())
        }

        let computer = SimilarityComputer()
        let queryEmbedding = createRandomEmbedding()

        measure {
            _ = computer.findTopK(query: queryEmbedding, in: store, k: 120)
        }
    }

    func testNormalizationPerformance() {
        let embeddings = (0..<1000).map { _ in createRandomEmbedding() }

        measure {
            for embedding in embeddings {
                _ = EmbeddingStore.normalize(embedding)
            }
        }
    }

    // MARK: - Helper Methods

    private func createRandomEmbedding() -> [Float32] {
        return (0..<512).map { _ in Float32.random(in: -1...1) }
    }
}
