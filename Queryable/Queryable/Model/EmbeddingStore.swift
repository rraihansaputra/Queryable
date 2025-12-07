//
//  EmbeddingStore.swift
//  Queryable
//
//  Optimized embedding storage with contiguous memory layout and pre-normalization
//  for fast similarity search.
//

import Foundation
import Accelerate
import CoreML

/// High-performance embedding storage using contiguous memory layout.
/// Embeddings are pre-normalized for O(1) cosine similarity via dot product.
final class EmbeddingStore {

    // MARK: - Constants

    static let embeddingDimension: Int = 512
    private static let version: Int = 1

    // MARK: - Storage

    /// Photo IDs in order, corresponding to embeddings
    private(set) var ids: [String]

    /// Contiguous storage for all embeddings: [emb0_0...emb0_511, emb1_0...emb1_511, ...]
    /// All embeddings are pre-normalized (unit length)
    private(set) var embeddings: [Float32]

    /// Fast lookup from ID to index
    private var idToIndex: [String: Int]

    // MARK: - Computed Properties

    var count: Int { ids.count }
    var isEmpty: Bool { ids.isEmpty }

    // MARK: - Initialization

    init() {
        self.ids = []
        self.embeddings = []
        self.idToIndex = [:]
    }

    init(ids: [String], embeddings: [Float32]) {
        precondition(embeddings.count == ids.count * Self.embeddingDimension,
                     "Embedding count mismatch: expected \(ids.count * Self.embeddingDimension), got \(embeddings.count)")
        self.ids = ids
        self.embeddings = embeddings
        self.idToIndex = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
    }

    // MARK: - Embedding Access

    /// Get embedding at index as a slice (no copy)
    func embeddingSlice(at index: Int) -> ArraySlice<Float32> {
        let start = index * Self.embeddingDimension
        let end = start + Self.embeddingDimension
        return embeddings[start..<end]
    }

    /// Get embedding for a photo ID
    func embedding(for id: String) -> [Float32]? {
        guard let index = idToIndex[id] else { return nil }
        return Array(embeddingSlice(at: index))
    }

    /// Check if an ID exists
    func contains(id: String) -> Bool {
        return idToIndex[id] != nil
    }

    /// Get index for ID
    func index(for id: String) -> Int? {
        return idToIndex[id]
    }

    // MARK: - Modification

    /// Add a new embedding (will be normalized)
    func add(id: String, embedding: [Float32]) {
        precondition(embedding.count == Self.embeddingDimension,
                     "Embedding dimension mismatch: expected \(Self.embeddingDimension), got \(embedding.count)")

        // Check if already exists
        if let existingIndex = idToIndex[id] {
            // Update existing embedding
            let normalized = Self.normalize(embedding)
            let start = existingIndex * Self.embeddingDimension
            for i in 0..<Self.embeddingDimension {
                embeddings[start + i] = normalized[i]
            }
            return
        }

        // Add new embedding
        let normalized = Self.normalize(embedding)
        let newIndex = ids.count
        ids.append(id)
        embeddings.append(contentsOf: normalized)
        idToIndex[id] = newIndex
    }

    /// Add a pre-normalized embedding (skip normalization for performance)
    func addNormalized(id: String, embedding: [Float32]) {
        precondition(embedding.count == Self.embeddingDimension,
                     "Embedding dimension mismatch")

        if let existingIndex = idToIndex[id] {
            let start = existingIndex * Self.embeddingDimension
            for i in 0..<Self.embeddingDimension {
                embeddings[start + i] = embedding[i]
            }
            return
        }

        let newIndex = ids.count
        ids.append(id)
        embeddings.append(contentsOf: embedding)
        idToIndex[id] = newIndex
    }

    /// Remove embedding by ID
    /// Note: This is O(n) as it requires reindexing. Use sparingly.
    @discardableResult
    func remove(id: String) -> Bool {
        guard let index = idToIndex[id] else { return false }

        // Remove from arrays
        ids.remove(at: index)
        let start = index * Self.embeddingDimension
        embeddings.removeSubrange(start..<(start + Self.embeddingDimension))

        // Rebuild index (only for items after removed index)
        idToIndex.removeValue(forKey: id)
        for i in index..<ids.count {
            idToIndex[ids[i]] = i
        }

        return true
    }

    /// Remove embeddings for IDs not in the provided set
    func retainOnly(validIds: Set<String>) -> Int {
        let idsToRemove = ids.filter { !validIds.contains($0) }

        if idsToRemove.isEmpty { return 0 }

        // Rebuild arrays without removed IDs
        var newIds: [String] = []
        var newEmbeddings: [Float32] = []
        newEmbeddings.reserveCapacity(validIds.count * Self.embeddingDimension)

        for (index, id) in ids.enumerated() {
            if validIds.contains(id) {
                newIds.append(id)
                let start = index * Self.embeddingDimension
                newEmbeddings.append(contentsOf: embeddings[start..<(start + Self.embeddingDimension)])
            }
        }

        self.ids = newIds
        self.embeddings = newEmbeddings
        self.idToIndex = Dictionary(uniqueKeysWithValues: newIds.enumerated().map { ($1, $0) })

        return idsToRemove.count
    }

    // MARK: - Normalization

    /// Normalize a vector to unit length using vDSP
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

    /// Normalize an MLShapedArray
    static func normalize(_ array: MLShapedArray<Float32>) -> [Float32] {
        return normalize(array.scalars)
    }

    // MARK: - Serialization

    /// Save to file
    func save(to url: URL) throws {
        var data = Data()

        // Version header
        var version = Int32(Self.version)
        data.append(Data(bytes: &version, count: MemoryLayout<Int32>.size))

        // Count
        var count = Int32(ids.count)
        data.append(Data(bytes: &count, count: MemoryLayout<Int32>.size))

        // IDs (length-prefixed strings)
        for id in ids {
            let idData = id.data(using: .utf8)!
            var length = Int32(idData.count)
            data.append(Data(bytes: &length, count: MemoryLayout<Int32>.size))
            data.append(idData)
        }

        // Embeddings (contiguous float array)
        embeddings.withUnsafeBytes { buffer in
            data.append(Data(buffer))
        }

        try data.write(to: url)
    }

    /// Load from file
    static func load(from url: URL) throws -> EmbeddingStore {
        let data = try Data(contentsOf: url)
        var offset = 0

        // Version
        let version = data.withUnsafeBytes { ptr -> Int32 in
            ptr.load(fromByteOffset: offset, as: Int32.self)
        }
        offset += MemoryLayout<Int32>.size

        guard version == Self.version else {
            throw EmbeddingStoreError.unsupportedVersion(Int(version))
        }

        // Count
        let count = data.withUnsafeBytes { ptr -> Int32 in
            ptr.load(fromByteOffset: offset, as: Int32.self)
        }
        offset += MemoryLayout<Int32>.size

        // IDs
        var ids: [String] = []
        ids.reserveCapacity(Int(count))

        for _ in 0..<count {
            let length = data.withUnsafeBytes { ptr -> Int32 in
                ptr.load(fromByteOffset: offset, as: Int32.self)
            }
            offset += MemoryLayout<Int32>.size

            let idData = data.subdata(in: offset..<(offset + Int(length)))
            guard let id = String(data: idData, encoding: .utf8) else {
                throw EmbeddingStoreError.invalidData
            }
            ids.append(id)
            offset += Int(length)
        }

        // Embeddings
        let embeddingCount = Int(count) * embeddingDimension
        var embeddings = [Float32](repeating: 0, count: embeddingCount)

        embeddings.withUnsafeMutableBytes { buffer in
            _ = data.copyBytes(to: buffer, from: offset..<(offset + embeddingCount * MemoryLayout<Float32>.size))
        }

        return EmbeddingStore(ids: ids, embeddings: embeddings)
    }
}

// MARK: - Errors

enum EmbeddingStoreError: Error, LocalizedError {
    case unsupportedVersion(Int)
    case invalidData

    var errorDescription: String? {
        switch self {
        case .unsupportedVersion(let v):
            return "Unsupported embedding store version: \(v)"
        case .invalidData:
            return "Invalid embedding data"
        }
    }
}

// MARK: - Migration from Legacy Format

extension EmbeddingStore {

    /// Migrate from legacy [String: MLMultiArray] format
    static func migrate(from legacyEmbeddings: [String: MLMultiArray]) -> EmbeddingStore {
        let store = EmbeddingStore()

        for (id, mlArray) in legacyEmbeddings {
            let shaped = MLShapedArray<Float32>(converting: mlArray)
            store.add(id: id, embedding: shaped.scalars)
        }

        return store
    }

    /// Migrate from legacy Embedding array (NSKeyedArchiver format)
    static func migrate(fromLegacyFile url: URL) throws -> EmbeddingStore {
        let data = try Data(contentsOf: url)
        guard let embeddings = try NSKeyedUnarchiver.unarchivedArrayOfObjects(
            ofClasses: [Embedding.self, MLMultiArray.self, NSString.self],
            from: data
        ) as? [Embedding] else {
            throw EmbeddingStoreError.invalidData
        }

        let store = EmbeddingStore()

        for embedding in embeddings {
            guard let id = embedding.id, let mlArray = embedding.embedding else { continue }
            let shaped = MLShapedArray<Float32>(converting: mlArray)
            store.add(id: id, embedding: shaped.scalars)
        }

        return store
    }
}
