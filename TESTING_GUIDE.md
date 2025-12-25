# Testing Search Optimizations on macOS

This guide walks you through testing the search performance optimizations without building the full iOS app, then verifying with a complete build.

---

## Overview of Changes

The optimizations include:
- **EmbeddingStore**: Contiguous memory layout with pre-normalized embeddings
- **SimilarityComputer**: Heap-based Top-K selection with parallel vDSP operations
- **PhotoSearcher**: Automatic migration from legacy format

---

## Phase 1: Unit Tests (No App Build Required)

### Option A: Run Swift Package Tests

1. **Create a Swift Package for testing**

```bash
cd /path/to/Queryable
mkdir -p Tests/QueryableTests
```

2. **Create Package.swift** in the project root:

```swift
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "QueryableTests",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "QueryableCore", targets: ["QueryableCore"]),
    ],
    targets: [
        .target(
            name: "QueryableCore",
            dependencies: [],
            path: "Queryable/Queryable",
            sources: [
                "Model/EmbeddingStore.swift",
                "Model/SimilarityComputer.swift",
                "Model/Embedding.swift"
            ]
        ),
        .testTarget(
            name: "QueryableCoreTests",
            dependencies: ["QueryableCore"],
            path: "Tests/QueryableTests"
        ),
    ]
)
```

3. **Copy the test file**:

```bash
cp Queryable/QueryableTests/EmbeddingStoreTests.swift Tests/QueryableTests/
```

4. **Run tests**:

```bash
swift test
```

### Option B: Run Standalone Swift Script (Recommended)

The `SearchPerformanceTest.swift` file can be run directly:

```bash
# On macOS with Swift installed
swift SearchPerformanceTest.swift
```

This tests:
- EmbeddingStore operations
- TopKHeap correctness
- SimilarityComputer performance
- Comparison with legacy implementation

### Option C: Run Python Tests (Cross-platform)

```bash
# Install numpy if needed
pip3 install numpy

# Run the test
python3 search_performance_test.py
```

Expected output:
```
Testing with 35000 embeddings (k=120)...
  Legacy avg:         0.1551s
  Optimized (numpy):  0.0011s  (147.7x speedup)
  Optimized (heap):   0.0168s  (9.2x speedup)
```

---

## Phase 2: Xcode Unit Tests

### Step 1: Add Test Target to Xcode Project

1. Open `Queryable/Queryable.xcodeproj` in Xcode
2. File → New → Target → "Unit Testing Bundle"
3. Name it `QueryableTests`
4. Add `EmbeddingStoreTests.swift` to the test target

### Step 2: Configure Test Target

In the test target's Build Settings:
- Set "Host Application" to `None`
- Add `Queryable` framework to "Link Binary With Libraries"

### Step 3: Run Tests

```bash
# From command line
xcodebuild test \
  -project Queryable/Queryable.xcodeproj \
  -scheme Queryable \
  -destination 'platform=iOS Simulator,name=iPhone 15' \
  -only-testing:QueryableTests

# Or in Xcode: Cmd+U
```

---

## Phase 3: Integration Testing with Real Embeddings

### Step 1: Export Existing Embeddings

If you have existing embeddings from the app, you can test with them:

1. **Locate embeddings file** on your device/simulator:
   ```
   ~/Library/Developer/CoreSimulator/Devices/<UUID>/data/Containers/Data/Application/<UUID>/Documents/embeddingData
   ```

2. **Copy to test location**:
   ```bash
   cp /path/to/embeddingData ./test_embeddings
   ```

### Step 2: Create Integration Test

Create `IntegrationTest.swift`:

```swift
import Foundation
import CoreML

// Load the actual embeddings and test migration + search
func testWithRealEmbeddings() {
    let url = URL(fileURLWithPath: "./test_embeddings")

    // Test migration
    do {
        print("Loading legacy embeddings...")
        let start = CFAbsoluteTimeGetCurrent()
        let store = try EmbeddingStore.migrate(fromLegacyFile: url)
        let loadTime = CFAbsoluteTimeGetCurrent() - start

        print("Loaded \(store.count) embeddings in \(loadTime)s")

        // Test search
        let query = Array(repeating: Float32(0.1), count: 512)
        let computer = SimilarityComputer()

        let searchStart = CFAbsoluteTimeGetCurrent()
        let results = computer.findTopK(query: query, in: store, k: 120)
        let searchTime = CFAbsoluteTimeGetCurrent() - searchStart

        print("Search completed in \(searchTime)s")
        print("Top result: \(results.first?.id ?? "none")")

    } catch {
        print("Error: \(error)")
    }
}

testWithRealEmbeddings()
```

Run it:
```bash
swift IntegrationTest.swift
```

---

## Phase 4: Build and Test Full App

### Step 1: Build for Simulator

```bash
xcodebuild build \
  -project Queryable/Queryable.xcodeproj \
  -scheme Queryable \
  -destination 'platform=iOS Simulator,name=iPhone 15' \
  -configuration Debug
```

### Step 2: Build for Device (Release)

```bash
xcodebuild build \
  -project Queryable/Queryable.xcodeproj \
  -scheme Queryable \
  -destination 'generic/platform=iOS' \
  -configuration Release
```

### Step 3: Run on Simulator

```bash
# Boot simulator
xcrun simctl boot "iPhone 15"

# Install app
xcrun simctl install booted /path/to/Queryable.app

# Launch app
xcrun simctl launch booted com.mazzystar.Queryable
```

---

## Phase 5: Verify Performance in App

### Enable Debug Logging

The optimized search path prints timing information:

```
Using optimized search path
Total photos in library: 35000
0.02 seconds used for filtering deleted photos.
Searching query = dog on beach
Text embedding computed
0.31 seconds used for optimized similarity search on 35000 embeddings.
0.001 seconds used for creating 120 PhotoAsset objects.
```

### Compare with Legacy

To compare, temporarily disable optimized search:

```swift
// In PhotoSearcher.swift, change:
private var useOptimizedSearch = false  // Force legacy path
```

Rebuild and compare the timing output.

### Expected Results

| Metric | Legacy | Optimized |
|--------|--------|-----------|
| 35k embeddings search | ~2.8s | ~0.3s |
| Top-K selection | ~0.2s | ~0.01s |
| Total search time | ~3.0s | ~0.35s |

---

## Troubleshooting

### Build Errors

1. **Missing files in target**
   - Ensure `EmbeddingStore.swift` and `SimilarityComputer.swift` are added to the Queryable target
   - Check File Inspector → Target Membership

2. **CoreML import errors**
   - These files require iOS 16+ / macOS 13+
   - Check deployment target in project settings

### Runtime Errors

1. **Embeddings not loading**
   - Check Documents directory permissions
   - Verify file exists: `embeddingDataV2`

2. **Migration not happening**
   - Legacy file `embeddingData` must exist
   - Check console for migration logs

### Performance Not Improved

1. **Still using legacy path**
   - Check `useOptimizedSearch` flag
   - Verify `embeddingStore.isEmpty` is false

2. **Main thread blocking**
   - Ensure `Task.detached` is being used
   - Check for `@MainActor` isolation issues

---

## Files Changed

| File | Changes |
|------|---------|
| `Model/EmbeddingStore.swift` | **NEW** - Optimized storage |
| `Model/SimilarityComputer.swift` | **NEW** - Fast similarity search |
| `ViewModel/PhotoSearcher.swift` | **MODIFIED** - Integration |
| `QueryableTests/EmbeddingStoreTests.swift` | **NEW** - Unit tests |

---

## Quick Verification Checklist

- [ ] Python test passes with >5x speedup
- [ ] Swift test passes (if Swift available)
- [ ] Xcode project builds without errors
- [ ] Unit tests pass in Xcode
- [ ] App launches on simulator
- [ ] Search completes in <0.5s for 35k photos
- [ ] Results are correct (same photos as legacy)
- [ ] Migration from legacy format works
