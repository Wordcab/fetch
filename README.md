# Fetch

**Private semantic search in under a millisecond. No server, account, API key, or telemetry.**

Fetch is an embedded Rust retrieval engine for local-first AI. It searches a
packed 1-bit graph, reranks the shortlist with the original embedding geometry,
and stores the complete index in one local `.fetch` file.

On a Ryzen 7 5800H, searching 100,000 384-dimensional vectors:

| Recall@10 vs exact | p50 | p95 | p99 |
|---:|---:|---:|---:|
| **99.43%** | **0.85 ms** | **1.01 ms** | **1.08 ms** |

That includes ANN search and full-vector cosine reranking. It excludes query embedding,
because Fetch does not choose an embedding model for you.

## Why Fetch

- **Sovereign by construction.** Native retrieval performs no network calls;
  browser retrieval runs inside the tab. Your embeddings, queries, metadata,
  and index stay on hardware you control.
- **Fast without gambling on the first approximation.** A compact binary HNSW
  graph finds candidates; cosine reranking restores the full embedding order.
- **One file, no service.** Build an immutable snapshot once, save it, and load it
  in-process in roughly 190 ms at 100k × 384 with a warm filesystem cache.
- **A real accuracy knob.** Change candidate depth per query. The benchmark below
  shows the latency/recall tradeoff instead of hiding it behind one headline.
- **Native and browser paths.** Use the native graph-backed engine from Rust, or a
  35 KB dependency-free WASM kernel for smaller browser-resident collections.

Fetch is deliberately not a hosted vector database. There is no daemon to
operate and no user data to monetize.

## Run it

You need Rust and a C++ build toolchain.

```bash
git clone https://github.com/Wordcab/fetch.git
cd fetch
cargo run --release --example quickstart
```

The example builds `demo.fetch`, loads it back from disk, and queries it.

To use the library directly:

```rust
use wordcab_fetch::{Config, FetchIndex};

// Row-major embeddings: [document_count, embedding_dimension].
let embeddings: Vec<f32> = embed_documents_locally(&documents);
let dim = 384;

let index = FetchIndex::build(
    &embeddings,
    documents.len(),
    dim,
    Config::default(),
)?;
index.save("knowledge.fetch")?;

// A later process can load the complete snapshot.
let index = FetchIndex::load("knowledge.fetch")?;
let query: Vec<f32> = embed_query_locally("how do I rotate an API key?");
let hits = index.search(&query, 10)?;

for hit in hits {
    println!("{}: {}", hit.score, documents[hit.id as usize]);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

Until the crate is published, point Cargo at this repository:

```toml
[dependencies]
wordcab-fetch = { git = "https://github.com/Wordcab/fetch" }
```

## Setting up a local index

The durable unit is an immutable `.fetch` snapshot:

1. Chunk your documents and embed every chunk with any local or self-hosted
   model. Keep the vectors in row-major order.
2. Keep document text and metadata in your own store. A Fetch result ID is the
   zero-based vector row, so `documents[hit.id]` is the entire mapping layer.
3. Call `FetchIndex::build`, then `save`. Index construction belongs off the
   request path; the 100k benchmark builds in about 49 seconds.
4. At startup, call `FetchIndex::load`. Embed queries with the same model and
   dimension used for the documents, then call `search`.
5. When documents or the embedding model change, build a new snapshot and swap
   files atomically.

Fetch normalizes document and query vectors internally. Zero vectors, non-finite
values, dimension mismatches, and malformed index files are rejected.

### What is inside `.fetch`?

```text
header | packed 1-bit HNSW graph | f16 rerank slab
```

At 100k × 384, the graph is 11.23 MiB and the complete file is 84.47 MiB. On
load, the rerank slab expands to `f32` for speed; estimated resident memory is
163.82 MiB. The format favors low query latency and high recall over the smallest
possible RAM footprint.

The native graph is powered by
[USearch](https://github.com/unum-cloud/USearch). Fetch adds the binary-candidate
plus full-vector-rerank layout, validation, the single-file snapshot API, and an
exact-baseline benchmark.

## Benchmarks

These numbers were measured on Linux, an AMD Ryzen 7 5800H, Rust release mode,
one query at a time, after 20 warm-up queries.

Dataset: 100,000 normalized 384-dimensional clustered synthetic vectors and 300
perturbed queries. Ground truth is exhaustive `f32` cosine top-10. This measures
index recall—not embedding-model quality, chunking quality, or end-to-end RAG.

| Rerank candidates | Recall@10 | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|
| 50 | 76.73% | 0.19 ms | 0.30 ms | 0.34 ms |
| 100 | 87.27% | 0.22 ms | 0.35 ms | 0.37 ms |
| 200 | 95.37% | 0.42 ms | 0.57 ms | 0.63 ms |
| **400 (default)** | **99.43%** | **0.85 ms** | **1.01 ms** | **1.08 ms** |
| 600 | 99.83% | 1.39 ms | 1.60 ms | 2.22 ms |

Reproduce the complete sweep:

```bash
cargo run --release --example benchmark -- \
  --count 100000 \
  --queries 300 \
  --candidates 50,100,200,400,600
```

Use `--json` for machine-readable output. Use `--keep-index path.fetch` to retain
the generated snapshot.

## Browser / WASM

The browser kernel uses a flat packed-bit scan plus float reranking. It is
separate from the native HNSW-backed `.fetch` format: pass it a `Float32Array`
when the app starts, typically loaded from IndexedDB or OPFS.
Pass normalized document embeddings; the low-level browser kernel ranks by dot
product and does not normalize them for you.

```bash
rustup target add wasm32-unknown-unknown
cargo build --release --package wordcab-fetch-wasm --target wasm32-unknown-unknown
node wasm/tests/smoke.mjs
node wasm/tests/benchmark.mjs
```

```js
import { loadFetch } from "./wasm/fetch.mjs";

const runtime = await loadFetch("./wordcab_fetch_wasm.wasm");
const index = runtime.fromVectors(vectors, documentCount, dimension);
const hits = index.search(query, { candidates: 400, topK: 10 });
index.free();
```

The Node 22 smoke workload at 100k × 384 measured 0.91 ms p50, 1.14 ms p99,
466 ms to build, and a 35,444-byte raw WASM binary. Its accuracy check is anchor
recovery on synthetic signed vectors, so it is intentionally not presented as
comparable to the native exact-recall benchmark.

To run the browser example:

```bash
python3 -m http.server 8080
```

Open `http://localhost:8080/wasm/examples/browser.html`.

## Scope

Fetch currently optimizes for read-heavy local retrieval:

- immutable snapshots, not live inserts or deletes;
- dense semantic retrieval, not BM25 or metadata filtering;
- bring-your-own embeddings, chunking, and document store;
- native Rust and a low-level JavaScript/WASM wrapper;
- one process, one index, no distributed layer.

Those constraints are the point: the query path remains small, inspectable, and
entirely yours.

## License

Apache-2.0.
