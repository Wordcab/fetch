import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { loadFetch } from "../fetch.mjs";

const count = Number(process.env.FETCH_COUNT ?? 100_000);
const dim = Number(process.env.FETCH_DIM ?? 384);
const queries = Number(process.env.FETCH_QUERIES ?? 500);
const candidates = Number(process.env.FETCH_CANDIDATES ?? 400);
const topK = 5;

const wasm = readFileSync(
  new URL("../../target/wasm32-unknown-unknown/release/wordcab_fetch_wasm.wasm", import.meta.url),
);
const runtime = await loadFetch(wasm);
const { vectors, queryVectors, anchors } = makeVectors(count, dim, queries);

const buildStarted = performance.now();
const index = runtime.fromVectors(vectors, count, dim);
const buildMs = performance.now() - buildStarted;

for (let query = 0; query < Math.min(20, queries); query++) {
  index.search(queryVectors.subarray(query * dim, (query + 1) * dim), { candidates, topK });
}

const latencies = [];
let anchorsFound = 0;
for (let query = 0; query < queries; query++) {
  const started = performance.now();
  const hits = index.search(queryVectors.subarray(query * dim, (query + 1) * dim), {
    candidates,
    topK,
  });
  latencies.push(performance.now() - started);
  if (hits.some((hit) => hit.id === anchors[query])) anchorsFound++;
}
index.free();

latencies.sort((left, right) => left - right);
const result = {
  count,
  dim,
  queries,
  candidates,
  anchor_recall_at_5: anchorsFound / queries,
  build_ms: buildMs,
  p50_ms: percentile(latencies, 50),
  p95_ms: percentile(latencies, 95),
  p99_ms: percentile(latencies, 99),
  wasm_bytes: wasm.length,
};
if (result.anchor_recall_at_5 !== 1) {
  throw new Error(`expected perfect anchor recovery, got ${result.anchor_recall_at_5}`);
}
console.log(JSON.stringify(result, null, 2));

function makeVectors(rows, dimensions, queryCount) {
  const vectors = new Float32Array(rows * dimensions);
  const queryVectors = new Float32Array(queryCount * dimensions);
  const anchors = new Uint32Array(queryCount);
  let state = 0x9e3779b9;

  function randomBit() {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return state & 1;
  }

  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < dimensions; column++) {
      vectors[row * dimensions + column] = randomBit() ? 1 : -1;
    }
  }
  for (let query = 0; query < queryCount; query++) {
    const anchor = (query * 7919) % rows;
    anchors[query] = anchor;
    queryVectors.set(
      vectors.subarray(anchor * dimensions, (anchor + 1) * dimensions),
      query * dimensions,
    );
    queryVectors[query * dimensions + ((anchor + query + 1) % dimensions)] *= -1;
  }
  return { vectors, queryVectors, anchors };
}

function percentile(sorted, value) {
  const rank = Math.ceil((value / 100) * (sorted.length - 1));
  return sorted[Math.min(sorted.length - 1, rank)];
}
