import { readFileSync } from "node:fs";
import { loadFetch } from "../fetch.mjs";

const wasm = readFileSync(
  new URL("../../target/wasm32-unknown-unknown/release/wordcab_fetch_wasm.wasm", import.meta.url),
);
const runtime = await loadFetch(wasm);
const vectors = new Float32Array([
  1, -1, -1, -1,
  -1, 1, -1, -1,
  -1, -1, 1, -1,
  -1, -1, -1, 1,
]);
const index = runtime.fromVectors(vectors, 4, 4);
const hits = index.search(new Float32Array([0.9, -1, -1, -1]), {
  candidates: 4,
  topK: 2,
});
index.free();

if (hits.length !== 2 || hits[0].id !== 0) {
  throw new Error(`unexpected results: ${JSON.stringify(hits)}`);
}
console.log(JSON.stringify({ ok: true, hits }, null, 2));
