export async function loadFetch(wasmSource) {
  const bytes = await loadBytes(wasmSource);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return new FetchModule(instance.exports);
}

async function loadBytes(source) {
  if (source instanceof ArrayBuffer) return source;
  if (ArrayBuffer.isView(source)) {
    return source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength);
  }
  if (typeof source === "string" || source instanceof URL) {
    const response = await fetch(source);
    if (!response.ok) throw new Error(`failed to fetch Fetch WASM: ${response.status}`);
    return response.arrayBuffer();
  }
  throw new TypeError("wasmSource must be a URL, ArrayBuffer, or Uint8Array");
}

export class FetchModule {
  constructor(exports) {
    this.exports = exports;
  }

  fromVectors(vectors, count, dim) {
    if (!(vectors instanceof Float32Array)) {
      throw new TypeError("vectors must be a Float32Array");
    }
    requirePositiveInteger(count, "count");
    requirePositiveInteger(dim, "dim");
    if (vectors.length !== count * dim) {
      throw new Error(`vectors length ${vectors.length} does not equal count * dim`);
    }

    const pointer = this.exports.alloc_f32(vectors.length);
    if (!pointer) throw new Error("could not allocate WASM vector buffer");
    try {
      new Float32Array(this.exports.memory.buffer, pointer, vectors.length).set(vectors);
      const indexPointer = this.exports.index_new(pointer, count, dim);
      if (!indexPointer) throw new Error("could not create Fetch index");
      return new FetchIndex(this.exports, indexPointer, count, dim);
    } finally {
      this.exports.free_f32(pointer, vectors.length);
    }
  }
}

export class FetchIndex {
  constructor(exports, pointer, count, dim) {
    this.exports = exports;
    this.pointer = pointer;
    this.count = count;
    this.dim = dim;
  }

  search(query, options = {}) {
    if (!this.pointer) throw new Error("index has been freed");
    if (!(query instanceof Float32Array)) {
      throw new TypeError("query must be a Float32Array");
    }
    if (query.length !== this.dim) {
      throw new Error(`query length ${query.length} does not equal index dim ${this.dim}`);
    }

    const candidates = options.candidates ?? 400;
    const topK = options.topK ?? 10;
    requirePositiveInteger(candidates, "candidates");
    requirePositiveInteger(topK, "topK");

    const queryPointer = this.exports.alloc_f32(this.dim);
    const idsPointer = this.exports.alloc_u32(topK);
    const scoresPointer = this.exports.alloc_f32(topK);
    if (!queryPointer || !idsPointer || !scoresPointer) {
      if (queryPointer) this.exports.free_f32(queryPointer, this.dim);
      if (idsPointer) this.exports.free_u32(idsPointer, topK);
      if (scoresPointer) this.exports.free_f32(scoresPointer, topK);
      throw new Error("could not allocate WASM search buffers");
    }

    try {
      new Float32Array(this.exports.memory.buffer, queryPointer, this.dim).set(query);
      const found = this.exports.index_search(
        this.pointer,
        queryPointer,
        candidates,
        topK,
        idsPointer,
        scoresPointer,
      );
      const ids = Array.from(new Uint32Array(this.exports.memory.buffer, idsPointer, found));
      const scores = Array.from(new Float32Array(this.exports.memory.buffer, scoresPointer, found));
      return ids.map((id, index) => ({ id, score: scores[index] }));
    } finally {
      this.exports.free_f32(queryPointer, this.dim);
      this.exports.free_u32(idsPointer, topK);
      this.exports.free_f32(scoresPointer, topK);
    }
  }

  free() {
    if (this.pointer) {
      this.exports.index_free(this.pointer);
      this.pointer = 0;
    }
  }
}

function requirePositiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new TypeError(`${name} must be a positive safe integer`);
  }
}
