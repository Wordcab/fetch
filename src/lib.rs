//! A small, local-first semantic retrieval engine.
//!
//! Fetch uses a packed 1-bit HNSW graph to find candidates, then reranks those
//! candidates with full vectors. Queries and indexes stay in-process.

use half::f16;
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use usearch::{b1x8, Index, IndexOptions, MetricKind, ScalarKind};

const MAGIC: [u8; 8] = *b"WCFETCH\0";
const FORMAT_VERSION: u32 = 1;
const HEADER_BYTES: u64 = 56;

/// Index construction and query defaults.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    /// Maximum graph connections per node.
    pub connectivity: usize,
    /// Candidate expansion used while building the graph.
    pub expansion_add: usize,
    /// Candidate expansion used while searching the graph.
    pub expansion_search: usize,
    /// Number of binary ANN candidates sent to full-vector reranking.
    pub candidates: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            connectivity: 6,
            expansion_add: 300,
            expansion_search: 120,
            candidates: 400,
        }
    }
}

/// One semantic search result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchHit {
    /// Zero-based row in the vectors used to build the index.
    pub id: u64,
    /// Cosine similarity after reranking.
    pub score: f32,
}

/// Errors returned by index construction, persistence, and search.
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("invalid .fetch index: {0}")]
    InvalidFormat(String),
    #[error("index backend error: {0}")]
    Backend(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Convenient result alias for Fetch operations.
pub type Result<T> = std::result::Result<T, FetchError>;

/// An immutable local semantic index.
pub struct FetchIndex {
    graph: Index,
    vectors: Vec<f32>,
    count: usize,
    dim: usize,
    config: Config,
}

impl FetchIndex {
    /// Builds an index from a row-major `[count, dim]` vector matrix.
    ///
    /// Vectors are cosine-normalized internally. Result IDs are the zero-based
    /// row numbers, so applications can keep text and metadata in any local
    /// store keyed by that row.
    pub fn build(vectors: &[f32], count: usize, dim: usize, config: Config) -> Result<Self> {
        validate_shape(vectors.len(), count, dim)?;
        validate_config(config)?;

        let graph = new_graph(dim, config)?;
        graph
            .reserve(count)
            .map_err(|error| backend_error("reserve graph", error))?;

        let bytes = dim.div_ceil(8);
        let mut normalized = vec![0.0f32; dim];
        let mut binary = vec![b1x8(0); bytes];
        let mut slab = Vec::with_capacity(
            count
                .checked_mul(dim)
                .ok_or_else(|| FetchError::InvalidInput("count * dim overflows usize".into()))?,
        );

        for row in 0..count {
            let source = &vectors[row * dim..(row + 1) * dim];
            normalize_into(source, &mut normalized, row)?;
            pack_sign_bits(&normalized, &mut binary);
            graph
                .add(row as u64, &binary)
                .map_err(|error| backend_error("add vector", error))?;
            slab.extend(normalized.iter().copied());
        }

        Ok(Self {
            graph,
            vectors: slab,
            count,
            dim,
            config,
        })
    }

    /// Loads a complete, immutable `.fetch` index from local storage.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file_len = path.metadata()?.len();
        let mut reader = BufReader::new(File::open(path)?);

        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(FetchError::InvalidFormat("bad magic bytes".into()));
        }

        let version = read_u32(&mut reader)?;
        if version != FORMAT_VERSION {
            return Err(FetchError::InvalidFormat(format!(
                "unsupported format version {version}"
            )));
        }

        let dim = usize_from_u32(read_u32(&mut reader)?, "dimension")?;
        let count = usize_from_u64(read_u64(&mut reader)?, "count")?;
        let config = Config {
            connectivity: usize_from_u32(read_u32(&mut reader)?, "connectivity")?,
            expansion_add: usize_from_u32(read_u32(&mut reader)?, "expansion_add")?,
            expansion_search: usize_from_u32(read_u32(&mut reader)?, "expansion_search")?,
            candidates: usize_from_u32(read_u32(&mut reader)?, "candidates")?,
        };
        validate_config(config)?;

        let graph_len = usize_from_u64(read_u64(&mut reader)?, "graph length")?;
        let slab_values = usize_from_u64(read_u64(&mut reader)?, "slab length")?;
        validate_shape(slab_values, count, dim)?;

        let slab_bytes = slab_values
            .checked_mul(2)
            .ok_or_else(|| FetchError::InvalidFormat("slab byte length overflows usize".into()))?;
        let expected_len = HEADER_BYTES
            .checked_add(graph_len as u64)
            .and_then(|value| value.checked_add(slab_bytes as u64))
            .ok_or_else(|| FetchError::InvalidFormat("file length overflows u64".into()))?;
        if expected_len != file_len {
            return Err(FetchError::InvalidFormat(format!(
                "file is {file_len} bytes; header describes {expected_len} bytes"
            )));
        }

        let mut graph_bytes = vec![0u8; graph_len];
        reader.read_exact(&mut graph_bytes)?;

        let mut raw_slab = vec![0u8; slab_bytes];
        reader.read_exact(&mut raw_slab)?;
        let vectors = raw_slab
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32())
            .collect();

        let graph = new_graph(dim, config)?;
        graph
            .load_from_buffer(&graph_bytes)
            .map_err(|error| backend_error("load graph", error))?;
        if graph.size() != count || graph.dimensions() != dim {
            return Err(FetchError::InvalidFormat(format!(
                "graph metadata mismatch: expected {count}x{dim}, got {}x{}",
                graph.size(),
                graph.dimensions()
            )));
        }

        Ok(Self {
            graph,
            vectors,
            count,
            dim,
            config,
        })
    }

    /// Saves the graph and reranking slab as one `.fetch` file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let graph_len = self.graph.serialized_length();
        let mut graph_bytes = vec![0u8; graph_len];
        self.graph
            .save_to_buffer(&mut graph_bytes)
            .map_err(|error| backend_error("serialize graph", error))?;

        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(&MAGIC)?;
        write_u32(&mut writer, FORMAT_VERSION)?;
        write_u32(&mut writer, u32_from_usize(self.dim, "dimension")?)?;
        write_u64(&mut writer, self.count as u64)?;
        write_u32(
            &mut writer,
            u32_from_usize(self.config.connectivity, "connectivity")?,
        )?;
        write_u32(
            &mut writer,
            u32_from_usize(self.config.expansion_add, "expansion_add")?,
        )?;
        write_u32(
            &mut writer,
            u32_from_usize(self.config.expansion_search, "expansion_search")?,
        )?;
        write_u32(
            &mut writer,
            u32_from_usize(self.config.candidates, "candidates")?,
        )?;
        write_u64(&mut writer, graph_len as u64)?;
        write_u64(&mut writer, self.vectors.len() as u64)?;
        writer.write_all(&graph_bytes)?;
        for value in &self.vectors {
            writer.write_all(&f16::from_f32(*value).to_bits().to_le_bytes())?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Searches with the candidate count stored in the index configuration.
    pub fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>> {
        self.search_with_candidates(query, top_k, self.config.candidates)
    }

    /// Searches with an explicit accuracy/latency tradeoff for this query.
    pub fn search_with_candidates(
        &self,
        query: &[f32],
        top_k: usize,
        candidates: usize,
    ) -> Result<Vec<SearchHit>> {
        if query.len() != self.dim {
            return Err(FetchError::InvalidInput(format!(
                "query has {} values; expected {}",
                query.len(),
                self.dim
            )));
        }
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if candidates == 0 {
            return Err(FetchError::InvalidInput(
                "candidates must be greater than zero".into(),
            ));
        }

        let mut normalized = vec![0.0f32; self.dim];
        normalize_into(query, &mut normalized, usize::MAX)?;
        let mut binary = vec![b1x8(0); self.dim.div_ceil(8)];
        pack_sign_bits(&normalized, &mut binary);

        let shortlist_len = candidates.max(top_k).min(self.count);
        let matches = self
            .graph
            .search(&binary, shortlist_len)
            .map_err(|error| backend_error("search graph", error))?;

        let mut reranked = Vec::with_capacity(matches.keys.len());
        for id in matches.keys {
            let row = usize::try_from(id).map_err(|_| {
                FetchError::InvalidFormat(format!("graph key {id} does not fit usize"))
            })?;
            if row >= self.count {
                return Err(FetchError::InvalidFormat(format!(
                    "graph key {id} is outside the vector slab"
                )));
            }
            let offset = row * self.dim;
            let score = dot(
                &normalized,
                &self.vectors[offset..offset.saturating_add(self.dim)],
            );
            reranked.push(SearchHit { id, score });
        }
        reranked.sort_unstable_by(|a, b| cmp_f32_desc(a.score, b.score));
        reranked.truncate(top_k.min(reranked.len()));
        Ok(reranked)
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn dimensions(&self) -> usize {
        self.dim
    }

    pub fn config(&self) -> Config {
        self.config
    }

    /// Serialized bytes used by the binary ANN graph, excluding the rerank slab.
    pub fn graph_bytes(&self) -> usize {
        self.graph.serialized_length()
    }

    /// Backend-reported graph memory plus the in-memory `f32` rerank slab.
    pub fn resident_bytes_estimate(&self) -> usize {
        self.graph.memory_usage().saturating_add(
            self.vectors
                .len()
                .saturating_mul(std::mem::size_of::<f32>()),
        )
    }

    /// SIMD instruction sets available to the native graph backend.
    pub fn hardware_acceleration(&self) -> String {
        self.graph.hardware_acceleration()
    }
}

fn new_graph(dim: usize, config: Config) -> Result<Index> {
    Index::new(&IndexOptions {
        dimensions: dim,
        metric: MetricKind::Hamming,
        quantization: ScalarKind::B1,
        connectivity: config.connectivity,
        expansion_add: config.expansion_add,
        expansion_search: config.expansion_search,
        multi: false,
    })
    .map_err(|error| backend_error("create graph", error))
}

fn validate_shape(len: usize, count: usize, dim: usize) -> Result<()> {
    if count == 0 {
        return Err(FetchError::InvalidInput(
            "count must be greater than zero".into(),
        ));
    }
    if dim == 0 {
        return Err(FetchError::InvalidInput(
            "dimension must be greater than zero".into(),
        ));
    }
    let expected = count
        .checked_mul(dim)
        .ok_or_else(|| FetchError::InvalidInput("count * dim overflows usize".into()))?;
    if len != expected {
        return Err(FetchError::InvalidInput(format!(
            "vector matrix has {len} values; expected {count} * {dim} = {expected}"
        )));
    }
    Ok(())
}

fn validate_config(config: Config) -> Result<()> {
    if config.connectivity == 0
        || config.expansion_add == 0
        || config.expansion_search == 0
        || config.candidates == 0
    {
        return Err(FetchError::InvalidInput(
            "all config values must be greater than zero".into(),
        ));
    }
    Ok(())
}

fn normalize_into(source: &[f32], output: &mut [f32], row: usize) -> Result<()> {
    let mut squared_norm = 0.0f32;
    for value in source {
        if !value.is_finite() {
            let label = if row == usize::MAX {
                "query".to_string()
            } else {
                format!("vector row {row}")
            };
            return Err(FetchError::InvalidInput(format!(
                "{label} contains a non-finite value"
            )));
        }
        squared_norm += value * value;
    }
    let norm = squared_norm.sqrt();
    if norm <= f32::EPSILON {
        let label = if row == usize::MAX {
            "query".to_string()
        } else {
            format!("vector row {row}")
        };
        return Err(FetchError::InvalidInput(format!(
            "{label} has zero magnitude"
        )));
    }
    for (target, value) in output.iter_mut().zip(source) {
        *target = *value / norm;
    }
    Ok(())
}

fn pack_sign_bits(vector: &[f32], output: &mut [b1x8]) {
    output.fill(b1x8(0));
    for (index, value) in vector.iter().enumerate() {
        if *value > 0.0 {
            output[index / 8].0 |= 1u8 << (index % 8);
        }
    }
}

fn dot(query: &[f32], vector: &[f32]) -> f32 {
    query
        .iter()
        .zip(vector)
        .map(|(left, right)| left * right)
        .sum()
}

fn cmp_f32_desc(left: f32, right: f32) -> Ordering {
    right.total_cmp(&left)
}

fn backend_error(context: &str, error: impl std::fmt::Display) -> FetchError {
    FetchError::Backend(format!("{context}: {error}"))
}

fn write_u32(writer: &mut impl Write, value: u32) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut impl Write, value: u64) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(reader: &mut impl Read) -> std::io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> std::io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn usize_from_u32(value: u32, field: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| FetchError::InvalidFormat(format!("{field} does not fit usize")))
}

fn usize_from_u64(value: u64, field: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| FetchError::InvalidFormat(format!("{field} does not fit usize")))
}

fn u32_from_usize(value: usize, field: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| FetchError::InvalidInput(format!("{field} does not fit u32")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn tiny_vectors() -> Vec<f32> {
        vec![
            1.0, 0.0, 0.0, 0.0, //
            0.9, 0.1, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
        ]
    }

    #[test]
    fn nearest_vector_is_ranked_first() {
        let index = FetchIndex::build(
            &tiny_vectors(),
            4,
            4,
            Config {
                candidates: 4,
                ..Config::default()
            },
        )
        .unwrap();
        let hits = index.search(&[0.99, 0.01, 0.0, 0.0], 2).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 0);
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn saved_index_round_trips() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("tiny.fetch");
        let original = FetchIndex::build(
            &tiny_vectors(),
            4,
            4,
            Config {
                candidates: 4,
                ..Config::default()
            },
        )
        .unwrap();
        original.save(&path).unwrap();

        let loaded = FetchIndex::load(&path).unwrap();
        let hits = loaded.search(&[0.0, 0.0, 0.95, 0.05], 2).unwrap();
        assert_eq!(loaded.count(), 4);
        assert_eq!(loaded.dimensions(), 4);
        assert_eq!(hits[0].id, 3);
    }

    #[test]
    fn malformed_shapes_and_queries_are_rejected() {
        assert!(FetchIndex::build(&[1.0, 2.0], 2, 2, Config::default()).is_err());
        let index = FetchIndex::build(
            &tiny_vectors(),
            4,
            4,
            Config {
                candidates: 4,
                ..Config::default()
            },
        )
        .unwrap();
        assert!(index.search(&[1.0, 0.0], 1).is_err());
        assert!(index.search(&[0.0; 4], 1).is_err());
    }

    #[test]
    fn corrupt_index_is_rejected() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("corrupt.fetch");
        std::fs::write(&path, b"not a fetch index").unwrap();
        assert!(FetchIndex::load(&path).is_err());
    }
}
