use clap::Parser;
use rand::prelude::*;
use rand_distr::StandardNormal;
use serde::Serialize;
use std::collections::HashSet;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use wordcab_fetch::{Config, FetchIndex};

#[derive(Parser, Debug)]
#[command(about = "Benchmark Fetch against an exact cosine top-k baseline")]
struct Args {
    #[arg(long, default_value_t = 100_000)]
    count: usize,
    #[arg(long, default_value_t = 384)]
    dim: usize,
    #[arg(long, default_value_t = 300)]
    queries: usize,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long, default_value_t = 1_024)]
    clusters: usize,
    #[arg(long, default_value_t = 0.05)]
    cluster_std: f32,
    #[arg(long, default_value_t = 0.01)]
    query_noise: f32,
    #[arg(long, default_value_t = 13)]
    seed: u64,
    #[arg(long, default_value_t = 6)]
    connectivity: usize,
    #[arg(long, default_value_t = 300)]
    expansion_add: usize,
    #[arg(long, default_value_t = 120)]
    expansion_search: usize,
    #[arg(long, value_delimiter = ',', default_value = "400")]
    candidates: Vec<usize>,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    keep_index: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    count: usize,
    dim: usize,
    queries: usize,
    top_k: usize,
    connectivity: usize,
    expansion_add: usize,
    expansion_search: usize,
    build_ms: f64,
    save_ms: f64,
    load_ms: f64,
    graph_bytes: usize,
    file_bytes: u64,
    resident_bytes_estimate: usize,
    acceleration: String,
    profiles: Vec<QueryProfile>,
}

#[derive(Debug, Serialize)]
struct QueryProfile {
    candidates: usize,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    recall_at_k: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.top_k == 0 || args.queries == 0 || args.clusters == 0 {
        return Err("top-k, queries, and clusters must be greater than zero".into());
    }

    let (vectors, queries) = clustered_vectors(
        args.count,
        args.dim,
        args.queries,
        args.clusters,
        args.cluster_std,
        args.query_noise,
        args.seed,
    );
    let exact = exact_top_k(&vectors, &queries, args.count, args.dim, args.top_k);
    if args.candidates.contains(&0) {
        return Err("candidate depths must be greater than zero".into());
    }
    let max_candidates = *args
        .candidates
        .iter()
        .max()
        .ok_or("at least one candidate depth is required")?;
    let config = Config {
        connectivity: args.connectivity,
        expansion_add: args.expansion_add,
        expansion_search: args.expansion_search,
        candidates: max_candidates,
    };

    let started = Instant::now();
    let index = FetchIndex::build(&vectors, args.count, args.dim, config)?;
    let build_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let graph_bytes = index.graph_bytes();
    let resident_bytes_estimate = index.resident_bytes_estimate();
    let acceleration = index.hardware_acceleration();

    let temporary = std::env::temp_dir().join(format!(
        "wordcab-fetch-benchmark-{}-{}.fetch",
        std::process::id(),
        args.seed
    ));
    let index_path = args.keep_index.as_ref().unwrap_or(&temporary);
    let started = Instant::now();
    index.save(index_path)?;
    let save_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let file_bytes = fs::metadata(index_path)?.len();
    drop(index);

    let started = Instant::now();
    let loaded = FetchIndex::load(index_path)?;
    let load_ms = started.elapsed().as_secs_f64() * 1_000.0;

    let mut profiles = Vec::with_capacity(args.candidates.len());
    for candidates in &args.candidates {
        for query in queries.iter().take(20) {
            let _ = loaded.search_with_candidates(query, args.top_k, *candidates)?;
        }

        let mut latencies = Vec::with_capacity(args.queries);
        let mut matched = 0usize;
        for (query, expected) in queries.iter().zip(&exact) {
            let started = Instant::now();
            let hits = loaded.search_with_candidates(query, args.top_k, *candidates)?;
            latencies.push(started.elapsed().as_secs_f64() * 1_000.0);
            let expected: HashSet<u64> = expected.iter().map(|id| *id as u64).collect();
            matched += hits.iter().filter(|hit| expected.contains(&hit.id)).count();
        }
        latencies.sort_by(f64::total_cmp);
        profiles.push(QueryProfile {
            candidates: *candidates,
            p50_ms: percentile(&latencies, 50.0),
            p95_ms: percentile(&latencies, 95.0),
            p99_ms: percentile(&latencies, 99.0),
            recall_at_k: matched as f64 / (args.queries * args.top_k) as f64,
        });
    }

    let result = BenchmarkResult {
        count: args.count,
        dim: args.dim,
        queries: args.queries,
        top_k: args.top_k,
        connectivity: config.connectivity,
        expansion_add: config.expansion_add,
        expansion_search: config.expansion_search,
        build_ms,
        save_ms,
        load_ms,
        graph_bytes,
        file_bytes,
        resident_bytes_estimate,
        acceleration,
        profiles,
    };

    if args.keep_index.is_none() {
        let _ = fs::remove_file(&temporary);
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("Fetch exact-baseline benchmark");
        println!("vectors       {} x {}", result.count, result.dim);
        println!("queries       {}", result.queries);
        println!("build         {:.1} ms", result.build_ms);
        println!(
            "save / load   {:.1} / {:.1} ms",
            result.save_ms, result.load_ms
        );
        println!("graph         {:.2} MiB", mib(result.graph_bytes as u64));
        println!("index file    {:.2} MiB", mib(result.file_bytes));
        println!(
            "resident est. {:.2} MiB",
            mib(result.resident_bytes_estimate as u64)
        );
        println!("acceleration  {}", result.acceleration);
        println!();
        println!(
            "{:>10} {:>12} {:>10} {:>10} {:>10}",
            "candidates",
            format!("recall@{}", result.top_k),
            "p50 ms",
            "p95 ms",
            "p99 ms"
        );
        for profile in &result.profiles {
            println!(
                "{:>10} {:>12.4} {:>10.3} {:>10.3} {:>10.3}",
                profile.candidates,
                profile.recall_at_k,
                profile.p50_ms,
                profile.p95_ms,
                profile.p99_ms
            );
        }
    }

    Ok(())
}

fn clustered_vectors(
    count: usize,
    dim: usize,
    query_count: usize,
    clusters: usize,
    cluster_std: f32,
    query_noise: f32,
    seed: u64,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centers = Vec::with_capacity(clusters);
    for _ in 0..clusters {
        let mut center = random_vector(dim, &mut rng);
        normalize(&mut center);
        centers.push(center);
    }

    let mut vectors = Vec::with_capacity(count * dim);
    for _ in 0..count {
        let center = &centers[rng.gen_range(0..clusters)];
        let mut vector = Vec::with_capacity(dim);
        for value in center {
            let noise: f32 = rng.sample(StandardNormal);
            vector.push(*value + noise * cluster_std);
        }
        normalize(&mut vector);
        vectors.extend(vector);
    }

    let mut queries = Vec::with_capacity(query_count);
    for _ in 0..query_count {
        let anchor = rng.gen_range(0..count);
        let mut query = vectors[anchor * dim..(anchor + 1) * dim].to_vec();
        for value in &mut query {
            let noise: f32 = rng.sample(StandardNormal);
            *value += noise * query_noise;
        }
        normalize(&mut query);
        queries.push(query);
    }
    (vectors, queries)
}

fn exact_top_k(
    vectors: &[f32],
    queries: &[Vec<f32>],
    count: usize,
    dim: usize,
    top_k: usize,
) -> Vec<Vec<usize>> {
    queries
        .iter()
        .map(|query| {
            let mut scores = Vec::with_capacity(count);
            for row in 0..count {
                let vector = &vectors[row * dim..(row + 1) * dim];
                scores.push((row, dot(query, vector)));
            }
            let keep = top_k.min(scores.len());
            if keep < scores.len() {
                scores.select_nth_unstable_by(keep, |left, right| right.1.total_cmp(&left.1));
                scores.truncate(keep);
            }
            scores.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
            scores.into_iter().map(|(id, _)| id).collect()
        })
        .collect()
}

fn random_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    (0..dim)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect()
}

fn normalize(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
    for value in vector {
        *value /= norm;
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((percentile / 100.0) * (sorted.len() - 1) as f64).ceil() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}
