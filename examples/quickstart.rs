use std::error::Error;
use wordcab_fetch::{Config, FetchIndex};

fn main() -> Result<(), Box<dyn Error>> {
    let dim = 4;
    let vectors = vec![
        1.0, 0.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, 0.0, // row 1
        0.0, 0.0, 1.0, 0.0, // row 2
        0.0, 0.0, 0.0, 1.0, // row 3
    ];

    let index = FetchIndex::build(
        &vectors,
        4,
        dim,
        Config {
            candidates: 4,
            ..Config::default()
        },
    )?;
    index.save("demo.fetch")?;

    let local = FetchIndex::load("demo.fetch")?;
    let hits = local.search(&[0.05, 0.90, 0.05, 0.0], 2)?;
    println!("{hits:#?}");
    Ok(())
}
