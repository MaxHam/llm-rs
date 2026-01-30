mod bpe;
use std::{fs::read_to_string, path::Path};

use bpe::BytePairEncoder;

fn main() {
    let path = Path::new("gutenberg_txts/corpus.txt");
    let corpus = read_to_string(path).expect("Failed to read corpus file");
    let bpe = BytePairEncoder::train(corpus.as_str(), 100);
    let vocab = bpe.vocabulary();

    // Store vocab as a JSON file (as a HashMap)
    let json = serde_json::to_string(&vocab).expect("Failed to serialize vocab");
    std::fs::write("vocab.json", json).expect("Failed to write vocab.json");
}
