use std::io::{self, Write};

use candle_core::Tensor;
use clap::Parser;
use llm_rs::bigram::Bigram;
use llm_rs::bpe::{TokenTranslation, Tokenizer};
use llm_rs::cli::{Cli, Model, TokenizerKind};
use llm_rs::dataset::Dataset;
use llm_rs::sampling::Generator;
use llm_rs::training::{Training, TrainingConfig};
use llm_rs::transformer::Transformer;

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let device = cli.device.to_device(cli.cuda_device)?;

    println!("Loading corpus from {}", cli.data);
    let corpus = std::fs::read_to_string(&cli.data)?;

    let tokenizer = match cli.tokenizer {
        TokenizerKind::Bpe => {
            println!("Training BPE tokenizer (vocab_size={})", cli.vocab_size);
            Tokenizer::train(&corpus, cli.vocab_size).expect("failed to train tokenizer")
        }
        TokenizerKind::Ascii => {
            println!("Using ASCII tokenizer (vocab_size=128)");
            Tokenizer::ascii()
        }
    };

    let actual_vocab = tokenizer.vocabulary.len();
    println!("Vocabulary size: {actual_vocab}");

    let mut dataset = Dataset::from_file(&cli.data, cli.train_ratio, &tokenizer, &device)?;
    println!(
        "Dataset: {} training tokens, {} validation tokens",
        dataset.training_size, dataset.validation_size
    );

    let training_config = TrainingConfig {
        num_epochs: cli.epochs,
        batch_size: cli.batch_size,
        lr: cli.lr,
        weight_decay: cli.weight_decay,
    };

    let mut generator: Box<dyn Generator> = match cli.model {
        Model::Transformer => {
            println!(
                "Initializing Transformer (seq={}, emb={}, blocks={}, heads={}, dropout={})",
                cli.max_seq_len, cli.n_emb, cli.n_blocks, cli.n_heads, cli.dropout
            );
            let model = Transformer::new(
                actual_vocab,
                &device,
                cli.max_seq_len,
                cli.n_emb,
                cli.n_blocks,
                cli.n_heads,
                cli.dropout,
            )?;
            println!(
                "Training for {} epochs (batch_size={}, lr={}, wd={})",
                cli.epochs, cli.batch_size, cli.lr, cli.weight_decay
            );
            model.train(&mut dataset, &training_config)?;
            Box::new(model)
        }
        Model::Bigram => {
            println!("Initializing Bigram model");
            let model = Bigram::new(actual_vocab, &device)?;
            println!(
                "Training for {} epochs (batch_size={}, lr={}, wd={})",
                cli.epochs, cli.batch_size, cli.lr, cli.weight_decay
            );
            model.train(&mut dataset, &training_config)?;
            Box::new(model)
        }
    };

    println!("\nChitchat with your LLM");
    println!("Type something and press enter. Ctrl+C to exit.\n");

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        let encoded = tokenizer.encode(input);
        let mut prompt = Tensor::from_tokens(&encoded, &device)?;
        prompt = prompt.unsqueeze(0)?;

        let output = generator.generate(prompt, cli.max_tokens, cli.temperature, cli.top_k)?;
        let decoded = tokenizer.decode(&output.to_tokens(&tokenizer));

        println!("{decoded}\n");
    }
}
