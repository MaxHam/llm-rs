use anyhow::Error;
use candle_core::Device;
use clap::{Parser, ValueEnum};

#[derive(ValueEnum, Clone, Debug)]
pub enum Model {
    Transformer,
    Bigram,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum TokenizerKind {
    Bpe,
    Ascii,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum DeviceKind {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl DeviceKind {
    pub fn to_device(&self) -> Result<Device, Error> {
    match self {
            DeviceKind::Cpu => Ok(Device::Cpu),

            #[cfg(feature = "cuda")]
            DeviceKind::Cuda => {
                // CUDA implementation
                // for simplicity, we just always pick one
                Ok(Device::Cuda(0))
            }
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "llm-rs", about = "Train and chat with small language models")]
pub struct Cli {
    /// Model architecture
    #[arg(short, long, default_value = "transformer")]
    pub model: Model,

    /// Path to training data (plain text file)
    #[arg(short, long, default_value = "./test_corpus.txt")]
    pub data: String,

    /// Tokenizer type
    #[arg(short, long, default_value = "bpe")]
    pub tokenizer: TokenizerKind,

    /// Compute device
    #[arg(long, default_value = "cpu")]
    pub device: DeviceKind,

    // ── Tokenizer ───────────────────────────────────────────────
    /// BPE vocabulary size (must be > 256 for bpe tokenizer)
    #[arg(long, default_value_t = 512)]
    pub vocab_size: u16,

    // ── Training hyperparameters ────────────────────────────────
    /// Number of training epochs
    #[arg(long, default_value_t = 256)]
    pub epochs: usize,

    /// Batch size
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,

    /// Learning rate
    #[arg(long, default_value_t = 1e-3)]
    pub lr: f64,

    /// AdamW weight decay
    #[arg(long, default_value_t = 0.0)]
    pub weight_decay: f64,

    /// Train/validation split ratio
    #[arg(long, default_value_t = 0.9)]
    pub train_ratio: f64,

    // ── Transformer architecture (ignored for bigram) ───────────
    /// Maximum sequence length / context window
    #[arg(long, default_value_t = 256)]
    pub max_seq_len: usize,

    /// Embedding dimension
    #[arg(long, default_value_t = 384)]
    pub n_emb: usize,

    /// Number of transformer blocks
    #[arg(long, default_value_t = 6)]
    pub n_blocks: usize,

    /// Number of attention heads
    #[arg(long, default_value_t = 6)]
    pub n_heads: usize,

    /// Dropout probability
    #[arg(long, default_value_t = 0.2)]
    pub dropout: f32,

    // ── Generation parameters ───────────────────────────────────
    /// Max new tokens to generate per prompt
    #[arg(long, default_value_t = 32)]
    pub max_tokens: usize,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.9)]
    pub temperature: f32,

    /// Top-k sampling (0 = disabled)
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,
}
