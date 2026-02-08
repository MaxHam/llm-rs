use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};

#[derive(Clone)]
struct GPTConfig {
    vocab_size: usize,
    n_embd: usize,
    device: Device
}

pub struct Transformer<'a> {
    config: &'a GPTConfig,
    tok_emb: Embedding,
}

impl<'a> Transformer<'a> {
    pub fn from_config(config: &'a GPTConfig) -> Result<Self> {
        // random initial weights
        let device = config.device.clone();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let tok_emb = candle_nn::embedding(
            config.vocab_size,
            config.n_embd,
            vb.pp("tok_emb"),
        )?;

        Ok(Self {
            config,
            tok_emb,
        })
    }

    fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let x = self.tok_emb.forward(idx)?;
        let w = self.tok_emb.embeddings();
        let (batch, tokens, embeddings) = x.dims3()?;
        let x2d = x.reshape((batch * tokens, embeddings))?;
        // (B*T, C)

        // Transpose for output projection
        let w_t = w.transpose(0, 1)?;
        let logits2d = x2d.matmul(&w_t)?;
        // (B*T, V)

        let logits = logits2d.reshape((batch, tokens, self.config.vocab_size))?;
        // (B, T, V)

        Ok(logits)
    }
}



#[test]
fn test_tok_emb() {
    // Given
    let config = GPTConfig {
        device: Device::Cpu,
        vocab_size: 64,
        n_embd: 32
    };
    let input = Tensor::from_vec(
        vec![1u32, 5, 42, 9],
        (1, 4),
        &config.device
    ).unwrap();
    // When
    let model = Transformer::from_config(&config).unwrap();
    let output = model.forward(&input).unwrap();

    // Then the shape should be (batch_size, sequence_length, vocab_size):
    // [
    //   token 0 logits over 64 vocab items,
    //   token 1 logits over 64 vocab items,
    //   token 2 logits over 64 vocab items,
    //   token 3 logits over 64 vocab items
    // ]
    // shape = (1, 4, 64)
    // Meaning:
    // output[batch_index][token_position][vocab_index] = logit for that token
    let shape = output.shape();
    assert_eq!(shape.dims(), &[1, 4, config.vocab_size]);
}
