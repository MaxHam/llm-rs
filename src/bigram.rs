use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
/// A simple Bigram language model.
///
/// This model stores a bigram probability table of shape `[vocab_size, vocab_size]`,
/// where each row corresponds to the probability distribution of the next token
/// given the current token.
///
/// In a bigram model, the prediction of the next token depends **only on the current token**,
/// ignoring any earlier context. This is a simple form of a Markov model for sequences.
pub struct Bigram {
    tok_emb: Embedding,
}

impl Bigram {
    pub fn new(vocab_size: usize, device: &Device) -> Result<Self> {
        let tok_emb_weights = Tensor::randn(0.0, 0.02f32, (vocab_size, vocab_size), &device)?;
        let tok_emb = Embedding::new(tok_emb_weights, vocab_size);
        Ok(Self { tok_emb })
    }

    pub fn generate(&self, mut idx: Tensor, max_new_tokens: usize) -> Result<Tensor> {
        for _ in 0..max_new_tokens {
            let logits = self.forward(&idx)?;
            let (_, seq_len, _) = logits.dims3()?;
            let last_logits = logits.i((.., seq_len - 1, ..))?;
            // best token
            let next_tensor = last_logits.argmax(0)?; // [1]

            // reshape to [1,1]
            let next_tensor = next_tensor.unsqueeze(0)?; 

            // append to sequence
            idx = Tensor::cat(&[&idx, &next_tensor], 1)?;
        }
        Ok(idx)
    }
}

impl Module for Bigram {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.tok_emb.forward(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Result, Tensor};
    use candle_nn::Embedding;

    #[test]
    fn test_forward_shape() -> Result<()> {
        // Given
        let device = Device::Cpu;
        let vocab_size = 5;
        let emb_weights = Tensor::randn(0.0f32, 0.02, (vocab_size, vocab_size), &device)?;
        let tok_emb = Embedding::new(emb_weights, vocab_size);
        let model = Bigram { tok_emb };
        let idx = Tensor::from_slice(&[0u32, 1, 2], &[3], &device)?; // seq_len=3

        // When
        let logits = model.forward(&idx)?;

        // Then
        assert_eq!(logits.shape().dims(), &[3, vocab_size]);

        Ok(())
    }
}
