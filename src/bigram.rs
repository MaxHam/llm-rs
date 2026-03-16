use candle_core::{DType, Device, Error, IndexOp, Result, Shape, Tensor};
use candle_nn::{AdamW, Embedding, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap, loss, ops};
use rand::{distr::{Distribution, weighted::WeightedIndex}, rngs::ThreadRng};

use crate::dataset::Dataset;

pub fn sample_multinomial(rng: &mut ThreadRng, prs: &Vec<f32>) -> candle_core::Result<u32> {
    let distribution = WeightedIndex::new(prs).map_err(Error::wrap)?;
    let next_token = distribution.sample(rng) as u32;

    Ok(next_token)
}

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
    vocab_size: usize,
    rng: ThreadRng,
    var_map: VarMap
}

impl Bigram {
    pub fn new(vocab_size: usize, device: &Device) -> Result<Self> {
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embeddings = var_builder
        .get((vocab_size, vocab_size), "embeddings")
        .unwrap();
        let tok_emb = Embedding::new(embeddings, vocab_size);
        let rng = rand::rng();
        Ok(Self { tok_emb, vocab_size, rng, var_map})
    }

    pub fn train(&self, dataset: &mut Dataset, num_epochs: usize, batch_size: usize) -> Result<()> {
        let mut optimizer = AdamW::new(self.var_map.all_vars(), ParamsAdamW::default())?;

        for epoch in 0..num_epochs {
            let (training_inputs, training_targets) =
                dataset.random_training_batch(self.vocab_size, batch_size)?;
            let logits = self.forward(&training_inputs)?;
            let (batch_size, time_size, channel_size) = logits.shape().dims3()?;
            let loss = loss::cross_entropy(
                &logits.reshape(Shape::from((batch_size * time_size, channel_size)))?,
                &training_targets.reshape(Shape::from((batch_size * time_size,)))?,
            )?;
            optimizer.backward_step(&loss)?;

            println!(
                "Epoch: {epoch:3} Train loss: {:8.5}",
                loss.to_scalar::<f32>()?
            );
        }

        Ok(())
    }

    pub fn generate(&mut self, mut idx: Tensor, max_new_tokens: usize) -> Result<Tensor> {
        for _ in 0..max_new_tokens {
            let logits = self.forward(&idx)?;
            let (_, seq_len, _) = logits.dims3()?;
            let last_logits = logits.i((.., seq_len - 1, ..))?;
            let probabilities = ops::softmax(&last_logits, 0)?;
            let probabilities = probabilities.squeeze(0)?;     
            let probs_vec = probabilities.to_vec1()?;
            let next_token = sample_multinomial(&mut self.rng, &probs_vec)?;
            // reshape to [1,1]
            let next_tensor = Tensor::from_slice(&[next_token], &[1, 1], &Device::Cpu)?; 
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

    #[test]
    fn test_forward_shape() -> Result<()> {
        // Given
        let device = Device::Cpu;
        let vocab_size = 5;
        let model = Bigram::new(vocab_size, &device)?;
        let idx = Tensor::from_slice(&[0u32, 1, 2], &[3], &device)?; // seq_len=3

        // When
        let logits = model.forward(&idx)?;

        // Then
        assert_eq!(logits.shape().dims(), &[3, vocab_size]);

        Ok(())
    }
}
