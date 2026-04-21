use candle_core::{Error, Tensor, Result};
use rand::{
    distr::{Distribution, weighted::WeightedIndex},
    rngs::ThreadRng,
};

pub fn sample_multinomial(rng: &mut ThreadRng, prs: &[f32]) -> Result<u32> {
    let distribution = WeightedIndex::new(prs).map_err(Error::wrap)?;
    let next_token = distribution.sample(rng) as u32;

    Ok(next_token)
}


pub trait Generator {
    fn generate(
        &mut self,
        idx: Tensor,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> Result<Tensor>;
}
