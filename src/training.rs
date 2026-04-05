use candle_core::Result;
use crate::dataset::Dataset;

pub struct TrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub weight_decay: f64,
}

pub trait Training {
    fn train(&self, dataset: &mut Dataset, config: &TrainingConfig) -> Result<()>;
}
