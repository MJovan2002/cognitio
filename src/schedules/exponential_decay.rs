use num_traits::Float;

use crate::schedules::LearningRateSchedule;

pub struct ExponentialDecay<T> {
    initial_learning_rate: T,
    decay_rate: T,
    decay_steps: u64,
    exp: fn(T, u64, u64) -> T,
    step: u64,
}

impl<T: Float + From<f64>> ExponentialDecay<T> {
    pub fn new(initial_learning_rate: T, decay_rate: T, decay_steps: u64, staircase: bool) -> Self {
        Self {
            initial_learning_rate,
            decay_rate,
            decay_steps,
            exp: match staircase {
                true => |decay_rate, decay_steps, step| decay_rate.powi((step / decay_steps) as i32),
                false => |decay_rate, decay_steps, steps| decay_rate.powf(T::from(steps as f64 / decay_steps as f64)),
            },
            step: 0,
        }
    }
}

impl<T: Float> LearningRateSchedule<T> for ExponentialDecay<T> {
    fn next(&mut self) -> T {
        let step = self.step;
        self.step += 1;
        self.initial_learning_rate * (self.exp)(self.decay_rate, self.decay_steps, step)
    }
}
