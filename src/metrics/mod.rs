pub trait Metric<State, Result> {
    fn update(&mut self, state: State);

    fn reset(&mut self);

    fn result(&self) -> Result;
}

impl<State: Clone, Result0, Result1, A: Metric<State, Result0>, B: Metric<State, Result1>> Metric<State, (Result0, Result1)> for (A, B) {
    fn update(&mut self, state: State) {
        self.0.update(state.clone());
        self.1.update(state);
    }

    fn reset(&mut self) {
        self.0.reset();
        self.1.reset();
    }

    fn result(&self) -> (Result0, Result1) {
        (self.0.result(), self.1.result())
    }
}
