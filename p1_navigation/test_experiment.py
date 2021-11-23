import numpy as np

from agent import BaseAgent
from envorinment import BaseEnvironment
from experiment import Experiment
from tempfile import TemporaryDirectory


class TestEnv(BaseEnvironment):

    def initialize(self, train_mode):
        self.state = 1

    def reset(self):
        return self.state

    def step(self, action):
        reward = 1 if action > 0 else -1
        self.state += 1
        next_state = self.state
        done = self.state > 1000
        return next_state, reward, done, None

    def close(self):
        pass


class TestAgent(BaseAgent):

    def initialize(self, train_mode):
        self.train_mode = train_mode

    def step(self, state, reward, next_state, done):
        pass

    def get_action(self, state, train_mode=True):
        return 1 if self.train_mode else 0

    def store(self, filename):
        pass

    def load(self, filename):
        pass


def test_experiment():
    env = TestEnv()
    agent = TestAgent()
    exp = Experiment(env, agent)
    scores = exp.train(200, max_t=5)
    assert len(scores) == 200
    assert np.allclose(scores, 5)

    with TemporaryDirectory() as dir:
        exp.store(dir)
        exp2 = Experiment(env, agent)
        exp2.load(dir)
        assert np.allclose(exp.train_scores, exp2.train_scores)

    scores = exp2.evaluate(2, 5)
    assert len(scores) == 2
    assert np.allclose(scores, -5)
