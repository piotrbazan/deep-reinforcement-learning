import numpy as np
import pytest

from agent import BaseAgent, DqnAgent
from envorinment import BaseEnvironment, BananaEnv
from experiment import Experiment
from tempfile import TemporaryDirectory

from model import DqnModel
from replay_buffer import ReplayBuffer
from strategy import LinearEpsilonGreedyStrategy


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


# @pytest.mark.skip(reason='Remove annotation to test agent learning')
def test_dqn_experiment():
    env = BananaEnv('Banana_Linux_NoVis/Banana.x86_64')
    model = DqnModel(input_dim=env.nS, output_dim=env.nA, hidden_dims=(64, 64))
    memory = ReplayBuffer(max_size=10_000)
    train_strategy = LinearEpsilonGreedyStrategy(eps_start=1., eps_min=.1, decay=.001)
    agent = DqnAgent(model, memory, train_strategy, ddqn=False, gamma=.9, batch_size=4, train_every=4, update_every=1, tau=1.)
    exp = Experiment(env, agent)
    exp.train(20)
