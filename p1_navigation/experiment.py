from pathlib import Path

import numpy as np
import pandas as pd

from agent import BaseAgent
from envorinment import BaseEnvironment


class Experiment:
    """
    Class to conduct experiments. Use train/evaluate.
    Results can be stored/loaded with store/load methods.
    """

    def __init__(self, env: BaseEnvironment, agent: BaseAgent):
        self.env = env
        self.agent = agent
        self.history = pd.DataFrame()

    def _run(self, num_episodes, max_t):
        history = []
        for e in range(num_episodes):
            state = self.env.reset()
            score = 0
            for t in range(max_t):
                action = self.agent.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                score += reward
                state = next_state
                if done:
                    break
            history.append({'episode': e, 'score': score, 'agent': self.agent.state_dict()})
            self.print_stats(history)
            self.agent.episode_end()
        self.history = self.parse_history(history)

    def train(self, episodes, max_t=1000):
        self.env.initialize(train_mode=True)
        self.agent.initialize(train_mode=True)
        self._run(episodes, max_t)

    def evaluate(self, episodes=1, max_t=1000):
        self.env.initialize(train_mode=False)
        self.agent.initialize(train_mode=False)
        self._run(episodes, max_t)

    def store(self, path):
        model_path = Path(path) / 'model'
        history_path = Path(path) / 'history.parquet'
        self.agent.store(model_path)
        self.history.to_parquet(history_path)

    def load(self, path):
        model_path = Path(path) / 'model'
        history_path = Path(path) / 'history.parquet'
        self.agent.load(model_path)
        self.history = pd.read_parquet(history_path)

    def parse_history(self, history):
        df = pd.DataFrame(history)
        df['agent_avg_loss'] = df['agent'].transform(lambda d: d['avg_loss'])
        df['agent_train_epsilon'] = df['agent'].transform(lambda d: d['train_strategy']['epsilon'])
        return df

    def print_stats(self, history):
        if len(history) % 5 == 0:
            last = history[-1]
            episode = last["episode"] + 1
            score = last["score"]
            loss = last["agent"]["avg_loss"]
            epsilon = last["agent"]["train_strategy"]["epsilon"]
            print(f'\rEpisode: {episode}, score: {score:.3f}, agent_avg_loss: {loss:.3f}, epsilon: {epsilon:.3f}', end='')
