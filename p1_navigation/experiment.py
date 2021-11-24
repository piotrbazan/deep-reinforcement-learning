from pathlib import Path

import numpy as np

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
        self.train_scores = []

    def _run(self, num_episodes, max_t):
        scores = []
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
            scores.append(score)
            self.print_stats(e, scores)
            self.agent.update()
        return scores

    def train(self, episodes, max_t=1000):
        self.env.initialize(train_mode=True)
        self.agent.initialize(train_mode=True)
        scores = self._run(episodes, max_t)
        self.train_scores = np.array(scores)
        return scores

    def evaluate(self, episodes=1, max_t=1000):
        self.env.initialize(train_mode=False)
        self.agent.initialize(train_mode=False)
        return self._run(episodes, max_t)

    def store(self, path):
        model_path = Path(path) / 'model'
        score_path = Path(path) / 'scores.npy'
        self.agent.store(model_path)
        np.save(str(score_path), self.train_scores)

    def load(self, path):
        model_path = Path(path) / 'model'
        score_path = Path(path) / 'scores.npy'
        self.agent.load(model_path)
        self.train_scores = np.load(str(score_path))

    def print_stats(self, episode_num, scores):
        if episode_num % 5 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'\rEpisode: {episode_num}, avg score: {avg_score :.3f}, agent stats: {self.agent.state_dict()}', end='')
