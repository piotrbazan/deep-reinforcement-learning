import numpy as np
from collections import defaultdict


def greedy_action(Q, state):
    return np.argmax(Q[state])


def epsilon_greedy_action(Q, state, eps, nA):
    if np.random.random() > eps:
        return greedy_action(Q, state)
    else:
        probs = np.ones(nA) * eps / nA
        best_a = np.argmax(Q[state])
        probs[best_a] = 1 - eps + eps / nA
        return np.random.choice(np.arange(nA), p=probs)


def random_action(nA):
    return np.random.choice(np.arange(nA))




class SarsaAgent:

    def __init__(self, nA, alpha=.1, gamma=1.):
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 1.
        self.next_action = None
        self.iter = 0

    def select_action(self, state):
        if self.next_action:
            return self.next_action
        else:
            return epsilon_greedy_action(self.Q, state, self.eps, self.nA)

    def calc_target(self, new_state):
        new_action = epsilon_greedy_action(self.Q, new_state, self.eps, self.nA)
        self.next_action = new_action
        return self.Q[new_state][new_action]

    def step(self, state, action, reward, new_state, done):
        self.iter += 1
        self.eps = max(1. / self.iter, .05)
        target = reward + self.gamma * self.calc_target(new_state)
        self.Q[state][action] *= (1 - self.alpha)
        self.Q[state][action] += self.alpha * target


class SarsaMax(SarsaAgent):

    def calc_target(self, new_state):
        """ Calculates best value in new state """
        return np.max(self.Q[new_state])


class SarsaExpected(SarsaMax):

    def calc_target(self, new_state):
        """ Calculates expected value in new state """
        probs = np.ones(self.nA) * self.eps / self.nA
        best_a = np.argmax(self.Q[new_state])
        probs[best_a] = 1 - self.eps + self.eps / self.nA
        return np.dot(self.Q[new_state], probs)
