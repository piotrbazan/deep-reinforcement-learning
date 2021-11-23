from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def initialize(self, train_mode):
        pass

    @abstractmethod
    def step(self, state, reward, next_state, done):
        pass

    @abstractmethod
    def get_action(self, state, train_mode=True):
        pass

    @abstractmethod
    def store(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass
