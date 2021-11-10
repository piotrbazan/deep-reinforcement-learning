from agent import *
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
agent = Sarsa()
avg_rewards, best_avg_reward = interact(env, agent)