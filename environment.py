import gym
from gym.spaces import Discrete, Box
from price_model import SimpleDemand, ReferenceDemand, CompetitionDemand
import numpy as np

class HiLoPricingEnv(gym.Env):
    def __init__(self, T, price_grid, model=SimpleDemand()):
        self.action_space = Discrete(len(price_grid))
        self.observation_space = Box(0, 10000, shape = (2*T, ), dtype=np.float32)
        self.T = T
        self.model = model
        self.price_grid = price_grid
        self.reset()

    def reset(self):
        self.state = np.repeat(0, 2*self.T)
        self.t = 0
        return self.state

    def step(self, action):
        next_state = np.repeat(0., len(self.state))
        next_state[0] = self.price_grid[action]
        next_state[1:self.T] = self.state[0:(self.T-1)]
        next_state[self.T+self.t] = 1
        reward = self.model.profit(self.price_grid[action])

        self.t += 1
        self.state = next_state
        return next_state, reward, self.t == (self.T-1), {}

class ReferenceDemandEnv(gym.Env):
    def __init__(self, T, price_grid, model=ReferenceDemand()):
        self.action_space = Discrete(len(price_grid))
        self.observation_space = Box(0, 10000, shape = (2*T, ), dtype=np.float32)
        self.T = T
        self.model = model
        self.price_grid = price_grid
        self.reset()

    def reset(self):
        self.state = np.repeat(0, 2*self.T)
        self.t = 0
        return self.state

    def step(self, action):
        next_state = np.repeat(0., len(self.state))
        next_state[0] = self.price_grid[action]
        next_state[1:self.T] = self.state[0:(self.T-1)]
        next_state[self.T+self.t] = 1
        if self.t == 0:
            reward = self.model.profit(next_state[0], next_state[0])
        else:
            reward = self.model.profit(next_state[0], next_state[1])
        self.t += 1
        self.state = next_state
        return next_state, reward, self.t == (self.T-1), {}


class CompetitionDemandEnv(gym.Env):
    def __init__(self, T, price_grid, model=CompetitionDemand()):
        self.action_space = Discrete(len(price_grid))
        self.observation_space = Box(0, 10000, shape = (2*T, ), dtype=np.float32)
        self.T = T
        self.model = model
        self.price_grid = price_grid
        self.reset()

    def reset(self):
        self.state = np.repeat(0, 2*self.T)
        self.t = 0
        self.com = 0.83
        return self.state

    def step(self, action):
        next_state = np.repeat(0., len(self.state))
        next_state[0] = self.price_grid[action]
        next_state[1:self.T] = self.state[0:(self.T-1)]
        next_state[self.T+self.t] = 1
        if self.t == 0:
            reward = self.model.profit(next_state[0], self.com,next_state[0])
        else:
            self.com = 0.83 #0.99*next_state[1] if self.com < next_state[1] else self.com
            reward = self.model.profit(next_state[0], self.com,next_state[1])
        self.t += 1
        self.state = next_state
        return next_state, reward, self.t == (self.T-1), {}
    
    def com_price(self, previous):
        if self.com>previous:
            pass