import math
import random
import numpy as np

class AnnealedEpsGreedyPolicy(object):
    def __init__(self, eps_start=0.9, eps_end=0.05, eps_decay=800):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def select_action(self, q_values):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) *\
                                     math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            return np.argmax(q_values)
        else:
            return random.randrange(len(q_values))

    def best_action(self, q_values):
        return np.argmax(q_values)