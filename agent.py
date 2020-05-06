import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

import numpy as np

import copy
from collections import namedtuple
from IPython.display import clear_output
import matplotlib.pyplot as plt

from exploration import AnnealedEpsGreedyPolicy
from buffer import ReplayBuffer
from environment import HiLoPricingEnv, ReferenceDemandEnv, CompetitionDemandEnv
from network import Critic
import utils

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Agent(object):
    def __init__(self,
                 price_grid,
                 T=30,
                 batch_size=512):
        self.batch_size = batch_size
        self.tau = 0.005
        self.gamma = 0.8
        self.T = T
        
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")

        self.network = Critic(2*T, len(price_grid)).to(self.device)
        self.network.train()
        
        self.target = copy.deepcopy(self.network).to(self.device)
        self.target.eval()

        self.optimizer = optim.AdamW(self.network.parameters(), 0.005)
        
        self.policy = AnnealedEpsGreedyPolicy()
        self.memory = ReplayBuffer(100000)
        self.env = ReferenceDemandEnv(T=T, price_grid=price_grid)

        self.price_grid = price_grid

        self.best_p = None
        self.best_performance = -1e6

    def step(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),\
                                    device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.stack(batch.reward)

        state_action_values = self.network(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0]

        expected_state_action_values = reward_batch[:, 0] + self.gamma * next_state_values

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def train(self, num_episodes=1000):
        self.return_trace = []
        self.p_trace = []
        for i_episode in range(num_episodes):
            state = self.env.reset()
            reward_trace = []
            p = []
            for t in range(self.T):
                with torch.no_grad():
                    q_values = self.network(self.to_tensor(state).to(self.device))
                action = self.policy.select_action(q_values.cpu().numpy())

                next_state, reward, done, _ = self.env.step(action)

                self.memory.push(self.to_tensor(state).to(self.device),
                                 self.to_tensor_long(action).to(self.device),
                                 self.to_tensor(next_state).to(self.device) if t != (self.T-1) else None,
                                 self.to_tensor([reward]).to(self.device))

                state = next_state

                self.step()

                reward_trace.append(reward)
                p.append(self.price_grid[action])

                self.target_update()
            
            self.return_trace.append(sum(reward_trace))

            if self.best_performance < sum(reward_trace):
                self.best_p = p
                self.best_performance = sum(reward_trace)

            self.p_trace.append(p)

            if i_episode % 50 == 30:
                clear_output(wait=True)
                print(f'Episode {i_episode} of {num_episodes} ({i_episode/num_episodes*100:.2f}%)')
                print(self.best_performance)
                print("\n")
                for i, p_ in enumerate(self.best_p):
                      print(i,"일 제안 가격 : ",p_)
                plt.figure(figsize=(16, 5))
                plt.plot(self.best_p)
                plt.show()
                utils.plot_return_trace(self.return_trace)

        fig = plt.figure(figsize=(16, 5))
        utils.plot_price_schedules(self.p_trace, 5, 1, fig.number)

    def to_tensor(self, x):
        return torch.from_numpy(np.array(x).astype(np.float32))
    
    def to_tensor_long(self, x):
        return torch.tensor([[x]], device=self.device, dtype=torch.long)

    def target_update(self):
        with torch.no_grad():
            for param, param_target in zip(self.network.parameters(), self.target.parameters()):
                param_target = (1-self.tau)*param_target + self.tau*param

    def evaluation(self):
        reward_history = []
        p_history = []
        for k in range(50):
            state = self.env.reset()
            reward_trace = []
            p = []
            for t in range(self.T):
                with torch.no_grad():
                    q_values = self.network(self.to_tensor(state).to(self.device))
                action = self.policy.best_action(q_values.cpu().numpy())

                next_state, reward, done, _ = self.env.step(action)

                state = next_state

                reward_trace.append(reward)
                p.append(self.price_grid[action])
            reward_history.append(sum(reward_trace))
            p_history.append(p)

        ind = np.argmax(reward_history)

        
        print("30일 간 수익 합계 : ", reward_history[ind])
        print("\n")
        for i, p_ in enumerate(p_history[ind]):

            print(i,"일 제안 가격 : ",p_)

        plt.figure(figsize=(16, 5))
        plt.plot(p_history[ind])
        plt.show()