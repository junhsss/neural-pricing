import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
      super().__init__()
      layers = [
              nn.Linear(state_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, action_size)
      ]
      self.model = nn.Sequential(*layers)

    def forward(self, x):
      q_values = self.model(x)
      return q_values  