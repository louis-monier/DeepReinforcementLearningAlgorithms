import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class VNetwork(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(VNetwork, self).__init__()

        self.l1 = nn.Linear(state_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        out = self.l3(x)
        return out


class QNetwork(nn.Module):
    def __init__(self, state_size, num_actions, hidden_size):
        super(QNetwork, self).__init__()

        # Q1
        self.l1_1 = nn.Linear(state_size + num_actions, hidden_size)
        self.l1_2 = nn.Linear(hidden_size, hidden_size)
        self.l1_3 = nn.Linear(hidden_size, 1)

        # Q2
        self.l2_1 = nn.Linear(state_size + num_actions, hidden_size)
        self.l2_2 = nn.Linear(hidden_size, hidden_size)
        self.l2_3 = nn.Linear(hidden_size, 1)

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)

        x1 = F.relu(self.l1_1(x))
        x1 = F.relu(self.l1_2(x1))
        out1 = self.l1_3(x1)

        x2 = F.relu(self.l2_1(x))
        x2 = F.relu(self.l2_2(x2))
        out2 = self.l2_3(x2)

        return out1, out2


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, num_actions, hidden_size, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.l1 = nn.Linear(state_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, num_actions)
        self.log_std = nn.Linear(hidden_size, num_actions)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action(self, state, device):
        state_v = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state_v)
        std = torch.exp(log_std)

        normal_distrib = Normal(mean, std)
        unbounded_action = normal_distrib.rsample()

        # Enforcing Action Bounds (cf' Appendix C)
        bounded_action = torch.tanh(unbounded_action)
        bounded_action = bounded_action.detach().cpu().numpy()[0]

        return bounded_action

    def sample(self, state_v):
        mean, log_std = self.forward(state_v)
        std = torch.exp(log_std)

        normal_distrib = Normal(mean, std)

        # Reparameterization trick (mean + std * N(0,1))
        u = normal_distrib.rsample()  # unbounded action
        action = torch.tanh(u)  # change of variable : bounded action

        # Enforcing Action Bounds (cf' Appendix C)
        log_prob = normal_distrib.log_prob(u) - torch.log(1 - action**2 + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob
