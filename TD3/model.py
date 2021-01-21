import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_size):
        super(Critic, self).__init__()

        # Q1
        self.l1_1 = nn.Linear(state_dim + num_actions, hidden_size)
        self.l1_2 = nn.Linear(hidden_size, hidden_size)
        self.l1_3 = nn.Linear(hidden_size, 1)

        # Q1
        self.l2_1 = nn.Linear(state_dim + num_actions, hidden_size)
        self.l2_2 = nn.Linear(hidden_size, hidden_size)
        self.l2_3 = nn.Linear(hidden_size, 1)

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)

        x1 = F.relu(self.l1_1(x))
        x1 = F.relu(self.l1_2(x1))
        q1 = self.l1_3(x1)

        x2 = F.relu(self.l2_1(x))
        x2 = F.relu(self.l2_2(x2))
        q2 = self.l2_3(x2)

        return q1, q2


class Actor(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size, max_action):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        unbounded_action = F.relu(self.linear3(x))

        # Bounded deterministic action between [-1;1]
        bounded_action = self.max_action * torch.tanh(unbounded_action)

        return bounded_action

    def get_action(self, state, device):
        state_v = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state_v)
        action = action.detach().cpu().numpy()[0]
        return action
