import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal


class ActorCriticDiscrete(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size):
        super(ActorCriticDiscrete, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(),
        )

        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.apply(init_weights)

    def forward(self):
        raise NotImplementedError

    def get_action(self, state):
        action_probs = self.actor(state)
        distrib = Categorical(action_probs)
        value = self.critic(state).detach().numpy().flatten()

        return distrib, value

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        distrib = Categorical(action_probs)

        action_log_probs = distrib.log_prob(action).unsqueeze(1)
        dist_entropy = distrib.entropy()

        value = self.critic(state)

        return action_log_probs, value, dist_entropy


class ActorCriticContinuous(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size, device, std=0.1):
        super(ActorCriticContinuous, self).__init__()
        self.device = device

        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.log_std = torch.full((num_actions,), std * std).to(device)

        self.apply(init_weights)

    def forward(self):
        raise NotImplementedError

    def get_action(self, state):
        action_mean = self.actor(state).detach().numpy()
        cov_mat = torch.diag(torch.exp(self.log_std)).to(self.device)
        distrib = MultivariateNormal(action_mean, cov_mat)

        value = self.critic(state)

        return distrib, value

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        cov_mat = torch.diag(torch.exp(self.log_std)).to(self.device)
        distrib = MultivariateNormal(action_mean, cov_mat)

        action_log_probs = distrib.log_prob(action).unsqueeze(1)
        dist_entropy = distrib.entropy()

        value = self.critic(state)

        return action_log_probs, value, dist_entropy


def init_weights(m, mean=0.0, std=0.1, bias=0.1):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=mean, std=std)
        nn.init.constant_(m.bias, bias)
