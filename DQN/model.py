import torch.nn as nn


# Fully connected neural network with one hidden layer
class DQN(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_actions)

        self.apply(init_weights)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        out = self.l2(x)
        return out


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.1)
        nn.init.constant_(m.bias, 0.1)
