import torch
import numpy as np
import scipy.signal
import time

class Storage:
    def __init__(self):
        self.states = []
        self.state_values = []
        self.next_value = 0.0
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.returns = np.array([])

    def clear_all(self):
        self.states.clear()
        self.state_values.clear()
        self.next_value = 0.0
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.returns = np.array([])


def compute_gae(data, gamma, lambd):
    # Compute target value using TD(lambda) estimator and advantage with GAE(lambda)

    vpreds = data.state_values + [data.next_value]
    rewards = data.rewards
    data.dones.append(False)
    dones = data.dones
    returns = []
    
    gae = 0
    for t in reversed(range(len(data.rewards))):
        nonterminal = 1 - dones[t + 1]
        delta = rewards[t] + gamma * vpreds[t + 1] * nonterminal - vpreds[t]
        gae = delta + gamma * lambd * nonterminal * gae
        returns.insert(0, gae + vpreds[t])
    data.returns = np.asarray(returns)
    
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def compute_gae_vec(data, gamma, lambd):
    # Vectorized version
    # Compute target value using TD(lambda) estimator and advantage with GAE(lambda)

    # data.state_values : list of numpy float
    # data.dones : list of bool
    # nonterminal : list of int
    # data.rewards : list of float
    # data.returns : numpy array

    # convert to numpy array
    nonterminal = 1 - np.asarray((data.dones), dtype=np.int32)
    vpreds = np.asarray(data.state_values + [data.next_value])
    rewards = np.asarray(data.rewards)

    # Vectorized version
    delta = rewards + gamma * vpreds[1:] * nonterminal[1:] - vpreds[:-1]
    gae = discount(delta, gamma * lambd)
    data.returns = np.array(gae + vpreds[:-1])
