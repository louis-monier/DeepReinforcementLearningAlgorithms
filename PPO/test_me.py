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
        self.returns = []

    def clear_all(self):
        self.states.clear()
        self.state_values.clear()
        self.next_value = 0.0
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.returns.clear()


def compute_gae(data, gamma, lambd):
    # Compute target value using TD(lambda) estimator and advantage with GAE(lambda)

    vpreds = data.state_values + [data.next_value]
    rewards = data.rewards
    data.dones.append(False)
    dones = data.dones

    gae = 0
    for t in reversed(range(len(data.rewards))):
        nonterminal = 1 - dones[t + 1]
        delta = rewards[t] + gamma * vpreds[t + 1] * nonterminal - vpreds[t]
        gae = delta + gamma * lambd * nonterminal * gae
        data.returns.insert(0, gae + vpreds[t])


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def compute_gae_vec(data, gamma, lambd):
    # Vectorized version
    # Compute target value using TD(lambda) estimator and advantage with GAE(lambda)

    # data.state_values : list of tensors
    # data.dones : list of bool
    # nonterminal : list of int
    # data.rewards : list of float
    # data.returns : list of tensors

    # convert to numpy array
    a = time.time()
    nonterminal = 1 - np.asarray(data.dones, dtype=np.int32)
    vpreds = np.asarray(data.state_values + [data.next_value])
    #vpreds_next = np.roll(vpreds, -1) * nonterminal
    rewards = np.asarray(data.rewards)# + [0.0])
    print("temps 1:", time.time() - a)

    # Vectorized version
    a = time.time()
    delta = rewards + gamma * vpreds[1:] * nonterminal[1:] - vpreds[:-1]
    #delta = rewards + gamma * vpreds_next  - vpreds
    print("temps 2:", time.time() - a)

    #a = time.time()
    #delta = delta[:-1]
    #print("temps 3:", time.time() - a)

    a = time.time()
    gae = discount(delta, gamma * lambd)
    print("temps 4:", time.time() - a)

    a = time.time()
    data.returns = np.array(gae + vpreds[:-1])
    print("temps 5:", time.time() - a)

data = Storage()
data.rewards = [1.0 for i in range(2000)]
data.dones = [False for i in range(2000)]
data.state_values = [np.random.randint(1,8) for i in range(2000)]
data.next_value = np.random.randint(1,8)

gamma = 0.99
lambd = 0.95

a = time.time()
compute_gae(data, gamma, lambd)
b = time.time()
print("temps boucle:", b-a)

a = time.time()
compute_gae_vec(data, gamma, lambd)
b = time.time()
print("temps vectorisé:", b-a)