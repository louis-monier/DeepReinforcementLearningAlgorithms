import scipy.signal
import numpy as np
import time

def discounts(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def compute_targets_and_gae(transitions, discount=0.9, lmbda = 0.9):
    advantages = []
    returns = []
    gae = 0
    for i in reversed(range(len(transitions['reward'])-1)):
        delta = transitions['reward'][i] + discount*transitions['value'][i+1] - transitions['value'][i]
        gae = delta + discount*lmbda*gae
        advantages.insert(0, gae)
        returns.insert(0, gae + transitions['value'][i])
    return advantages, returns


def compute_targets_and_gae2(transitions, discount=0.9, lmbda = 0.9):
    a = time.time()
    values = np.asarray(transitions['value'])
    values_next = np.roll(transitions['value'], -1)
    rewards = np.asarray(transitions['reward'])
    print("temps 1:", time.time() - a)

    a = time.time()
    deltas = rewards + discount*values_next - values
    print("temps 2:", time.time() - a)

    a = time.time()
    deltas = deltas[:-1]
    print("temps 3:", time.time() - a)

    a = time.time()
    adv = discounts(deltas, gamma=(discount*lmbda))
    print("temps 4:", time.time() - a)

    a = time.time()
    returns = adv+transitions['value'][:-1]
    print("temps 5:", time.time() - a)

    return adv, returns

transitions = {
    'reward': [1.0 for i in range(10000)],
    'terminal': [False for i in range(10000)],
    'value': [np.random.randint(1,8) for i in range(10000)],
}


a = time.time()
compute_targets_and_gae(transitions)
b = time.time()
print("temps boucle:", b-a)

a = time.time()
compute_targets_and_gae2(transitions)
b = time.time()
print("temps vectorisé:", b-a)