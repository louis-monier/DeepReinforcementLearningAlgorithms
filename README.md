# Implementatations of STOA Deep Reinforcement Learning algorithms

This repository contains PyTorch implementations from scratch of the state-of-the-art deep reinforcement learning algorithms:
* [Deep Q-Network (DQN)](https://arxiv.org/pdf/1312.5602.pdf)
* [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
* [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
* [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1801.01290.pdf)

## Installation
This project requires Python 3.5+, PyTorch 1.0.1+ and Gym 0.16+.

```
$ git clone https://github.com/louis-monier/DeepReinforcementLearningAlgorithms.git
$ pip3 install -r requirements.txt
```

***
## Deep Q-Network (DQN)

To run the code:
```sh
$ cd DQN
$ python3 run_experiment.py
```

For Boltzmann exploration, simply add:
```
$ python3 run_experiment.py --boltzmann
```

Training parameters can be changed in the cfg/config_dqn.yaml file.

***
## Proximal Policy Optimization (PPO)

Pytorch implementation of the paper : [Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/pdf/1707.06347.pdf)

To run the code (env with discrete action space):
```sh
$ cd PPO
$ python run_experiment.py --env CartPole-v1
```
Or (env with continuous action space):
```
$ python run_experiment.py --env LunarLanderContinuous-v2
```

You can change the training parameters in the cfg/config_ppo.yaml file.

***
## Twin Delayed DDPG (TD3)

To run the code:
```sh
$ cd TD3
$ python run_experiment.py
```

You can change the training parameters in the cfg/config_td3.yaml file.

***
## Soft Actor-Critic (SAC)

### Training

To run the modern version of the SAC algorithm: clipped double-Q trick + no the extra value function. This implementation follows the pseudo-code available at [Spinning Up (Open AI)](https://spinningup.openai.com/en/latest/algorithms/sac.html#spinup.sac_pytorch):

```sh
$ cd SAC
$ python3 run_experiment.py
```

Older version of SAC algorithm:
```
$ python3 run_experiment.py --old_agent
```

You can change the training parameters in the cfg/config_sac.yaml file.

***
References:
* Mnih, Kavukcuoglu, Silver, Graves, Antonoglou, Wierstra & Riedmiller. (2013). Playing Atari with Deep Reinforcement Learning. [PDF](https://arxiv.org/pdf/1312.5602.pdf).
