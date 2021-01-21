import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

from utils import ReplayBuffer, Checkpoint
from model import DQN


class DQNAgent:
    def __init__(self, env, render, config_info):
        self.env = env
        self._reset_env()
        self.render = render

        # Set seeds
        self.seed = 0
        env.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device in use : {self.device}")

        # Define checkpoint
        checkpoint = Checkpoint(self.device, **config_info)

        # Create / load checkpoint dict
        (
            self.ckpt,
            self.path_ckpt_dict,
            self.path_ckpt,
            config,
        ) = checkpoint.manage_checkpoint()

        # Unroll useful parameters from config dict
        self.batch_size = config["training"]["batch_size"]
        self.max_timesteps = config["training"]["max_timesteps"]
        self.replay_size = config["training"]["replay_size"]
        self.start_temp = config["training"]["start_temperature"]
        self.final_temp = config["training"]["final_temperature"]
        self.decay_temp = config["training"]["decay_temperature"]
        self.gamma = config["training"]["gamma"]
        self.early_stopping = config["training"]["early_stopping"]
        self.update_frequency = config["training"]["update_frequency"]
        self.eval_frequency = config["training"]["eval_frequency"]

        # Define state and action dimension spaces
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Define Q-network and target Q-network
        self.network = DQN(state_dim, action_dim, **config["model"]).to(self.device)
        self.target_network = DQN(state_dim, action_dim, **config["model"]).to(
            self.device
        )

        # Loss and optimizer
        self.criterion = nn.MSELoss()
        lr = config["optimizer"]["learning_rate"]
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Load network's weight if resume training
        checkpoint.load_weights(
            self.ckpt, self.network, self.target_network, self.optimizer
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.replay_size)

        self.transition = namedtuple(
            "transition",
            field_names=["state", "action", "reward", "done", "next_state"],
        )

    def _reset_env(self):
        self.state, self.done = self.env.reset(), False
        self.episode_reward = 0.0

    def play_step(self, temperature=1):
        reward_signal = None

        # Boltmann exploration
        state_v = torch.tensor(self.state, dtype=torch.float32).to(self.device)
        q_values = self.network(state_v)
        probas = Categorical(F.softmax(q_values / temperature, dim=0))
        action = probas.sample().item()

        # Perform one step in the environment
        next_state, reward, self.done, _ = self.env.step(action)

        # Create a tuple for the new transition
        new_transition = self.transition(
            self.state, action, reward, self.done, next_state
        )

        # Add transition to the replay buffer
        self.replay_buffer.store_transition(new_transition)

        self.state = next_state
        self.episode_reward += reward

        if self.render:
            self.env.render()

        if self.done:
            reward_signal = self.episode_reward
            self._reset_env()

        return reward_signal

    def train(self):

        # Initializations
        all_episode_rewards = []
        episode_timestep = 0
        best_mean_reward = None
        episode_num = 0
        temp = self.start_temp  # start epsilon to explore while filling the buffer
        writer = SummaryWriter(log_dir=self.path_ckpt, comment="-dqn")

        # Evaluate untrained policy
        evaluations = [self.eval_policy()]

        # Training loop
        for t in range(int(self.max_timesteps)):
            episode_timestep += 1

            # -> is None if episode is not terminated
            # -> is episode reward when episode is terminated
            reward_signal = self.play_step(temp)

            # when episode is terminated
            if reward_signal is not None:
                episode_reward = reward_signal

                mean_reward = np.mean(all_episode_rewards[-10:])

                print(
                    f"Timestep [{t + 1}/{int(self.max_timesteps)}] ; "
                    f"Episode num : {episode_num + 1} ; "
                    f"Episode length : {episode_timestep} ; "
                    f"Reward : {episode_reward:.2f} ; "
                    f"Mean reward {mean_reward:.2f}"
                )

                # Save episode's reward & reset counters
                all_episode_rewards.append(episode_reward)
                episode_timestep = 0
                episode_num += 1

                # Save checkpoint
                self.ckpt["episode_num"] = episode_num
                self.ckpt["all_episode_rewards"].append(episode_reward)
                self.ckpt["optimizer_state_dict"] = self.optimizer.state_dict()
                torch.save(self.ckpt, self.path_ckpt_dict)

                writer.add_scalar("episode reward", episode_reward, t)
                writer.add_scalar("mean reward", mean_reward, t)

                # Save network if performance is better than average
                if best_mean_reward is None or best_mean_reward < mean_reward:
                    self.ckpt["best_mean_reward"] = mean_reward
                    self.ckpt["model_state_dict"] = self.network.state_dict()
                    self.ckpt[
                        "target_model_state_dict"
                    ] = self.target_network.state_dict()
                    if best_mean_reward is not None:
                        print(f"Best mean reward updated : {best_mean_reward}")
                    best_mean_reward = mean_reward

                # Criterion to early stop training
                if mean_reward > self.early_stopping:
                    self.plot_reward()
                    print(f"Solved in {t + 1}  timesteps!")
                    break

            # Fill the replay buffer
            if len(self.replay_buffer) < self.replay_size:
                continue
            else:
                # Adjust exploration parameter
                temp = np.maximum(
                    self.final_temp, self.start_temp - (t / self.decay_temp)
                )
            writer.add_scalar("temperature", temp, t)

            # Get the weights of the network before update
            weights_network = self.network.state_dict()

            # when it's time perform a batch gradient descent
            if t % self.update_frequency == 0:
                # Backward and optimize
                self.optimizer.zero_grad()
                batch = self.replay_buffer.sample_buffer(self.batch_size)
                loss = self.train_on_batch(batch)
                loss.backward()
                self.optimizer.step()

            # Synchronize target network
            self.target_network.load_state_dict(weights_network)

            # Evaluate episode
            if (t + 1) % self.eval_frequency == 0:
                evaluations.append(self.eval_policy())
                np.save(self.path_ckpt, evaluations)

    def train_on_batch(self, batch_samples):
        # Unpack batch_size of transitions randomly drawn from the replay buffer
        states, actions, rewards, dones, next_states = batch_samples

        # Transform np arrays into tensors and send them to device
        states_v = torch.tensor(states).to(self.device)
        next_states_v = torch.tensor(next_states).to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        dones_bool = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Vectorized version
        q_vals = self.network(states_v)  # dim=batch_size x num_actions
        # Get the Q-values corresponding to the action
        q_vals = q_vals.gather(1, actions_v.view(-1, 1))
        q_vals = q_vals.view(1, -1)[0]

        target_next_q_vals = self.target_network(next_states_v)
        # Max action of the target Q-values
        target_max_next_q_vals, _ = torch.max(target_next_q_vals, dim=1)
        # If state is terminal
        target_max_next_q_vals[dones_bool] = 0.0
        # No update of the target during backpropagation
        target_max_next_q_vals = target_max_next_q_vals.detach()

        # Bellman approximation for target Q-values
        target_q_vals = rewards_v + self.gamma * target_max_next_q_vals

        return self.criterion(q_vals, target_q_vals)

    def eval_policy(self, eval_episodes=10):
        # Runs policy for X episodes and returns average reward
        # A fixed seed is used for the eval environment
        self.env.seed(self.seed + 100)

        avg_reward = 0.0
        temperature = 1
        for _ in range(eval_episodes):
            self._reset_env()
            reward_signal = None
            while reward_signal is None:
                reward_signal = self.play_step(temperature)
            avg_reward += reward_signal

        avg_reward /= eval_episodes

        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward

    def plot_reward(self):
        plt.plot(self.ckpt["all_episode_rewards"])
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Reward evolution for {self.env.unwrapped.spec.id} Gym environment")
        plt.tight_layout()
        path_fig = os.path.join(self.path_ckpt, "figure.png")
        plt.savefig(path_fig)
        print(f"Figure saved to {path_fig}")
        plt.show()
