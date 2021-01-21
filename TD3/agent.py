import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import namedtuple
import itertools
import matplotlib.pyplot as plt

from model import Critic, Actor
from utils import *


class TD3Agent:
    def __init__(self, env, render, config_info):
        self.env = env
        self._reset_env()
        self.render = render

        # Create run folder to store parameters, figures, and tensorboard logs
        self.path_runs = create_run_folder(config_info)

        # Extract training parameters from yaml config file
        param = load_training_parameters(config_info["config_param"])
        self.train_param = param["training"]

        # Define device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device in use : {self.device}")

        # Define state and action dimension spaces
        state_dim = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        # Define models
        hidden_size = param["model"]["hidden_size"]
        self.critic = Critic(state_dim, self.num_actions, hidden_size).to(self.device)
        self.target_critic = Critic(state_dim, self.num_actions, hidden_size).to(
            self.device
        )
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.policy = Actor(
            state_dim, self.num_actions, hidden_size, self.max_action
        ).to(self.device)
        self.target_policy = Actor(
            state_dim, self.num_actions, hidden_size, self.max_action
        ).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())

        # Define loss criterion
        self.criterion = nn.MSELoss()

        # Define optimizers
        lr = float(param["optimizer"]["learning_rate"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(param["training"]["replay_max_size"])

        self.transition = namedtuple(
            "transition",
            field_names=["state", "action", "reward", "done", "next_state"],
        )

        # Useful variables
        self.batch_size = param["training"]["batch_size"]
        self.gamma = param["training"]["gamma"]
        self.tau = param["training"]["tau"]
        self.start_step = param["training"]["start_step"]
        self.max_timesteps = param["training"]["max_timesteps"]
        self.noise_policy = param["training"]["noise_policy"]
        self.noise_clip = param["training"]["noise_clip"]
        self.noise_explor = param["training"]["noise_explor"]
        self.update_freq_policy = param["training"]["update_freq_policy"]
        self.eval_freq = param["training"]["eval_freq"]
        self.num_eval_episodes = param["training"]["num_eval_episodes"]

    def _reset_env(self):
        # Reset the environment and initialize episode reward
        self.state, self.done = self.env.reset(), False
        self.episode_reward = 0.0
        self.episode_step = 0

    def train(self):
        # Main training loop
        total_timestep = 0
        all_episode_rewards = []
        all_mean_rewards = []
        update = 0

        # Create tensorboard writer
        writer = SummaryWriter(log_dir=self.path_runs, comment="-td3")

        for episode in itertools.count(1, 1):
            self._reset_env()

            while not self.done:
                # trick to improve exploration at the start of training
                if self.start_step > total_timestep:
                    action = self.env.action_space.sample()  # Sample random action
                else:
                    policy_action = self.policy.get_action(self.state, self.device)
                    add_noise_action = np.random.normal(
                        loc=0, scale=self.noise_explor, size=self.num_actions
                    )
                    noisy_action = policy_action + add_noise_action
                    action = np.clip(noisy_action, -self.max_action, self.max_action)

                # Fill the replay buffer up with transitions
                if (
                    len(self.replay_buffer) > self.batch_size
                    and total_timestep > self.start_step
                ):

                    batch = self.replay_buffer.sample_buffer(self.batch_size)

                    # Update parameters of all the networks
                    self.train_on_batch(batch, update, writer)

                if self.render:
                    self.env.render()

                # Perform one step in the environment
                next_state, reward, self.done, _ = self.env.step(action)
                total_timestep += 1
                self.episode_step += 1
                self.episode_reward += reward

                # Create a tuple for the new transition
                new_transition = self.transition(
                    self.state, action, reward, self.done, next_state
                )

                # Append transition to the replay buffer
                self.replay_buffer.store_transition(new_transition)

                self.state = next_state

            if total_timestep > self.max_timesteps:
                break

            mean_reward = np.mean(all_episode_rewards[-50:])
            all_episode_rewards.append(self.episode_reward)
            all_mean_rewards.append(mean_reward)

            print(
                "Episode n°{} ; total timestep [{}/{}] ; episode steps {} ; "
                "reward {} ; mean reward {}".format(
                    episode,
                    total_timestep,
                    self.max_timesteps,
                    self.episode_step,
                    round(self.episode_reward, 2),
                    round(mean_reward, 2),
                )
            )

            writer.add_scalar("reward", self.episode_reward, episode)
            writer.add_scalar("mean reward", mean_reward, episode)

            # Let's evaluate TD3
            if episode % self.eval_freq == 0:
                avg_eval_return = self.eval()
                writer.add_scalar("eval/reward", avg_eval_return, episode)

        ### END
        # Save networks' weights
        path_critic = os.path.join(self.path_runs, "critic.pth")
        path_actor = os.path.join(self.path_runs, "actor.pth")
        torch.save(self.critic.state_dict(), path_critic)
        torch.save(self.policy.state_dict(), path_actor)

        # Plot reward
        self.plot_reward(all_episode_rewards, all_mean_rewards)

        # Close all
        writer.close()
        self.env.close()

    def train_on_batch(self, batch_samples, update, writer):
        # Unpack batch_size of transitions randomly drawn from the replay buffer
        update += 1

        (
            state_batch,
            action_batch,
            reward_batch,
            done_int_batch,
            next_state_batch,
        ) = batch_samples

        # Transform np arrays into tensors and send them to device
        state_batch = torch.tensor(state_batch).to(self.device)
        next_state_batch = torch.tensor(next_state_batch).to(self.device)
        action_batch = torch.tensor(action_batch).to(self.device)
        reward_batch = torch.tensor(reward_batch).unsqueeze(1).to(self.device)
        done_int_batch = torch.tensor(done_int_batch).unsqueeze(1).to(self.device)

        ### Update Q
        with torch.no_grad():
            add_noise = torch.clamp(
                torch.randn_like(action_batch) * self.noise_policy,
                min=-self.noise_clip,
                max=self.noise_clip,
            )
            next_action = torch.clamp(
                self.target_policy(next_state_batch) + add_noise,
                min=-self.max_action,
                max=self.max_action,
            )

            target_q1, target_q2 = self.target_critic(next_state_batch, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward_batch + self.gamma * (1 - done_int_batch) * target_q

        # Estimated state-action values
        q1, q2 = self.critic(state_batch, action_batch)
        q1_loss = self.criterion(q1, target_q)
        q2_loss = self.criterion(q2, target_q)

        # Losses and optimizers
        self.critic_opt.zero_grad()
        q1_loss.backward()
        self.critic_opt.step()

        self.critic_opt.zero_grad()
        q2_loss.backward()
        self.critic_opt.step()

        writer.add_scalar("loss/q1", q1_loss.item(), update)
        writer.add_scalar("loss/q2", q2_loss.item(), update)

        if update % self.update_freq_policy == 0:
            action = self.policy(state_batch)
            q1 = self.critic(state_batch, action)
            policy_loss = -q1.mean()

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            writer.add_scalar("loss/policy", policy_loss.item(), update)

            # update target networks
            soft_update(self.target_critic, self.critic, self.tau)
            soft_update(self.target_policy, self.policy, self.tau)

    def eval(self):
        # Runs policy for X episodes and returns average reward

        # eval_env.seed(seed + 100)
        avg_reward = 0.0
        for _ in range(self.num_eval_episodes):
            self._reset_env()
            while not self.done:
                action = self.policy.get_action(np.array(self.state))
                nex_state, reward, self.done, _ = self.env.step(action)
                avg_reward += reward
                self.state = nex_state

        avg_reward /= self.num_eval_episodes

        print(f"Evaluation over {self.num_eval_episodes} episodes: {avg_reward:.3f}")
        return avg_reward

    def plot_reward(self, data, mean_data):
        plt.plot(data, label="reward")
        plt.plot(mean_data, label="mean reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Reward evolution for {self.env.unwrapped.spec.id} Gym environment")
        plt.tight_layout()
        plt.legend()

        path_fig = os.path.join(self.path_runs, "figure.png")
        plt.savefig(path_fig)
        print(f"Figure saved to {path_fig}")

        plt.show()
