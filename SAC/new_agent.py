import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple
import itertools
import matplotlib.pyplot as plt

from models import QNetwork, PolicyNetwork
from utils import *


class NewSACAgent:
    def __init__(self, env, render, config_info):
        self.env = env
        self.render = render
        self._reset_env()

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
        num_actions = env.action_space.shape[0]

        # Define models
        hidden_size = param["model"]["hidden_size"]
        self.q_net = QNetwork(state_dim, num_actions, hidden_size).to(self.device)

        self.target_q_net = QNetwork(state_dim, num_actions, hidden_size).to(
            self.device
        )
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.policy_net = PolicyNetwork(state_dim, num_actions, hidden_size).to(
            self.device
        )

        # Define loss criterion
        self.q_criterion = nn.MSELoss()

        # Define optimizers
        lr = float(param["optimizer"]["learning_rate"])
        self.q_opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.policy_opt = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(param["training"]["replay_size"])

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
        self.alpha = param["training"]["alpha"]

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
        writer = SummaryWriter(log_dir=self.path_runs, comment="-sac")

        for episode in itertools.count(1, 1):
            self._reset_env()

            while not self.done:
                # trick to improve exploration at the start of training
                if self.start_step > total_timestep:
                    action = self.env.action_space.sample()  # Sample random action
                else:
                    action = self.policy_net.get_action(
                        self.state, self.device
                    )  # Sample action from policy

                # Fill the replay buffer up with transitions
                if len(self.replay_buffer) > self.batch_size:
                    batch = self.replay_buffer.sample_buffer(self.batch_size)

                    # Update parameters of all the networks
                    q1_loss, q2_loss, policy_loss = self.train_on_batch(batch)
                    writer.add_scalar("loss/q1", q1_loss, update)
                    writer.add_scalar("loss/q2", q2_loss, update)
                    writer.add_scalar("loss/policy", policy_loss, update)
                    update += 1

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

            mean_reward = np.mean(all_episode_rewards[-100:])
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
                    round(mean_reward, 2)
                )
            )

            writer.add_scalar("reward", self.episode_reward, episode)
            writer.add_scalar("mean reward", mean_reward, episode)

        # Save networks' weights
        path_critic = os.path.join(self.path_runs, "critic.pth")
        path_actor = os.path.join(self.path_runs, "actor.pth")
        torch.save(self.q_net.state_dict(), path_critic)
        torch.save(self.policy_net.state_dict(), path_actor)

        # Plot reward
        self.plot_reward(all_episode_rewards, all_mean_rewards)

        # Close all
        writer.close()
        self.env.close()

    def train_on_batch(self, batch_samples):
        # Unpack batch_size of transitions randomly drawn from the replay buffer
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
            next_state_action, next_state_log_pi = self.policy_net.sample(
                next_state_batch
            )
            q1_next_target, q2_next_target = self.target_q_net(
                next_state_batch, next_state_action
            )
            min_qf_next_target = (
                torch.min(q1_next_target, q2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward_batch + (1 - done_int_batch) * self.gamma * (
                min_qf_next_target
            )
        # Estimated state-action values
        q1, q2 = self.q_net(state_batch, action_batch)
        q1_loss = self.q_criterion(q1, next_q_value)
        q2_loss = self.q_criterion(q2, next_q_value)

        ### Update policy
        pi, log_pi = self.policy_net.sample(state_batch)
        q1_pi, q2_pi = self.q_net(state_batch, pi)
        # Mitigate positive bias in the policy improvement step
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_pi - min_q_pi).mean()

        # Losses and optimizers
        self.q_opt.zero_grad()
        q1_loss.backward()
        self.q_opt.step()

        self.q_opt.zero_grad()
        q2_loss.backward()
        self.q_opt.step()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        soft_update(self.target_q_net, self.q_net, self.tau)

        return q1_loss.item(), q2_loss.item(), policy_loss.item()

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
