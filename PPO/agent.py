import gym
import torch.nn as nn
import torch.optim
import time

from model import ActorCriticDiscrete, ActorCriticContinuous
from utils import *


class PPOAgent:
    def __init__(self, env, render, config_info):
        self.env = env
        self.render = render
        self._reset_env()

        # Extract training parameters from yaml config file
        config = load_training_parameters(config_info["config_param"])

        # Create a run folder to store parameters, figures, and tensorboard logs
        self.path_runs = create_run_folder(config_info)

        # Define device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device in use : {self.device}")

        # Define state, action dimension spaces & model
        state_dim = env.observation_space.shape[0]
        hidden_size = config["model"]["hidden_size"]
        if isinstance(env.action_space, gym.spaces.Box):
            num_action = env.action_space.shape[0]
            self.AC = ActorCriticContinuous(
                input_size=state_dim,
                num_actions=num_action,
                hidden_size=hidden_size,
                device=self.device,
                std=0.1,
            )
        else:
            num_action = env.action_space.n
            self.AC = ActorCriticDiscrete(
                input_size=state_dim, num_actions=num_action, hidden_size=hidden_size
            )

        # Define optimizer
        lr = config["optimizer"]["learning_rate"]
        self.optimizer = torch.optim.Adam(self.AC.parameters(), lr=lr)

        # Call Storage class to store collected data
        self.storage = Storage()

        # Useful training param
        self.max_timesteps = config["training"]["max_timesteps"]
        self.batch_size = config["training"]["batch_size"]
        self.update_timestep = config["training"]["update_timestep"]
        self.early_stopping = config["training"]["early_stopping"]
        self.epsilon_clip = config["training"]["epsilon_clip"]
        self.num_epochs = config["training"]["num_epochs"]
        self.log_iterval = config["training"]["log_interval"]
        self.gamma = config["training"]["gamma"]
        self.lambd = config["training"]["lambda"]

    def _reset_env(self):
        # Reset the environment and initialize episode reward
        self.state = self.env.reset()
        self.done = True  # marks if we're on first timestep of an episode
        self.curr_ep_reward = 0.0
        self.curr_ep_len = 0

    def train(self):

        # Main training function
        episode = 0
        timesteps = 0
        tstart = time.time()
        early_stop = None

        all_ep_rewards = []  # rewards cumulated of completed episodes
        all_ep_lens = []  # lengths of episodes
        all_ep_mean_reward = []
        all_ep_mean_len = []

        # Reset env before starting training
        self._reset_env()
        epsilon = 0
        while timesteps <= int(self.max_timesteps) and not early_stop:

            self.state = torch.FloatTensor(self.state).to(self.device)
            if np.random.random() < epsilon:
                action = self.env.action_space.sample()
            else:
                distrib_action, vpred = self.AC.get_action(self.state)
                sampled_action = distrib_action.sample()
                log_prob = distrib_action.log_prob(sampled_action)
                action = sampled_action.cpu().numpy()

            if timesteps > 0 and timesteps % self.update_timestep == 0:
                print("Update time!")
                # Need to pack the last value for GAE
                self.storage.next_value = vpred * (1 - self.done)

                # Compute gae of current transitions
                t_loop = time.time()
                compute_gae(self.storage, self.gamma, self.lambd)
                tf_loop = time.time()
                print("Loop : ", tf_loop - t_loop, "s")

                # Vectorized version is faster !
                #t_vec = time.time()
                #compute_gae_vec(self.storage, self.gamma, self.lambd)
                #tf_vec = time.time()
                #print("Vectorized : ", tf_vec - t_vec, "s")

                # Compare time ratio
                #print("Vectorized version is : ", np.round((tf_loop - t_loop)/(tf_vec - t_vec), 2), " times faster!")

                # Training on collected transitions with current policy
                self.train_on_batch()

                # Dump all transitions
                self.storage.clear_all()

            self.storage.states.append(self.state)
            self.storage.state_values.append(vpred)
            self.storage.actions.append(torch.tensor(action))
            self.storage.log_probs.append(log_prob)
            self.storage.dones.append(self.done)

            self.state, reward, self.done, _ = self.env.step(action)
            self.storage.rewards.append(reward)

            self.curr_ep_reward += reward
            self.curr_ep_len += 1
            if self.done:
                episode += 1
                all_ep_rewards.append(self.curr_ep_reward)
                all_ep_lens.append(self.curr_ep_len)

                mean_reward = np.mean(all_ep_rewards[-100:])
                all_ep_mean_reward.append(mean_reward)
                mean_len = np.mean(all_ep_lens[-100:])
                all_ep_mean_len.append(mean_len)
                print(
                    "Timesteps [{}/{}] ; Episode {} ; "
                    "Reward : {:.1f} ; Ep length : {} ; "
                    "Mean reward {:.1f} ; Mean ep len {}".format(
                        timesteps,
                        int(self.max_timesteps),
                        episode,
                        self.curr_ep_reward,
                        self.curr_ep_len,
                        mean_reward,
                        int(mean_len),
                    )
                )

                # Criterion to early stop training
                if mean_reward > self.early_stopping:
                    print(f"Solved in {episode} episodes!")
                    early_stop = True

                self._reset_env()

            if self.render and timesteps > 60000:
                self.env.render()

            timesteps += 1

        path_model = os.path.join(self.path_runs, "model.pth")
        torch.save(self.AC.state_dict(), path_model)
        plot_reward(all_ep_rewards, all_ep_mean_reward, self.env, self.path_runs)

        print("Time elapsed : ", time.time() - tstart, "s")
        self.env.close()

    def train_on_batch(self):

        # Convert lists to tensors
        states = torch.stack(self.storage.states).to(self.device)
        actions = torch.stack(self.storage.actions).to(self.device)
        log_probs = (
            torch.stack(self.storage.log_probs).detach().to(self.device)
        )

        # vpreds = torch.stack(self.storage.vpreds).detach().to(self.device)
        #returns = torch.stack(self.storage.returns).detach().to(self.device)

        vpreds = torch.Tensor(self.storage.state_values).detach().to(self.device)
        returns = torch.Tensor(self.storage.returns).detach().to(self.device)

        # Standardized returns function estimate
        returns = (returns - torch.mean(returns)) / (torch.std(returns) + 1e-5)

        eps_clip = self.epsilon_clip
        criterion = nn.MSELoss()

        # Create an array with batch indices
        batchs_inds = np.arange(self.update_timestep)
        np.random.shuffle(batchs_inds)  # Shuffle randomly the indices

        # Main optimization loop
        for _ in range(self.num_epochs):

            # Loop over mini-batches
            for batch_start in range(0, self.update_timestep, self.batch_size):
                current_batch_inds = batchs_inds[
                    batch_start : (batch_start + self.batch_size)
                ]
                current_batch_inds = torch.from_numpy(current_batch_inds).to(
                    self.device
                )

                b_old_states = states[current_batch_inds]
                b_old_vpreds = vpreds[current_batch_inds]
                b_old_actions = actions[current_batch_inds]
                b_old_log_probs = log_probs[current_batch_inds].unsqueeze(1)
                b_returns = returns[current_batch_inds]

                new_log_probs, new_vpreds, dist_entropy = self.AC.evaluate(
                    b_old_states, b_old_actions
                )

                entropy = dist_entropy.mean()

                # PPO ratio
                ratios = torch.exp(new_log_probs - b_old_log_probs)
                b_advantages = b_returns - new_vpreds.detach()

                surr1 = ratios * b_advantages
                surr2 = (
                    torch.clamp(ratios, 1.0 - eps_clip, 1.0 + eps_clip) * b_advantages
                )
                loss_actor = -torch.min(surr1, surr2).mean()

                loss_critic = criterion(b_old_vpreds, b_returns)

                # Final loss
                loss = loss_actor + 0.5 * loss_critic - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
