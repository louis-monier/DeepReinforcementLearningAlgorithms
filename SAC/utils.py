import yaml
import numpy as np
from collections import deque
from datetime import datetime
import shutil
import os


class ReplayBuffer:
    def __init__(self, replay_size):
        self.replay_buffer = deque(maxlen=int(replay_size))

    def __len__(self):
        return len(self.replay_buffer)

    def store_transition(self, transition):
        return self.replay_buffer.append(transition)

    def sample_buffer(self, batch_size):
        # draw batch_size indices from the replay_buffer
        samples_indices = np.random.choice(
            len(self.replay_buffer), batch_size, replace=False
        )

        # draw samples
        states, actions, rewards, dones, next_states = zip(
            *[self.replay_buffer[inds] for inds in samples_indices]
        )

        # formatting
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.uint8)
        next_states = np.array(next_states, dtype=np.float32)

        return states, actions, rewards, dones, next_states

def create_run_folder(config_info):
    # Create the folder runs/**time of run**
    ckpt_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path_runs = os.path.join(
        config_info["prefix_path"], config_info["path_runs"], ckpt_name
    )
    os.makedirs(path_runs, exist_ok=True)

    # Copy and paste the config param to keep track
    path_config = os.path.join(path_runs, "config.yaml")
    shutil.copy2(config_info["config_param"], path_config)

    print(f"Initialized run folder {path_runs}")

    return path_runs

def load_training_parameters(parameter_dict):
    # Extract the training parameters
    with open(parameter_dict, "r") as f:
        training_parameters = yaml.safe_load(f)
    return training_parameters

def soft_update(target_net, net, tau):
    for target_param, param in zip(target_net.parameters(),
                                   net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1.0 - tau) * target_param.data)

