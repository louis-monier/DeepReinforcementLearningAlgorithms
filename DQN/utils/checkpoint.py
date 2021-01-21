import torch
import os
import yaml
from datetime import datetime
import shutil


class Checkpoint:
    def __init__(self, device, config_param, prefix_path, path_ckpts, resume):
        self.device = device
        self.config_param = config_param
        self.prefix_path = prefix_path
        self.path_ckpts = path_ckpts
        self.resume = resume

    def init_ckpt(self):
        check_dict = {
            "episode_num": 0,
            "model_state_dict": None,
            "target_model_state_dict": None,
            "optimizer_state_dict": None,
            "all_episode_rewards": [],
            "best_mean_reward": [],
        }
        return check_dict

    def manage_checkpoint(self):
        if self.resume:
            path_ckpt = os.path.join(self.prefix_path, self.path_ckpts, self.resume)
            print(f"Loading checkpoint {path_ckpt}")
            path_config = os.path.join(path_ckpt, "config.yaml")
            path_ckpt_dict = os.path.join(path_ckpt, "ckpt.pt")
            checkpoint = torch.load(path_ckpt_dict, map_location=self.device)

        else:
            ckpt_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path_ckpt = os.path.join(self.prefix_path, self.path_ckpts, ckpt_name)
            os.makedirs(path_ckpt, exist_ok=True)
            path_config = os.path.join(path_ckpt, "config.yaml")
            shutil.copy2(self.config_param, path_config)
            path_ckpt_dict = os.path.join(path_ckpt, "ckpt.pt")
            checkpoint = self.init_ckpt()
            print(f"Initialized checkpoint {path_ckpt}")

        print(f"\nConfig file: {path_config}")

        with open(path_config, "r") as f:
            config = yaml.safe_load(f)

        return checkpoint, path_ckpt_dict, path_ckpt, config

    def load_weights(self, checkpoint, network, target_network, optimizer):
        if self.resume:
            # Load the network's weights
            network.load_state_dict(checkpoint["model_state_dict"])
            target_network.load_state_dict(checkpoint["target_model_state_dict"])
            print("Network's weights loaded")

            # Load the optimizer parameters
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
