import yaml
import os
import shutil
from datetime import datetime


def load_training_parameters(parameter_dict):
    # Extract the training parameters
    with open(parameter_dict, "r") as f:
        training_parameters = yaml.safe_load(f)
    return training_parameters


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
