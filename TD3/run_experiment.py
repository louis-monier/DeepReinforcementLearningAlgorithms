import gym
import argparse

import agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO")

    # OpenAI gym environment name
    parser.add_argument(
        "--env", default="LunarLanderContinuous-v2", help="Gym environment"
    )

    # Config parameters
    parser.add_argument(
        "--config",
        type=str,
        default="cfg/config_TD3.yaml",
        help="Path to the config file",
    )

    parser.add_argument(
        "--path_runs", type=str, default="runs", help="Path to runs folder"
    )

    parser.add_argument(
        "--prefix_path", type=str, default="", help="Path prefix",
    )

    parser.add_argument("--render", action="store_true", help="Visualize training")

    args = parser.parse_args()

    config_info = {
        "config_param": args.config,
        "prefix_path": args.prefix_path,
        "path_runs": args.path_runs,
    }

    # Create environment
    env = gym.make(args.env)

    # Initialize agent
    agent = agent.TD3Agent(env, args.render, config_info)

    # Training
    agent.train()
