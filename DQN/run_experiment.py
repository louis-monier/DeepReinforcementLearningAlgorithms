import gym
import argparse

import agent, agent_boltzmann


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN")

    # OpenAI gym environment name
    parser.add_argument("--env", default="CartPole-v1", help="Gym environment")

    # Config parameters
    parser.add_argument(
        "--config",
        type=str,
        default="cfg/config_dqn.yaml",
        help="Path to the config file",
    )

    # Keep track of the training progression
    parser.add_argument(
        "--path_ckpts", type=str, default="ckpts", help="Path to checkpoints folder"
    )

    parser.add_argument(
        "--prefix_path", type=str, default="", help="Path prefix",
    )

    parser.add_argument(
        "-r", "--resume", type=str, help="Name of the checkpoint to resume"
    )

    parser.add_argument(
        "--boltzmann", action="store_true", help="Boltzmann exploration"
    )

    parser.add_argument("--render", action="store_true", help="Visualize training")

    args = parser.parse_args()

    config_info = {
        "config_param": args.config,
        "prefix_path": args.prefix_path,
        "path_ckpts": args.path_ckpts,
        "resume": args.resume,
    }

    # Create environment
    env = gym.make(args.env)

    # Initialize agent
    if args.boltzmann:
        agent = agent_boltzmann.DQNAgent(env, args.render, config_info)
    else:
        agent = agent.DQNAgent(env, args.render, config_info)

    # Launch training
    print(f"\nTraining on {env.unwrapped.spec.id}..\n")
    agent.train()

    # Visualize reward evolution
    agent.plot_reward()
