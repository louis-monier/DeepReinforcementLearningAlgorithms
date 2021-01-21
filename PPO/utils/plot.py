import matplotlib.pyplot as plt
import os


def plot_reward(data, mean_data, env, path_runs):
    plt.plot(data, label="reward")
    plt.plot(mean_data, label="mean reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Reward evolution for {env.unwrapped.spec.id} Gym environment")
    plt.tight_layout()
    plt.legend()

    path_fig = os.path.join(path_runs, "figure.png")
    plt.savefig(path_fig)
    print(f"Figure saved to {path_fig}")

    plt.show()
