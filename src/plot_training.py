import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from constants import GRID_ALPHA


def plot_training_data(train_dir: str):
    # train_data = pd.read_csv(f"{train_dir}/0.monitor.csv", skiprows=0, header=1)
    evals_data = np.load(f"{train_dir}/evaluations.npz")

    plt.figure(1)
    plt.plot(
        evals_data["timesteps"],
        np.mean(evals_data["results"], axis=1),
        label="Avg Reward",
    )
    plt.fill_between(
        evals_data["timesteps"],
        np.max(evals_data["results"], axis=1),
        np.min(evals_data["results"], axis=1),
        color="blue",
        alpha=0.2,
    )
    plt.scatter(
        evals_data["timesteps"],
        np.max(evals_data["results"], axis=1),
        color="red",
        marker=".",
        label="Max Value",
    )
    plt.scatter(
        evals_data["timesteps"],
        np.min(evals_data["results"], axis=1),
        color="green",
        marker=".",
        label="Min Value",
    )

    plt.ylabel("Reward")
    plt.xlabel("Timestep")
    plt.title("Rewards for Evaluations")
    plt.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
    plt.legend()

    plt.figure(2)
    plt.plot(
        evals_data["timesteps"],
        np.mean(evals_data["ep_lengths"], axis=1),
        label="Avg Ep Len",
    )
    plt.fill_between(
        evals_data["timesteps"],
        np.max(evals_data["ep_lengths"], axis=1),
        np.min(evals_data["ep_lengths"], axis=1),
        color="blue",
        alpha=0.2,
    )
    plt.scatter(
        evals_data["timesteps"],
        np.max(evals_data["ep_lengths"], axis=1),
        color="red",
        marker=".",
        label="Max Value",
    )
    plt.scatter(
        evals_data["timesteps"],
        np.min(evals_data["ep_lengths"], axis=1),
        color="green",
        marker=".",
        label="Min Value",
    )


    plt.ylabel("Episode Length")
    plt.xlabel("Timestep")
    plt.title("Episode Lengths for Evaluations")
    plt.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    plot_training_data(train_dir="logs/ars/BipedalWalker-v3_4")
