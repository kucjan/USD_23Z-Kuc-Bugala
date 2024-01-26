import matplotlib.pyplot as plt
import numpy as np

from src.constants import GRID_ALPHA


def plot_eval_data(algo: str, env: str, exp_id: int, save: bool = False) -> None:
    train_dir = f"../logs/{algo}/{env}_{exp_id}"
    evals_data = np.load(f"{train_dir}/evaluations.npz")

    exp_id = train_dir.split("_")[-1]

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
    # plt.title("Rewards for Evaluations")
    plt.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
    plt.legend()

    plt.savefig(f"../output/{algo}/training_eval_reward_{exp_id}.png")

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
    # plt.title("Episode Lengths for Evaluations")
    plt.grid(axis="y", linestyle="--", alpha=GRID_ALPHA)
    plt.legend()

    plt.savefig(f"../output/{algo}/training_eval_eplen_{exp_id}.png")

    plt.show()
