import numpy as np
import pandas as pd


def evaluate_best_model(train_dir: str) -> dict:
    evals_data = np.load(f"{train_dir}/evaluations.npz")

    timesteps = evals_data["timesteps"]
    results = evals_data["results"]
    ep_lengths = evals_data["ep_lengths"]

    avg_results = np.mean(results, axis=1)

    best_timestep_index = np.argmax(avg_results)
    best_timestep = timesteps[best_timestep_index]

    best_results = results[best_timestep_index]
    best_ep_lengths = ep_lengths[best_timestep_index]

    return {
        "best_timestep": best_timestep,
        "min_reward": np.min(best_results),
        "max_reward": np.max(best_results),
        "avg_reward": np.mean(best_results),
        "std_reward": np.std(best_results),
        "min_ep_length": np.min(best_ep_lengths),
        "max_ep_length": np.max(best_ep_lengths),
        "avg_ep_length": np.mean(best_ep_lengths),
        "std_ep_length": np.std(best_ep_lengths),
    }
