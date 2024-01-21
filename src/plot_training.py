import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_training_data(train_dir: str):
    train_data = pd.read_csv(f"{train_dir}/0.monitor.csv", skiprows=0, header=1)
    evals_data = np.load(f"{train_dir}/evaluations.npz")

    plt.figure(1)
    plt.plot(train_data.t, train_data.r)
    plt.figure(2)
    plt.plot(train_data.t, train_data.l)
    plt.figure(3)
    plt.plot(evals_data["timesteps"], np.mean(evals_data["results"], axis=1))
    plt.figure(4)
    plt.plot(evals_data["timesteps"], np.mean(evals_data["ep_lengths"], axis=1))
    plt.show()


if __name__ == "__main__":
    plot_training_data(train_dir="logs/ars/BipedalWalker-v3_3")
