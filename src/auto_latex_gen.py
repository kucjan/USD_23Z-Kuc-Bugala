import pandas as pd
import numpy as np
import sys
import yaml
import os

from evaluation import evaluate_best_model


def get_exp_ids(logs_dir: str, env: str) -> list:
    folders = os.listdir(logs_dir)

    ids = [int(folder.split("_")[-1]) for folder in folders if env in folder]
    return ids


def scientific_notation(x):
    return "{:.2e}".format(x)


if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            raise ValueError(
                "You need to specify input args: algorithm name, environment name and experiment id"
            )

        algo = sys.argv[1]
        env = sys.argv[2]
        exp_ids = get_exp_ids(f"logs/{algo}", env)

        if env == "InvertedDoublePendulumBulletEnv-v0":
            used_params = [
                "gamma",
                "learning_rate",
                "noise_std",
                "policy_kwargs",
                "gradient_steps",
            ]
        elif env == "LunarLander-v2":
            used_params = [
                "learning_rate",
                "n_delta",
                "n_top",
                "delta_std",
                "policy_kwargs"
            ]
        else:
            raise Exception("Add used params to the script. Returning None.")

        results_cols = [
            "best_timestep",
            "min_reward",
            "max_reward",
            "avg_reward",
            "std_reward",
        ]

        params_table = pd.DataFrame(columns=["exp_id"] + used_params + ["avg_reward"])
        results_table = pd.DataFrame(columns=["exp_id"] + results_cols)

        for exp_id in exp_ids:
            PATH = f"logs/{algo}/{env}_{exp_id}/{env}/"
            ARGS_FILE = PATH + "args.yml"

            with open(ARGS_FILE, "r") as stream:
                args_yaml = stream.read()True

            args_lines = args_yaml.split("\n")[1:]

            for ind, line in enumerate(args_lines):
                if line == "  - - conf_file":
                    break
            # print(args_lines)
            params_file_path = args_lines[ind + 1].replace(" ", "").replace("-", "").replace("<<<<<<<<HEAD:","")
            # print(params_file_path)
            if params_file_path == "null": continue
            with open(params_file_path, "r") as params_file:
                params_dict = yaml.safe_load(params_file)[env]

            params_dict = {param: params_dict[param] for param in used_params}

            best_model_eval = evaluate_best_model(
                train_dir=f"logs/{algo}/{env}_{exp_id}"
            )

            params_dict["avg_reward"] = best_model_eval["avg_reward"]
            params_dict["exp_id"] = exp_id

            best_model_eval = {name: best_model_eval[name] for name in results_cols}
            best_model_eval["exp_id"] = exp_id

            params_table.loc[len(params_table.index)] = params_dict
            results_table.loc[len(results_table.index)] = best_model_eval

        params_table["learning_rate"] = params_table["learning_rate"].apply(
            scientific_notation
        )

        print(
            params_table.sort_values(by="avg_reward", ascending=False, ignore_index=True)
            .to_latex(
                escape=True,
                column_format="|l|" + "|".join("c" * len(params_table.columns)) + "|",
                float_format="%.2f",
            )
            .replace("\\\\", "\\\\ \hline")
        )
        print(
            results_table.sort_values(by="avg_reward", ascending=False, ignore_index=True)
            .to_latex(
                escape=True,
                column_format="|l|" + "|".join("c" * len(results_table.columns) + "|"),
                float_format="%.2f",
            )
            .replace("\\\\", "\\\\ \hline")
        )

    except ValueError as e:
        print(e)
