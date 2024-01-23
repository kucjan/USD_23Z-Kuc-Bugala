import sys
import yaml
import os

sys.path.append("..")
from src.plot_training import plot_eval_data


if __name__ == "__main__":
    try:
        do_training = False
        if len(sys.argv) == 5:
            do_training = True
        elif len(sys.argv) < 4:
            raise ValueError(
                "You need to specify input args: algorithm name, environment name and experiment id"
            )

        algo = sys.argv[1]
        env = sys.argv[2]
        exp_id = sys.argv[3]

        PATH = f"../logs/{algo}/{env}_{exp_id}/{env}/"
        ARGS_FILE = PATH + "args.yml"

        with open(ARGS_FILE, "r") as stream:
            args_yaml = stream.read()

        args_lines = args_yaml.split("\n")[1:]

        args_lines = [
            line.replace("-", "", 1).replace(" ", "")
            if ind == 0 or ind % 2 == 1
            else line.replace(" ", "").replace("_", "-")
            for ind, line in enumerate(args_lines)
        ]

        args_dict = {
            args_lines[i]: args_lines[i + 1] for i in range(0, len(args_lines) - 1, 2)
        }

        used_args_list = [
            "--algo",
            "--env",
            "--conf-file",
            "--eval-freq",
            "--eval-episodes",
            "--n-eval-envs",
            "--seed",
        ]

        args_dict = {
            arg: (f"../{args_dict[arg]}" if arg == "--conf-file" else args_dict[arg])
            for arg in used_args_list
        }

        print("TRAINING PARAMS given for experiment:")
        with open(args_dict["--conf-file"], "r") as stream:
            params_yaml = yaml.safe_load(stream)
        print(params_yaml)

        print("\n" + "-" * 20 + "\n")

        if do_training:
            print("REPRODUCING TRAINING:")
            args_string = " ".join(
                [f"{key} {value}" for key, value in args_dict.items()]
            )

            os.system(f"python -m rl_zoo3.train {args_string}")

            print("\n" + "-" * 20 + "\n")

        print("PLOTTING EVALUATION DATA:")
        plot_eval_data(train_dir=f"../logs/{algo}/{env}_{exp_id}")
        print("\n" + "-" * 20 + "\n")

        print("DISPLAYING BEST MODEL BEHAVIOUR SIMULATION:")
        os.system(
            f"python -m rl_zoo3.enjoy --algo {algo} --env {env} -f ../logs/ --exp-id {exp_id} --load-best"
        )
    except ValueError as e:
        print(e)