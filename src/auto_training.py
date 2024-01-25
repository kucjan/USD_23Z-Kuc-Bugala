import random
import yaml
import os


def generate_params_file(
    base_params: dict,
    params_num: int,
    algo: str,
    env: str,
    gamma: float,
    learning_rate: float,
    noise_std: float,
    policy_kwargs: dict,
    gradient_steps: int,
) -> str:
    new_params = base_params
    new_params[env]["gamma"] = gamma
    new_params[env]["learning_rate"] = learning_rate
    new_params[env]["noise_std"] = noise_std
    new_params[env]["policy_kwargs"] = f"dict(net_arch={policy_kwargs})"
    new_params[env]["gradient_steps"] = gradient_steps

    file_path = f"params/pendulum/{algo}_{params_num}.yml"
    with open(file_path, "w") as new_params_file:
        yaml.dump(new_params, new_params_file)

    return file_path


if __name__ == "__main__":
    ALGO = "ddpg"
    ENV = "InvertedDoublePendulumBulletEnv-v0"
    BASE_PARAMS_NUM = 20

    EVAL_FREQ = 1000
    EVAL_EPISODES = 100
    N_EVAL_ENVS = 1

    NUM_OF_TESTS = 8

    params_num = 21

    with open(f"params/pendulum/{ALGO}_{BASE_PARAMS_NUM}.yml", "r") as base_params_yaml:
        base_params_dict = yaml.safe_load(base_params_yaml)

    gamma_values = [0.98, 0.96, 9]
    learning_rate_values = [5e-4]
    noise_std_values = [0.05, 0.01]
    policy_kwargs_values = ["[300, 200]", "[400, 400]", "[500, 400]"]
    gradient_steps_values = [1, 2]

    selected_combinations = random.sample(
        [
            (gamma, learning_rate, noise_std, policy_kwargs, gradient_steps)
            for gamma in gamma_values
            for learning_rate in learning_rate_values
            for noise_std in noise_std_values
            for policy_kwargs in policy_kwargs_values
            for gradient_steps in gradient_steps_values
        ],
        NUM_OF_TESTS,
    )

    for train_it, combination in enumerate(selected_combinations):
        print("\n" + "-" * 20 + "\n")
        print(f"TRAINING NUM {train_it+1}")
        print("\n" + "-" * 20 + "\n")

        gamma, learning_rate, noise_std, policy_kwargs, gradient_steps = combination

        conf_file = generate_params_file(
            base_params_dict,
            params_num,
            ALGO,
            ENV,
            gamma,
            learning_rate,
            noise_std,
            policy_kwargs,
            gradient_steps,
        )

        os.system(
            f"python -m rl_zoo3.train --algo {ALGO} --env {ENV} --conf-file {conf_file} --eval-freq {EVAL_FREQ} --eval-episodes {EVAL_EPISODES} --n-eval-envs {N_EVAL_ENVS} -P"
        )

        params_num += 1
