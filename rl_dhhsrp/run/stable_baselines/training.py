import os

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from gym.wrappers import TimeLimit
from DHHSRPEnvironment import DHHSRP

import matplotlib.pyplot as plt
import numpy as np
import argparse

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_results(log_folder, title="Learning Curve"):
    x, y = ts2xy(load_results(log_folder), "timesteps")
    print(x)
    print(y)
    # y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()
    fig.savefig(log_folder + "/log.jpg")

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, filename : str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, filename)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance_type", type=int, default=2, help="Which folder to read input from"
    )
    parser.add_argument(
        "--output_folder", type=str, default=".",
        help="Which folder to write logs and output, generate if not exist",
    )
    parser.add_argument(
        "--timesteps", type=int, default=10000, help="Number of training timesteps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Number of sample for each NN updating",
    )
    parser.add_argument(
        "--discount_factor", type=float, default=0.99, help="Discount factor."
    )
    parser.add_argument(
        "--NN_size", type=int, default=128, help="Size of each hidden layer"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate of deep Q network"
    )
    parser.add_argument(
        "--obj", type=str, default="patient",
        help="patient: maximize number of patient. visit: maximize number of visit",
    )
    parser.add_argument(
        "--alg", type=str, default="DQN", help="Name of algorithm to use"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Print log of training process or not"
    )
    args = parser.parse_args()

    instance_folder_dict = {
        0: "../../enviroment/instances/uniform/150/",
        1: "../../enviroment/instances/uniform/240/",
        2: "../../enviroment/instances/uniform/360/",
        3: "../../enviroment/instances/simplify/240/",
    }
    instance_folder = instance_folder_dict[args.instance_type]

    seed = 0
    log_dir = args.output_folder
    os.makedirs(log_dir, exist_ok=True)

    obj_type = args.obj == "visit"
    env = TimeLimit(
        DHHSRP(instance_folder, reward_type=obj_type), max_episode_steps=2000
    )
    env = Monitor(env, log_dir)
    check_env(env, warn=True)

    # Instantiate the agent
    callback = SaveOnBestTrainingRewardCallback(check_freq=5e3, log_dir=log_dir, filename=args.alg + "_model")
    # Neural net architechture
    policy_kwargs = dict(
        net_arch=[
            dict(pi=[args.NN_size, args.NN_size], vf=[args.NN_size, args.NN_size])
        ]
    )

    # Define model
    model = None
    model_switcher = {
        "DQN": DQN(
            "MlpPolicy", env,
            verbose=args.verbose,
            learning_rate=args.lr,
            gamma=args.discount_factor, seed=seed,
            exploration_fraction=0.33,
            exploration_initial_eps=1,
            gradient_steps=10,
            batch_size=args.batch_size,
            policy_kwargs=dict(net_arch=[args.NN_size, args.NN_size]),
        ),
        "PPO": PPO(
            "MlpPolicy", env,
            verbose=args.verbose, learning_rate=args.lr,
            gamma=args.discount_factor, seed=seed,
            ent_coef=0.3,
            policy_kwargs=policy_kwargs,
        ),
        "A2C": A2C(
            "MlpPolicy", env,
            verbose=args.verbose, learning_rate=args.lr,
            gamma=args.discount_factor, seed=seed,
            policy_kwargs=policy_kwargs,
            )
    }
    model = model_switcher[args.alg]
    # Train the agent
    model.learn(args.timesteps, log_interval=2e4, callback=callback)
    # Save the agent
    model.save(args.alg + "_dhhsrp_last_model")
    # model = DQN.load("dqn_dhhcsrp", env=env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
    print(mean_reward, std_reward)
    plot_results(log_dir, title=args.alg + " Learning Curve")