import os
from stable_baselines.common.env_checker import check_env
from stable_baselines import A2C, DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench.monitor import Monitor

from gym.wrappers import TimeLimit
from DHHSRPEnvironment import DHHSRP
from callbacks import SaveOnBestTrainingRewardCallback, SaveTestCallback, plot_results

import numpy as np
import argparse

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 256],
                                           layer_norm=False,
                                           feature_extraction="mlp")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance_type", type=str, default='uniform',
        choices=['uniform', 'cluster', 'simplify'],
        help="Type of instances",
    )
    parser.add_argument(
        "--arr_rate", type=int, default=360,
        choices=[150, 240, 360],
        help="Type of instances",
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
        "--NN_size", type=int, default=256, help="Size of each hidden layer"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate of deep Q network"
    )
    parser.add_argument(
        "--obj", type=str, default="patient",
        choices=['patient', 'visit'],
        help="patient: maximize number of patient. visit: maximize number of visit",
    )
    parser.add_argument(
        "--alg", type=str, default="DQN", 
        choices=["DQN", "A2C", "PPO"],
        help="Name of algorithm to use"
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Print log of training process or not"
    )
    
    args = parser.parse_args()

    instance_folder = "../../enviroment/instances/" + args.instance_type + '/' + str(args.arr_rate) + '/'

    seed = 0
    log_dir = args.output_folder
    os.makedirs(log_dir, exist_ok=True)

    env = TimeLimit(
        DHHSRP(instance_folder, reward_type=args.obj, cap_heur = False), max_episode_steps=5000
    )
    env = Monitor(env, log_dir)
    check_env(env, warn=True)

    # Define model
    model = DQN(
            CustomDQNPolicy,
            env,
            verbose=args.verbose,
            learning_rate=args.lr,
            gamma=args.discount_factor, seed=seed,
            learning_starts=0,
            exploration_fraction=0.3,
            exploration_initial_eps=1,
            exploration_final_eps=0.05,
            train_freq=20,
            target_network_update_freq=2000,
            batch_size=args.batch_size,
            policy_kwargs=dict(dueling=False)
        )

    # Callbacks
    callback_train = SaveOnBestTrainingRewardCallback(check_freq=5e3, log_dir=log_dir, filename="DQN_model")
    callback_test = SaveTestCallback(check_freq=2e4, log_dir=log_dir, filename="DQN_model", instance_dir=instance_folder)
    # Train the agent
    model.learn(args.timesteps, log_interval=2e4, callback=[callback_train, callback_test])
    # Save the agent
    model.save(log_dir + '/' + args.alg + "_dhhsrp_last_model")

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=5)
    print(mean_reward, std_reward)
    plot_results(log_dir, title=args.alg + " Learning Curve")
