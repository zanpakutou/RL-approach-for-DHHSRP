import os
import optuna
import gym
import numpy as np

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from gym.wrappers import TimeLimit
from DHHSRPEnvironment import DHHSRP
import torch
torch.cuda.is_available = lambda : False

def optimize_ppo(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 1024, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.97, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-4, 1e-1),
        #'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'n_epochs': int(trial.suggest_loguniform('noptepochs', 8, 24)),
        'gae_lambda': trial.suggest_uniform('lam', 0.8, 1.)
    }


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = optimize_ppo(trial)
    env = TimeLimit(
        DHHSRP(instance_dir="../../enviroment/instances/uniform/240/", reward_type=0), max_episode_steps=3000
    )
    NN_size = 256
    policy_kwargs = dict(
        net_arch=[
            dict(pi=[NN_size, NN_size//2], vf=[NN_size, NN_size//2])
        ]
    )
    model = PPO('MlpPolicy', env, verbose=0, batch_size = 256, policy_kwargs=policy_kwargs, **model_params)
    model.learn(2000000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)

    return -1 * mean_reward


if __name__ == '__main__':
    study = optuna.create_study()
    try:
        study.optimize(optimize_agent, n_trials=100, n_jobs=3)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')