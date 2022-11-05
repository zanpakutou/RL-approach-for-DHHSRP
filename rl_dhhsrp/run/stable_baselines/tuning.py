import os
from stable_baselines import  DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench.monitor import Monitor

from gym.wrappers import TimeLimit
from DHHSRPEnvironment import DHHSRP
from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from agent.feature_engineering import FeatureExtractor
from utils.logger import Logger
from callbacks import SaveOnBestTrainingRewardCallback, SaveTestCallback, plot_results

import optuna
import matplotlib.pyplot as plt
import numpy as np


class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 256],
                                           layer_norm=False,
                                           feature_extraction="mlp")

def optimize_ppo(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'exploration_fraction':trial.suggest_categorical("exploration_fraction", [0.1, 0.3, 0.5]),
        'train_freq':trial.suggest_categorical("train_freq", [10,20,50,100]),
        'target_network_update_freq': trial.suggest_categorical("target_network_update_freq", [ 500, 1500, 3000]), 
        'batch_size':trial.suggest_categorical("batch_size", [ 64, 256, 1024]),
        'buffer_size':trial.suggest_categorical("buffer_size", [ 100000, 1000000])
    }

def run_stable_baselines(no: int, instance_dir, model, obj):
    env_type = TimeLimit(DHHSRP(instance_dir, reward_type=0), max_episode_steps=10000)
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "context.in")

    sched = Schedule(env)
    requests = env.get_request()
    feature_extractor = FeatureExtractor(env, sched)
    ans_rl = total_request = 0
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                total_request = total_request + 1
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(
                    request, current_time, weekly_deadline=True
                )
                if valid == True:
                    state = feature_extractor.get_feature(
                        request, current_time, min_cost_insertion
                    )
                    action, _states = model.predict(state)

                    if action == 0 and week > 3:
                        continue
                    else:
                        sched.accept_checked_request(
                            request, min_cost_insertion, weekly_deadline=True
                        )
                        ans_rl = ans_rl + reward(obj, request)

    return ans_rl

def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    model_params = optimize_ppo(trial)
    env = TimeLimit(
        DHHSRP(instance_dir="../../enviroment/instances/uniform/240/", reward_type='patient'), max_episode_steps=5000
    )

    model = DQN(CustomDQNPolicy, env, verbose=0, policy_kwargs=dict(dueling=False), **model_params)

    callback_test = SaveTestCallback(check_freq=2e4, log_dir='tuning', filename="DQN_model", instance_dir="../../enviroment/instances/uniform/240/")
    # Train the agent
    model.learn(3000000, callback=[callback_test])

    return -1 * callback_test.best_test_result


if __name__ == '__main__':
    study = optuna.create_study()
    try:
        study.optimize(optimize_agent, n_trials=100, n_jobs=6)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')