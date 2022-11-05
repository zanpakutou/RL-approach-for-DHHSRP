import os
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.bench.monitor import Monitor, load_results
from stable_baselines.results_plotter import ts2xy

from gym.wrappers import TimeLimit
from DHHSRPEnvironment import DHHSRP
from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from agent.feature_engineering import FeatureExtractor
from utils.logger import Logger

import matplotlib.pyplot as plt
import numpy as np

def reward(obj: str, request: Request):
    if obj == "visit":
        return request.require_time[0] * request.require_time[1]
    elif obj == "patient":
        return 1
    return None

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

def plot_results(log_folder, title="Learning Curve"):
    x, y = ts2xy(load_results(log_folder), "timesteps")
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title)
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
                    print(
                        f"Num timesteps: {self.num_timesteps} - Best mean reward: {self.best_mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
        return True

class SaveTestCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, instance_dir: str, filename : str, obj = 'patient', verbose=1):
        super(SaveTestCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, filename)
        self.best_test_result = -np.inf
        self.instance_dir = instance_dir
        self.obj = obj

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            test_pool = [900, 901, 902, 903, 904, 905, 906]
            test_res = 0

            for no in test_pool:
                ans = run_stable_baselines(no, instance_dir = self.instance_dir, model=self.model, obj=self.obj)
                test_res  = test_res + ans
            test_res = test_res / len(test_pool)

            if self.verbose > 0:
                print(
                    f"Test result: {test_res:.2f}"
                )
            logger = Logger(dir=self.log_dir, use=['test'])
            logger.write_test_log(episode=self.n_calls, score = test_res)
            
            # New best model, you could save the agent here
            if test_res > self.best_test_result:
                self.best_test_result = test_res
                self.model.save(self.save_path + '_' + format(test_res, '.1f'))

            return True