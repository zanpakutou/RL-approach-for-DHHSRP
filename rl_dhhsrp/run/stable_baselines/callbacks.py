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

def run_stable_baselines(no: int, instance_dir,  model, obj, epsilon = 0, cap_heur = False, nb_nurse = 6):
    env_type = TimeLimit(DHHSRP(instance_dir, reward_type=obj, nb_nurse = nb_nurse), max_episode_steps=10000)
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "/../../context_" + str(nb_nurse) + ".in")

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
                    request, current_time, weekly_deadline=True, capacity_heur = cap_heur
                )
                if valid == True:
                    state = feature_extractor.get_feature(
                        request, current_time, min_cost_insertion, capacity_heur = cap_heur
                    )
                    if (np.random.random_sample() < epsilon):
                        action = np.random.randint(2)
                    else:
                        action, _states = model.predict(state)

                    if action == 0 and week > 3:
                        continue
                    else:
                        sched.accept_checked_request(
                            request, min_cost_insertion, weekly_deadline=True, capacity_heur = cap_heur
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
    def __init__(self, check_freq: int, log_dir: str, instance_dir: str, filename : str, obj = 'patient', total_timesteps = 1e6, cap_heur = False, verbose=1, nb_nurse = 6):
        super(SaveTestCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, filename)
        self.best_test_result = -np.inf
        self.instance_dir = instance_dir
        self.total_timesteps = total_timesteps
        self.obj = obj
        self.cap_heur = cap_heur
        self.nb_nurse = nb_nurse

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:

        fraction = self.n_calls / self.total_timesteps
        epsilon = 1 + fraction * (0.05 - 1)
        if self.n_calls % self.check_freq == 0:
            test_pool = [950, 951, 952, 953, 954]
            test_res = 0
            test_epsilon = 0
            for no in test_pool:
                ans = run_stable_baselines(no, instance_dir = self.instance_dir, model=self.model, obj=self.obj, cap_heur = self.cap_heur, nb_nurse=self.nb_nurse)
                test_res  = test_res + ans
                ans_epsilon = run_stable_baselines(no, instance_dir = self.instance_dir, model=self.model, epsilon=epsilon, obj=self.obj, cap_heur =self.cap_heur,nb_nurse=self.nb_nurse)
                test_epsilon = test_epsilon + ans_epsilon
            test_res = test_res / len(test_pool)
            test_epsilon = test_epsilon/len(test_pool
            )
            if self.verbose > 0:
                print(
                    f"Test result: {test_res:.2f}"
                )
            logger = Logger(dir=self.log_dir, use=['test', 'eval'])
            logger.write_test_log(episode=self.n_calls, score = test_res)
            logger.write_eval_log(episode=self.n_calls, score = test_epsilon)
            
            # New best model, you could save the agent here
            if test_res > self.best_test_result:
                self.best_test_result = test_res
                self.model.save(self.save_path + '_' + format(test_res, '.1f'))

            return True