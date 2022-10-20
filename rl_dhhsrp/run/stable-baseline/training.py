import os

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from gym.wrappers import TimeLimit
from CustomEnvironment import DHHSRP

import matplotlib.pyplot as plt
import numpy as np
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    print(x)
    print(y)
    #y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()
    fig.savefig("log.jpg")

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True

log_dir = "/tmp/mornitor/"
os.makedirs(log_dir, exist_ok=True)
env = TimeLimit(DHHSRP('../../enviroment/instances/new_instances/uniform/150/'),   max_episode_steps=300)
# Logs will be saved in log_dir/monitor.csv
env = Monitor(env, log_dir)
check_env(env, warn=True)

# Instantiate the agent
callback = SaveOnBestTrainingRewardCallback(check_freq=5e3, log_dir=log_dir)
model = DQN("MlpPolicy", env, verbose=1,  exploration_fraction = 0.25, exploration_initial_eps = 1, seed = 0,\
            gradient_steps=10, learning_rate = 0.00001)
# Train the agent
model.learn(total_timesteps=int(5e5), log_interval = 2e4,  callback=callback)
# Save the agent
model.save("dqn_dhhcsrp")
del model  # delete trained model to demonstrate loading


model = DQN.load("dqn_dhhcsrp", env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
print(mean_reward)


# Helper from the library
plot_results(log_dir)
