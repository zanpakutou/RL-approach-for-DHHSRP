import sys, os

sys.path.append(os.path.abspath(os.path.join("..")))

from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from agent.feature_engineering import FeatureExtractor
from agent.ddqn import DDQNAgent
from utils.utils import mean, stdev
from greedy.greedy import SBA
from enviroment.patient_generator import PatientGenerator
from config.config import Config
from stable_baselines import DQN
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from gym.wrappers import TimeLimit
from run.stable_baselines.DHHSRPEnvironment_nurse import DHHSRP
import numpy as np

instance_type = 'cluster'
nb_nurse = '1'
instance_dir = '../enviroment/instances/' + instance_type + '/150/'

#n150_cluster_0.995_2x256
def run_stable_baselines(
    no: int,
    model_path="/home/quy/Repos/RL_DHHSRP/rl_dhhsrp/run/stable_baselines/n150_"  + instance_type + "_0.995_2x256_1/DQN_model_best_model",
    nb_nurse=nb_nurse
):
    config = Config()
    env_type = TimeLimit(DHHSRP(instance_dir, reward_type=0, nb_nurse= nb_nurse), max_episode_steps=2000)
    model = DQN.load(model_path, env=env_type)
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "/../../context_" + str(nb_nurse) + ".in")
    
    location_count = [[0 for i in range(80)] for j in range(80)]
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
                        x, y = request.location
                        location_count[x][y] = location_count[x][y] + 1
                        sched.accept_checked_request(
                            request, min_cost_insertion, weekly_deadline=True
                        )

    return location_count

def location_count_func(nb_nurse: int, model_path):
    location_count = None
    for i in range(000, 999):
        if (location_count is None):
            location_count = np.array(run_stable_baselines(i, nb_nurse = nb_nurse, model_path=model_path))
        else:
            location_count = location_count + np.array(run_stable_baselines(i, nb_nurse = nb_nurse, model_path=model_path))

    return location_count

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(14, 5)
plt.subplots_adjust(left=0.01,
                bottom=0.05,
                right=1,
                top=0.98,)


nb_nurses=['1', '6']
model_path_= [instance_type + "_0.995_2x256_1/DQN_model_best_model", instance_type + "_0.995_2x256/DQN_model_best_model",]
index=0

for ax in axes.flat:
    model_path="/home/quy/Repos/RL_DHHSRP/rl_dhhsrp/run/stable_baselines/n150_" + model_path_[index]
    im = ax.imshow(location_count_func(nb_nurses[index], model_path), vmin=0, vmax=510)
    circ = Circle((40, 40), 1.2, color = "coral")
    ax.add_patch(circ)
    index=index+1

fig.colorbar(im, ax=axes.ravel().tolist(), pad=0.05)


#plt.savefig("location_distribution_" + instance_type + '_' + nb_nurse + ".jpg", pad_inches=0, dpi=1500)
plt.savefig("test")
plt.show()