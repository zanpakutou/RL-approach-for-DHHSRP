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
from run.stable_baselines.assignment_env import DHHSRP
import numpy as np

instance_type = 'C'
nb_nurse = '3'
instance_dir = '../enviroment/instances/' + nb_nurse + "_nurse/" + instance_type + '/340/'

def is_cluster(i, j):
    check = False
    check = check or (i in range(8, 19) and j in range(20, 31))
    check = check or (i in range(52, 68) and j in range(64, 80))
    check = check or (i in range(43, 64) and j in range(27, 48))
    return check

def run_stable_baselines(
    no: int,
    model_path,
    nb_nurse=nb_nurse
):
    config = Config()
    env_type = TimeLimit(DHHSRP(instance_dir, reward_type=0, nb_nurse= nb_nurse), max_episode_steps=2000)
    print(instance_dir)
    model = DQN.load(model_path, env=env_type)
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "/../../context.in")
    
    location_count = [[0 for i in range(60)] for j in range(60)]
    sched = Schedule(env)
    requests = env.get_request()
    feature_extractor = FeatureExtractor(env, sched)
    ans_rl = total_request = 0
    in_cluster = 0
    out_cluster = 0 
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                x, y = request.location
                if (is_cluster(x, y)):
                    in_cluster = in_cluster + 1
                else:
                    out_cluster = out_cluster + 1
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
                        
                        location_count[x][y] = location_count[x][y] + 1
                        sched.accept_checked_request(
                            request, min_cost_insertion, weekly_deadline=True
                        )
               

    return np.array(location_count), in_cluster, out_cluster


def location_count_func(nb_nurse: int, model_path):
    location_count = None
    in_cluster = 0
    out_cluster = 0 
    i = o = 0
    for it in range(000, 999):
        if (location_count is None):
            location_count, i, o = np.array(run_stable_baselines(it, nb_nurse = nb_nurse, model_path=model_path))
        else:
            x, i, o = np.array(run_stable_baselines(it, nb_nurse = nb_nurse, model_path=model_path))
            location_count = location_count + x

        in_cluster = in_cluster + i
        out_cluster = out_cluster + o
        print(it)
    print(in_cluster, out_cluster)    
    return location_count

fig, axes = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(7, 5)
plt.subplots_adjust(left=0.01,
                bottom=0.05,
                right=1,
                top=0.98,)

model_path="../run/stable_baselines/23_02_10/" + instance_type + "_340_3/DQN_model_best_model"
im = axes.imshow(location_count_func(nb_nurse, model_path))
circ = Circle((10, 10), 1.2, color = "coral")
axes.add_patch(circ)
circ = Circle((30, 30), 1.2, color = "coral")
axes.add_patch(circ)
circ = Circle((40, 50), 1.2, color = "coral")
axes.add_patch(circ)

fig.colorbar(im, pad=0.05)

plt.savefig("location_distribution_" + instance_type + '_' + nb_nurse + "2.jpg", pad_inches=0, dpi=1500)
#plt.savefig("test")
plt.show()