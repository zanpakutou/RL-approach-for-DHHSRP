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

from gym.wrappers import TimeLimit
from run.stable_baselines.DHHSRPEnvironment import DHHSRP
import numpy as np

instance_dir = '../enviroment/instances/cluster/150/'
nb_nurse = 6

def run_stable_baselines(
    no: int,
    model_path="/home/quy/Repos/Quy_11_11/Quy/2022_11_7/ddqn/cluster-150-6-patient-True-15000000-512-0.997/DQN_model_518.4",
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
    accept = []
    reject = []
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
                        week, day, hour, dis = state[0], state[1], state[2], state[3]
                        reject.append((week * 5 + day) * 8 + hour, dis)
                        continue
                    else:
                        x, y = request.location
                        week, day, hour, dis = state[0], state[1], state[2], state[3]
                        accept.append((week * 5 + day) * 8 + hour, dis)
                        sched.accept_checked_request(
                            request, min_cost_insertion, weekly_deadline=True
                        )

    return accept, reject 

location_count = None
plt.subplots_adjust(left=0.05,
                bottom=0.05,
                right=0.95,
                top=0.95,)

for i in range(990, 999):
    accept, reject = 
    if (location_count is None):
        location_count = np.array(run_stable_baselines(i))
    else:
        location_count = location_count + np.array(run_stable_baselines(i))

color_map = plt.imshow(location_count, interpolation = None, origin="upper")
plt.colorbar()
plt.savefig("location_distribution_cluster_6.jpg", pad_inches=0, dpi=1500)
plt.show()