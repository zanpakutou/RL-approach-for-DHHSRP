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
from run.stable_baselines.assignment_env import DHHSRP
import numpy as np

instance_dir = '../enviroment/instances/3_nurse/U/150/'
nb_nurse = 6

def run_stable_baselines(
    no: int,
    model_path="../run/stable_baselines/23_02_10/U_150_3/DQN_model_best_model",
):
    config = Config()
    env_type = TimeLimit(DHHSRP(instance_dir, reward_type=0, nb_nurse= nb_nurse), max_episode_steps=2000)
    model = DQN.load(model_path, env=env_type)
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "/../../context.in")
    
    location_count = [[0 for i in range(60)] for j in range(60)]
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
                        _week, _day, _hour = state[0][0], state[0][1], state[0][2]
                        _dis = []
                        for n in range(6):
                            if state[0][9 + n]:
                                _dis.append(state[0][6 + n])
                        dis = sum(_dis)/len(_dis)
                        dis = dis / (_week * _day)
                        reject.append([24 * _week * _day *_hour, dis * 1357.64501988])
                        continue
                    else:
                        if (week > 3):
                            (valid, min_cost_insertion) = sched.check_feasible(
                                request, current_time, spec_nurse = action - 1, weekly_deadline=True
                            )
                            x, y = request.location
                            _dis = []
                            _week, _day, _hour = state[0][0], state[0][1], state[0][2]
                            for n in range(3):
                                if state[0][9 + n]:
                                    _dis.append(state[0][6 + n])
                            dis = sum(_dis)/len(_dis)
                            dis = dis / (_week * _day)
                            accept.append([24 * _week * _day * _hour, dis * 1357.64501988])
                        if (valid):
                            sched.accept_checked_request(
                                request, min_cost_insertion, weekly_deadline=True
                            )
                        else:
                            print("invalid")
    print(len(reject))

    return accept, reject 

location_count = None

fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.subplots_adjust(left=0.12,
                bottom=0.1,
                right=0.95,
                top=0.95,)

accept = []
reject = []

for i in range(988, 989):
    print(i)
    _accept, _reject = run_stable_baselines(i)
    accept = accept + _accept
    reject = reject + _reject

accept = np.array(accept)
reject = np.array(reject)
reject
#print(reject)
#ax.set_yticklabels([])
plt.scatter(accept[:,0], accept[:,1], c="red", marker="o", linewidth=1, label="accept")
plt.scatter(reject[:,0], reject[:,1], c="blue", marker="x", linewidth=1, label="reject")

plt.xlabel('Total required service time (hours)', fontsize=15)
plt.ylabel('Cheapest insertion cost', fontsize=15)

plt.legend(loc='upper left', borderaxespad=0., bbox_to_anchor=(0.05, 0.9), fontsize=13)

plt.savefig("time_distribution_cluster_6.jpg", pad_inches=0, dpi=1500)
plt.show()
