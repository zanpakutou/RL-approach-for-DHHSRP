import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '..')))

from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from agent.feature_engineering import FeatureExtractor
from agent.ddqn import DDQNAgent
from utils.utils import mean, stdev
from greedy.greedy import SBA
from enviroment.patient_generator import PatientGenerator
from config.config import Config

import csv
import numpy as np

state_size = 16
action_size = 2

instance_dir  = "../../enviroment/instances/new_instances/uniform/360/"
results = []
config = Config()

agent = DDQNAgent(config)
#agent.load("../base/save/tdhhsrp-ddqn-239.0.h5")
agent.epsilon = 0

for no in range(0, 10):
    print(no)
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "context.in")
    sched = Schedule(env)
    requests = env.get_request()
    ans_greedy = 0
    total_request = 0
    total_valid = 0
    stat_DH = []
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                total_request = total_request + 1
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True)
                if valid == True :
                    ans_greedy = ans_greedy + 1
                    sched.accept_checked_request(request, min_cost_insertion, weekly_deadline = True)
    stat_DH = sched.get_metrics()
    stat_DH  = stat_DH + [ans_greedy, ans_greedy/total_request]
    print(ans_greedy, total_request)
    ##########################################################################################
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "context.in")
    sched = Schedule(env)
    requests = env.get_request()
    ans_greedy_cap = 0
    stat_CH = []
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True, capacity_heur = True)
                if valid == True :
                    ans_greedy_cap = ans_greedy_cap + 1
                    sched.accept_checked_request(request, min_cost_insertion, weekly_deadline = True)
    stat_CH = sched.get_metrics()
    stat_CH = stat_CH + [ans_greedy_cap, ans_greedy_cap/total_request]
    print(ans_greedy_cap, total_request)
    ##########################################################################################
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "context.in")
    sched = Schedule(env)
    requests = env.get_request()
    sba = SBA(sched, PatientGenerator(), 3, 10)
    ans_sba = 0
    stat_SBA = []
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True)
                if valid == True :
                    action = [0]
                    if (week <= 3):
                        action = [1]
                        sched.accept_checked_request(request, min_cost_insertion, weekly_deadline = True)
                        continue
                    else:
                        action = sba.act(request, current_time)
                    if (action[0] == 0):
                      continue
                    ans_sba = ans_sba + 1
                    sched.accept_request(request, current_time, action[1], weekly_deadline = True)
    stat_SBA = sched.get_metrics()
    stat_SBA = stat_SBA + [ans_sba, ans_sba/total_request]
    print(ans_sba, total_request)
    ##########################################################################################
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "context.in")
    sched = Schedule(env)
    requests = env.get_request()
    feature_extractor = FeatureExtractor(env, sched)
    ans_rl = 0
    stat_RL = []
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True)
                if valid == True :
                    state = feature_extractor.get_feature(
                        request, current_time, min_cost_insertion
                    )
                    state = np.reshape(state, [1, state_size])
                    if (week <= 3):
                        action = 1
                    else:
                        action = agent.act(state)
                    
                    if (action == 0):
                      continue
                    ans_rl = ans_rl + 1
                    sched.accept_checked_request(request, min_cost_insertion, weekly_deadline = True)
    stat_RL = sched.get_metrics()
    stat_RL = stat_RL + [ans_rl, ans_rl/total_request]
    print(ans_rl, total_request)
    
    results.append(stat_DH + stat_CH + stat_SBA + stat_RL)
    print("----------------------------------------")
#results= np.array(results).T.tolist()
with open('result.csv', 'w') as f:
    write = csv.writer(f)
    for line in results:
        write.writerow(line)
stats = []
for i in results:
    stats.append(mean(i))
print(stats)