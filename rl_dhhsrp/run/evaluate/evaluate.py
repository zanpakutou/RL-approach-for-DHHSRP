import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '..')))

from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
#from agent.feature_engineering import FeatureExtractor
#from agent.ddqn import DDQNAgent
from utils.utils import mean, stdev
from greedy.greedy import SBA
from enviroment.patient_generator import PatientGenerator
import csv

import numpy as np

#p = PatientGenerator()
#for i in range(0,50):
#    p.generate_instance("enviroment/test_greedy/" + str(i) + ".in")

state_size = 22
action_size = 2

instance_dir  = "../../enviroment/instances/old_instances/test/"
results = []
for no in range(0, 10):
    print(no)
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "context.in")
    sched = Schedule(env)
    requests = env.get_request()
    ans_greedy = 0
    total_request = 0
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
    sba = SBA(sched, PatientGenerator())
    ans_sba = 0
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True)
                if valid == True :
                    action = sba.act(request, current_time)
                    if (action[0] == 0):
                      continue
                    ans_sba = ans_sba + 1
                    sched.accept_request(request, current_time, action[1], weekly_deadline = True)
    stat_SBA = sched.get_metrics()
    stat_SBA = stat_SBA + [ans_sba, ans_sba/total_request]
    print(ans_sba, total_request)
    ##########################################################################################
    '''env = PatientRequest()
    env.make("enviroment/instances/test/" + str(no) + ".in", "enviroment/instances/context.in")
    sched = Schedule(env)
    requests = env.get_request()
    feature_extractor = FeatureExtractor(env, sched)

    ans_RL = 0
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True)
                if valid == False :
                        continue
                state = feature_extractor.get_feature(request, current_time)
                #print(state)
                state = np.reshape(state, [1, state_size])
                action = agent.act(state)
                if action == 0:
                    continue
                else:
                    sched.accept_request(request, current_time, weekly_deadline = True)
                    ans_RL = ans_RL + 1'''
    results.append(stat_DH)
    results.append(stat_CH)
    results.append(stat_SBA)
    #print(results)
    #print(str(ans_greedy) + ',' + str(ans_greedy_cap) + ',' + str(ans_sba))
results=[*zip(*results)]
with open('result.csv', 'w') as f:
    write = csv.writer(f)
    for line in results:
        write.writerow(line)
stats = []
for i in results:
    stats.append(mean(i))
print(stats)