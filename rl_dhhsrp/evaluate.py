from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from enviroment import utils
from agent.feature_engineering import FeatureExtractor
from agent.ddqn import DDQNAgent
from greedy.greedy import SBA
from enviroment.patient_generator import PatientGenerator

import numpy as np

#p = PatientGenerator()
#for i in range(0,500):
#    p.generate_instance("instances/" + str(i) + ".in")

state_size = 7
action_size = 2
agent = DDQNAgent(state_size, action_size)
agent.load("save/dhhsrp-ddqn-670.h5")
agent.epsilon = 0.0001


for no in range(0, 10):
    env = PatientRequest()
    env.make("enviroment/instances/test/" + str(no) + ".in", "enviroment/instances/context.in")
    sched = Schedule(env)

    feature_extractor = FeatureExtractor(env, sched)
    requests = env.get_request()

    ans_greedy = 0
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time)

                if valid == True :
                    ans_greedy = ans_greedy + 1
                    sched.accept_request(request, current_time)
    ##########################################################################################
    env = PatientRequest()
    env.make("enviroment/instances/test/" + str(no) + ".in", "enviroment/instances/context.in")
    sched = Schedule(env)

    feature_extractor = FeatureExtractor(env, sched)
    requests = env.get_request()
    sba = SBA(sched, PatientGenerator())
    ans_sba = 0
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time)
                if valid == True :
                    action = sba.act(request, current_time)
                    if (action == 0):
                      continue
                    ans_sba = ans_sba + 1
                    sched.accept_request(request, current_time)
    ##########################################################################################
    env = PatientRequest()
    env = PatientRequest()
    env.make("enviroment/instances/test/" + str(no) + ".in", "enviroment/instances/context.in")
    sched = Schedule(env)
    requests = env.get_request()
    feature_extractor = FeatureExtractor(env, sched)

    ans_RL = 0
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time)
                if valid == False :
                        continue

                state = feature_extractor.get_feature(request, current_time)
                #print(state)
                state = np.reshape(state, [1, state_size])
                action = agent.act(state)
                action = agent.act(state)
                if action == 0:
                    continue
                else:
                    sched.accept_request(request, current_time)
                    ans_RL = ans_RL + 1
    print(str(ans_greedy) + ',' + str(ans_sba) + ',' + str(ans_RL))