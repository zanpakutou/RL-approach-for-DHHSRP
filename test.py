from patient_generator import PatientGenerator 
from enviroment import Enviroment
from schedule import Schedule
from feature_engineering import FeatureExtractor
from dqn import DQNAgent
import numpy as np
import utils

#p = PatientGenerator()
#for i in range(0,100):
#    p.generate_instance("instances/" + str(i) + ".in")

state_size = 9
action_size = 2
agent = DQNAgent(state_size, action_size)
agent.load("save/dhhsrp-dqn.h5")
agent.epsilon = 0.0001


for no in range(99, 100):
    env = Enviroment()
    env.make("instances/test/" + str(no) + ".in", "instances/context.in")
    sched = Schedule(env)

    feature_extractor = FeatureExtractor(env, sched)
    requests = env.get_request()

    ans = 0
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time)

                if valid == True :
                    ans = ans + 1
                    sched.accept_request(request, current_time)
    print(ans)
    ########################################################################################
    env = Enviroment()
    env.make("instances/test/" + str(no) + ".in", "instances/context.in")
    sched = Schedule(env)
    requests = env.get_request()
    feature_extractor = FeatureExtractor(env, sched)

    ans = 0
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time)
                if valid == False :
                        continue

                state = feature_extractor.get_feature(request, current_time)
                state = np.reshape(state, [1, state_size])
                #print(state)
                action = agent.act(state)
                if action == 0:
                    continue
                else:
                    sched.accept_request(request, current_time)
                    ans = ans + 1
    print(ans)
    print("---------------------------------------------------")
            