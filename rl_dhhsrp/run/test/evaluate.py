import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '..')))

from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from agent.feature_engineering import FeatureExtractor
from agent.ddqn import DDQNAgent
# from utils.utils import mean, stdev
from greedy.greedy import SBA
from enviroment.patient_generator import PatientGenerator
from config.config import Config
from stable_baselines3 import A2C, PPO, DQN

from gym.wrappers import TimeLimit
from run.stable_baselines.DHHSRPEnvironment import DHHSRP
# import torch
# torch.cuda.is_available = lambda : False

import csv
# import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, default=".",
    help='Which folder to write logs and output, generate if not exist')
parser.add_argument('--instance_type', type=str, default="simplify",
    choices=['uniform', 'cluster', 'simplify'],
    help='Which folder of customer request need to be run')
parser.add_argument('--inter_arrival_rate', type=int, default=360,
    choices=[60, 150, 240, 360],
    help='Instance"s arrival rate')
parser.add_argument('--nb_scenario', type=int, default=5,
    help='Number of simulation to be make in each decision')
parser.add_argument('--obj', type=str, default='patient', 
    choices=['patient', 'visit'],
    help='patient: maximize number of patient. visit: maximize number of visit')

def run_DH_greedy(no: int):
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "context.in")
    sched = Schedule(env)
    requests = env.get_request()
    ans_greedy = total_request = 0

    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                total_request = total_request + 1
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True)
                if valid == True :
                    ans_greedy = ans_greedy + 1
                    sched.accept_checked_request(request, min_cost_insertion, weekly_deadline = True)

    print("DH schedule \t" + str(ans_greedy) + " per " + str(total_request) + " requests")
    return sched.get_metrics() + [ans_greedy, ans_greedy/total_request]

def run_CH_greedy(no: int):
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "context.in")
    
    sched = Schedule(env)
    requests = env.get_request()
    ans_greedy_cap = total_request = 0

    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                total_request = total_request + 1
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True, capacity_heur = True)
                if valid == True :
                    ans_greedy_cap = ans_greedy_cap + 1
                    sched.accept_checked_request(request, min_cost_insertion, weekly_deadline = True)
    print("CH schedule \t" + str(ans_greedy_cap) + " per " + str(total_request) + " requests")
    return sched.get_metrics() + [ans_greedy_cap, ans_greedy_cap/total_request]

def run_SBA_greedy(no : int, nb_scen=10, inter_arrival_rate=360, instance_type='uniform'):
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "context.in")
    sched = Schedule(env)
    requests = env.get_request()

    look_up_scensize = {60: 24, 150: 9, 240: 6, 360: 3}
    scen_size = look_up_scensize[inter_arrival_rate]
    sba = SBA(sched, PatientGenerator(mode=args.instance_type), nb_scen, scen_size * 5)
    ans_sba = total_request = 0
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                total_request = total_request + 1
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True)
                if valid == True :
                    action = [0]
                    if (week <= 3):
                        action = [1]
                        ans_sba = ans_sba + 1
                        sched.accept_checked_request(request, min_cost_insertion, weekly_deadline = True)
                        continue
                    else:
                        action = sba.act(request, current_time)
                    if (action[0] == 0):
                      continue
                    ans_sba = ans_sba + 1
                    sched.accept_request(request, current_time, action[1], weekly_deadline = True)
    print("SBA schedule \t" + str(ans_sba) + " per " + str(total_request) + " requests")
    return sched.get_metrics()+ [ans_sba, ans_sba/total_request]

def run_RL(no : int, model_path = "../base/save/test.h5"):
    config = Config()
    agent = DDQNAgent(config).load(model_path)
    agent.epsilon = 0
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "context.in")
    sched = Schedule(env)
    requests = env.get_request()
    feature_extractor = FeatureExtractor(env, sched)
    ans_rl = total_request = 0
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                total_request = total_request + 1
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True)
                if valid == True :
                    state = feature_extractor.get_feature(
                        request, current_time, min_cost_insertion
                    )
                    action = agent.act(state)
                    if action == 0 and week > 3:
                        continue
                    else:
                        sched.accept_checked_request(
                            request, min_cost_insertion, weekly_deadline=True
                        )
                        ans_rl = ans_rl + 1

    print("RL schedule \t" + str(ans_rl) + " per " + str(total_request) + " requests")
    return sched.get_metrics() + [ans_rl, ans_rl/total_request]

def run_stable_baselines(no: int, model_path = "../stable_baselines/0-5000000-512-216-0.00001-0.999-patient-PPO/PPO_model"):
    config = Config()
    env_type = TimeLimit(
        DHHSRP(instance_dir, reward_type=0), max_episode_steps=2000
    )
    model = PPO.load(model_path, env=env_type)
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "context.in")

    sched = Schedule(env)
    requests = env.get_request()
    feature_extractor = FeatureExtractor(env, sched)
    ans_rl = total_request = 0
    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                total_request = total_request + 1
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(request, current_time, weekly_deadline = True)
                if valid == True :
                    state = feature_extractor.get_feature(
                        request, current_time, min_cost_insertion
                    )
                    action, _states = model.predict(state)
                    
                    if action == 0 and week > 3:
                        continue
                    else:
                        sched.accept_checked_request(
                            request, min_cost_insertion, weekly_deadline=True
                        )
                        ans_rl = ans_rl + 1

    print("RL schedule \t" + str(ans_rl) + " per " + str(total_request) + " requests")
    return sched.get_metrics() + [ans_rl, ans_rl/total_request]

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    print('output folder: ', args.output_folder)
    instance_dir  = "../../enviroment/instances/" + args.instance_type + '/' + str(args.inter_arrival_rate) + "/"
    print('instance dir: ', instance_dir)

    results = []
    for no in range(950, 951):
        print(no)
        stat_DH = run_DH_greedy(no)
        stat_CH = run_CH_greedy(no)
        stat_SBA = run_SBA_greedy(no, nb_scen=args.nb_scenario, inter_arrival_rate=args.inter_arrival_rate)
        #stat_RL= run_stable_baselines(no)
 
        results.append(stat_DH + stat_CH + stat_SBA)
        print("----------------------------------------")
 
    with open(args.output_folder + '/result_' + str(args.instance_type) + '_' + str(args.inter_arrival_rate) + '_' + str(args.nb_scenario) + '.csv', 'w', newline='', encoding='utf-8') as f:
        write = csv.writer(f)
        for line in results:
            write.writerow(line)