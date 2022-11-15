import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "..")))

from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from agent.feature_engineering import FeatureExtractor
from agent.ddqn import DDQNAgent
# from utils.utils import mean, stdev
from greedy.greedy import SBA
from enviroment.patient_generator import PatientGenerator
from config.config import Config
from stable_baselines import DQN

from gym.wrappers import TimeLimit
from run.stable_baselines.DHHSRPEnvironment import DHHSRP

import csv
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output_folder",
    type=str,
    default=".",
    help="Which folder to write logs and output, generate if not exist",
)
parser.add_argument(
    "--instance_type",
    type=str,
    default="uniform",
    choices=["uniform", "cluster", "simplify"],
    help="Which folder of customer request need to be run",
)
parser.add_argument(
    "--arr_rate",
    type=int,
    default=360,
    choices=[90, 150, 240, 360],
    help='Instance"s arrival rate',
)
parser.add_argument(
    "--nb_nurse",
    type=int,
    default=6,
    choices=[1, 6, 12],
    help="Number of nurse",
)
parser.add_argument(
    "--nb_scenario",
    type=int,
    default=5,
    help="Number of simulation to be make in each decision",
)
parser.add_argument(
    "--obj",
    type=str,
    default="patient",
    choices=["patient", "visit"],
    help="patient: maximize number of patient. visit: maximize number of visit",
)

args = parser.parse_args()
filename = args.output_folder + "/result_"\
    + str(args.instance_type)\
    + "_" + str(args.arr_rate)\
    + "_" + str(args.nb_scenario)\
    + "_" + str(args.obj)\
    + "_" + str(args.nb_nurse)

def reward(obj: str, request: Request):
    if args.obj == "visit":
        return request.require_time[0] * request.require_time[1]
    elif args.obj == "patient":
        return 1
    return None

def run_DH_greedy(no: int):
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "/../../context_" + str(args.nb_nurse) + ".in")
    sched = Schedule(env)
    requests = env.get_request()
    ans_greedy = total_request = 0
    acc_rate_dict = {'accept' : {}, 'total' : {}}

    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                total_request = total_request + 1
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(
                    request, current_time, weekly_deadline=True
                )
                if (request.require_time not in acc_rate_dict['total'].keys()):
                    acc_rate_dict['total'][request.require_time] = 0
                    acc_rate_dict['accept'][request.require_time]= 0

                acc_rate_dict['total'][request.require_time] = acc_rate_dict['total'][request.require_time] + 1
                if valid == True:
                    acc_rate_dict['accept'][request.require_time] = acc_rate_dict['accept'][request.require_time] + 1 
                    ans_greedy = ans_greedy + reward(args.obj, request)
                    sched.accept_checked_request(
                        request, min_cost_insertion, weekly_deadline=True
                    )

    print(
        "DH schedule \t" + str(ans_greedy) + " per " + str(total_request) + " requests"
    )
    return sched.get_metrics() + [ans_greedy, ans_greedy / total_request], acc_rate_dict

def run_CH_greedy(no: int):
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "/../../context_" + str(args.nb_nurse) + ".in")

    sched = Schedule(env)
    requests = env.get_request()
    ans_greedy_cap = total_request = 0
    acc_rate_dict = {'accept' : {}, 'total' : {}}

    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                total_request = total_request + 1
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(
                    request, current_time, weekly_deadline=True, capacity_heur=True
                )
                if (request.require_time not in acc_rate_dict['total'].keys()):
                    acc_rate_dict['total'][request.require_time] = 0
                    acc_rate_dict['accept'][request.require_time]= 0
                acc_rate_dict['total'][request.require_time] = acc_rate_dict['total'][request.require_time] + 1
                if valid == True:
                    acc_rate_dict['accept'][request.require_time] = acc_rate_dict['accept'][request.require_time] + 1 
                    ans_greedy_cap = ans_greedy_cap + reward(args.obj, request)
                    sched.accept_checked_request(
                        request, min_cost_insertion, weekly_deadline=True
                    )

    print(
        "CH schedule \t"
        + str(ans_greedy_cap)
        + " per "
        + str(total_request)
        + " requests"
    )
    return  sched.get_metrics() + [ans_greedy_cap, ans_greedy_cap / total_request], acc_rate_dict

def run_SBA_greedy(
    no: int,
    nb_scen=10,
    inter_arrival_rate=360,
    capacity_heur=False,
):
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "/../../context_" + str(args.nb_nurse) + ".in")
    sched = Schedule(env)
    requests = env.get_request()

    look_up_scensize = {90: 16, 150: 9, 240: 6, 360: 3}
    scen_size = look_up_scensize[inter_arrival_rate]
    sba = SBA(sched, PatientGenerator(mode=args.instance_type), capacity_heur=capacity_heur, num_scen=nb_scen, obj = args.obj, scen_size=scen_size * 5)
    ans_sba = total_request = 0
    decision_made = valid_req = decision_time = 0
    acc_rate_dict = {'accept' : {}, 'total' : {}}

    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                total_request = total_request + 1
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(
                    request,
                    current_time,
                    weekly_deadline=True,
                    capacity_heur=capacity_heur,
                )
                if (request.require_time not in acc_rate_dict['total'].keys()):
                    acc_rate_dict['total'][request.require_time] = 0
                    acc_rate_dict['accept'][request.require_time]= 0
                acc_rate_dict['total'][request.require_time] = acc_rate_dict['total'][request.require_time] + 1
                if valid == True:
                    action = [0]
                    if week <= 3:
                        action = [1]
                        ans_sba = ans_sba + reward(args.obj, request)
                        sched.accept_checked_request(
                            request,
                            min_cost_insertion,
                            weekly_deadline=True,
                            capacity_heur=capacity_heur,
                        )
                        continue
                    else:
                        st = time.time()
                        action = sba.act(request, current_time)
                        decision_time = decision_time + time.time() - st
                        decision_made = decision_made + 1
                    valid_req = valid_req + 1
                        
                    if action[0] == 0:
                        continue
                    acc_rate_dict['accept'][request.require_time] = acc_rate_dict['accept'][request.require_time] + 1 
                    ans_sba = ans_sba + reward(args.obj, request)
                    sched.accept_request(
                        request, current_time, action[1], weekly_deadline=True, capacity_heur = capacity_heur
                    )
    print(
        "SBA " + str(capacity_heur) + " schedule \t"
        + str(ans_sba) + " per " + str(total_request) + " requests"
    )
    heur_name = 'CH' if capacity_heur else 'DH'


    return sched.get_metrics() + [
        ans_sba,
        ans_sba / total_request,
        decision_time / decision_made,
        valid_req,
    ], acc_rate_dict

def run_stable_baselines(
    no: int,
    model_path="../stable_baselines/DQN_150/DQN_dhhsrp_last_model",
):
    config = Config()
    env_type = TimeLimit(DHHSRP(instance_dir, reward_type=0, nb_nurse = args.nb_nurse), max_episode_steps=2000)
    model = DQN.load(model_path, env=env_type)
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "/../../context_" + str(args.nb_nurse) + ".in")

    sched = Schedule(env)
    requests = env.get_request()
    feature_extractor = FeatureExtractor(env, sched)
    ans_rl = total_request = valid_req = 0
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
                        sched.accept_checked_request(
                            request, min_cost_insertion, weekly_deadline=True
                        )
                        ans_rl = ans_rl + reward(args.obj, request)
                    valid_req = valid_req + 1

    print("RL schedule \t" + str(ans_rl) + " per " + str(total_request) + " requests")
    return sched.get_metrics() + [ans_rl, ans_rl / total_request, valid_req]


if __name__ == "__main__":
    os.makedirs(args.output_folder, exist_ok=True)
    print('output folder: ', args.output_folder)
    instance_dir = (
        "../../enviroment/instances/"
        + args.instance_type
        + "/"
        + str(args.arr_rate)
        + "/"
    )
    print('instance dir: ', instance_dir)

    g_DH_dict = {}
    g_CH_dict = {}
    g_SDH_dict = {}
    g_SCH_dict = {}

    with open(
        filename + ".csv", "w",
        newline="", encoding="utf-8",
    ) as f:
        write = csv.writer(f)
        for no in range(950, 976):
            print(no)
            stat_DH, DH_dict = run_DH_greedy(no)
            stat_CH, CH_dict = run_CH_greedy(no)
            stat_SBA_DH, SDH_dict = run_SBA_greedy(
                no, nb_scen=args.nb_scenario, inter_arrival_rate=args.arr_rate
            )
            
            stat_SBA_CH, SCH_dict = run_SBA_greedy(
                no,nb_scen=args.nb_scenario,
                inter_arrival_rate=args.arr_rate,capacity_heur=True,
            )
            #stat_RL= run_stable_baselines(no, model_path = "/home/quy/Repos/Quy_11_11/Quy/2022_11_7/ddqn/cluster-360-1-patient-True-15000000-512-0.997/DQN_model_130.2")
            write.writerow(stat_DH + stat_CH + stat_SBA_DH + stat_SBA_CH)
            f.flush()
            print("----------------------------------------")

            g_DH_dict[no] = DH_dict
            g_CH_dict[no] = CH_dict
            g_SDH_dict[no] = SDH_dict
            g_SCH_dict[no] = SCH_dict

            np.save(filename + '_DH_.npy', g_DH_dict)
            np.save(filename + '_CH_.npy', g_CH_dict)
            np.save(filename + '_SBA_DH.npy', g_SDH_dict)
            np.save(filename + '_SBA_CH.npy', g_SCH_dict)
    
    #g_DH_dict = np.load(filename + '_DH_.npy', allow_pickle='TRUE')
    #g_CH_dict = np.load(filename + '_CH_.npy', allow_pickle='TRUE')
    #g_SDH_dict = np.load(filename + '_SBA_DH.npy', allow_pickle='TRUE')
    #_SCH_dict = np.load(filename + '_SBA_CH.npy', allow_pickle='TRUE')
