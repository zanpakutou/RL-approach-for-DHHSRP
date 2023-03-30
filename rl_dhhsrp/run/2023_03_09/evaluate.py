import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "..")))

from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from agent.feature_engineering import FeatureExtractor
from agent.ddqn import DDQNAgent
from utils.utils import MAX_VAL
from greedy.greedy import SBA
from enviroment.patient_generator import PatientGenerator
from config.config import Config
from stable_baselines import DQN

from gym.wrappers import TimeLimit
from run.stable_baselines.assignment_env import DHHSRP

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
    default="U",
    choices=["U", "C", "UC"],
    help="Which folder of customer request need to be run",
)
parser.add_argument(
    "--arr_rate",
    type=int,
    help='Instance"s arrival rate',
)
parser.add_argument(
    "--nb_nurse",
    type=int,
    default=3,
    choices=[1, 3, 12, 24],
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
parser.add_argument(
    "--instance_id",
    type=int,
    default=950,
)

parser.add_argument(
    "--switch_CH",
    type=int,
    default=0,
)

args = parser.parse_args()
filename = args.output_folder + "/result_"\
    + str(args.instance_type)\
    + "_" + str(args.arr_rate)\
    + "_" + str(args.nb_scenario)\
    + "_" + str(args.obj)\
    + "_" + str(args.nb_nurse)\
    + "_" + str(args.instance_id)\
    + "_" + str(args.switch_CH)
def reward(obj: str, request: Request):
    if obj == "visit":
        return request.require_time[0] * request.require_time[1]
    elif obj == "patient":
        return 1
    return None

def run_offline_DH(no: int):
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "/../../context.in")
    sched = Schedule(env)
    requests = env.get_request()
    ans_greedy = nb_patient = total_request = 0

    for week in range(env.nb_weeks):
        _requests = []
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                total_request = total_request + 1
                _requests.append(request)
        is_insertable = True
        current_time = (week, 0, 0)
        while is_insertable:
            min_cost = MAX_VAL
            is_insertable = False
            index = 0
            best_index = -1
            insertions = [0] * len(_requests)

            for _request in _requests:
                (valid, min_cost_insertion) = sched.check_feasible(
                    _request, current_time, weekly_deadline=True )
                
                if valid == True:
                    heuristic = min_cost_insertion[0][0] / reward(args.obj, _requests[best_index])
                    if heuristic < min_cost:
                        min_cost = heuristic
                        best_index = index
                    is_insertable = True

                insertions[index] = min_cost_insertion
                index = index + 1

            if is_insertable:
                if best_index < 0:
                    print("??")
                sched.accept_checked_request(
                    _requests[best_index],
                    insertions[best_index],
                    current_time,
                    weekly_deadline=True,
                )
                ans_greedy = ans_greedy + reward(args.obj, _requests[best_index])
                nb_patient = nb_patient + 1
                _requests.remove(_requests[best_index])

    print(
        "DH offline schedule \t" + str(ans_greedy) + " per " + str(total_request) + " requests"
    )
    return sched.get_metrics() + [ans_greedy, total_request, nb_patient]

def run_DH_greedy(no: int):
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "/../../context.in")
    sched = Schedule(env)
    requests = env.get_request()
    ans_greedy = nb_patient = total_request = 0

    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                total_request = total_request + 1
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(
                    request, current_time, weekly_deadline=True
                )
                if valid == True:
                    ans_greedy = ans_greedy + reward(args.obj, request)
                    nb_patient = nb_patient + 1
                    sched.accept_checked_request(
                        request, min_cost_insertion, weekly_deadline=True
                    )

    print(
        "DH schedule \t" + str(ans_greedy) + ", " + str(nb_patient) + " per " + str(total_request) + " requests"
    )
    return sched.get_metrics() + [ans_greedy, total_request, nb_patient]

def run_CH_greedy(no: int):
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "/../../context.in")

    sched = Schedule(env)
    requests = env.get_request()
    nb_patient = ans_greedy_cap = total_request = 0

    for week in range(env.nb_weeks):
        for day in range(env.day_per_week):
            for request in requests[week][day]:
                total_request = total_request + 1
                current_time = (week, day, request.current_time)
                (valid, min_cost_insertion) = sched.check_feasible(
                    request, current_time, weekly_deadline=True, capacity_heur=True
                )
                if valid == True:
                    ans_greedy_cap = ans_greedy_cap + reward(args.obj, request)
                    nb_patient = nb_patient + 1
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
    return  sched.get_metrics() + [ans_greedy_cap, total_request, nb_patient]

def run_SBA_greedy(
    no: int,
    nb_scen=10,
    inter_arrival_rate=360,
    capacity_heur=False,
):
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "/../../context.in")
    sched = Schedule(env)
    requests = env.get_request()

    scen_size = 5*(env.working_tw[1] - env.working_tw[0])/inter_arrival_rate
    sba = SBA(sched, PatientGenerator(mode=args.instance_type), capacity_heur=capacity_heur, num_scen=nb_scen, obj = args.obj, scen_size=int(scen_size * 5))
    nb_patient = ans_sba = total_request = 0
    decision_made = valid_req = decision_time = 0

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

                if valid == True:
                    action = [0]
                    valid_req = valid_req + 1
                    if week <= 3:
                        action = [1]
                        ans_sba = ans_sba + reward(args.obj, request)
                        nb_patient = nb_patient + 1
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
                    
                        
                    if action[0] == 0:
                        continue

                    ans_sba = ans_sba + reward(args.obj, request)
                    nb_patient = nb_patient + 1
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
        total_request,
        decision_time / decision_made,
        valid_req,
        nb_patient
    ]

def run_stable_baselines(
    no: int,
    model_path,
):
    config = Config()
    env_type = TimeLimit(DHHSRP(instance_dir, reward_type=0, nb_nurse = args.nb_nurse), max_episode_steps=5000)
    model = DQN.load(model_path, env=env_type)
    env = PatientRequest()
    env.make(instance_dir + str(no) + ".in", instance_dir + "/../../context.in")

    sched = Schedule(env)
    requests = env.get_request()
    print(env.nb_weeks)
    feature_extractor = FeatureExtractor(env, sched)
    nb_patient = ans_rl = total_request = valid_req = 0
    decision_time = decision_made = 0
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
                    st = time.time()
                    action, _states = model.predict(state)
                    decision_time = decision_time + time.time() - st
                    decision_made = decision_made + 1
                    valid_req = valid_req + 1
                    
                    if action == 0 and week > 3:
                        continue
                    else:
                        if (week > 3):
                            (valid, min_cost_insertion) = sched.check_feasible(
                                request, current_time, spec_nurse = action - 1, weekly_deadline=True
                            )

                        if (valid):
                            sched.accept_checked_request(
                                request, min_cost_insertion, weekly_deadline=True
                            )
                            ans_rl = ans_rl + reward(args.obj, request)
                            nb_patient = nb_patient + 1
                        else:
                            print("invalid")

    print("RL schedule \t" + str(ans_rl) + " per " + str(total_request) + " requests")
    return sched.get_metrics() + [ans_rl, total_request, decision_time/decision_made, valid_req, nb_patient]

if __name__ == "__main__":
    os.makedirs(args.output_folder, exist_ok=True)
    print('output folder: ', args.output_folder)
    #parent_dir='/home/tusan/projects/def-roussea5/tusan/software/RL-approach-for-DHHSRP/rl_dhhsrp'
    parent_dir = "../../"
    instance_dir = (
        parent_dir
        + "/enviroment/instances/" 
        + str(args.nb_nurse) 
        + '_nurse/' 
        + args.instance_type 
        + '/' + str(args.arr_rate) 
        + '/'
    )
    print('instance dir: ', instance_dir)

    header = []
    if (args.switch_CH == 1):
        header = ["ch_sum_travel","ch_avg_travel","ch_dev_travel","ch_sum_service","ch_avg_service","ch_dev_service","ch_obj","ch_rate","ch_visit",\
                "sba_ch_sum_travel","sba_ch_avg_travel","sba_ch_dev_travel","sba_ch_sum_service", "sba_ch_avg_service","sba_ch_dev_service","sba_ch_obj",\
                "sba_ch_rate","sba_ch_time","sba_ch_valid","sba_ch_visit","rl_sum_travel","rl_avg_travel", "rl_dev_travel","rl_sum_service","rl_avg_service",\
                "rl_dev_service","rl_obj","rl_rate","rl_time","rl_valid","rl_visit", "id"
            ]
    else:
        header = ["dho_sum_travel","dho_avg_travel","dho_dev_travel","dho_sum_service","dho_avg_service","dho_dev_service","dho_obj","dho_rate",\
        "dho_visit","dh_sum_travel","dh_avg_travel","dh_dev_travel","dh_sum_service","dh_avg_service","dh_dev_service","dh_obj","dh_rate","dh_visit",\
        "sba_dh_sum_travel","sba_dh_avg_travel","sba_dh_dev_travel","sba_dh_sum_service","sba_dh_avg_service","sba_dh_dev_sercive","sba_dh_obj",\
        "sba_dh_rate","sba_dh_time","sba_dh_valid","sba_dh_visit",]

    with open(
        filename + ".csv", "w",
        newline="", encoding="utf-8",
    ) as f:
        write = csv.writer(f)
        write.writerow(header)
        for no in range(args.instance_id, args.instance_id + 1):
            print(no)
            
            xxx = "../stable_baselines/23_02_10/"
            yyy = args.instance_type + '_' + str(args.arr_rate) + "_" + str(args.nb_nurse)+ "/DQN_model_best_model"
            
            print(args.switch_CH)
            ans = []
            if (args.switch_CH == 1):
                stat_CH = run_CH_greedy(no)
                stat_SBA_CH = run_SBA_greedy(
                    no, nb_scen=args.nb_scenario,
                    inter_arrival_rate=args.arr_rate, capacity_heur=True,
                )
                stat_RL= run_stable_baselines(no, model_path=xxx + yyy)
                ans = stat_CH + stat_SBA_CH + stat_RL
            else:
                stat_DH_off = run_offline_DH(no)
                stat_DH = run_DH_greedy(no)
                stat_SBA_DH = run_SBA_greedy(
                   no, nb_scen=args.nb_scenario, inter_arrival_rate=args.arr_rate
                )
                ans = stat_DH_off + stat_DH + stat_SBA_DH
            write.writerow(ans + [no])
            f.flush()
            print("----------------------------------------")