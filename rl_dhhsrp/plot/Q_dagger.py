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
from sklearn import tree
from sklearn.model_selection import cross_val_score
import graphviz
import random

import numpy as np

class SuperviseDistillation():
    def __init__(self):
        self.tree = tree.DecisionTreeClassifier(max_depth=3)
        self.data = []
    
    def store(self, obs, act):
        """ not batched """
        self.data.append({'s':obs, 'a':act})
    
    def simulation(self, instance_no : int):
        nb_nurse = 3
        instance_type = 'U'
        arr_rate = 150
        instance_dir = (
            "../enviroment/instances/" 
            + str(nb_nurse) 
            + '_nurse/' 
            + instance_type 
            + '/' + str(arr_rate) 
            + '/'
        )
        xxx = "../run/stable_baselines/08_03_20/"
        yyy = instance_type + '_' + str(arr_rate) + "_" + str(nb_nurse)+ "/DQN_model_best_model"
        model_path=xxx + yyy

        env_type = TimeLimit(DHHSRP(instance_dir, reward_type=0, nb_nurse = nb_nurse), max_episode_steps=5000)
        model = DQN.load(model_path, env=env_type)
        env = PatientRequest()

        env.make(instance_dir + str(instance_no) + ".in", instance_dir + "/../../context.in")

        sched = Schedule(env)
        requests = env.get_request()

        feature_extractor = FeatureExtractor(env, sched)
        for week in range(env.nb_weeks):
            for day in range(env.day_per_week):
                for request in requests[week][day]:
                    current_time = (week, day, request.current_time)
                    (valid, min_cost_insertion) = sched.check_feasible(
                        request, current_time, weekly_deadline=True
                    )
                    if valid == True:
                        state = feature_extractor.get_feature(
                            request, current_time, min_cost_insertion
                        )
                        action, _states = model.predict(state)
                        self.store(state, action)
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
                            else:
                                print("invalid")

    def collect_data(self):
        for instance_no in range(10):
            #Do simulations
            print(instance_no)
            self.simulation(instance_no)
            
        return 0
    
    def train(self):
        X = [item['s'][0] for item in self.data]
        y = [item['a'] for item in self.data]

        self.tree.fit(X, y)
        
        return 0
    
    def plot(self):
        fig = plt.figure(figsize=(25,18))
        plt.subplots_adjust(left=0, bottom=0, right=0.6, top=0.7, wspace=0, hspace=0)
        _ = tree.plot_tree(self.tree,
        #request_info + [remaining_time, is_post_state, valid] + insert_cost + valids + future_visit + total_idle_time_avai + total_travel_time
            feature_names=['week','day', 'hour', 'remain time', 'post state', 'valid',  'cost 1', 'cost 2', 'cost 3', 'validity 1', 'validity 2', 'validity 3',
                           'future visit 1', 'future visit 2', 'future visit 3', 'idle 1', 'idle 2', 'idle 3', 'travel 1',
                           'travel 2', 'travel 3'],
            class_names=['reject', 'nurse 0', 'nurse 1', 'nurse 2'],
            filled=True,
            impurity=False # Adds color accoding to class
            );
        
        fig.savefig("decistion_tree_3.png")
        print(tree.export_text(self.tree))
    
    def evaluate(self):
        return 0

class Q_Dagger():
    def __init__(self):
        self.data = []
        self.tree = []
        self.nb_tree = 0
        self.iteration = 1000

        supervise_object = SuperviseDistillation()
        supervise_object.collect_data()
        supervise_object.train()
        self.tree.append(supervise_object.tree)

    def store(self, obs, act, loss):
        """ not batched """
        self.data.append({'s':obs, 'a':act, 'l':loss})
        
    def simulation(self, instance_no):
        nb_nurse = 3
        instance_type = 'U'
        arr_rate = 255
        instance_dir = (
            "../enviroment/instances/" 
            + str(nb_nurse) 
            + '_nurse/' 
            + instance_type 
            + '/' + str(arr_rate) 
            + '/'
        )
        xxx = "../run/stable_baselines/23_02_10/"
        yyy = instance_type + '_' + str(arr_rate) + "_" + str(nb_nurse)+ "/DQN_model_best_model"
        model_path=xxx + yyy

        env_type = TimeLimit(DHHSRP(instance_dir, reward_type=0, nb_nurse = nb_nurse), max_episode_steps=5000)
        model = DQN.load(model_path, env=env_type)
        env = PatientRequest()
        env.make(instance_dir + str(instance_no) + ".in", instance_dir + "/../../context.in")

        sched = Schedule(env)
        requests = env.get_request()

        feature_extractor = FeatureExtractor(env, sched)
        for week in range(env.nb_weeks):
            for day in range(env.day_per_week):
                for request in requests[week][day]:
                    current_time = (week, day, request.current_time)
                    (valid, min_cost_insertion) = sched.check_feasible(
                        request, current_time, weekly_deadline=True
                    )
                    if valid == True:
                        state = feature_extractor.get_feature(
                            request, current_time, min_cost_insertion
                        )
                        #TODO test predict_value function
                        #Q-dagger : get best and current action value
                        label_action, q_value = model.predict(state)
                        q_value = q_value[0]
                        state = feature_extractor.get_feature(
                            request, current_time, min_cost_insertion
                        )

                        action = self.tree[self.nb_tree].predict(state)
                        self.store(state, label_action, max(np.max(q_value) - q_value[action][0], 0.01))
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
                            else:
                                print("invalid")
        return 0
    
    def cross_validate(self):
        X = [item['s'][0] for item in self.data]
        y = [item['a'] for item in self.data]
        best_tree = self.tree[0]
        best_score = 0

        for clf in self.tree:
            scores = cross_val_score(clf, X, y, cv=6)
            avg = scores.mean()
            if avg > best_score:
                best_tree = clf
                best_score = avg
                print(avg)
        return best_tree
    
    def train(self):
        X = [item['s'][0] for item in self.data]
        y = [item['a'] for item in self.data]
        weigh = [item['l'] for item in self.data]
        s = sum(weigh)
        normalize_w = [item/s for item in weigh]
        a = [i for i in range(len(X))]
        select = np.random.choice(a, size=(len(X)//3), replace=False, p=normalize_w)
        X_train = [X[ind] for ind in select]
        y_train = [y[ind] for ind in select]
        print(len(X_train))
        self.tree.append(tree.DecisionTreeClassifier(max_depth=3))
        self.tree[-1].fit(X_train, y_train)

        return 0
    
    def run(self):
        for instance_no in range(500):
            print(instance_no)
            #Do simulations
            self.simulation(instance_no)
            self.train()
        return self.cross_validate()
            
    def plot(self):
        fig = plt.figure(figsize=(25,18))
        plt.subplots_adjust(left=0, bottom=0, right=0.6, top=0.7, wspace=0, hspace=0)
        final_tree = self.run()
        _ = tree.plot_tree(final_tree,
            feature_names=['week','day', 'hour', 'remain time', 'post state', 'valid',  'cost 0', 'cost 1', 'cost 2', 'validity 0', 'validity 1', 'validity 2',
                           'future visit 0', 'future visit 1', 'future visit 2', 'idle 0', 'idle 1', 'idle 2', 'travel 0',
                           'travel 1', 'travel 2'],
            class_names=['nurse 0', 'nurse 1', 'nurse 2'],
            filled=True,
            impurity=False # Adds color accoding to class
            );
        
        fig.savefig("decistion_tree_3.png")
        print(tree.export_text(final_tree))
    
supervise_object = Q_Dagger()
supervise_object.plot()