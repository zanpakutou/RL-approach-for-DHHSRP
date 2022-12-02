import numpy as np
import gym
from gym import spaces
import sys, os

sys.path.append(os.path.abspath(os.path.join("..", "..")))

from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from utils.utils import set_seed
from utils.logger import Logger
from agent.feature_engineering import FeatureExtractor
from agent.ddqn import DDQNAgent
from config.config import Config

class DHHSRP(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    ACCEPT = 1
    REJECT = 0

    def __init__(self, instance_dir = "../../enviroment/instances/uniform/150/", reward_type = 'patient', is_episodic=False, cap_heur = False, nb_nurse = 6):
        super(DHHSRP, self).__init__()
        self.is_episodic = is_episodic
        nb_weeks = 145
        if (self.is_episodic):
            nb_weeks = 22

        
        self.instance_dir = instance_dir
        self.reward_type = reward_type
        self.cap_heur = cap_heur
        self.nb_nurse = nb_nurse
        self.env = PatientRequest()
        self.env.make(self.instance_dir + str(0) + ".in", self.instance_dir + "/../../context_" + str(self.nb_nurse) + ".in", nb_weeks = nb_weeks)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=2,
                                                shape=(1, 5 * self.env.nb_nurses + 6,), dtype=np.float64) #10 * self.env.nb_nurses + 6

    def find_next_request(self, request_position):
        next_position = (None, None, None)
        c_week, c_day, c_index = request_position
        for week in range(c_week, self.env.nb_weeks):
            for day in range(0, self.env.day_per_week):
                if (week == c_week and day < c_day):
                    continue;
                for index in range(len(self.requests[week][day])):
                    if (week == c_week and day == c_day and index <= c_index):
                        continue;
                    request = self.requests[week][day][index]
                    current_time = (week, day, request.current_time)
                    new_request_position = ( week, day, index )

                    return (current_time, new_request_position, request)

        return (None, None,None)

    def reset(self):
        instance_no = np.random.randint(900)
        self.env = PatientRequest()
        nb_weeks = 145
        if (self.is_episodic):
            nb_weeks = 22
            
        self.env.make(self.instance_dir + str(instance_no) + ".in", self.instance_dir + "/../../context_" + str(self.nb_nurse) + ".in", nb_weeks = nb_weeks)
        self.env.scheduling_horizon = 150
        self.sched = Schedule(self.env)
        self.requests = self.env.get_request()
        self.feature_extractor = FeatureExtractor(self.env, self.sched)
        self.is_post_state = True
        self.current_time =  (0, 0, 0)
        self.request_position = (0, 0, -1)
        initial_state = None

        for week in range(0, 4):
            for day in range(self.env.day_per_week):
                for request in self.requests[week][day]:
                    current_time = (week, day, request.current_time)
                    (valid, min_cost_insertion) = self.sched.check_feasible(request, current_time, weekly_deadline = True)
                    if valid == True :
                        self.sched.accept_checked_request(request, min_cost_insertion, weekly_deadline = True)
                        initial_state = self.feature_extractor.get_feature(request, current_time, min_cost_insertion, is_post_state = True, obj = self.reward_type);
                        self.request_position = (week, day, len(self.requests[week][day]) - 1)

        return initial_state


    def step(self, action):
        obs = []
        reward = 0
        done = False
        infor = {} 
        if (self.is_post_state == True):
            self.current_time, self.request_position, next_request = self.find_next_request(self.request_position) #???
            (valid, min_cost_insertion) = self.sched.check_feasible(next_request, self.current_time, weekly_deadline = True, capacity_heur = self.cap_heur)
            obs = self.feature_extractor.get_feature(request=next_request, current_time=self.current_time, min_cost_insertion=min_cost_insertion, valid=valid, capacity_heur = self.cap_heur, obj = self.reward_type);
            self.is_post_state = False
            reward = 0
        else:
            week, day, index = self.request_position;
            request = self.requests[week][day][index];
            (valid, min_cost_insertion) = self.sched.check_feasible(request, self.current_time, weekly_deadline = True, capacity_heur = self.cap_heur)
            if action == self.REJECT or valid == False:
                obs = self.feature_extractor.get_feature(request=request, current_time=self.current_time, min_cost_insertion=min_cost_insertion, is_post_state = True, capacity_heur = self.cap_heur, obj = self.reward_type);
                reward = 0
            elif action == self.ACCEPT:
                self.sched.accept_checked_request(request, min_cost_insertion, weekly_deadline = True, capacity_heur = self.cap_heur)
                obs = self.feature_extractor.get_feature(request=request, current_time=self.current_time, min_cost_insertion=min_cost_insertion, is_post_state = True, capacity_heur = self.cap_heur, obj = self.reward_type);
                reward = 1
                if (self.reward_type == 'visit'):
                        reward = request.require_time[0] * request.require_time[1]
            else:
                raise ValueError("Received invalid action={} which is not part of the action space".format(action))
            self.is_post_state = True
        
        week, day, index = self.current_time
        if week >= self.env.nb_weeks -  3:
            if (self.is_episodic):
                done = True

        return obs, reward, done, infor

    def render(self, mode='console'):
        #agent is represented as a cross, rest as a dot
        print("->" , self.request_position, self.current_time, self.is_post_state)

    def close(self):
        pass