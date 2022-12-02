from enviroment.schedule import Schedule, Visit
from utils.utils import distance
import numpy as np
from math import sqrt

class FeatureExtractor:
    def __init__(self, env, schedule):
        self.env = env
        self.schedule = schedule
    def get_extends_feature(self, request, current_time, min_cost_insertion, is_post_state = False, capacity_heur = False, valid = True, obj = "patient"):
        max_consider_week = self.env.max_required_week;
        total_time = ((self.env.working_tw[1] - self.env.working_tw[0]) * self.env.day_per_week * max_consider_week)
        working_hours = (self.env.working_tw[1] - self.env.working_tw[0])//60
        # Nurse 's resource
        total_idle_time_avai = []
        total_travel_time = []
        future_visit = []
        next_nurses = [0] * self.env.nb_nurses
        for nurse in range(0, self.env.nb_nurses):
            future_visit.append(0)

            end_week = min(self.env.scheduling_horizon, current_time[0] + max_consider_week + 1)
            for week in range(current_time[0] + 1, end_week):
                _total_idle_time_avai = 0
                _total_travel_time = 0
                for day in range(0, self.env.day_per_week):
                    future_visit[nurse] = future_visit[nurse] + len(self.schedule.planned_routes[nurse][week][day].visit) - 2
                    prev_visit = Visit((-1, -1), -1, -1)
                    for visit in self.schedule.planned_routes[nurse][week][day].visit:
                        if prev_visit.st < 0:
                            prev_visit = visit
                            continue
                        travel_time = distance(prev_visit.pos, visit.pos)
                        idle_time = visit.st - prev_visit.ed - travel_time
                        _total_idle_time_avai = _total_idle_time_avai + idle_time
                        _total_travel_time = _total_travel_time + travel_time
                        prev_visit = visit
                total_idle_time_avai = total_idle_time_avai+ [_total_idle_time_avai / (total_time)]
                total_travel_time = total_travel_time + [3 * _total_travel_time / total_time]    

            #total_travel_time[nurse] = 3 * total_travel_time[nurse] / (total_time)
            #total_idle_time_avai[nurse] = total_idle_time_avai[nurse] / (total_time)
            future_visit[nurse] = future_visit[nurse] / (self.env.day_per_week * self.env.max_required_week * working_hours)
        # Location & Eligibility
        cheapest_insertion_cost =  min_cost_insertion[0][0] / (80 * sqrt(2) * self.env.max_day_per_week * self.env.max_required_week)

        if (obj == 'visit'):
            cheapest_insertion_cost = cheapest_insertion_cost / (request.require_time[0] * request.require_time[1])
        request_info = [request.require_time[0] / self.env.max_required_week, \
            request.require_time[1] / self.env.max_day_per_week, \
            request.require_time[2] / self.env.max_required_hour
        ]
        next_nurses[min_cost_insertion[1]] = 1
        if valid == False:
            cheapest_insertion_cost = 1
        if (is_post_state):
            cheapest_insertion_cost = 0
            request_info = [0, 0, 0]
            next_nurses[min_cost_insertion[1]] = 0
            valid = False
            
        remaining_time = 1 - (current_time[1] * 24 * 60 + current_time[2])/(self.env.day_per_week * 24 * 60)
        
        features = request_info + [cheapest_insertion_cost]+\
            next_nurses + future_visit + [is_post_state] + [valid] +\
            total_idle_time_avai + total_travel_time

        return np.reshape(features, [1, len(features)])
    def get_single_feature(self, request, current_time, min_cost_insertion, is_post_state = False, capacity_heur = False, valid = True, obj = "patient"):
        max_consider_week = self.env.max_required_week;
        total_time = ((self.env.working_tw[1] - self.env.working_tw[0]) * self.env.day_per_week * max_consider_week)
        working_hours = (self.env.working_tw[1] - self.env.working_tw[0])//60
        # Nurse 's resource
        total_idle_time_avai = []
        total_travel_time = []
        future_visit = []
        next_nurses = [0] * self.env.nb_nurses
        for nurse in range(0, self.env.nb_nurses):
            total_idle_time_avai.append(0)
            total_travel_time.append(0)
            future_visit.append(0)

            end_week = min(self.env.scheduling_horizon, current_time[0] + max_consider_week + 1)
            for week in range(current_time[0] + 1, end_week):
                for day in range(0, self.env.day_per_week):
                    future_visit[nurse] = future_visit[nurse] + len(self.schedule.planned_routes[nurse][week][day].visit) - 2
                    prev_visit = Visit((-1, -1), -1, -1)
                    for visit in self.schedule.planned_routes[nurse][week][day].visit:
                        if prev_visit.st < 0:
                            prev_visit = visit
                            continue
                        travel_time = distance(prev_visit.pos, visit.pos)
                        idle_time = visit.st - prev_visit.ed - travel_time
                        total_idle_time_avai[nurse] = total_idle_time_avai[nurse] + idle_time
                        total_travel_time[nurse] = total_travel_time[nurse] + travel_time
                        prev_visit = visit
                    
            total_travel_time[nurse] = 3 * total_travel_time[nurse] / (total_time)
            total_idle_time_avai[nurse] = total_idle_time_avai[nurse] / (total_time)
            future_visit[nurse] = future_visit[nurse] / (self.env.day_per_week * self.env.max_required_week * working_hours)
        # Location & Eligibility
        cheapest_insertion_cost =  min_cost_insertion[0][0] / (80 * sqrt(2) * self.env.max_day_per_week * self.env.max_required_week)
        if (obj == 'visit'):
            cheapest_insertion_cost = cheapest_insertion_cost / (request.require_time[0] * request.require_time[1])
        request_info = [request.require_time[0] / self.env.max_required_week, \
            request.require_time[1] / self.env.max_day_per_week, \
            request.require_time[2] / self.env.max_required_hour
        ]
        select_nurse = min_cost_insertion[1]
        if valid == False:
            cheapest_insertion_cost = 1

        if (is_post_state):
            cheapest_insertion_cost = 0
            request_info = [0, 0, 0]
            valid = False
            
        remaining_time = 1 - (current_time[1] * 24 * 60 + current_time[2])/(self.env.day_per_week * 24 * 60)
        
        features = request_info + [cheapest_insertion_cost] + \
            [future_visit[select_nurse]] + [is_post_state] + [valid] +\
            [total_idle_time_avai[select_nurse]] + [total_travel_time[select_nurse]]
        return np.reshape(features, [1, len(features)])
    def get_normal_feature(self, request, current_time, min_cost_insertion, is_post_state = False, capacity_heur = False, valid = True, obj = "patient"):
        max_consider_week = self.env.max_required_week;
        total_time = ((self.env.working_tw[1] - self.env.working_tw[0]) * self.env.day_per_week * max_consider_week)
        working_hours = (self.env.working_tw[1] - self.env.working_tw[0])//60
        # Nurse 's resource
        total_idle_time_avai = []
        total_travel_time = []
        future_visit = []
        next_nurses = [0] * self.env.nb_nurses
        for nurse in range(0, self.env.nb_nurses):
            total_idle_time_avai.append(0)
            total_travel_time.append(0)
            future_visit.append(0)

            end_week = min(self.env.scheduling_horizon, current_time[0] + max_consider_week + 1)
            for week in range(current_time[0] + 1, end_week):
                for day in range(0, self.env.day_per_week):
                    future_visit[nurse] = future_visit[nurse] + len(self.schedule.planned_routes[nurse][week][day].visit) - 2
                    prev_visit = Visit((-1, -1), -1, -1)
                    for visit in self.schedule.planned_routes[nurse][week][day].visit:
                        if prev_visit.st < 0:
                            prev_visit = visit
                            continue
                        travel_time = distance(prev_visit.pos, visit.pos)
                        idle_time = visit.st - prev_visit.ed - travel_time
                        total_idle_time_avai[nurse] = total_idle_time_avai[nurse] + idle_time
                        total_travel_time[nurse] = total_travel_time[nurse] + travel_time
                        prev_visit = visit
                    
            total_travel_time[nurse] = 3 * total_travel_time[nurse] / (total_time)
            total_idle_time_avai[nurse] = total_idle_time_avai[nurse] / (total_time)
            future_visit[nurse] = future_visit[nurse] / (self.env.day_per_week * self.env.max_required_week * working_hours)
        # Location & Eligibility
        cheapest_insertion_cost =  min_cost_insertion[0][0] / (80 * sqrt(2) * self.env.max_day_per_week * self.env.max_required_week)

        request_info = [request.require_time[0] / self.env.max_required_week, \
            request.require_time[1] / self.env.max_day_per_week, \
            request.require_time[2] / self.env.max_required_hour
        ]
        
        next_nurses[min_cost_insertion[1]] = cheapest_insertion_cost
        if valid == False:
            cheapest_insertion_cost = 1
            next_nurses[min_cost_insertion[1]] = 0
        if (is_post_state):
            cheapest_insertion_cost = 0
            request_info = [0, 0, 0]
            next_nurses[min_cost_insertion[1]] = 0
            valid = False
        remaining_time = 1 - (current_time[1] * 24 * 60 + current_time[2])/(self.env.day_per_week * 24 * 60)
        
        features = request_info + [remaining_time] + \
            next_nurses + future_visit + [is_post_state] + [valid] +\
            total_idle_time_avai + total_travel_time
        return np.reshape(features, [1, len(features)])
    def get_nurse_choosing_feature(self, request, current_time, min_cost_insertion, is_post_state = False, capacity_heur = False, valid = True, obj = "patient"):
        max_consider_week = self.env.max_required_week;
        total_time = ((self.env.working_tw[1] - self.env.working_tw[0]) * self.env.day_per_week * max_consider_week)
        working_hours = (self.env.working_tw[1] - self.env.working_tw[0])//60
        # Nurse 's resource
        total_idle_time_avai = []
        total_travel_time = []
        future_visit = []
        next_nurses = [0] * self.env.nb_nurses
        valids = [0] * self.env.nb_nurses
        insert_cost = [0] * self.env.nb_nurses

        for nurse in range(0, self.env.nb_nurses):
            total_idle_time_avai.append(0)
            total_travel_time.append(0)
            future_visit.append(0)
            end_week = min(self.env.scheduling_horizon, current_time[0] + max_consider_week + 1)
            if (valid == True and is_post_state == False):
                (_valid, _min_cost_insertion) = self.schedule.check_feasible(request, current_time, spec_nurse = nurse, weekly_deadline = True, capacity_heur = capacity_heur)
                if (_valid):
                    valids[nurse] = 1
                    insert_cost[nurse] = _min_cost_insertion[0][0]/ (80 * sqrt(2) * self.env.max_day_per_week * self.env.max_required_week)

            
            for week in range(current_time[0] + 1, end_week):
                for day in range(0, self.env.day_per_week):
                    future_visit[nurse] = future_visit[nurse] + len(self.schedule.planned_routes[nurse][week][day].visit) - 2
                    prev_visit = Visit((-1, -1), -1, -1)
                    for visit in self.schedule.planned_routes[nurse][week][day].visit:
                        if prev_visit.st < 0:
                            prev_visit = visit
                            continue
                        travel_time = distance(prev_visit.pos, visit.pos)
                        idle_time = visit.st - prev_visit.ed - travel_time
                        total_idle_time_avai[nurse] = total_idle_time_avai[nurse] + idle_time
                        total_travel_time[nurse] = total_travel_time[nurse] + travel_time
                        prev_visit = visit
                    
            total_travel_time[nurse] = 3 * total_travel_time[nurse] / (total_time)
            total_idle_time_avai[nurse] = total_idle_time_avai[nurse] / (total_time)
            future_visit[nurse] = future_visit[nurse] / (self.env.day_per_week * self.env.max_required_week * working_hours)
        # Location & Eligibility
        cheapest_insertion_cost =  min_cost_insertion[0][0] / (80 * sqrt(2) * self.env.max_day_per_week * self.env.max_required_week)

        request_info = [request.require_time[0] / self.env.max_required_week, \
            request.require_time[1] / self.env.max_day_per_week, \
            request.require_time[2] / self.env.max_required_hour
        ]

        next_nurses[min_cost_insertion[1]] = cheapest_insertion_cost
        if valid == False:
            cheapest_insertion_cost = 1
            insert_cost = [0] * self.env.nb_nurses
            valids = [0] * self.env.nb_nurses
        if (is_post_state):
            cheapest_insertion_cost = 0
            request_info = [0, 0, 0]
            insert_cost = [0] * self.env.nb_nurses
            valids = [0] * self.env.nb_nurses
            valid = False

        
        remaining_time = 1 - (current_time[1] * 24 * 60 + current_time[2])/(self.env.day_per_week * 24 * 60)
        
        features = request_info + [remaining_time, is_post_state, valid] +\
            insert_cost + valids + future_visit +\
            total_idle_time_avai + total_travel_time

        return np.reshape(features, [1, len(features)])
    def get_feature(self, request, current_time, min_cost_insertion, is_post_state = False, capacity_heur = False, valid = True, obj = "patient"):
        return self.get_nurse_choosing_feature(request, current_time, min_cost_insertion, is_post_state = is_post_state, capacity_heur = capacity_heur, valid = valid, obj = obj);