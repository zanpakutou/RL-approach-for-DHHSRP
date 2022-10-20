from enviroment.schedule import Schedule, Visit
from utils.utils import distance
import numpy as np
from math import sqrt

class FeatureExtractor:
    def __init__(self, env, schedule):
        self.env = env
        self.schedule = schedule
    def get_feature(self, request, current_time, min_cost_insertion, is_post_state = False, capacity_heur = False):
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
                    

            total_travel_time[nurse] = total_travel_time[nurse] / (total_time)
            total_idle_time_avai[nurse] = total_idle_time_avai[nurse] / (total_time)
            future_visit[nurse] = future_visit[nurse] / (self.env.day_per_week * self.env.max_required_week * working_hours)
        # Location & Eligibility
        cheapest_insertion_cost = 0
        if (capacity_heur == False):
            cheapest_insertion_cost = min_cost_insertion[0][0] / (80 * sqrt(2) * self.env.max_day_per_week * self.env.max_required_week)
        else :
            cheapest_insertion_cost = min_cost_insertion[0][3] / (80 * sqrt(2) * self.env.max_day_per_week * self.env.max_required_week)

        request_info = [request.require_time[0] / self.env.max_required_week, \
            request.require_time[1] / self.env.max_day_per_week, \
            request.require_time[2] / self.env.max_required_hour
        ]
        next_nurses[min_cost_insertion[1]] = 1
        if (is_post_state):
            cheapest_insertion_cost = 0
            request_info = [0, 0, 0]
            next_nurses[min_cost_insertion[1]] = 0
            
        remaining_time = 1 - (current_time[1] * 24 * 60 + current_time[2])/(self.env.day_per_week * 24 * 60)
        

        features = request_info + [cheapest_insertion_cost] + [remaining_time] + \
            next_nurses + future_visit + \
            total_idle_time_avai + total_travel_time
        return np.reshape(features, [1, len(features)])