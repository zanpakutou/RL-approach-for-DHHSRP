from enviroment.schedule import Schedule, Visit
from utils.utils import distance
import numpy as np
from math import sqrt


class FeatureExtractor:
    def __init__(self, env, schedule):
        self.env = env
        self.schedule = schedule

    def get_feature(
        self,
        request,
        current_time,
        is_post_state=False,
        capacity_heur=False,
        valid=True,
        obj='visit'
    ):
        max_consider_week = self.env.max_required_week
        total_time = (
            (self.env.working_tw[1] - self.env.working_tw[0])
            * self.env.day_per_week
            * max_consider_week
        )

        # Nurse 's resource
        total_idle_time_avai = []
        total_travel_time = []
        valids = [0] * self.env.nb_nurses
        insert_cost = [0] * self.env.nb_nurses

        for nurse in range(0, self.env.nb_nurses):
            total_idle_time_avai.append(0)
            total_travel_time.append(0)
            end_week = min(
                self.env.scheduling_horizon, current_time[0] + max_consider_week + 1
            )
            if valid == True and is_post_state == False:
                (_valid, _min_cost_insertion) = self.schedule.check_feasible(
                    request,
                    current_time,
                    spec_nurse=nurse,
                    weekly_deadline=True,
                    capacity_heur=capacity_heur,
                )
                if _valid:
                    valids[nurse] = 1
                    if (obj == 'visit'):
                        insert_cost[nurse] = _min_cost_insertion[0][0] / (
                            60 
                            * request.require_time[1]
                            * self.env.max_required_week
                        )
                    else:
                        insert_cost[nurse] = _min_cost_insertion[0][0] / (
                            60 
                            * self.env.max_day_per_week
                            * self.env.max_required_week
                        )

            for week in range(current_time[0] + 1, end_week):
                for day in range(0, self.env.day_per_week):
                    prev_visit = Visit((-1, -1), -1, -1)
                    for visit in self.schedule.planned_routes[nurse][week][day].visit:
                        if prev_visit.st < 0:
                            prev_visit = visit
                            continue
                        travel_time = distance(prev_visit.pos, visit.pos)
                        idle_time = visit.st - prev_visit.ed - travel_time
                        total_idle_time_avai[nurse] = (
                            total_idle_time_avai[nurse] + idle_time
                        )
                        total_travel_time[nurse] = (
                            total_travel_time[nurse] + travel_time
                        )
                        prev_visit = visit

            total_travel_time[nurse] = 3 * total_travel_time[nurse] / (total_time)
            total_idle_time_avai[nurse] = total_idle_time_avai[nurse] / (total_time)


        # Location & Eligibility
        request_info = [
            request.require_time[0] / self.env.max_required_week,
            request.require_time[1] / self.env.max_day_per_week,
            request.require_time[2] / self.env.max_required_hour,
        ]

        if valid == False:
            insert_cost = [0] * self.env.nb_nurses
            valids = [0] * self.env.nb_nurses
        if is_post_state:
            request_info = [0, 0, 0]
            insert_cost = [0] * self.env.nb_nurses
            valids = [0] * self.env.nb_nurses
            valid = False

        remaining_time = 1 - (current_time[1] * 24 * 60 + current_time[2]) / (
            self.env.day_per_week * 24 * 60
        )

        features = (
            request_info
            + [remaining_time, is_post_state, valid]
            + insert_cost
            + valids
            + total_idle_time_avai
            + total_travel_time
        )

        return np.reshape(features, [1, len(features)])
