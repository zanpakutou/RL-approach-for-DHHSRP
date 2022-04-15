from enviroment import Enviroment
from schedule import Schedule, Visit
from utils import distance 

class FeatureExtractor:
    def __init__(self, env, schedule):
        self.env = env
        self.schedule = schedule
    def get_feature(self, request, current_time):
        decision_point = 1440 * (self.env.day_per_week * current_time[0] + current_time[1])\
                        + request.current_time
        #Request information
        require_weeks = request.require_time[0]
        require_days = request.require_time[1]
        require_hours = request.require_time[2]
        #Nurse's resource
        total_idle_time = 0
        avg_idle_time = 0
        count_idle = 0
        ocupied_rate = 0
        count_nurse = 0
        for nurse in range(0, self.env.nb_nurses):
            #if (self.env.qual[nurse] < request.require_skill):
            #    continue
            count_nurse = count_nurse + 1
            for week in range(0, self.env.nb_weeks):
                for day in range(0, self.env.day_per_week):
                    if (week == current_time[0] and day < current_time[1]):
                        continue
                    prev_visit = Visit((-1, -1), -1, -1)
                    for visit in self.schedule.planned_routes[nurse][week][day].visit:
                        if prev_visit.st < 0 :
                            prev_visit = visit
                            continue
                        
                        travel_time = distance(prev_visit.pos, visit.pos)
                        idle_time = visit.st - prev_visit.ed - travel_time
                        total_idle_time = total_idle_time + idle_time
                        count_idle = count_idle + 1
                        prev_visit = visit
                        
                        
        avg_idle_time = total_idle_time / count_idle
        ocupied_rate = total_idle_time / (count_nurse * self.env.day_per_week \
            * self.env.nb_weeks * (self.env.working_tw[1] - self.env.working_tw[0]))
        #Location & Eligibility
        (valid, min_cost_insertion) = self.schedule.check_feasible(request, current_time)
        cheapest_insertion_cost = min_cost_insertion[0]

        return [decision_point, require_weeks, require_days, require_hours, total_idle_time,\
                avg_idle_time, ocupied_rate, cheapest_insertion_cost, valid]