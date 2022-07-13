from enviroment.schedule import Schedule, Visit
from enviroment.utils import distance 

class FeatureExtractor:
    def __init__(self, env, schedule):
        self.env = env
        self.schedule = schedule
    def get_feature(self, request, current_time, is_check = False):
        total_time = ((self.env.working_tw[1] - self.env.working_tw[0])*self.env.day_per_week*self.env.scheduling_horizon)

        #Request information
        require_weeks = request.require_time[0]/self.env.scheduling_horizon
        require_days = request.require_time[1]/self.env.day_per_week
        require_hours = request.require_time[2]/2
        #Nurse's resource
        total_idle_time_avai = []
        avgl_idle_time_avai = []
        cheapest_insertion_cost = []
        count_idle = 0
        ocupied_rate = []
        count_nurse = 0
        for nurse in range(0, self.env.nb_nurses):
            total_idle_time_avai.append(0)
            avgl_idle_time_avai.append(0)
            ocupied_rate.append(0)
            cheapest_insertion_cost.append(1)
            
            if (self.env.qual[nurse] < request.require_skill):
                continue

            check = self.schedule.check_feasible(request, current_time, spec_nurse = nurse, weekly_deadline = True)
            if (check[0] == True):
                cheapest_insertion_cost[nurse] = check[1][0] / (80 * 2.83 * 3) #80 * sqet(2) * 2 * max_week
            if (is_check):
                if (check[0] == False):
                    continue
                
                
            count_nurse = count_nurse + 1
            count_idle = 0
            end_week = min(self.env.scheduling_horizon, current_time[0] + 4)
            for week in range(current_time[0], end_week):
                for day in range(0, self.env.day_per_week):
                    if (week == current_time[0] and day < current_time[1]):
                        continue
                    prev_visit = Visit((-1, -1), -1, -1)
                    for visit in self.schedule.planned_routes[nurse][week][day].visit:
                        if prev_visit.st < 0 :
                            prev_visit = visit
                            continue
                        if (week == current_time[0] and day == current_time[1] and visit.st < current_time[2]):
                            prev_visit = visit
                            continue
                        travel_time = distance(prev_visit.pos, visit.pos)
                        idle_time = visit.st - prev_visit.ed - travel_time
                        total_idle_time_avai[nurse] = total_idle_time_avai[nurse] + idle_time
                        count_idle = count_idle + 1
                        prev_visit = visit
                        
            ocupied_rate[nurse] = total_idle_time_avai[nurse] / (self.env.day_per_week \
                * (end_week - current_time[0]) * (self.env.working_tw[1] - self.env.working_tw[0]))
            total_idle_time_avai[nurse]  = total_idle_time_avai[nurse] / (total_time)
            avgl_idle_time_avai[nurse] = total_idle_time_avai[nurse] / count_idle
            cheapest_insertion_cost[nurse] = min(cheapest_insertion_cost[nurse], 1)
        #Location & Eligibility
        (valid, min_cost_insertion) = self.schedule.check_feasible(request, current_time)
        #if (request.require_skill < 0):
            #cheapest_insertion_cost = [1,1,1,1,1,1]
        return [require_weeks, require_days, require_hours] \
            + cheapest_insertion_cost + total_idle_time_avai + avgl_idle_time_avai + ocupied_rate