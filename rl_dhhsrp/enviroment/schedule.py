from enviroment.patient_request import PatientRequest, Request
from enviroment.utils import day_patterns, distance

class Visit:
    def __init__(self, position, st, ed):
        self.pos = position
        self.st = st
        self.ed = ed
class Route:
    def __init__(self, depot, working_tw):
        self.visit = [Visit(depot, working_tw[0], working_tw[0]), Visit(depot, working_tw[1], working_tw[1])]
    def get_travel_time(self):
        return self.visit[-2].ed
    def insert(self, position, start_time, end_time, checking = False):
        """ Insert a visit (position, start_time, end_time) into current route
            If checking = True then just return the increasing cost without modify the route
            If the insertion is infeasible, return -1
        """
        min_cost = 10000000000
        save_pos = -1
        for pos in range(1, len(self.visit)):
            prev = self.visit[pos - 1]
            next_ = self.visit[pos]
            if (not prev.ed <= start_time) or  (not (end_time <= next_.st)):
                continue 
            if not (prev.ed + distance(prev.pos, position) <= start_time):
                continue
            if not (end_time + distance(position, next_.pos) <= next_.st):
                continue
            insert_cost = distance(prev.pos, position) + distance(position, next_.pos)
            if (insert_cost < min_cost):
                save_pos = pos
                min_cost = insert_cost
        if save_pos > -1 : 
            if checking == False :
                self.visit.insert(save_pos, Visit(position, start_time, end_time))
            return min_cost
        else:
            return -1
class Schedule:
    def __init__(self, env):
        self.planned_routes = [[[Route(env.nurse_depot, env.working_tw) for i in range(env.day_per_week)] \
                                for j in range (env.scheduling_horizon)] for k in range(env.nb_nurses)] #nurses(weeks(days)))
        self.weeks = []
        self.horizon = env.scheduling_horizon
        self.work_tw = env.working_tw
        self.nb_nurses = env.nb_nurses
        self.qual = env.qual
        
    def check_feasible(self, request, current_time):
        min_cost_insertion =  (1000000000, -1, -1, (-1), -1)
        
        for pattern in day_patterns:
            if (len(pattern) != request.require_time[1]):
                continue
            max_start_week = self.horizon - request.require_time[0]
            for start_week in range(current_time[0], max_start_week + 1):
                is_st_week_ok = False
                for time in range(0, 1440, 15):
                    is_time_ok = True
                    #Out of working window of nurse
                    if (not self.work_tw[0] <= time <= self.work_tw[1]):
                        continue
                    if (not self.work_tw[0] <= (time + request.require_time[2] * 60) <= self.work_tw[1]):
                        continue
                    #Is not start of time slot
                    if (time % 15 != 0):
                        continue

                    #Check for each nurse
                    for nurse in range(0, self.nb_nurses):
                        if (self.qual[nurse] < request.require_skill):
                            continue
                        nurse_is_ok = True
                        nurse_total_cost = 0
                        #For each nurse, visit time, day pattern, start week, calculate the increasing cost
                        for week in range(start_week, start_week + request.require_time[0]):
                            if (nurse_is_ok == False):
                                break
                            for day in pattern:
                                #Check valid timestamps
                                if (week == current_time[0] and day < current_time[1]):
                                    nurse_is_ok = False
                                    break
                                if (week == current_time[0] and day == current_time[1] and time <= current_time[1]):
                                    nurse_is_ok = False
                                    break
                                #Check valid insertion
                                cost = self.planned_routes[nurse][week][day].insert(request.location, time, time + request.require_time[2]*60, checking = True)
                                if (cost < 0):
                                    nurse_is_ok = False
                                    break
                                nurse_total_cost = nurse_total_cost + cost;
                        
                        if (nurse_is_ok and nurse_total_cost < min_cost_insertion[0]):
                            min_cost_insertion = (nurse_total_cost, nurse, time, pattern, start_week)                  
        if (min_cost_insertion[0] < 999999900):
            return (True, min_cost_insertion)
        return (False, min_cost_insertion)
        
    def accept_request(self, request, current_time):
        """Update the planned routes after accept the request
        """
        (ok, min_cost_insertion) = self.check_feasible(request, current_time)
        if ok == False:
            return False
        (total_cost, nurse, time, pattern, start_week) = min_cost_insertion
        
        for week in range(start_week, start_week + request.require_time[0]):
            for day in pattern:
                self.planned_routes[nurse][week][day].insert(request.location, time, time + request.require_time[2] * 60)        
        return True
        
    def is_feasible(self):
        for week in range(self.horizon):
            for day in range(len(self.planned_routes[0][week])):
                for nurse in range(self.nb_nurses):
                    route = self.planned_routes[nurse][week][day]
                    for pos in range(1, len(route.visit)):
                        prev = route.visit[pos - 1]
                        curr = route.visit[pos]
                        if (prev.ed + distance(prev.pos, curr.pos) > curr.st):
                            return False
                    
        return True