from enviroment import Enviroment
from enviroment import Request
from utils import day_patterns, distance

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
        for pos in range(1, len(self.visit)):
            prev = self.visit[pos - 1]
            next_ = self.visit[pos]
            if (not prev.ed <= start_time) or  (not (end_time <= next_.st)):
                continue 
            if not (prev.ed + distance(prev.pos, position) <= start_time):
                continue
            if not (end_time + distance(position, next_.pos) <= next_.st):
                continue
            if (pos < len(self.visit) - 1):
                if checking == False :
                    self.visit.insert(pos, Visit(position, start_time, end_time))
                return 0
            else:
                if checking == False :
                    self.visit.insert(pos, Visit(position, start_time, end_time))
                return end_time - prev.ed
            
        return -1
class Schedule:
    def __init__(self, env):
        self.planned_routes = [[[Route(env.nurse_depot, env.working_tw) for i in range(env.day_per_week)] \
                                for j in range (env.nb_weeks)] for k in range(env.nb_nurses)] #nurses(weeks(days)))
        self.weeks = []
        self.horizon = env.nb_weeks
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
                    if (not self.work_tw[0] <= (time + request.require_time[2]*60) <= self.work_tw[1]):
                        continue
                    #Is not start of time slot
                    if (time % 15 != 0):
                        continue
                    
                    #Check for c_p weeks
                    for week in range(start_week, start_week + request.require_time[0]):
                        is_ok = False
                        for nurse in range(0, self.nb_nurses):
                            if (self.qual[nurse] < request.require_skill):
                                continue
                            _is_ok = True
                            total_cost = True
                            for day in pattern:
                                #Check valid timestamps
                                if (week == current_time[0] and day < current_time[1]):
                                    _is_ok = False
                                    continue
                                if (week == current_time[0] and day == current_time[1] and time <= current_time[1]):
                                    _is_ok = False
                                    continue
                                #Check valid insertion
                                cost = self.planned_routes[nurse][week][day].insert(request.location, time, time + request.require_time[2]*60, checking = True)
                                if (cost < 0):
                                    _is_ok = False
                                total_cost += cost;
                            if (_is_ok and total_cost < min_cost_insertion[0]): 
                                min_cost_insertion = (total_cost, nurse, time, pattern, start_week)
                            
                            is_ok = is_ok | _is_ok
                        is_time_ok = is_time_ok & is_ok
                    is_st_week_ok = is_st_week_ok | is_time_ok
                    
                if (is_st_week_ok):
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