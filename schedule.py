from enviroment import Enviroment
from enviroment import Request
import utils

class Visit:
    def __init__(self, position, st, ed):
        self.pos = position
        self.st = st
        self.ed = ed
class Route:
    def __init__(self, depot, working_tw):
        self.visit = [Visit(depot, working_tw[0], working_tw[0]), Visit(depot, working_tw[1], working_tw[1])]
    def get_travel_time(self):
        return self.visit[:-2].ed
    def insert(self, position, start_time, end_time, checking = False):
        """ Insert a visit (position, start_time, end_time) into current route
            If checking = True then just return the increasing cost without modify the route
            If the insertion is infeasible, return -1
        """
        position  = 0
        for pos in range(1, len(self.visit)):
            prev = self.visit[pos - 1]
            next_ = self.visit[pos]
            if not (prev.ed + distance(prev.pos, position) <= start_time):
                continue
            if not (end_time + distance(position, next_.pos) <= next_.ed):
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
        self.planned_routes = [[[Route(env.nurse_depot, env.working_tw)]\
                            * env.day_per_week] * env.nb_weeks] * env.nb_nurses #nurses(weeks(days)))
        self.weeks = []
        self.horizon = env.nb_weeks
        self.work_tw = env.working_tw
        self.nb_nurses = env.nb_nurses
        self.qual = env.qual
       
    def check_feasible(self, request, current_time):
        for pattern in day_patterns:
            if (len(pattern) != request.require_time[1]):
                continue
            max_start_week = self.horizon - request.require_time[0]
            for start_week in range(current_time[0], max_start_week + 1):
                is_st_week_ok = False
                for time in range(0, 1440):
                    is_time_ok = True
                    #Out of working window of nurse
                    if (time not in self.work_tw):
                        continue
                    if ((time + request.require_time[2] - 1) not in self.work_tw):
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
                            for day in pattern:
                                cost = self.planned_routes[nurse][week][day].insert(request.location, time, time + request.require_time[2] - 1, checking = True) 
                                if (cost < 0):
                                    _is_ok = False
                            is_ok = is_ok | _is_ok
                        is_time_ok = is_time_ok & is_ok
                    is_st_week_ok = is_st_week_ok | is_time_ok
                    
                if (is_st_week_ok):
                    return true
        return
    def accept_request(self, request, days, time):
        """Update the planned routes after accept the request
        """
        return