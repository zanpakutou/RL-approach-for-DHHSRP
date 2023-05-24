from enviroment.patient_request import PatientRequest, Request
from utils.utils import day_patterns, distance, MAX_VAL, mean, stdev
import operator
import numpy as np
import statistics

delta = 15
duration = 30  # end_time - start_time

def best_case_func(p, s):
    best_case = -1
    case1 = (s.st - p.st - duration - delta) // (duration + delta)
    case2 = (
        s.st - p.st - delta * np.ceil(distance(s.pos, p.pos) / delta)
    ) // duration - 1
    return min(case1, case2)


def flexible_capacity_func(p, s):
    return (s.st - p.st) // delta - (duration // delta + 1) * (best_case_func(p, s) + 1)


def reduce_flexible_cap_func(p, s, q):
    c = 0
    y = q.st - (p.st + duration) / delta > 1
    z = s.st - (q.st + duration) / delta > 1
    if best_case_func(p, q) == 0 and y:
        c = min(flexible_capacity_func(p, s), y + c)
    if best_case_func(q, s) == 0 and z:
        c = min(flexible_capacity_func(p, s), z + c)
    return c


class Visit:
    def __init__(
        self,
        position,
        st,
        ed,
    ):

        self.pos = position
        self.st = st
        self.ed = ed

class Route:
    def __init__(self, depot, working_tw):
        self.visit = [
            Visit(depot, working_tw[0], working_tw[0]),
            Visit(depot, working_tw[1], working_tw[1]),
        ]

    def get_travel_time(self):
        return self.visit[-2].ed

    def insert(
        self,
        position,
        start_time,
        end_time,
        checking=False,
    ):
        """Insert a visit (position, start_time, end_time) into current route
        If checking = True then just return the increasing cost without modify the route
        If the insertion is infeasible, return -1
        """

        min_cost = MAX_VAL
        save_pos = -1
        for pos in range(1, len(self.visit)):
            prev = self.visit[pos - 1]
            next_ = self.visit[pos]
            if not prev.ed <= start_time or not end_time <= next_.st:
                continue
            if not prev.ed + distance(prev.pos, position) <= start_time:
                continue
            if not end_time + distance(position, next_.pos) <= next_.st:
                continue

            insert_cost = (
                distance(prev.pos, position)
                + distance(position, next_.pos)
                - distance(prev.pos, next_.pos)
            )
            if insert_cost < min_cost:
                save_pos = pos
                min_cost = insert_cost
        if save_pos > -1:
            if checking == False:
                self.visit.insert(save_pos, Visit(position, start_time, end_time))
            return (min_cost,)
        else:
            return (-1,)

    def capacity_criteria(
        self,
        position,
        start_time,
        end_time,
        checking=False,
    ):

        (p, s) = (0, 0)
        best_case = -MAX_VAL
        save_pos = -1
        for pos in range(1, len(self.visit)):
            prev = self.visit[pos - 1]
            next_ = self.visit[pos]
            if not prev.ed <= start_time or not end_time <= next_.st:
                continue
            if not prev.ed + distance(prev.pos, position) <= start_time:
                continue
            if not end_time + distance(position, next_.pos) <= next_.st:
                continue
            p = prev
            s = next_
            q = Visit(position, start_time, end_time)
            best_case = (
                best_case_func(p, s) - best_case_func(p, q) - best_case_func(q, s)
            )
            reduce_flexible_cap = reduce_flexible_cap_func(p, s, q)
            insert_cost = (
                distance(prev.pos, position)
                + distance(position, next_.pos)
                - distance(prev.pos, next_.pos)
            )
            save_pos = pos
        if best_case > -MAX_VAL:
            if checking == False:
                self.visit.insert(save_pos, Visit(position, start_time, end_time))
            return (best_case, reduce_flexible_cap, 0, insert_cost)
        else:
            return (-1, -1)


class Schedule:
    def __init__(self, env):
        self.planned_routes = [
            [
                [
                    Route(env.nurse_depot[k], env.working_tw)
                    for i in range(env.day_per_week)
                ]
                for j in range(env.scheduling_horizon)
            ]
            for k in range(env.nb_nurses)
        ]

        # nurses(weeks(days)))

        self.weeks = []
        self.horizon = env.scheduling_horizon
        self.work_tw = env.working_tw
        self.nb_nurses = env.nb_nurses
        self.qual = env.qual
        self.count_accept_pat = [0] * 9
        duration = env.max_required_hour

    def check_feasible(
        self,
        request,
        current_time,
        spec_nurse=-1,
        weekly_deadline=False,
        capacity_heur=False,
    ):

        min_cost_insertion = (
            (MAX_VAL,),
            -1,
            -1,
            [-1],
            -1,
            -1,
        )
        if capacity_heur == True:
            min_cost_insertion = (
                (MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL),
                -1,
                -1,
                [-1],
                -1,
                -1,
            )

        pat_id = -1
        for pattern in day_patterns:
            pat_id = pat_id + 1
            if len(pattern) != request.require_time[1]:
                continue
            max_start_week = self.horizon - request.require_time[0]
            if weekly_deadline == True:
                max_start_week = min(max_start_week, current_time[0] + 1)
            for start_week in range(current_time[0] + 1, max_start_week + 1):
                is_st_week_ok = False
                for time in range(0, 1440, 15):
                    is_time_ok = True

                    # Out of working window of nurse
                    if not self.work_tw[0] <= time <= self.work_tw[1]:
                        continue
                    if (
                        not self.work_tw[0]
                        <= time + request.require_time[2] * 15
                        <= self.work_tw[1]
                    ):
                        continue

                    # Is not start of time slot
                    if time % 15 != 0:
                        continue

                    # Check for each nurse
                    for nurse in range(0, self.nb_nurses):
                        if self.qual[nurse] < request.require_skill:
                            continue
                        if spec_nurse > -1 and nurse != spec_nurse:
                            continue
                        nurse_is_ok = True
                        nurse_total_cost = (0,)
                        if capacity_heur:
                            nurse_total_cost = (0, 0, 0, 0)

                        # For each nurse, visit time, day pattern, start week, calculate the increasing cost
                        for week in range(
                            start_week, start_week + request.require_time[0]
                        ):
                            if nurse_is_ok == False:
                                break
                            for day in pattern:
                                # Check valid timestamps
                                if week == current_time[0] and day < current_time[1]:
                                    nurse_is_ok = False
                                    break
                                if (
                                    week == current_time[0]
                                    and day == current_time[1]
                                    and time <= current_time[1]
                                ):
                                    nurse_is_ok = False
                                    break

                                # Check valid insertion
                                if capacity_heur == True:
                                    cost = self.planned_routes[nurse][week][
                                        day
                                    ].capacity_criteria(
                                        request.location,
                                        time,
                                        time + request.require_time[2] * 15,
                                        checking=True,
                                    )
                                else:
                                    cost = self.planned_routes[nurse][week][day].insert(
                                        request.location,
                                        time,
                                        time + request.require_time[2] * 15,
                                        checking=True,
                                    )

                                if cost[0] < 0:
                                    nurse_is_ok = False
                                    break

                                nurse_total_cost = tuple(
                                    map(operator.add, nurse_total_cost, cost)
                                )

                        if capacity_heur == True:
                            nurse_total_cost = (
                                nurse_total_cost[0],
                                nurse_total_cost[1],
                                self.count_accept_pat[pat_id],
                                nurse_total_cost[3],
                            )
                        if nurse_is_ok and nurse_total_cost < min_cost_insertion[0]:
                            min_cost_insertion = (
                                nurse_total_cost,
                                nurse,
                                time,
                                pattern,
                                start_week,
                                pat_id,
                            )
        if capacity_heur == True:
            min_cost_insertion = (
                (min_cost_insertion[0][3],),
                min_cost_insertion[1],
                min_cost_insertion[2],
                min_cost_insertion[3],
                min_cost_insertion[4],
                min_cost_insertion[5],
            )

        if min_cost_insertion[0] < (MAX_VAL - 1, MAX_VAL, MAX_VAL, MAX_VAL):
            return (True, min_cost_insertion)

        return (False, min_cost_insertion)

    def accept_request(
        self,
        request,
        current_time,
        spec_nurse=-1,
        weekly_deadline=True,
        capacity_heur=False,
    ):
        """Update the planned routes after accept the request"""

        (ok, min_cost_insertion) = self.check_feasible(
            request,
            current_time,
            spec_nurse,
            weekly_deadline=weekly_deadline,
            capacity_heur=capacity_heur,
        )
        if ok == False:
            return False

        (
            total_cost,
            nurse,
            time,
            pattern,
            start_week,
            pat_id,
        ) = min_cost_insertion
        self.count_accept_pat[pat_id] = self.count_accept_pat[pat_id] + 1
        for week in range(start_week, start_week + request.require_time[0]):
            for day in pattern:
                self.planned_routes[nurse][week][day].insert(
                    request.location, time, time + request.require_time[2] * 15
                )
        return True

    def accept_checked_request(
        self,
        request,
        min_cost_insertion,
        spec_nurse=-1,
        weekly_deadline=False,
        capacity_heur=False,
    ):
        """Update the planned routes after accept the request that checked above"""

        (
            total_cost,
            nurse,
            time,
            pattern,
            start_week,
            pat_id,
        ) = min_cost_insertion
        self.count_accept_pat[pat_id] = self.count_accept_pat[pat_id] + 1
        for week in range(start_week, start_week + request.require_time[0]):
            for day in pattern:
                check = None
                if capacity_heur == True:
                    check = self.planned_routes[nurse][week][day].capacity_criteria(
                        request.location, time, time + request.require_time[2] * 15
                    )
                else:
                    check = self.planned_routes[nurse][week][day].insert(
                        request.location, time, time + request.require_time[2] * 15
                    )
                if check[0] < -0.5:
                    print("??? check fail")
        return True

    def is_feasible(self):
        for week in range(self.horizon):
            for day in range(len(self.planned_routes[0][week])):
                for nurse in range(self.nb_nurses):
                    route = self.planned_routes[nurse][week][day]
                    for pos in range(1, len(route.visit)):
                        prev = route.visit[pos - 1]
                        curr = route.visit[pos]
                        if prev.ed + distance(prev.pos, curr.pos) > curr.st:
                            return False
        return True

    def get_metrics(self):
        nurse_travel_time = []
        nurse_workload = []
        for nurse in range(self.nb_nurses):
            travel_time = 0
            workload = 0
            for week in range(self.horizon):
                for day in range(len(self.planned_routes[0][week])):
                    t = 0
                    w = 0
                    route = self.planned_routes[nurse][week][day]
                    for pos in range(1, len(route.visit)):
                        prev = route.visit[pos - 1]
                        curr = route.visit[pos]
                        travel_time = travel_time + distance(prev.pos, curr.pos)
                        workload = workload + curr.ed - curr.st
                        t = t + distance(prev.pos, curr.pos)
                        w = w + curr.ed - curr.st

            nurse_travel_time.append(travel_time)
            nurse_workload.append(workload)
        if self.nb_nurses <= 2:
            return [sum(nurse_travel_time), sum(nurse_workload)]
        return [
            sum(nurse_travel_time),
            mean(nurse_travel_time),
            stdev(nurse_travel_time),
            sum(nurse_workload),
            mean(nurse_workload),
            stdev(nurse_workload),
        ]