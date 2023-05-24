from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from utils.utils import MAX_VAL
from enviroment.patient_generator import PatientGenerator
from copy import copy, deepcopy


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


class SBA:
    def __init__(
        self,
        schedule,
        patient_generator,
        capacity_heur=False,
        obj="patient",
        num_scen=15,
        scen_size=20,
    ):
        self.sched = schedule
        self.pat_generator = patient_generator
        self.nb_scenarios = num_scen
        self.avg_request = scen_size
        self.capacity_heur = capacity_heur
        self.obj = obj

    def act(self, request, current_time):
        is_accepted = False
        accept_time = []
        for scen in range(self.nb_scenarios):
            schedulue = deepcopy(self.sched)
            # Generate a scenario
            request.current_time = current_time
            requests = [request]
            c_week, c_day, c_hour = current_time
            for no in range(self.avg_request):
                (
                    delay_time,
                    require_time,
                    require_skill,
                    location,
                ) = self.pat_generator.generate_patient()
                c_hour = c_hour + int(delay_time)
        
                while (c_hour >= 990):
                    c_hour = c_hour - 510
                    c_day = c_day + 1
                    if (c_day >= 5):
                        c_day = 0
                        c_week = c_week + 1
                if (c_hour < 990):
                    requests.append(
                        Request(
                            current_time=(c_week, c_day, c_hour),
                            require_time=require_time,
                            require_skill=require_skill,
                            location=location,
                        )
                    )

            # Cheapest insertion heuristic
            is_insertable = True
            while is_insertable:
                min_cost = MAX_VAL
                is_insertable = False
                index = 0
                best_index = -1
                insertions = [0] * len(requests)
                for _request in requests:
                    (valid, min_cost_insertion) = schedulue.check_feasible(
                        _request,
                        _request.current_time,
                        weekly_deadline=True,
                        capacity_heur=self.capacity_heur,
                    )

                    if valid == True:
                        heuristic = min_cost_insertion[0][0]
                        if self.obj == "visit":
                            week, day, hour = (
                                _request.require_time[0],
                                _request.require_time[1],
                                _request.require_time[2],
                            )
                            heuristic = heuristic / (week * day)

                        if heuristic < min_cost:
                            min_cost = heuristic
                            best_index = index
                        is_insertable = True
                    insertions[index] = min_cost_insertion
                    index = index + 1

                if is_insertable:
                    if best_index == 0:
                        is_accepted = True
                        accept_time.append(insertions[0][1])
                        break

                    schedulue.accept_checked_request(
                        requests[best_index],
                        insertions[best_index],
                        weekly_deadline=True,
                    )
                    requests.remove(requests[best_index])

        if is_accepted:
            return (1, most_frequent(accept_time))
        else:
            return (0, 0)
