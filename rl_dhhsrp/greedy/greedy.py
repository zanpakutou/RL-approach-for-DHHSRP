from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from enviroment import utils
from enviroment.patient_generator import PatientGenerator
from copy import copy, deepcopy

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i

    return num
    
class SBA:
    def __init__(self, schedule, patient_generator):
        self.sched = schedule
        self.pat_generator = patient_generator
        self.nb_scenarios = 10
        self.avg_request_p_day = 12 * 5
    def act(self, request, current_time):
        is_accepted = False
        accept_time = []
        for scen in range(self.nb_scenarios):
            schedulue = deepcopy(self.sched)
            #Generate a scenario
            requests = [request]
            for no in range(self.avg_request_p_day):
                (delay_time, require_time, require_skill, location) = self.pat_generator.generate_patient()
                requests.append(Request(current_time = current_time, require_time = require_time, require_skill = require_skill, location = location))
            #Cheapest insertion heuristic
            is_insertable = True
            current_time = current_time
            while(is_insertable):
                min_cost = 100000000
                is_insertable = False
                index = 0
                best_index = -1
                for _request in requests:
                    (valid, min_cost_insertion) = schedulue.check_feasible(_request, current_time)
                    if (valid == True):
                        if (min_cost_insertion[0] < min_cost):
                            min_cost = min_cost_insertion[0]
                            best_index = index
                        is_insertable = True
                    index = index + 1

                if (is_insertable):
                    if (best_index == 0):
                        (valid, min_cost_insertion) = schedulue.check_feasible(request, current_time)
                        if (valid == True):
                            is_accepted = True
                            accept_time.append(min_cost_insertion[1])
                        else:
                            print("???", min_cost_insertion)
                        break
                        
                    schedulue.accept_request(requests[best_index], current_time)
                    requests.remove(requests[best_index])

        if (is_accepted):
            return (1,most_frequent(accept_time))
        else:
            return (0, 0)
                
                