from patient_generator import PatientGenerator 
from enviroment import Enviroment
from schedule import Schedule
from feature_engineering import FeatureExtractor
import utils

p = PatientGenerator()
for i in range(0,100):
    p.generate_instance("instances/" + str(i) + ".in")

"""env = Enviroment()
env.make("instances/train/52.in", "instances/context.in")

sched = Schedule(env)

feature_extractor = FeatureExtractor(env, sched)

requests = env.get_request()

ans = 0

for week in range(env.nb_weeks):
    for day in range(env.day_per_week):
        for request in requests[week][day]:
            current_time = (week, day, request.current_time)
            (valid, min_cost_insertion) = sched.check_feasible(request, current_time)
            if valid == True :
                ans = ans + 1
                print(ans, feature_extractor.get_feature(request, current_time))
                sched.accept_request(request, current_time)
"""