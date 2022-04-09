from patient_generator import PatientGenerator 
from enviroment import Enviroment
from schedule import Schedule
from feature_engineering import FeatureExtractor
import utils 
#p = PatientGenerator()
#for i in range(0,100):
#    p.generate_instance("instances/" + str(i) + ".in")

env = Enviroment()
env.make("instances/1.in", "instances/context.in")

sched = Schedule(env)

feature_extractor = FeatureExtractor(env, sched)

requests = env.get_request()
current_time = (0, 0, 20)

print(feature_extractor.get_feature(requests[0][0][0], current_time)) 
sched.accept_request(requests[0][0][0], current_time)
print(feature_extractor.get_feature(requests[0][0][0], current_time)) 