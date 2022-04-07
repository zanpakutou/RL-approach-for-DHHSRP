from patient_generator import PatientGenerator 
from enviroment import Enviroment
#p = PatientGenerator()
#for i in range(0,100):
#    p.generate_instance("instances/" + str(i) + ".in")

env = Enviroment()
env.make("instances/1.in", "instances/context.in")
print(env.qual)
print(env.nurse_depot)

print(env.nb_weeks)
print(env.day_per_week)
print(env.requests)
