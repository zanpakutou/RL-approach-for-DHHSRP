from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from utils import utils
from enviroment.patient_generator import PatientGenerator, cluster_location_generator, uniform_location_generator
import numpy as np

p = PatientGenerator(arrival_rate = 240, loc_gen = cluster_location_generator)
for i in range(0,1000):
    print("generate ", i)
    p.generate_instance("enviroment/instances/new_instances/cluster/240/" + str(i) + ".in")