from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from utils import utils
from enviroment.patient_generator import PatientGenerator

import numpy as np

p = PatientGenerator()
for i in range(0,610):
    print("generate ", i)
    p.generate_instance("enviroment/instances/new_instances/cluster/150/" + str(i) + ".in")