from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from utils import utils
from enviroment.patient_generator import PatientGenerator, c_location_generator, u_location_generator, uc_location_generator
import numpy as np
import os

for nb_nurse in [24]:
    for arr in [40]:
        os.makedirs("enviroment/instances/" + str(nb_nurse) + "_nurse/U/" + str(arr), exist_ok=True)
        os.makedirs("enviroment/instances/" + str(nb_nurse) + "_nurse/C/" + str(arr), exist_ok=True)
        os.makedirs("enviroment/instances/" + str(nb_nurse) + "_nurse/UC/" + str(arr), exist_ok=True)
        p = PatientGenerator(arrival_rate = arr, loc_gen = u_location_generator)
        for i in range(0,1000):
            print("generate ", i)
            p.generate_instance("enviroment/instances/" + str(nb_nurse) + "_nurse/U/" + str(arr) + "/" + str(i) + ".in")
        p = PatientGenerator(arrival_rate = arr, loc_gen = c_location_generator)
        for i in range(0,1000):
            print("generate ", i)
            p.generate_instance("enviroment/instances/" + str(nb_nurse) + "_nurse/C/" + str(arr) + "/" + str(i) + ".in")
        p = PatientGenerator(arrival_rate = arr, loc_gen = uc_location_generator)
        for i in range(0,1000):
            print("generate ", i)
            p.generate_instance("enviroment/instances/" + str(nb_nurse) + "_nurse/UC/" + str(arr) + "/" + str(i) + ".in")
