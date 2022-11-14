import sys, os

sys.path.append(os.path.abspath(os.path.join("..")))

from enviroment.patient_request import PatientRequest, Request
from enviroment.schedule import Schedule
from agent.feature_engineering import FeatureExtractor
from utils.utils import mean, stdev
from greedy.greedy import SBA
from enviroment.patient_generator import PatientGenerator
from config.config import Config
import matplotlib.pyplot as plt

from gym.wrappers import TimeLimit
import numpy as np

instance_type = 'uniform'
arr_rate = '240'
obj = "patient"
nb_nurse = '1'
use_ch = 'False'
sba_dir = "/home/quy/Repos/Quy_11_11/Quy/2022_11_8/sba/uniform/"
run_dir = '/result_' + instance_type + '_' + str(arr_rate) + '_' + '20' + '_' + obj + '_' + nb_nurse + '_DH_.npy';
file_dir = sba_dir + run_dir

acceptance_dir = np.load(file_dir, allow_pickle=True)
print(acceptance_dir)