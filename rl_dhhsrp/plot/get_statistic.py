import matplotlib.pyplot as plt
import matplotlib
import math
import sys
import csv
from statistics import mean 
from matplotlib import rcParams

instance_type = 'uniform'
obj = "patient"
nb_nurse = '6'
use_ch = 'False'
rl_dir = "/home/quy/Repos/RL_DHHSRP/rl_dhhsrp/run/test/"

for instance_type in ['uniform', 'cluster']:
    for arr_rate in ['150', '240', '360']:
        sba_dir = "/home/quy/Repos/Quy_11_11/Quy/2022_11_8/sba/" + instance_type 
        sba_file = sba_dir + '/result_' + instance_type + '_' + str(arr_rate) + '_' + '20' + '_' + obj + '_' + nb_nurse + '.csv';
        with open(sba_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            ch=[]
            dh=[]
            sba_ch=[]
            sba_dh=[]
            time_sba_dh=[]
            time_sba_ch=[]
            
            mean_work_load = []
            stdev_work_load = []
            mean_travel_time = []
            stdev_traveling_time = []
 
            line_count = 0
            for row in csv_reader:
                if (nb_nurse == '1'):
                    dh.append(int(row[2]))
                    ch.append(int(row[6]))
                    sba_dh.append(int(row[10]))
                    sba_ch.append(int(row[16]))
                    time_sba_dh.append(float(row[12]))
                    time_sba_ch.append(float(row[18]))
                else:
                    dh.append(int(row[6]))
                    ch.append(int(row[14]))
                    sba_dh.append(int(row[22]))
                    sba_ch.append(int(row[32]))
                    time_sba_dh.append(float(row[24]))
                    time_sba_ch.append(float(row[34]))
                line_count = line_count + 1
        dh_, ch_, sba_dh_, sba_ch_ = mean(dh), mean(ch),  mean(sba_dh), mean(sba_ch)

        rl_file = rl_dir + '/result_' + instance_type + '_' + str(arr_rate) + '_' + '1' + '_' + obj + '_' + nb_nurse + '.csv';
        with open(rl_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            ch=[]
            dh=[]
            rl=[]
            time_rl = []
            line_count = 0
            for row in csv_reader:
                if (nb_nurse == '1'):
                    rl.append(int(row[10]))
                    time_rl.append(float(row[11]))
                else:
                    rl.append(int(row[22]))
                    time_rl.append(float(row[23]))
                line_count = line_count + 1
        rl_ = mean(rl)
        
        #print((sba_dh_  - dh_)/dh_,',',(sba_ch_- dh_)/dh_, ',', (rl_- dh_)/dh_)
        print(mean(time_sba_dh),',', mean(time_sba_ch),',',mean(time_rl))
        #print(time_sba_dh)