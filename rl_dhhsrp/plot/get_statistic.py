import matplotlib.pyplot as plt
import matplotlib
import math
import sys
import csv
from statistics import mean 
from matplotlib import rcParams

instance_type = 'uniform'
obj = "visit"
nb_nurse = '6'
use_ch = 'False'
rl_dir = "/home/quy/Repos/Experiments_result/test_rl/"

print("dho,dh,ch,sba_dh,sba_dh/valid,sba_dh time,sba_ch,sba_ch/valid,sba_ch time,rl,rl/valid,rl time")
for nb_nurse in ['12']:
    for instance_type in ['U']:
        for arr_rate in ['255', '100', '60']:
            sba_dir = "/home/quy/Repos/Experiments_result/2023_02_28/2023_02_28/" 
            sba_file = sba_dir + '/result_' + instance_type + '_' + str(arr_rate) + '_' + '75' + '_' + obj + '_' + nb_nurse + '.csv';
            with open(sba_file) as csv_file:
                ch=[] 
                dh=[]
                dh_o = []
                sba_ch=[]
                sba_dh=[]
                rl = []
                time_sba_dh=[]
                time_sba_ch=[]
                
                mean_work_load = []
                stdev_work_load = []
                mean_travel_time = []
                stdev_traveling_time = []

                dh_o_travel = []
                dh_travel = []
                ch_travel = []
                sba_dh_travel = []
                sba_ch_travel = []
                rl_travel = []

                dh_o_wload = []
                dh_wload = []
                ch_wload = []
                sba_dh_wload = []
                sba_ch_wload = []
                rl_wload = []

                dh_o_accept = []
                dh_accept = []
                ch_accept = []
                sba_dh_accept = []
                sba_ch_accept = []
                rl_accept = []

                dh_o_rate = []
                dh_rate = []
                ch_rate = []
                sba_dh_rate = []
                sba_ch_rate = []
                rl_rate = []

                total_req = 0

                time_rl = []
                time_sba_dh = []
                time_sba_ch = []

                sba_dh_rate_per_valid = []
                sba_ch_rate_per_valid = []
                rl_rate_per_valid = []

                sba_dh_rate_per_valid = []
                sba_ch_rate_per_valid = []
                rl_rate_per_valid = []

                sba_dh_valid = []
                sba_ch_valid = []
                rl_valid = []

                line_count = 0
                for row in csv.DictReader(csv_file):
                    dh_o.append(int(row['dho_obj']))
                    dh.append(int(row['dh_obj']))
                    ch.append(int(row['ch_obj']))
                    sba_dh.append(int(row['sba_dh_obj']))
                    sba_ch.append(int(row['sba_ch_obj']))
                    rl.append(int(row['rl_obj']))

                    dh_o_travel.append(float(row['dho_sum_travel']))
                    dh_travel.append(float(row['dh_sum_travel']))
                    ch_travel.append(float(row['ch_sum_travel']))
                    sba_dh_travel.append(float(row['sba_dh_sum_travel']))
                    sba_ch_travel.append(float(row['sba_ch_sum_travel']))
                    rl_travel.append(float(row['rl_sum_travel']))

                    dh_o_accept.append(int(row['dho_visit']))
                    dh_accept.append(int(row['dh_visit']))
                    ch_accept.append(int(row['ch_visit']))
                    sba_dh_accept.append(int(row['sba_dh_visit']))
                    sba_ch_accept.append(int(row['sba_ch_visit']))
                    rl_accept.append(int(row['rl_visit']))

                    total_req = int(row['rl_rate'])

                    dh_o_wload.append(float(row['dho_sum_workload']))
                    dh_wload.append(float(row['dh_sum_workload']))
                    ch_wload.append(float(row['ch_sum_workload']))
                    sba_dh_wload.append(float(row['sba_dh_sum_workload']))
                    sba_ch_wload.append(float(row['sba_ch_sum_workload']))
                    rl_wload.append(float(row['rl_sum_workload']))

                    dh_o_rate.append((dh_o_accept[-1]/ total_req))
                    dh_rate.append(dh_accept[-1]/total_req)
                    ch_rate.append(ch_accept[-1]/total_req) 
                    sba_dh_rate.append(sba_dh_accept[-1]/total_req)
                    sba_ch_rate.append(sba_ch_accept[-1]/total_req)
                    rl_rate.append(rl_accept[-1]/total_req) 

                    sba_dh_valid.append(int(row['sba_dh_valid']))
                    sba_ch_valid.append(int(row['sba_ch_valid']))
                    rl_valid.append(int(row['rl_valid'])) 
                    
                    sba_dh_rate_per_valid.append(sba_dh_accept[-1]/sba_dh_valid[-1])
                    sba_ch_rate_per_valid.append(sba_ch_accept[-1]/sba_ch_valid[-1])
                    rl_rate_per_valid.append(rl_accept[-1]/rl_valid[-1])
                    
                    time_sba_dh.append(float(row['sba_dh_time']))
                    time_sba_ch.append(float(row['sba_ch_time']))
                    time_rl.append(float(row['rl_time']))

                print(mean(dh_o_rate)*100,',', mean(dh_rate)*100,',', mean(ch_rate)*100,',', mean(sba_dh_rate) * 100,',', mean(sba_dh_rate_per_valid) * 100,',',mean(time_sba_dh)\
                    ,',',mean(sba_ch_rate) * 100,',',mean(sba_ch_rate_per_valid) * 100,',',mean(time_sba_ch),',',mean(rl_rate) * 100,\
                    ',',mean(rl_rate_per_valid) * 100,',',mean(time_rl))
