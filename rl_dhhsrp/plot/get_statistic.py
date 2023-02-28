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
rl_dir = "/home/quy/Repos/Experiments_result/test_rl/"
dev_dh = []
dev_ch = []
for nb_nurse in ['1', '6']:
    for instance_type in ['uniform', 'cluster']:
        for arr_rate in ['150', '240', '360']:
            sba_dir = "/home/quy/Repos/Experiments_result/sba/" + instance_type 
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

                dh_rate = []
                ch_rate = []
                sba_dh_rate = []
                sba_ch_rate = []
                line_count = 0
                for row in csv_reader:
                    line_count = line_count + 1
                    if (line_count <= 6):
                        continue;
                    if (nb_nurse == '1'):
                        dh.append(int(row[2]))
                        ch.append(int(row[6]))
                        sba_dh.append(int(row[10]))
                        sba_ch.append(int(row[16]))
                        time_sba_dh.append(float(row[12]))
                        time_sba_ch.append(float(row[18]))

                        dh_rate.append(float(row[3]))
                        ch_rate.append(float(row[7]))
                        sba_dh_rate.append(float(row[11]))
                        sba_ch_rate.append(float(row[17]))
                    else:
                        dh.append(int(row[6]))
                        ch.append(int(row[14]))
                        sba_dh.append(int(row[22]))
                        sba_ch.append(int(row[32]))
                        time_sba_dh.append(float(row[24]))
                        time_sba_ch.append(float(row[34]))

                        dh_rate.append(float(row[7]))
                        ch_rate.append(float(row[15]))
                        sba_dh_rate.append(float(row[23]))
                        sba_ch_rate.append(float(row[33]))
            #print(len(time_sba_ch))
            #print(nb_nurse, instance_type, arr_rate, line_count)
            dh_, ch_, sba_dh_, sba_ch_ = mean(dh), mean(ch),  mean(sba_dh), mean(sba_ch)
            rl_file = rl_dir + '/result_' + instance_type + '_' + str(arr_rate) + '_' + '1' + '_' + obj + '_' + nb_nurse + '.csv';
            line_count = 0
            with open(rl_file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                ch=[]
                dh=[]
                rl=[]
                rl_rate = []
                time_rl = []
                for row in csv_reader:
                    line_count = line_count + 1
                    if (line_count <= 6):
                        continue;
                    if (nb_nurse == '1'):
                        rl.append(int(row[10]))
                        rl_rate.append(float(row[11]))
                        time_rl.append(float(row[12]))
                        
                    else:
                        rl.append(int(row[22]))
                        rl_rate.append(float(row[23]))
                        time_rl.append(float(row[24]))
                    
            rl_ = mean(rl)
            #print(len(time_rl))
            #print((sba_dh_  - dh_)/dh_ * 100,',',(sba_ch_- dh_)/dh_ * 100, ',', (rl_- dh_)/dh_ * 100) 
            #print(mean(time_sba_dh),',', mean(time_sba_ch),',',mean(time_rl))
            print(mean(dh_rate),',', (ch_  - dh_)/dh_ * 100,',', mean(ch_rate),',', (sba_dh_  - dh_)/dh_ * 100,',',mean(sba_dh_rate),',',mean(time_sba_dh),',', (sba_ch_- dh_)/dh_ * 100,\
                 ',',mean(sba_dh_rate),',', mean(time_sba_ch),',',(rl_- dh_)/dh_ * 100,',', mean(rl_rate), ',', mean(time_rl))

            
            for i in range(len(time_rl)):
                dev_dh.append((rl[i] - sba_dh[i])/sba_dh[i])
                dev_ch.append((rl[i] - sba_ch[i])/sba_ch[i]) 
                print(dev_dh[i], dev_ch[i], i)
print(">>", mean(dev_dh) * 100, mean(dev_ch) * 100)