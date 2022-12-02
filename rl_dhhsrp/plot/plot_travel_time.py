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

fig, axes = plt.subplots(nrows=2, ncols=3, sharey=True)
fig.set_size_inches(14, 7)
plt.subplots_adjust(left=0.075,
                bottom=0.05,
                right=0.99,
                top=0.93,)
medianprops = dict(linestyle='-', linewidth=2.5, color='darkgreen')
for i_index, instance_type in enumerate(['uniform', 'cluster']):
    for a_index, arr_rate in enumerate(['150', '240', '360']):
        sba_dir = "/home/quy/Repos/Experiments_result/sba/" + instance_type 
        sba_file = sba_dir + '/result_' + instance_type + '_' + str(arr_rate) + '_' + '20' + '_' + obj + '_' + nb_nurse + '.csv';
        with open(sba_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            dh_workload=[]
            ch_workload=[]
            sba_dh_workload=[]
            sba_ch_workload=[]

            dev_dh_workload=[]
            dev_ch_workload=[]
            dev_sba_dh_workload=[]
            dev_sba_ch_workload=[]

            dh_travel=[]
            ch_travel=[]
            sba_dh_travel=[]
            sba_ch_travel=[]

            dev_dh_travel=[]
            dev_ch_travel=[]
            dev_sba_dh_travel=[]
            dev_sba_ch_travel=[]
            line_count = 0
            
            for row in csv_reader:
                line_count = line_count + 1
                if (line_count <= 6):
                    continue;
                dh_workload.append(float(row[4]))
                ch_workload.append(float(row[12]))
                sba_dh_workload.append(float(row[20]))
                sba_ch_workload.append(float(row[30]))

                dev_dh_workload.append(float(row[5]))
                dev_ch_workload.append(float(row[13]))
                dev_sba_dh_workload.append(float(row[21]))
                dev_sba_ch_workload.append(float(row[31]))

                dh_travel.append(float(row[1]))
                ch_travel.append(float(row[9]))
                sba_dh_travel.append(float(row[17]))
                sba_ch_travel.append(float(row[27]))

                dev_dh_travel.append(float(row[2]))
                dev_ch_travel.append(float(row[10]))
                dev_sba_dh_travel.append(float(row[18]))
                dev_sba_ch_travel.append(float(row[28]))

                line_count = line_count + 1

        rl_file = rl_dir + '/result_' + instance_type + '_' + str(arr_rate) + '_' + '1' + '_' + obj + '_' + nb_nurse + '.csv';
        with open(rl_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            rl_workload=[]
            dev_rl_workload=[]
            rl_travel=[]
            dev_rl_travel=[]
            time_rl = []
            line_count = 0
            for row in csv_reader:
                line_count = line_count + 1
                if (line_count <= 6):
                    continue;
                rl_workload.append(float(row[20]))
                dev_rl_workload.append(float(row[21]))
                rl_travel.append(float(row[17]))
                dev_rl_travel.append(float(row[18]))

        print(len(rl_workload), len(dev_ch_travel))
        stats = [{
            "label": 'DH',  # not required
            "mean":  mean(dh_travel),  # not required
            "whislo": mean(dh_travel) - mean(dev_dh_travel),  # required
            "whishi": mean(dh_travel) + mean(dev_dh_travel),  # required
            "med": mean(dh_travel),
            "q1": mean(dh_travel),
            "q3": mean(dh_travel),
            },
            {
            "label": 'CH',  # not required
            "mean":  mean(ch_travel),  # not required
            "whislo": mean(ch_travel) - mean(dev_ch_travel),  # required
            "whishi": mean(ch_travel) + mean(dev_ch_travel),  # required
            "med": mean(ch_travel),
            "q1": mean(ch_travel),
            "q3": mean(ch_travel),
            },
            {
            "label": 'SBA-DH',  # not required
            "mean":  mean(sba_dh_travel),  # not required
            "whislo": mean(sba_dh_travel) - mean(dev_sba_dh_travel),  # required
            "whishi": mean(sba_dh_travel) + mean(dev_sba_dh_travel),  # required
            "med": mean(sba_dh_travel),
            "q1": mean(sba_dh_travel),
            "q3": mean(sba_dh_travel),
            },
            {    
            "label": 'SBA-CH',  # not required
            "mean":  mean(sba_ch_travel),  # not required
            "whislo": mean(sba_ch_travel) - mean(dev_sba_ch_travel),  # required
            "whishi": mean(sba_ch_travel) + mean(dev_sba_ch_travel),  # required
            "med": mean(sba_ch_travel),
            "q1": mean(sba_ch_travel),
            "q3": mean(sba_ch_travel),
            },
            {
            "label": 'DDQN',  # not required
            "mean":  mean(rl_travel),  # not required
            "whislo": mean(rl_travel) - mean(dev_rl_travel),  # required
            "whishi": mean(rl_travel) + mean(dev_rl_travel),  # required
            "med": mean(rl_travel),
            "q1": mean(rl_travel),
            "q3": mean(rl_travel),
            },
            
        ]
        #print(mean(rl_workload), mean(sba_ch_workload), mean(sba_dh_workload), mean(dh_workload))
        axes[i_index][a_index].bxp(stats,showfliers = False, showbox = False, medianprops=medianprops)
        axes[i_index][a_index].tick_params(axis='x', labelsize=13)
axes[0][0].set_title("arrival rate = 150", fontsize=15)
axes[0][1].set_title("arrival rate = 240", fontsize=15)
axes[0][2].set_title("arrival rate = 360", fontsize=15)

axes[0][0].set_ylabel('Uniform',labelpad = 15, fontsize=18)
axes[1][0].set_ylabel('Cluster',labelpad = 15, fontsize=18)

plt.savefig("travel.jpg", pad_inches=0, dpi=700)
plt.show()