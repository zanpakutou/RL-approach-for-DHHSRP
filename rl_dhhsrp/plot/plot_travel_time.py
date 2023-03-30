import matplotlib.pyplot as plt
import matplotlib
import math
import sys
import csv
from statistics import mean 
from matplotlib import rcParams

instance_type = 'uniform'
obj = "visit"
nb_nurse = '3'
use_ch = 'False'
rl_dir = "/home/quy/Repos/Experiments_result/2023_02_28/2023_02_28"

fig, axes = plt.subplots(nrows=3, ncols=3, sharey=True)
fig.set_size_inches(14, 7)
plt.subplots_adjust(left=0.075,
                bottom=0.05,
                right=0.99,
                top=0.93,)
medianprops = dict(linestyle='-', linewidth=2.5, color='darkgreen')
for i_index, instance_type in enumerate(["U", "C", "UC"]):
    for a_index, arr_rate in enumerate(['150', '255', '340']):
        sba_dir = "/home/quy/Repos/Experiments_result/2023_02_28/2023_02_28"
        sba_file = sba_dir + '/result_' + instance_type + '_' + str(arr_rate) + '_' + '75' + '_' + obj + '_' + nb_nurse + '_.csv';
        with open(sba_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            dh_workload=[]
            ch_workload=[]
            sba_dh_workload=[]
            sba_ch_workload=[]
            rl_workload = []

            dev_dh_workload=[]
            dev_ch_workload=[]
            dev_sba_dh_workload=[]
            dev_sba_ch_workload=[]
            dev_rl_workload = []

            dh_travel=[]
            ch_travel=[]
            sba_dh_travel=[]
            sba_ch_travel=[]
            rl_travel = []

            dev_dh_travel=[]
            dev_ch_travel=[]
            dev_sba_dh_travel=[]
            dev_sba_ch_travel=[]
            dev_rl_travel=[]

            line_count = 0
            
            for row in csv.DictReader(csv_file):
                line_count = line_count + 1

                dh_workload.append(float(row['dh_sum_service'])/float(row['dh_obj']))
                ch_workload.append(float(row['ch_sum_service'])/float(row['ch_obj']))
                sba_dh_workload.append(float(row['sba_dh_sum_service'])/float(row['sba_dh_obj']))
                sba_ch_workload.append(float(row['sba_ch_sum_service'])/float(row['sba_ch_obj']))
                rl_workload.append(float(row['rl_sum_service'])/float(row['rl_obj']))

                dev_dh_workload.append(float(row['dh_dev_service'])/float(row['dh_obj']))
                dev_ch_workload.append(float(row['ch_dev_service'])/float(row['dh_obj']))
                dev_sba_dh_workload.append(float(row['sba_dh_dev_sercive'])/float(row['dh_obj']))
                dev_sba_ch_workload.append(float(row['sba_ch_dev_service'])/float(row['dh_obj']))
                dev_rl_workload.append(float(row['rl_dev_service'])/float(row['dh_obj']))

                dh_travel.append(float(row['dh_sum_travel'])/float(row['dh_obj']))
                ch_travel.append(float(row['ch_sum_travel'])/float(row['ch_obj']))
                sba_dh_travel.append(float(row['sba_dh_sum_travel'])/float(row['sba_dh_obj']))
                sba_ch_travel.append(float(row['sba_ch_sum_travel'])/float(row['sba_ch_obj']))
                rl_travel.append(float(row['rl_sum_travel'])/float(row['rl_obj']))

                dev_dh_travel.append(float(row['dh_dev_travel'])/float(row['dh_obj']))
                dev_ch_travel.append(float(row['ch_dev_travel'])/float(row['dh_obj']))
                dev_sba_dh_travel.append(float(row['sba_dh_dev_travel'])/float(row['dh_obj']))
                dev_sba_ch_travel.append(float(row['sba_ch_dev_travel'])/float(row['dh_obj']))
                dev_rl_travel.append(float(row['rl_dev_travel'])/float(row['dh_obj']))

                line_count = line_count + 1

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
        print(mean(rl_travel), mean(sba_ch_travel), mean(sba_dh_travel), mean(dh_travel))
        axes[i_index][a_index].bxp(stats,showfliers = False, showbox = False, medianprops=medianprops)
        axes[i_index][a_index].tick_params(axis='x', labelsize=13)

axes[0][0].set_title("high", fontsize=15)
axes[0][1].set_title("medium", fontsize=15)
axes[0][2].set_title("low", fontsize=15)

axes[0][0].set_ylabel('U',labelpad = 15, fontsize=18)
axes[1][0].set_ylabel('C',labelpad = 15, fontsize=18)
axes[2][0].set_ylabel('UC',labelpad = 15, fontsize=18)

plt.savefig("travel_3.jpg", pad_inches=0, dpi=700)
plt.show()