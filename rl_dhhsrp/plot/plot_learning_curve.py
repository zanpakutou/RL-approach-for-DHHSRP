import matplotlib.pyplot as plt
import matplotlib
import math
import sys
import csv
from statistics import mean 
from matplotlib import rcParams

instance_type = 'U'
arr_rate = '360'
obj = "visit"
nb_nurse = '3'
use_ch = 'False'
parent_dir = "../run/stable_baselines/23_02_10/"
sba_dir = "../run/test/result_rl_visit/"
run_dir = "/home/quy/Repos/2022_11_27/" + instance_type + "/DQN_" + arr_rate + "/"

def get_ddqn_test_content(dir):
    _dir = dir + "/TEST"
    my_file = open(_dir, "r")
    content = my_file.read()

    content_list = []
    for line in content.split("\n"):
        if (len(line) <= 0):
            continue
        epoch = []
        for item in line.split(" "):
            epoch.append(float(item))
        if (len(epoch) >= 2):
            content_list.append(epoch)
            #if (content_list[-1][0] > 8e6):
            #    break
    print(dir)
    content_list.pop(0)
    content_list.pop(0)
    content_list.pop(0)
    content_list.pop(0)
    content_list.pop(0)
    content_list.pop(0)
    content_list.pop(0)
    content_list.pop(0)
    content_list.pop(0)
    content_list.pop(0)
    my_file.close()
    return content_list

def get_sba_result(dir, nb_scen = '20'):
    _dir = dir + '/result_' + instance_type + '_' + str(arr_rate) + '_' + nb_scen + '_' + obj + '_' + nb_nurse + '_.csv';
    print(_dir)
    with open(_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        ch=[]
        dh_o=[]
        dh=[]
        sba_ch=[]
        sba_dh=[]
        line_count = 0
        for row in csv.DictReader(csv_file):
            dh_o.append(int(row['dho_obj']))
            dh.append(int(row['dh_obj']))
            ch.append(int(row['ch_obj']))
            sba_dh.append(int(row['sba_dh_obj']))
            sba_ch.append(int(row['sba_ch_obj']))

            line_count = line_count + 1
            if (line_count >= 5):
                break;
        if (line_count < 5):
            print("Not enough lines in csv")
            return 500, 500, 500, 500, 500

    return mean(dh_o), mean(dh), mean(ch),  mean(sba_dh), mean(sba_ch)


fig, ax = plt.subplots(3, 2)
fig.set_size_inches(10, 12)
#fig.tight_layout()

arr_dict={0:'340', 1:'255', 2:'150'}
instance_types={0:'U', 1:'C'}

#rcParams['axes.titlepad'] = 20 
#fig.suptitle(instance_type + ' location distribution', fontsize = 22)
plt.subplots_adjust(left=0.07,
                bottom=0.1,
                right=0.98,
                top=0.90,
                wspace=0.3,
                hspace=0.5)

for i in range(3):
    arr_rate = arr_dict[i]
    for j in range(2):
        instance_type = instance_types[j]
        run_dir = parent_dir + "/" + instance_type  + "_" + arr_rate + "_" + nb_nurse + "/"
        content_list = get_ddqn_test_content(run_dir)
        dh_o, dh, ch, sba_dh, sba_ch = get_sba_result(dir=sba_dir, nb_scen = '75')
        it = [i[0] for i in content_list]
        obj_ = [i[1] for i in content_list]

        ax[i][j].axhline(y=dh, xmin=0.0, xmax=1.0, color='blue', label="DH")
        ax[i][j].axhline(y=ch, xmin=0.0, xmax=1.0, color='green', label="CH")
        ax[i][j].axhline(y=sba_ch, xmin=0.0, xmax=1.0, color='purple', label="SBA-CH")
        ax[i][j].axhline(y=sba_dh, xmin=0.0, xmax=1.0, color='orange', label="SBA-DH")
        ax[i][j].axhline(y=dh_o, xmin=0.0, xmax=1.0, color='black', label="DHO")
        ax[i][j].plot(it, obj_, color = "red", label= "DDQN")

        ax[i][j].tick_params(axis='both', which='major', width=3, length= 15)
        ax[i][j].tick_params(axis='both', which='minor', width=3, length= 8)
        ax[i][j].set_title("arrival rate = " + arr_rate, fontsize=18)

#plt.title('Avg gaps in log scale to reference solutions per iteration', pad=0, fontsize=26)
plt.legend(loc='upper center', borderaxespad=0., bbox_to_anchor=(-0.2, 4.5), fontsize=16, ncol=6, fancybox=True)
'''ax[0][0].set_ylabel('150',fontsize=18)
ax[1][0].set_ylabel('240',fontsize=18)
ax[2][0].set_ylabel('360',fontsize=18)'''
ax[2][0].set_xlabel('Uniform',labelpad = 15, fontsize=21)
ax[2][1].set_xlabel('Cluster',labelpad = 15, fontsize=21)

plt.show()
fig.savefig("test_policy" + str(nb_nurse) + ".jpg")