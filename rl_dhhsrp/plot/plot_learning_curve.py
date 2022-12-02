import matplotlib.pyplot as plt
import matplotlib
import math
import sys
import csv
from statistics import mean 
from matplotlib import rcParams

instance_type = 'uniform'
arr_rate = '240'
obj = "patient"
nb_nurse = '1'
use_ch = 'False'
parent_dir = "/home/quy/Repos/Experiments_result/Result_nurse_action/1_nurse/"#"/home/quy/Repos/Quy_11_17/2022_11_14/ddqn/"
sba_dir = "/home/quy/Repos/Experiments_result/Quy_11_11/Quy/2022_11_8/sba/"
#run_dir = instance_type + "-" + arr_rate + "-" + nb_nurse + "-" + obj + "-" + use_ch + "-15000000-2048-0.995" 
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
            if (content_list[-1][0] > 8e6):
                break
    content_list.pop(0)
    my_file.close()
    return content_list

def get_sba_result(dir, nb_scen = '20'):
    _dir = dir + '/result_' + instance_type + '_' + str(arr_rate) + '_' + nb_scen + '_' + obj + '_' + nb_nurse + '.csv';
    with open(_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        ch=[]
        dh=[]
        sba_ch=[]
        sba_dh=[]
        line_count = 0
        for row in csv_reader:
            if (nb_nurse == '1'):
                dh.append(int(row[2]))
                ch.append(int(row[6]))
                sba_dh.append(int(row[10]))
                sba_ch.append(int(row[16]))
            else:
                dh.append(int(row[6]))
                ch.append(int(row[14]))
                sba_dh.append(int(row[22]))
                sba_ch.append(int(row[32]))
            line_count = line_count + 1
            if (line_count >= 5):
                break;
        if (line_count < 5):
            print("Not enough lines in csv")
            return 500, 500, 500, 500

    return mean(dh), mean(ch),  mean(sba_dh), mean(sba_ch)


fig, ax = plt.subplots(3, 2)
fig.set_size_inches(10, 12)
#fig.tight_layout()

arr_dict={0:'150', 1:'240', 2:'360'}
instance_types={0:'uniform', 1:'cluster'}
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
        run_dir = instance_type + "-" + arr_rate + "-" + nb_nurse + "-" + obj + "-" + use_ch + "-15000000-2048-0.995" 
        run_dir = parent_dir + "n" + arr_rate + "_" + instance_type + "_0.995_2x256/"
        content_list = get_ddqn_test_content(run_dir)
        dh, ch, sba_dh, sba_ch = get_sba_result(dir=sba_dir + instance_type, nb_scen = '20')
        it = [i[0] for i in content_list]
        obj_ = [i[1] for i in content_list]

        ax[i][j].axhline(y=dh, xmin=0.0, xmax=1.0, color='blue', label="DH")
        ax[i][j].axhline(y=ch, xmin=0.0, xmax=1.0, color='green', label="CH")
        ax[i][j].axhline(y=sba_ch, xmin=0.0, xmax=1.0, color='purple', label="SBA-CH")
        ax[i][j].axhline(y=sba_dh, xmin=0.0, xmax=1.0, color='orange', label="SBA-DH")
        ax[i][j].plot(it, obj_, color = "red", label= "DDQN")

        ax[i][j].tick_params(axis='both', which='major', width=3, length= 15)
        ax[i][j].tick_params(axis='both', which='minor', width=3, length= 8)
        ax[i][j].set_title("arrival rate = " + arr_rate, fontsize=18)

#plt.title('Avg gaps in log scale to reference solutions per iteration', pad=0, fontsize=26)
plt.legend(loc='upper center', borderaxespad=0., bbox_to_anchor=(-0.2, 4.5), fontsize=16, ncol=5, fancybox=True)
'''ax[0][0].set_ylabel('150',fontsize=18)
ax[1][0].set_ylabel('240',fontsize=18)
ax[2][0].set_ylabel('360',fontsize=18)'''
ax[2][0].set_xlabel('Uniform',labelpad = 15, fontsize=21)
ax[2][1].set_xlabel('Cluster',labelpad = 15, fontsize=21)

plt.show()
fig.savefig("test_policy.jpg")