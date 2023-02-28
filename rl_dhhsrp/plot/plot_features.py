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
nb_nurse = '6'
use_ch = 'False'
parent_dir = "/home/quy/Repos/Experiments_result/Result_nurse_action/6_nurse/"
feature_dir = "/home/quy/Repos/RL_DHHSRP/rl_dhhsrp/run/stable_baselines/feature_test/"

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
            if (epoch[0] < 2e5):
                continue
            content_list.append(epoch)
            if (content_list[-1][0] > 2e7):
                break
    my_file.close()
    return content_list

fig, ax = plt.subplots()
fig.set_size_inches(12, 9)
#fig.tight_layout()

arr_dict={0:'150', 1:'240', 2:'360'}
instance_types={0:'uniform', 1:'cluster'}
#rcParams['axes.titlepad'] = 20 parent_dir
#fig.suptitle(instance_type + ' location distribution', fontsize = 22)
plt.subplots_adjust(left=0.05,
                bottom=0.12,
                right=0.98,
                top=0.89,
                wspace=0.3,
                hspace=0.5)

arr_rate = "150"
instance_type = 'uniform'

run_dir = parent_dir + "n" + arr_rate + "_" + instance_type + "_0.995_2x256/"
full = get_ddqn_test_content(run_dir)
it_full = [i[0] for i in full]
obj_full = [i[1] for i in full]

run_dir = feature_dir  + "no_request/n" + arr_rate + "_" + instance_type + "_0.995_2x256_6/"
no_request = get_ddqn_test_content(run_dir)
it_req = [i[0] for i in no_request]
obj_req = [i[1] for i in no_request]

run_dir = feature_dir + "no_nurse2/"
no_nurse = get_ddqn_test_content(run_dir)
it_nurse = [i[0] for i in no_nurse]
obj_nurse = [i[1] for i in no_nurse]
-0.2
run_dir = feature_dir + "no_decision2/"
decision = get_ddqn_test_content(run_dir)
it_decision = [i[0] for i in decision]
obj_decision = [i[1] for i in decision]

ax.plot(it_full, obj_full, color = "red", label= "Full features")
ax.plot(it_req, obj_req, color = "royalblue", label= "No patient's request information")
ax.plot(it_nurse, obj_nurse, color = "goldenrod", label= "No nurse resources information")
ax.plot(it_decision, obj_decision, color = "seagreen", label= "no additional information")

ax.tick_params(axis='both', which='major', width=3, length= 15)
ax.tick_params(axis='both', which='minor', width=3, length= 8)
#ax.set_title("arrival rate = " + arr_rate, fontsize=18)

#plt.title('Avg gaps in log scale to reference solutions per iteration', pad=0, fontsize=26)
plt.legend(loc='upper center', borderaxespad=0., bbox_to_anchor=(0.48, 1.13), fontsize=19, ncol=2, fancybox=True)
ax.set_xlabel('Number of timesteps',fontsize=17, labelpad=20)
#ax.set_xlabel('Number of timesteps',fontsize=16)
'''ax[0][0].set_ylabel('150',fontsize=18)
ax[1][0].set_ylabel('240',fontsize=18)
ax[2][0].set_ylabel('360',fontsize=18)
ax[2][0].set_xlabel('Uniform',labelpad = 15, fontsize=21)
ax[2][1].set_xlabel('Cluster',labelpad = 15, fontsize=21)'''

plt.show()
fig.savefig("feature_test.jpg")