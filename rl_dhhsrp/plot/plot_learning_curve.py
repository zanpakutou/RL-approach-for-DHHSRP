import matplotlib.pyplot as plt
import matplotlib
import math
import sys
import csv
from statistics import mean 
from matplotlib import rcParams

instance_type = 'cluster'
arr_rate = '240'
obj = "patient"
nb_nurse = '1'
use_ch = 'True'
parent_dir = "/home/quy/Repos/Quy_11_14/Quy/2022_11_11/ddqn/"
sba_dir = "/home/quy/Repos/Quy_11_11/Quy/2022_11_8/sba/"
run_dir = instance_type + "-" + arr_rate + "-" + nb_nurse + "-" + obj + "-" + use_ch + "-20000000-512-0.999" 

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
plt.subplots_adjust(left=0.05,
                bottom=0.1,
                right=0.98,
                top=0.85,
                wspace=0.3,
                hspace=0.5)

for i in range(3):
    arr_rate = arr_dict[i]
    for j in range(2):
        instance_type = instance_types[j]
        run_dir = instance_type + "-" + arr_rate + "-" + nb_nurse + "-" + obj + "-" + use_ch + "-20000000-512-0.999" 

        content_list = get_ddqn_test_content(parent_dir + run_dir)
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
        ax[i][j].title.set_text("arrival rate = " + arr_rate)

        #plt.title('Avg gaps in log scale to reference solutions per iteration', pad=45, fontsize=26)
plt.legend(loc='upper center', borderaxespad=0., bbox_to_anchor=(-0.2, 4.4), fontsize=15, ncol=5, fancybox=True)
'''ax[0][0].set_ylabel('150',fontsize=18)
ax[1][0].set_ylabel('240',fontsize=18)
ax[2][0].set_ylabel('360',fontsize=18)'''
ax[2][0].set_xlabel('Uniform',labelpad = 20, fontsize=19)
ax[2][1].set_xlabel('Cluster',labelpad = 20, fontsize=19)

plt.show()
fig.savefig(instance_type + "_patient.jpg")