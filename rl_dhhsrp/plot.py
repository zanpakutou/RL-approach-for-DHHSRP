import matplotlib.pyplot as plt
import matplotlib
import math

my_file = open("train_log", "r")
content = my_file.read()

content_list = []
for line in content.split("\n"):
    if (len(line) <= 0):
        continue
    epoch = []
    for item in line.split(" "):
        epoch.append(float(item))
        
    content_list.append(epoch)
my_file.close()

matplotlib.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (10, 5)

it = [i[0] for i in content_list]
obj = [i[1] for i in content_list]

fig, ax = plt.subplots()
ax.plot(it, obj, color = "red", label= "DDQN")

#ax.legend(bbox_to_anchor=(0.95, 0.95),
#                         loc='upper right', borderaxespad=0.)
ax.tick_params(axis='both', which='major', width=3, length= 15)
ax.tick_params(axis='both', which='minor', width=3, length= 8)

#plt.title('Avg gaps in log scale to reference solutions per iteration', pad=45, fontsize=26)
plt.ylabel('Cumul Reward',fontsize=20)
plt.xlabel('Number of epochs',labelpad = 15, fontsize=20)
plt.show()
fig.savefig("log.jpg")