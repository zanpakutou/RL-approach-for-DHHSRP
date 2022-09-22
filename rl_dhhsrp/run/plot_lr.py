import matplotlib.pyplot as plt
import matplotlib
import math

my_file = open("LR", "r")
content = my_file.read()

content_list = []
for line in content.split("\n"):
    if (len(line) <= 0):
        continue
    epoch = []
    for item in line.split(" "):
        epoch.append(item)
        
    content_list.append(epoch)
my_file.close()

matplotlib.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (10, 5)

it = []
lr = []
is_apend = False
for content in content_list:
    if is_apend :
        lr.append(float(content[0]))
        is_apend = False
    if len(content) == 3:
        it.append(float(content[2]))
        is_apend = True
   

fig, ax = plt.subplots()
ax.plot(it, lr, color = "blue", label= "DDQN")

#ax.legend(bbox_to_anchor=(0.95, 0.95),
#                         loc='upper right', borderaxespad=0.)
ax.tick_params(axis='both', which='major', width=3, length= 15)
ax.tick_params(axis='both', which='minor', width=3, length= 8)

#plt.title('Avg gaps in log scale to reference solutions per iteration', pad=45, fontsize=26)
plt.ylabel('Cumul Reward',fontsize=20)
plt.xlabel('Number of epochs',labelpad = 15, fontsize=20)
plt.show()
fig.savefig("lr.jpg")