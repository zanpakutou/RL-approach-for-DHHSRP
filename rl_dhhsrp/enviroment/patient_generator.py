import numpy as np

def cluster_location_generator(max_location_coor):
    box_1 = (8 , 20, 18, 30)
    box_2 = (52, 64, 67, 79)
    box_3 = (43, 27, 63, 47)
    if (np.random.randint(0, 2) == 0):
        order = np.random.randint(0, 725)
        box_order = 1
        x = y = 0
        if (order < 100):
            box_order = 1
            x = order // 10 + box_1[0]
            y = order %  10 + box_1[1]

        elif (order < 325):
            box_order = 2
            x = (order - 100) // 15 + box_2[0]
            y = (order - 100) %  15 + box_2[1]
        elif (order < 725):
            box_order = 3
            x = (order - 325) // 20 + box_3[0]
            y = (order - 325) %  20 + box_3[1]
        return (x, y)
    else:
        return (np.random.randint(0, max_location_coor), np.random.randint(0, max_location_coor))
def uniform_location_generator(max_location_coor):
    return (np.random.randint(0, max_location_coor), np.random.randint(0, max_location_coor))
class PatientGenerator:
    def __init__(self, arrival_rate = 360, loc_gen = uniform_location_generator):
        self.arrival_rate = 150;
        self.horizon = 20
        self.max_weeks = 4
        self.max_days = 3
        self.max_hours = 2
        self.max_skill = 3
        self.max_location_coor = 80
        self.loc_gen = loc_gen
        np.random.seed(333)
        
    def generate_patient(self):  
        next_request_time = np.random.exponential(self.arrival_rate)
        nb_weeks = np.random.randint(1, self.max_weeks + 1)
        nb_days = np.random.randint(1, self.max_days + 1)
        nb_hours = np.random.randint(1, self.max_hours + 1)
        require_skill = np.random.randint(1, self.max_skill + 1)
        location = self.loc_gen(self.max_location_coor)
        return (next_request_time, (nb_weeks, nb_days, nb_hours), require_skill, location)
    
    def generate_instance(self, dir):
        with open(dir, 'w') as f:
            for week in range(0, self.horizon):
                f.write("---> Week : " +  str(week) + "\n")
                for day in range(0, 5):
                    f.write("--> Day : " +  str(day) + "\n")
                    current_time = 0
                    count = 0
                    while (current_time < 24 * 60):
                        (delay_time, require_time, require_skill, location) = self.generate_patient()
                        current_time += int(delay_time)
                        if (current_time >= 24 * 60):
                            break
                        f.write("-> Request " + str(count) + '\n')
                        f.write(str(current_time) +  '\n')
                        f.write(str(require_time) + '\n')
                        f.write(str(require_skill) + '\n')
                        f.write(str(location) + '\n')
                        count = count + 1
        f.close()
                        