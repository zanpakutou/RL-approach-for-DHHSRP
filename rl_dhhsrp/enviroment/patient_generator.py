import numpy as np

def c_location_generator(max_location_coor):
    box_1 = (3 , 15, 8, 20)
    box_2 = (18, 30, 23, 35)
    box_3 = (33, 47, 43, 57)
    order = np.random.randint(0, 269)
    box_order = 1
    x = y = 0
    if (order < 25):
        box_order = 1
        x = order // 5 + box_1[0]
        y = order %  5 + box_1[1]
    elif (order < 169):
        box_order = 2
        x = (order - 25) // 12 + box_2[0]
        y = (order - 25) %  12 + box_2[1]
    elif (order < 269):
        box_order = 3
        x = (order - 169) // 10 + box_3[0]
        y = (order - 169) %  10 + box_3[1]
    return (x, y)
    
def u_location_generator(max_location_coor):
    return (np.random.randint(0, max_location_coor), np.random.randint(0, max_location_coor))

def uc_location_generator(max_location_coor):
    if (np.random.uniform(low=0.0, high=1.0) > 0.7):
        return u_location_generator(max_location_coor)
    else:
        return c_location_generator(max_location_coor)

class PatientGenerator:
    def __init__(self, arrival_rate = 150, loc_gen = u_location_generator, mode=None):
        self.arrival_rate = arrival_rate;
        self.horizon = 360
        self.min_week = 4
        self.max_weeks = 4
        self.max_days = 3
        self.min_hours = 2
        self.max_hours = 2
        self.max_skill = 3
        self.max_location_coor = 60
        self.loc_gen = loc_gen
        np.random.seed(333)

        if (mode=='uniform'):
            None
        elif (mode=='cluster'):
            self.loc_gen = c_location_generator
        elif (mode=='simplify'):
            self.min_week = 4
            self.max_hours = 1
        
    def generate_patient(self):  
        next_request_time = np.random.exponential(self.arrival_rate)
        nb_weeks = np.random.randint(self.min_week, self.max_weeks + 1)#$np.random.choice(np.arange(1, 5), p=[0.05, 0.15, 0.3, 0.5])
        nb_days = np.random.choice(np.arange(1, 4), p=[0.05, 0.35, 0.6]) #np.random.randint(1, self.max_days + 1) 
        nb_hours = np.random.randint(self.min_hours, self.max_hours + 1)
        require_skill = np.random.randint(1, self.max_skill + 1)
        location = self.loc_gen(self.max_location_coor)
        return (next_request_time, (nb_weeks, nb_days, nb_hours), require_skill, location)
    
    def generate_instance(self, dir):
        with open(dir, 'w') as f:
            current_time = int(np.random.exponential(self.arrival_rate)) % 510 + 480
            for week in range(0, self.horizon):
                f.write("---> Week : " +  str(week) + "\n")
                for day in range(0, 5):
                    f.write("--> Day : " +  str(day) + "\n")
                    count = 0
                    while (current_time < 990):
                        (delay_time, require_time, require_skill, location) = self.generate_patient()
                        f.write("-> Request " + str(count) + '\n')
                        f.write(str(current_time) +  '\n')
                        f.write(str(require_time) + '\n')
                        f.write(str(require_skill) + '\n')
                        f.write(str(location) + '\n')
                        count = count + 1
                        current_time += int(delay_time)
                        
                    if (current_time >= 990):
                        current_time = current_time - 510

        f.close()