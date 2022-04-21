import numpy as np


class PatientGenerator:
    def __init__(self):
        self.arrival_rate = 25;
        self.horizon = 10
        self.max_weeks = 4
        self.max_days = 3
        self.max_hours = 2
        self.max_skill = 3
        self.max_location_coor = 80
        np.random.seed(333)
        
    def generate_patient(self):  
        next_request_time = np.random.exponential(self.arrival_rate)
        nb_weeks = np.random.randint(1, self.max_weeks + 1)
        nb_days = np.random.randint(1, self.max_days + 1)
        nb_hours = np.random.randint(1, self.max_hours + 1)
        require_skill = np.random.randint(1, self.max_skill + 1)
        location = (np.random.randint(0, self.max_location_coor), np.random.randint(0, self.max_location_coor))
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
                        