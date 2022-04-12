import re

class Request:
    def __init__(self, current_time = 0, require_time = (0, 0, 0), require_skill = 1, location = (0, 0)):
        self.current_time = current_time
        self.require_time = require_time #weeks, days/w, hours/d
        self.require_skill = require_skill
        self.location = location
    def __str__(self):
        return "(" + str(self.current_time) + " " + str(self.require_time) + " " + str(self.require_skill)\
        + " " + str(self.location) + ")"
    def __repr__(self):
        return "(" + str(self.current_time) + " " + str(self.require_time) + " " + str(self.require_skill)\
        + " " + str(self.location) + ")"
class Enviroment:
    def __init__(self):
        #Init context parameter
        self.nb_nurses = 0
        self.working_tw = (0, 1439)
        self.qual = []
        self.nurse_depot = (0, 0)
        #Init patient parameter
        self.nb_weeks = 0
        self.day_per_week = 0
        self.requests = []
        
        
    def make(self, patient_dir, context_dir):
        """Read param for patients from 'patient_dir' and
            param for nurses from context_dir
        """
        print(patient_dir)
        try:
            context_file = open(context_dir, "r")
            patient_file = open(patient_dir, "r")

            for line in context_file.readlines():
                if "Number of nurses:" in line:
                    self.nb_nurses = int(re.search(r'\d+', line).group())
                if "Working window:" in line:
                    buff = list(map(int, re.findall(r'\d+', line)))
                    self.working_tw = (buff[0], buff[1])
                if "Level of qualification:" in line:
                    buff = list(map(int, re.findall(r'\d+', line)))
                    self.qual = buff
                if "Nurse location:" in line:
                    buff = list(map(int, re.findall(r'\d+', line)))
                    self.nurse_depot = (buff[0], buff[1])
            cnt = 0
            for line in patient_file.readlines():
                if "---> Week :" in line:
                    self.nb_weeks = self.nb_weeks + 1
                    self.day_per_week = 0
                    self.requests.append([])
                    continue
                if "--> Day :" in line:
                    self.day_per_week = self.day_per_week + 1
                   
                    self.requests[-1].append([])
                    continue
                if "-> Request" in line:
                    cnt = 0
                    continue
                cnt = cnt + 1
                if cnt == 1:
                    current_time = int(line)
                if cnt == 2:
                    buff = list(map(int, re.findall(r'\d+', line)))
                    require_time = (buff[0], buff[1], buff[2])
                if cnt == 3:
                    require_skill = int(line)
                if cnt == 4:
                    buff = list(map(int, re.findall(r'\d+', line)))
                    location = (buff[0], buff[1])
                    r = Request(current_time, require_time, require_skill, location)
                    self.requests[-1][-1].append(r)
                
        except FileNotFoundError:
            print("The file doesn't exist")
        finally:
            patient_file.close()
            context_file.close()
        
    def get_request(self):
        return self.requests
    def accept_request():
        return 