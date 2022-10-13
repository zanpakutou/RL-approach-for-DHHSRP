import numpy as np

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Logger(metaclass=SingletonMeta):
    def __init__(self):
        self.train_log   = open("LOG", "w")
        self.test_log    = open("TEST", "w")
        self.lr_log      = open("LR", "w")

    def init_lr_log(self, episode):
        self.lr_log.write("episodes : {}\n".format(episode))

    def write_lr_log(self, lr):
        self.lr_log.write(str(lr) + '\n')

    def write_train_log(self, state, pred=[]):
        if len(pred) != 0:
            self.train_log.write(str(np.around(state, decimals=4)) + '\n')
            self.train_log.write(str(np.around(pred, decimals=4))  + '\n')
        else:
            self.train_log.write(state)
  
    def write_test_log(self, episode, score):
        self.test_log.write(str(episode) + " " + str(score))
    def flush_log(self):
        self.train_log.flush()
        self.test_log.flush()
        self.lr_log.flush()