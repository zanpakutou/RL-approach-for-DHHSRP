import numpy as np
np.set_printoptions(precision=4, suppress=True)

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Logger(metaclass=SingletonMeta):
    def __init__(self, dir="", use=['train', 'test', 'lr']):
        if ('train' in use):
            self.train_log   = open(dir + "/LOG", "w")
        if ('test' in use):
            self.test_log    = open(dir + "/TEST", "w")
        if ('eval' in use):
            self.eval_log    = open(dir + "/EVAL", "w")

    def init_lr_log(self, episode):
        self.lr_log.write("episodes : {}\n".format(episode))

    def write_eval_log(self, episode, score):
        self.eval_log.write(str(episode) + " " + str(score) + '\n')
        self.eval_log.flush()

    def write_train_log(self, state, pred=[]):
        if len(pred) != 0:
            self.train_log.write(str(np.around(state, decimals=4)) + '\n')
            self.train_log.write(str(np.around(pred, decimals=4))  + '\n')
        else:
            self.train_log.write(state)
  
    def write_test_log(self, episode, score):
        self.test_log.write(str(episode) + " " + str(score) + '\n')
        self.test_log.flush()

    def flush_log(self):
        self.train_log.flush()
        self.lr_log.flush()

    def close_log(self):
        self.train_log.close()
        self.test_log.close()
        self.lr_log.close()