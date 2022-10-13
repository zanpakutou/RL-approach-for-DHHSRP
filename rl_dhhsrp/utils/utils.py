from math import sqrt
from typing import TextIO
import statistics
day_patterns = [[0], [1], [2], [3], [4],\
                [0, 2], [1, 3], [2, 4], [0, 2, 4]]
MAX_VAL = 2 ** 30


def distance(A, B):
    return sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
def mean(a):
    return statistics.mean(a)
def stdev(a):
    return statistics.stdev(a)
def set_seed(seed_value):
    seed_value = seed_value
    import os
    import random
    import numpy as np
    import tensorflow as tf
    from keras import backend as K
    
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)