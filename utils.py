import math 

day_patterns = [(0),(1), (2), (3), (4), (5),\
                (0, 2), (1, 3), (2, 4), (3, 5), (1, 3, 5)]

def distance(A, B):
    return math.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)