from math import sqrt

day_patterns = [[0],[1], [2], [3], [4],\
                [0, 2], [1, 3], [2, 4], [0, 2, 4]]

def distance(A, B):
    return sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)