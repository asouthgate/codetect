import math
import numpy as np

def logsumexp(logls):
    m = max(logls)
    sumo = 0
    for l in logls:
        sumo += np.exp(l-m)
    return m + np.log(sumo) 

def approx(a,b,eps=0.000001):
    if math.fabs(a-b) < eps:
        return True
    return False

def ham(s1,s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])

