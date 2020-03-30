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

c2i = {"A":0, "C":1, "G":2, "T":3, "-":4, "M":4, "R":4, "Y":4, "S":4, "K":4, "W":4, "V":4, "H":4, "N":4, "X":4}

def str_c2i(s):
    return tuple([c2i[c] for c in s])

def rev_comp(s):
    s2 = ""
    for c in s[::-1]:
        if c == "A":
            s2 += "T"
        elif c == "T":
            s2 += "A"
        elif c == "G":
            s2 += "C"
        elif c == "C":
            s2 += "G"
        else:
            s2 += "N"
    return s2

def ham_early_bail(s1, s2, min_d):
    sumo = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            sumo += 1   
        if sumo >= min_d:
            return sumo
    return sumo
