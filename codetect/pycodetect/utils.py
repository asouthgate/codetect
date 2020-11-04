from Bio.SeqIO import FastaIO
import math
import numpy as np
import sys

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

def ham_nogaps(s1,s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i] and 4 not in [s1[i],s2[i]]])

def ham_nogaps_str(s1,s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i] and "N" not in [s1[i],s2[i]]])

def c2i(c):
    if c == "A": return 0
    elif c == "C": return 1
    elif c == "G": return 2
    elif c == "T": return 3
    else: return 4

i2c = lambda x: "ACGT"[x]

def str_c2i(s):
    return tuple([c2i(c) for c in s])

def str_i2c(s):
    return "".join(["ACGT"(c) for c in s])

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

def only_ACGT(c):
    if c in "ACGT":
        return c
    return "N"

def str_only_ACGT(s):
    return "".join([only_ACGT(c) for c in s])
