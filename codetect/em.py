#import pymc3 as pm
from io import StringIO
import math
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#import theano.tensor as tt
import sys
import random
np.set_printoptions(threshold=sys.maxsize)

c2i = {"A":0, "C":1, "G":2, "T":3}

class EM():
    def __init__(self, X, C, M, V_INDEX, CONSENSUS, NM_CACHE):
        self.X = X
        self.C = C
        self.M = M
        self.V_INDEX = V_INDEX
        self.CONSENSUS = CONSENSUS
        self.NM_CACHE = NM_CACHE
        self.MIN_THRESHOLD = 0.00001

    def PXi_condZi(self,xi,zi,g,v):
        logp = np.float64(0)
        pos,read = xi
        if zi == 0:
            for ki, c in enumerate(read):
                if c == self.CONSENSUS[ki+pos]:
                    logp += np.log((1-g))
                else:
                    logp += np.log(g)
        elif zi == 1:
            for ki, c in enumerate(read):
                bind = c2i[c]

#                assert v[pos+ki,bind] != 0.0
                logp += np.log(v[pos+ki,bind])
        return np.exp(logp)

    def calTi_pair(self,xi,pi,g,v):
        a = self.PXi_condZi(xi,0,g,v)
#        print(a,pi,g)
        b = self.PXi_condZi(xi,1,g,v)
        c = pi*a + (1-pi)*b
        t1i = (pi * a) / c
        t2i = ((1-pi) * b) / c
        return np.array([t1i,t2i])

    def recalc_T(self,pi,g,v):
        res = []
        for i in range(len(self.X)):
            pair = self.calTi_pair(self.X[i],pi,g,v)
            res.append(pair)
        return np.array(res)

    def recalc_gamma(self,T):
        # sum over reads, calculate the number of mismatches
#        numos = [T[i,0]*self.NM_CACHE[i]*self.C[i] for i,Xi in enumerate(self.X)]
#        denos = [T[i,0]*len(Xi[1])*self.C[i] for i,Xi in enumerate(self.X)]
#        lens = [len(Xi[1]) for i,Xi in enumerate(self.X)]
#        newgt = sum(numos)/sum(denos)
#        print(T)
#        print(self.NM_CACHE)
#        print(self.C)
        newgt = sum([(T[i,0]*self.NM_CACHE[i]*self.C[i])/len(Xi[1]) for i,Xi in enumerate(self.X)])/sum(self.C)
        assert len(self.NM_CACHE) == len(self.C) == len(T)
        assert 0 <= newgt <= 1,newgt
        return newgt

    def recalc_V(self,T):
        # Regularize by claiming that the probability of a mismatch can never be less than MIN_THRESHOLD
        newv = np.zeros((len(self.V_INDEX),4))
        assert len(self.V_INDEX) == len(self.M)
        for k in range(len(self.V_INDEX)):
            for c in range(4):
                # recalc Vi
                sumo = 0
                # Iterate over reads that mismatch at position k
                # THIS IS THE PROBABILITY THAT THEY ARE NOT THE SAME
                for ri in self.V_INDEX[k,c]:
                    sumo += (T[ri,1] * self.C[ri])
                assert sum(T[:,1]) > 0
                assert np.isfinite(sumo), sumo
                newv[k,c] = sumo
            newv[k] += self.MIN_THRESHOLD
            assert sum(newv[k]) != 0,(k,newv[k])
            newv[k] /= sum(newv[k])
            assert sum(newv[k]) > 0.99999, (newv[k], sum(newv[k]))
        return newv

    def recalc_pi(self,T):
        return sum([T[i,0]*self.C[i] for i in range(len(T))])/sum(self.C)

    def expected_d(self,v):
        sumo = 0
        for ci, c in enumerate(self.CONSENSUS):
            alts = [v[ci,j] for j in range(4) if j != c2i[c]]
            sumo += sum(alts)
        return sumo
 
    def do(self,testZ,truepi,trueham,truepid):
        for i in range(len(self.X)):
            print(testZ[i], self.X[i], self.C[i], self.NM_CACHE[i])

        vt = self.M
        pit = 0.99
        gt = 0.01

        for xi, tup in enumerate(self.X):
            pos,read = tup
            for k,c in enumerate(read):
                assert xi in self.V_INDEX[k+pos][c2i[c]],(xi,tup)
                assert self.M[pos+k,c2i[c]] > 0, (self.M[pos+k], pos, k,c)

        for m in self.M:
            if sum([np.isnan(q) for q in m]) == 0:
                assert sum(m) > 0.98, m

        assert len(self.V_INDEX) == len(self.M)
        assert 0 <= gt <= 1,gt

        for t in range(100):
            print()
            print("******ITERATION %d" %t)
            print("TRUEPI", truepi)
            print("TRUEHAM", trueham, truepid)
            print("ESTPI",pit)
            print("ESTGT",gt)
            print("ESTED", self.expected_d(vt))
            Tt = self.recalc_T(pit,gt,vt)
#            for i in range(len(self.X)):
#                print(testZ[i], self.X[i], self.C[i], self.NM_CACHE[i], Tt[i])

            assert sum(Tt[:,1]) > 0
            assert sum(Tt[:,0]) > 0
            pit = self.recalc_pi(Tt)
            gt = self.recalc_gamma(Tt)
            vt = self.recalc_V(Tt)     
#            foo=bar
            
                

