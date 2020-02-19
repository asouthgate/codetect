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

c2i = {c:i for i,c in enumerate("ACGT")}

class EM():
    def __init__(self, X, M, V_INDEX, CONSENSUS):
        self.X = X
        self.N_READS = sum([Xi.count for Xi in self.X])
        self.M = M
        self.V_INDEX = V_INDEX
        self.CONSENSUS = CONSENSUS
        self.MIN_THRESHOLD = 0.001

    def calTi_pair(self,Xi,pi,g,v):
        a = Xi.Pmajor(g)
        b = Xi.Pminor(v)
        assert 0 <= a <= 1
        assert 0 <= b <= 1
        c = pi*a + (1-pi)*b
        t1i = (pi * a) / c
        t2i = ((1-pi) * b) / c
#        print(str(Xi)[:50],Xi.z,Xi.nm)
#        print(a,b,c)
#        print(pi*a/c, (1-pi)*b/c)
#        print()
#        print(a,b,c)
        tp = np.array([t1i,t2i])
        assert sum(tp) > 0.999
        return tp
 
    def recalc_T(self,pi,g,v):
        res = []
        for Xi in self.X:
            pair = self.calTi_pair(Xi,pi,g,v)
            res.append(pair)
        return np.array(res)

    def recalc_gamma(self,T):
        numo = sum([T[i,0]*Xi.nm/len(Xi.base_pos_pairs) for i,Xi in enumerate(self.X)])
        deno = sum([T[i,0] for i,Xi in enumerate(self.X)])
        newgt = numo/deno
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
                for ri in self.V_INDEX[k][c]:
                    sumo += (T[ri,1])
                assert sum(T[:,1]) > 0
                assert np.isfinite(sumo), sumo
                newv[k,c] = sumo
            newv[k] += self.MIN_THRESHOLD
            assert sum(newv[k]) != 0,(k,newv[k])
            newv[k] /= sum(newv[k])
            assert sum(newv[k]) > 0.99999, (newv[k], sum(newv[k]))
        return newv

    def recalc_pi(self,T):
        return sum([T[i,0]*self.X[i].count for i in range(len(T))])/self.N_READS

    def expected_d(self,v):
        sumo = 0
        assert len(self.CONSENSUS) == len(v)
        for ci, c in enumerate(self.CONSENSUS):
            alts = [v[ci,j] for j in range(4) if j != c2i[c]]
            sumo += sum(alts)
        return sumo

    def do(self, N_ITS):
        assert len(self.X) > 0

        vt = self.M
#        vt = np.ones(self.M.shape)
#        vt *= 0.25
        pit = 0.99
        gt = 0.01

        print(type(vt))

        for i, Xi in enumerate(self.X):
            for pos,bk in Xi.get_aln():
                assert Xi.i2c(bk) != "-"
                assert i in self.V_INDEX[pos][bk]
                assert self.M[pos,bk] > 0

        assert len(self.CONSENSUS) == len(vt)

        for m in self.M:
            if sum([np.isnan(q) for q in m]) == 0:
                assert sum(m) > 0.98, m

        assert len(self.V_INDEX) == len(self.M)
        assert 0 <= gt <= 1,gt

#        print(self.M)
#        print(self.V_INDEX[:10])

        def ham(s1, s2):
            return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])
 
#        for Xi in self.X:
#            print("-"*Xi.pos + Xi.get_string(), ham(Xi.get_string(), self.CONSENSUS), Xi.nm)

        for t in range(N_ITS):
            Tt = self.recalc_T(pit,gt,vt)
#            print(Tt)
            pit = self.recalc_pi(Tt)
            gt = self.recalc_gamma(Tt)
#            gt = 0.02
            vt = self.recalc_V(Tt)     
            # constrain vt
#            for i in range(len(vt)):
#                vt[i] *= (1-(4/3)*gt)
#                vt[i] += (1/3)*gt
#                print(vt[i])
#                assert sum(vt[i]) > 0.9999, sum(vt[i])
#            print([Xi.z for Xi in self.X])
#            print(Tt[:,0])
#            print(Tt[:,1])
#            vt2 = self.recalc_V2(Tt)
#            self.MIN_THRESHOLD = gt/3
#            edp = self.expected_p(Tt) - gt
            edp = self.expected_d(vt) - (self.MIN_THRESHOLD*3*len(vt))
#            inds = [i for i in range(len(self.CONSENSUS)) if vt[i][c2i[self.CONSENSUS[i]]] != max(vt[i])]
#            print(len(inds))
#            for i in inds:
#                print(vt[i], self.M[i])
#            edp2 = self.expected_d(vt2)
#            print(t,pit,gt,edp, end="\r", flush=True)
            print("********",t,pit,gt,edp)
#            print("vt", vt[:5])


        print(t,pit,gt,edp,"       ")

