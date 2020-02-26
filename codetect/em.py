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
def ham(s1, s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])

class EM():
    def __init__(self, X, M, V_INDEX, CONSENSUS, EPS):
        self.X = X
        self.N_READS = sum([Xi.count for Xi in self.X])
        self.M = M
        self.V_INDEX = V_INDEX
        self.CONSENSUS = CONSENSUS
        self.MIN_THRESHOLD = 0.001
        self.EPSILON = EPS

    def print_debug_info(self, Tt):
        for i in range(20):
            print(self.X[i].z, Tt[i])

    def calTi_pair(self,Xi,pi,g,v):
        a = Xi.Pmajor(g)
        b = Xi.Pminor(v)
        assert 0 <= a <= 1
        assert 0 <= b <= 1
        c = pi*a + (1-pi)*b
        t1i = (pi * a) / c
        t2i = ((1-pi) * b) / c
#        print(str(Xi)[:50],Xi.z,Xi.nm)
#        print()
#        print(a,b,c)
        tp = np.array([t1i,t2i])
        assert sum(tp) > 0.999
        return tp

    def calTi_pair2(self,Xi,pi,g,st,mu):
        a = Xi.logPmajor(g)
        b = Xi.logPminor2(st,mu)
#        assert 0 <= a <= 1, a
#        assert 0 <= b <= 1, b
        l1 = a
        l2 = b
        lw1 = np.log(pi)
        lw2 = np.log(1-pi)
        alpha = max([l1 + lw1, l2 + lw2])
        exp1 = np.exp(l1 + lw1 - alpha)
        exp2 = np.exp(l2 + lw2 - alpha)
        c = exp1 + exp2
        t1i = exp1/c
        t2i = exp2/c
#        c = pi*a + (1-pi)*b
#        t1i = (pi * a) / c
#        t2i = ((1-pi) * b) / c
#        print(str(Xi)[:50],Xi.z,Xi.nm)
#        print("a=",a,"b=",b,"c=",c)
#        print("t=",[t1i,t2i])
        tp = np.array([t1i,t2i])
        assert sum(tp) > 0.999, sum(tp)
        return tp
 
    def recalc_T(self,pi,g,v):
        res = []
        for Xi in self.X:
            pair = self.calTi_pair(Xi,pi,g,v)
            res.append(pair)
        return np.array(res)

    def recalc_T2(self,pi,g,st,mu):
        res = []
        for Xi in self.X:
            pair = self.calTi_pair2(Xi,pi,g,st,mu)
            res.append(pair)
        return np.array(res)

    def recalc_mu(self,T, S):
        numo = sum([T[i,1]*Xi.count*Xi.cal_ham(S) for i,Xi in enumerate(self.X)])
        deno = sum([T[i,1]*Xi.count*len(Xi.base_pos_pairs) for i,Xi in enumerate(self.X)])
        assert deno > 0
        newmu = numo/deno
        assert 0 <= newmu <= 1,newmu
        return min(newmu,0.5)

    def recalc_gamma(self,T):
        numo = sum([T[i,0]*Xi.count*Xi.nm for i,Xi in enumerate(self.X)])
        deno = sum([T[i,0]*Xi.count*len(Xi.base_pos_pairs) for i,Xi in enumerate(self.X)])
        newgt = numo/deno
        assert 0 <= newgt <= 1,newgt
        return newgt

    def regularize_st(self,ststar,wmat,diff):
     # IF THE MAXIMUM STRING IS TOO CLOSE, GET THE MAXIMUM STRING SUBJECT TO CONSTRAINTS
        maxalts = []
        for k, bw in enumerate(wmat):
            # IF THE MAXIMUM IS NOT THE REFERENCE, SKIP
            if ststar[k] == self.CONSENSUS[k]:
                maxalt = max([j for j in range(4) if j != self.CONSENSUS[k]], key=lambda x:bw[x])
#                assert self.CONSENSUS[k] != maxalt
#                assert bw[ststar[k]] >= bw[maxalt]
#                assert bw[self.CONSENSUS[k]] >= bw[maxalt], (k,self.CONSENSUS[k], bw, maxalt)
                if bw[maxalt] > 0:
                    loss = bw[ststar[k]]-bw[maxalt]
                    maxalts.append([k,maxalt,loss])
                    assert maxalt != self.CONSENSUS[k]
        maxalts = np.array(maxalts)
        # Assume sorts small to high, take the last diff
        toflip = maxalts[np.argsort(maxalts[:,2])][0:diff]
        for k,maxalt,loss in toflip:
            assert self.CONSENSUS[int(k)] != maxalt
            ststar[int(k)] = maxalt
            assert ststar[int(k)] != self.CONSENSUS[int(k)]
#            print(k,maxalt,w,wmat[int(k)])
        return ststar        

    def get_weight_base_array(self, T):
        baseweights = np.zeros((len(self.CONSENSUS), 4))
        # FIRST CALCULATE THE MOST WEIGHTY BASE FOR EACH POSITION
        for k in range(len(self.V_INDEX)):
            v = np.zeros(4)
            totalTk = 0
            for j,rl in enumerate(self.V_INDEX[k]):
                for ri in rl:
                    Xri = self.X[ri]
                    rib = Xri.base_pos_pairs[k-Xri.pos][1]
                    baseweights[k,rib] += T[ri,1]
                    totalTk += T[ri,1]
            baseweights[k] /= totalTk
        return baseweights

    def recalc_st(self,T,minh):
        # BUILD THE MAXIMUM STRING
        baseweights = self.get_weight_base_array(T)
        ststar = []
        for bi,bw in enumerate(baseweights):
            maxi = max([j for j in range(4)], key=lambda x:bw[x])
            ststar.append(maxi)
#            print(bi,bw,self.CONSENSUS[bi],maxi)
        diff = ham(ststar,self.CONSENSUS)-minh
        if diff >= 0:
            return ststar
        else:
            return self.regularize_st(ststar,baseweights,diff)
#            return ststar
 
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
            alts = [v[ci,j] for j in range(4) if j != c]
            sumo += sum(alts)
        return sumo

    def init_st(self,M):
        st = [c for c in self.CONSENSUS]
        second_best = []
        for vi,vt in enumerate(M):
            stups = sorted([j for j in range(4)],key=lambda j:vt[j])
            sb = stups[-2]
            if vt[sb] > 0:
                second_best.append((vt[sb],vi,sb))
        second_best = sorted(second_best,key=lambda x:x[0])
        for val,vi,sb in second_best[-self.EPSILON:]:
            st[vi] = sb
        return st

    def do2(self, N_ITS, debug_minor, debug=False):

        pit = 0.5
        gt = 0.01
        mut = 0.01
        st = self.init_st(self.M)
        

        # Assertions
        for row in self.M:
            for v in row:
                assert not np.isnan(v)
        assert len(self.X) > 0
        for i, Xi in enumerate(self.X):
            for pos,bk in Xi.get_aln():
                assert Xi.i2c(bk) != "-"
                assert i in self.V_INDEX[pos][bk]
                assert self.M[pos,bk] > 0
        for m in self.M:
            if sum([q for q in m]) > 0:
                assert sum(m) > 0.98, m
        assert len(self.V_INDEX) == len(self.M)
        assert 0 <= gt <= 1,gt
        assert ham(st, self.CONSENSUS) >= self.EPSILON, ham(st, self.CONSENSUS)

        for t in range(N_ITS):
            sys.stderr.write("Iteration:%d,%f,%f,%d\n" % (t,pit,gt,ham(st,self.CONSENSUS)))
            assert ham(st, self.CONSENSUS) >= self.EPSILON
            Tt = self.recalc_T2(pit,gt,st,mut)
            if sum(Tt[:,1]) == 0:
                sys.stderr.write("No coinfection detected.\n")
                return False,st,Tt

            pit = self.recalc_pi(Tt)
            gt = self.recalc_gamma(Tt)
            st = self.recalc_st(Tt, self.EPSILON)     
            mut = self.recalc_mu(Tt, st)


        if pit > 0.99:
            sys.stderr.write("No coinfection detected!\n")
            return False, st, Tt
        
        sys.stderr.write("Coinfection detected!\n")
        return True, st, Tt
