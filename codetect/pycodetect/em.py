from io import StringIO
import math
import numpy as np
import sys
import random
import pycodetect.plotter as plotter
from pycodetect.utils import ham, c2i, logsumexp
np.set_printoptions(threshold=sys.maxsize)

class EM():
    """ Expectation Maximization object for parameter estimation.
    
    Args:
        rd: a ReadData object.
        min_d: minimum distance of estimated string from consensus.
    """
    def __init__(self, ds, min_d):
        self.ds = ds
        self.X = ds.X
        self.n_reads = sum([Xi.count for Xi in self.X])
        self.M = ds.M
        self.V_index = ds.V_INDEX
        self.min_cov = 0
        self.consensus = ds.get_consensus()
        self.min_threshold = 0.001
        self.min_freq = 0.03
        self.min_d = min_d

    def calc_log_likelihood(self,st,g0,g1,pi):
        """
        Calculate the log likelihood of data given parameters.

        Args:
            st: current string.
            g0: cluster 0 gamma.
            g1: cluster 1 gamma.
            pi: cluster proportion.
        """
        # We now seek the log likelihood 
        # = log(P(X_i | Zi=1,theta)pi + P(Xi | Zi=2,theta)(1-pi))
        # Use logsumexp
        # TODO: THIS IS WRONG
        sumo = 0
        for i,Xi in enumerate(self.X):
            a = Xi.logPmajor(g0)
            b = Xi.logPminor2(g1,st)
            lw1 = np.log(pi)
            lw2 = np.log(1-pi)
            sumo += logsumexp([a + lw1, b + lw2])
        return sumo

    def print_debug_info(self, Tt, st):
        inds = sorted([i for i in range(len(self.X))], key = lambda i : self.X[i].pos)
        for i in inds:
            print(self.X[i].pos, self.X[i].z, Tt[i], self.X[i].cal_ham(self.consensus), self.X[i].cal_ham(st))


    def calTi_pair2(self,Xi,pi,g0,g1,st,changed_inds):
        # TODO: depreciate; import calculator function
        """ Calculate the ith membership conditional probability array element.
        
        Args:
            pi: mixture model proportion
            g0: gamma parameter for cluster 0
            st: cluster 1 string
            g1: gamma parameter for cluster 1

        Returns:
            tp: T array
            lp: L array of Xi given cluster j
        """
        a = Xi.logPmajor(g0)
        b = Xi.logPminor2(g1,st,changed_inds)
#        assert 0 <= a <= 1, a
#        assert 0 <= b <= 1, b
#        print(a,b)
        # pi*e^L1 + (1-pi)e^L2 = e^(L1+logpi) + e^(L2+log(1-pi))
        # 
        l1 = a
        l2 = b
        lw1 = np.log(pi)
        lw2 = np.log(1-pi)
#        alpha = max([l1 + lw1, l2 + lw2])
        exp1 = np.exp(l1 + lw1)
        exp2 = np.exp(l2 + lw2)
#        exp1 = np.exp(l1 + lw1 - alpha)
#        exp2 = np.exp(l2 + lw2 - alpha)
#        assert exp1 > 0
#        assert exp2 > 0
        c = exp1 + exp2
        assert 0 < c <= 1.01,c
        t1i = exp1/c
        t2i = exp2/c
#        c = pi*a + (1-pi)*b
#        t1i = (pi * a) / c
#        t2i = ((1-pi) * b) / c
#        print(str(Xi)[:50],Xi.z,Xi.nm)
#        print("a=",a,"b=",b,"c=",c)
#        print("t=",[t1i,t2i])
        tp = np.array([t1i,t2i])
#        assert t1i > 0, t1i
#        assert t2i > 0, t2i
        assert sum(tp) > 0.999, sum(tp)
        return tp, np.log(c)
 
    def recalc_T2(self,pi,g,st,mu,changed_inds=None):
        res = []
        # Also calculate the log likelihood while we're at it
        Lt = 0
        for Xi in self.X:
            pairT, logL = self.calTi_pair2(Xi,pi,g,mu,st,changed_inds)
            res.append(pairT)
            Lt += logL
        return np.array(res), Lt

    def recalc_mu(self,T, S):
        numo = sum([T[i,1]*Xi.count*Xi.cal_ham(S) for i,Xi in enumerate(self.X)])
        deno = sum([T[i,1]*Xi.count*len(Xi.get_aln()) for i,Xi in enumerate(self.X)])
        assert deno > 0
        newmu = numo/deno
        assert 0 <= newmu <= 1,newmu
        return newmu

    def recalc_gamma(self,T):
        nms = [Xi.nm_major for Xi in self.X]
        Ti0s = T[:,0]
        numo = sum([T[i,0]*Xi.count*Xi.nm_major for i,Xi in enumerate(self.X)])
        deno = sum([T[i,0]*Xi.count*len(Xi.get_aln()) for i,Xi in enumerate(self.X)])
        newgt = numo/deno
        assert 0 <= newgt <= 1,newgt
        return newgt

    def regularize_st(self,ststar,wmat,diff):
     # IF THE MAXIMUM STRING IS TOO CLOSE, GET THE MAXIMUM STRING SUBJECT TO CONSTRAINTS
        maxalts = []
        for k in self.ds.VALID_INDICES:
            bw = wmat[k]
            # IF THE MAXIMUM IS NOT THE REFERENCE, SKIP
            if ststar[k] == self.consensus[k]:
                maxalt = max([j for j in range(4) if j != self.consensus[k]], key=lambda x:bw[x])
#                assert self.consensus[k] != maxalt
#                assert bw[ststar[k]] >= bw[maxalt]
#                assert bw[self.consensus[k]] >= bw[maxalt], (k,self.consensus[k], bw, maxalt)
                if bw[maxalt] > 0:
                    loss = bw[ststar[k]]-bw[maxalt]
                    maxalts.append([k,maxalt,loss])
                    assert maxalt != self.consensus[k]
        maxalts = np.array(maxalts)
        # Assume sorts small to high, take the last -diff, recall
        # diff is negative
        toflip = maxalts[np.argsort(maxalts[:,2])][0:-diff]
        for k,maxalt,loss in toflip:
            assert self.consensus[int(k)] != maxalt
            ststar[int(k)] = int(maxalt)
            assert ststar[int(k)] != self.consensus[int(k)]
#            print(k,maxalt,w,wmat[int(k)])
        return ststar        

    def get_weight_base_array(self, T):
        baseweights = np.zeros((len(self.consensus), 4))
        # FIRST CALCULATE THE MOST WEIGHTY BASE FOR EACH POSITION
        for k in self.ds.VALID_INDICES:
            v = np.zeros(4)
            totalTk = 0
            for j,rl in enumerate(self.V_index[k]):
                for ri in rl:
                    Xri = self.X[ri]
                    assert k in Xri.map, (k,Xri.map)
                    assert j == Xri.map[k]
                    baseweights[k,j] += T[ri,1]
                    totalTk += T[ri,1]
            if totalTk > 0:
                baseweights[k] /= totalTk
        return baseweights

    def recalc_st(self,T,minh):
        # BUILD THE MAXIMUM STRING
        baseweights = self.get_weight_base_array(T)
        ststar = [c for c in self.consensus]
        for bi in self.ds.VALID_INDICES:
            bw = baseweights[bi]
            maxi = max([j for j in range(4) if len(self.V_index[bi][j]) > self.min_cov], key=lambda x:bw[x])
            if sum(bw) > 0:
                ststar[bi] = maxi
            else:
                ststar[bi] = self.consensus[bi]
        diff = ham(ststar,self.consensus)-minh
        if diff >= 0:
            return ststar
        else:
            return self.regularize_st(ststar,baseweights,diff)
#            return ststar
 
    def recalc_V(self,T):
        # Regularize by claiming that the probability of a mismatch can never be less than MIN_THRESHOLD
        newv = np.zeros((len(self.V_index),4))
        assert len(self.V_index) == len(self.M)
        for k in range(len(self.V_index)):
            for c in range(4):
                # recalc Vi
                sumo = 0
                # Iterate over reads that mismatch at position k
                # THIS IS THE PROBABILITY THAT THEY ARE NOT THE SAME
                for ri in self.V_index[k][c]:
                    sumo += (T[ri,1])
                assert sum(T[:,1]) > 0
                assert np.isfinite(sumo), sumo
                newv[k,c] = sumo
            newv[k] += self.min_threshold
            assert sum(newv[k]) != 0,(k,newv[k])
            newv[k] /= sum(newv[k])
            assert sum(newv[k]) > 0.99999, (newv[k], sum(newv[k]))
        return newv

    def recalc_pi(self,T):
        return sum([T[i,0]*self.X[i].count for i in range(len(T))])/self.n_reads

    def expected_d(self,v):
        sumo = 0
        assert len(self.consensus) == len(v)
        for ci, c in enumerate(self.consensus):
            alts = [v[ci,j] for j in range(4) if j != c]
            sumo += sum(alts)
        return sumo

    def init_st_random(self,M):
        st = [c for c in self.consensus]
        for vi,vt in enumerate(M):
            st[vi] = np.random.choice([j for j in range(4)],p=vt)
        return st

    def init_st(self,M):
        st = [c for c in self.consensus]
        second_best = []
        for vi in self.ds.VALID_INDICES:
            vt = M[vi]
            stups = sorted([j for j in range(4)],key=lambda j:vt[j])
            sb = stups[-2]
            if vt[sb] > 0.0 and len(self.V_index[vi][sb]) > self.min_cov:
                second_best.append((vt[sb],vi,sb))
        second_best = sorted(second_best,key=lambda x:x[0])
        c = 0
#        for val,vi,sb in second_best[-len(self.consensus)//3:]:
        for val,vi,sb in second_best[::-1]:
            if c > self.min_d and val < self.min_freq:
                break
            c += 1
            st[vi] = sb
        return st

    def check_st(self, st):
        for i in range(len(st)):
            if i not in self.ds.VALID_INDICES:
                assert st[i] == self.consensus[i]

    def calc_L0(self):
        g = self.recalc_gamma(np.array([[1,0] for j in range(len(self.ds.X))]))
        return self.calc_log_likelihood(self.ds.get_consensus(),g,g,1)

    def do2(self, N_ITS=None, random_init=False, debug=False, debug_minor=None, max_pi=1.0, min_pi=0.5, fixed_st=None, mu_equals_gamma=True):
        pit = 0.5
        gt = 0.01
        mut = 0.01
        if fixed_st is None:
            if random_init:
                st = self.init_st_random(self.M)
            else:
                st = self.init_st(self.M)
        else:
            st = fixed_st
        # Assertions
        for row in self.M:
            for v in row:
                assert not np.isnan(v)
        assert len(self.X) > 0
        for i, Xi in enumerate(self.X):
            for pos,bk in Xi.get_aln():
#                assert Xi.i2c(bk) != "-"
                assert i in self.V_index[pos][bk]
                assert self.M[pos,bk] > 0
        for m in self.M:
            if sum([q for q in m]) > 0:
                assert sum(m) > 0.98, m
        assert len(self.V_index) == len(self.M)
        assert 0 <= gt <= 1,gt
        assert ham(st, self.consensus) >= self.min_d, ham(st, self.consensus)


        trace = []
        t = 0
        Lt = self.calc_log_likelihood(st,gt,mut,pit)
        changed_inds = None
        while True:
            if N_ITS is not None:
                if t > N_ITS: 
                    break
            trace.append([t, Lt, pit, gt, mut, st])
            self.check_st(st)
            assert pit <= max_pi
            sys.stderr.write("Iteration:%d" % t + str([Lt,pit,gt,mut,ham(st,self.consensus)]) + "\n")
            assert ham(st, self.consensus) >= self.min_d
            Ltold = Lt
            Tt,Lt = self.recalc_T2(pit,gt,st,mut,changed_inds)
#            if pit == 1:
#                break
#                sys.stderr.write("No coinfection detected.\n")
#                return self.calc_log_likelihood(st,gt,mut,pit),False,st,pit,gt


#            Ltval = self.calc_log_likelihood(st,gt,mut,pit)
#            assert np.abs(Lt-Ltval) < 0.000001, (Lt,Ltval)

            if debug:
                plotter.plot_genome(self.ds,Tt,st,debug_minor)
#            self.print_debug_info(Tt,st)
            self.st = st
            self.Tt = Tt
            self.gt = gt
            self.pit = pit
            if sum(Tt[:,1]) == 0:
                break
#                sys.stderr.write("No coinfection detected.\n")
#                return self.calc_log_likelihood(st,gt,mut,pit), False, st, pit, gt 

            old_pi = pit
            pit = self.recalc_pi(Tt)
#            pit = max(min_pi,min(max_pi, pit))
            pit = max(min_pi,min(max_pi,pit))
            gt = self.recalc_gamma(Tt)
            gt = min(max(gt, 0.0001), 0.05)
            old_st = st
            if fixed_st is None:
                st = self.recalc_st(Tt, self.min_d)     
            else:
                st = fixed_st
            changed_inds = [sti for sti in range(len(st)) if st[sti] != old_st[sti]]
            if np.abs(Ltold-Lt) < 0.001 and np.abs(old_pi-pit) < 0.001 and old_st == st:
                break
            mut = gt
            if not mu_equals_gamma:
                mut = self.recalc_mu(Tt, st)
                mut = min(max(mut, 0.0001), 0.05)
            t += 1
        trace.append([t, Lt, pit, gt, mut, st])
        self.check_st(st)
        assert pit <= max_pi
        sys.stderr.write("Iteration:%d" % t + str([Lt,pit,gt,mut,ham(st,self.consensus)]) + "\n")



        if debug:
            plotter.plot_genome(self.ds,Tt,st,debug_minor)

#        /cif pit > 0.99:
#            sys.stderr.write("No coinfection detected!\n")
#            return self.calc_log_likelihood(st,gt,mut,pit), False, st, Tt, pit, gt
        
#        sys.stderr.write("Coinfection detected!\n")
        return trace
