from io import StringIO
import math
import numpy as np
import sys
import random
import pycodetect.plotter as plotter
from pycodetect.utils import ham, c2i, logsumexp, ham_nogaps
np.set_printoptions(threshold=sys.maxsize)

# TODO: why does EM() have ReadAlnData state? It should not have that. 
#       currently, for example, consensus has more than one representation. Here, and in 
#       RAD.

class EM():
    """ Expectation Maximization object for parameter estimation.
    
    Args:
        rd: a ReadData object.
        min_d: minimum distance of estimated string from consensus.
    """
    def __init__(self, rd, min_d):
        self.rd = rd
        self.n_reads = sum([Xi.count for Xi in self.rd.X])
        self.min_cov = 0
        self.consensus = rd.get_consensus()
        self.min_d = min_d

    def calc_log_likelihood(self, st, g0, g1, pi):
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
        sumo = 0
        for i,Xi in enumerate(self.rd.X):
            # TODO: replace read calculating its own LL with cache?
            a = Xi.logPmajor(g0)
            b = Xi.logPminor2(g1, st)
            lw1 = np.log(pi)
            lw2 = np.log(1-pi)
            sumo += logsumexp([a + lw1, b + lw2])
        return sumo

    # TODO: replace with logging
    def print_debug_info(self, Tt, st):
        inds = sorted([i for i in range(len(self.rd.X))], key = lambda i : self.X[i].pos)
        for i in inds:
            print(self.rd.X[i].pos, self.X[i].z, Tt[i], self.X[i].cal_ham(self.consensus), self.X[i].cal_ham(st))

    def calTi_pair2(self, Xi, pi, g0, g1, st, prev_st, changed_inds):
        # TODO: depreciate; import calculator function
        """ Calculate the ith membership conditional probability array element.
        
        Args:
            Xi: ith ReadAln object
            pi: mixture model proportion
            g0: gamma parameter for cluster 0
            st: cluster 1 string
            g1: gamma parameter for cluster 1
            prev_st: previous cluster 1 string
            changed_inds: indices such that st[i] != prev_st[i]        

        Returns:
            tp: T array: P(Z_i = k | X_i)
            lp: Likelihood of Xi given cluster j
        """
        if changed_inds is None:
            assert st == prev_st
        a = Xi.logPmajor(g0)
        b = Xi.logPminor2(g1,st,prev_st,changed_inds)
        l1 = a
        l2 = b
        lw1 = np.log(pi)
        lw2 = np.log(1-pi)

        exp1 = np.exp(l1 + lw1)
        exp2 = np.exp(l2 + lw2)

        c = exp1 + exp2
        assert 0 < c <= 1.01:

        t1i = exp1/c
        t2i = exp2/c

        tp = np.array([t1i,t2i])
        assert sum(tp) > 0.999, sum(tp)
        return tp, np.log(c)
 
    def recalc_T(self, pi, g1, st, g2, prev_st, changed_inds=None):
        """ Recalculate the T array given new parameters.

        Args:
            pi: mixing proportion
            g1: gamma of cluster 1
            g2: gamma of cluster 2
            st: proposed s2
            prev_st: previous s2
            changed_inds: indices such that st[i] != prev_st[i]        

        Returns:
            Tt: T array at current iteration
            Lt: Likelihood at current iteration            
        """

        res = []
        # Also calculate the log likelihood while we're at it
        Lt = 0
        for Xi in self.rd.X:
            pairT, logL = self.calTi_pair2(Xi,pi,g1,g2,st,prev_st,changed_inds)
            res.append(pairT)
            Lt += logL
        Tt = np.array(res)
        return Tt, Lt

    def recalc_gk(self, T, S, k):
        # TODO: replace cal_ham with a call to a cache
        """ Recalculate gamma for the kth cluster. """
        numo = sum([T[i,k] * Xi.count * Xi.cal_ham(S) for i, Xi in enumerate(self.rd.X)])
        deno = sum([T[i,k] * Xi.count * len(Xi.get_aln()) for i, Xi in enumerate(self.rd.X)])
        assert deno > 0
        newg = numo/deno
        assert 0 <= newg <= 1, newg
        return newg

    def regularize_st(self, ststar, wmat, diff):
        """ For some st that breaks constraints, flip back bases optimally until
            constraints are satisfied. 
        
            Args:
                ststar: string to regularize
                wmat: weight matrix
                diff: number of excess SNPs

            Returns:
                ststar, with diff bases reverted        
        """
        maxalts = []
        for k in self.ds.VALID_INDICES:
            bw = wmat[k]
            # IF THE MAXIMUM IS NOT THE REFERENCE, SKIP
            if ststar[k] == self.consensus[k]:
                maxalt = max([j for j in range(4) if j != self.consensus[k]], key=lambda x:bw[x])
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
        return ststar        

    def get_weight_base_array(self, T):
        """ Computes an array of weights W corresponding to:
            W[i,j] = \sum_ri T_ri x delta(ri,j), where
            delta(ri,j) is 1 if read i has base j at that
            position.
        
        Args: 
            T: T array with probability of membership

        Returns:
            W: a weight array in |X| x 4 
        """
        baseweights = np.zeros((len(self.consensus), 4))
        for k in self.ds.VALID_INDICES:
            v = np.zeros(4)
            totalTk = 0
            for j,rl in enumerate(self.rd.V_INDEX[k]):
                for ri in rl:
                    Xri = self.rd.X[ri]
                    assert k in Xri.map, (k,Xri.map)
                    assert j == Xri.map[k]
                    baseweights[k,j] += T[ri,1]
                    totalTk += T[ri,1]
            if totalTk > 0:
                baseweights[k] /= totalTk
        return baseweights

    def recalc_st(self,T,min_d):
        """ Calculating the string that maximizes Q 

        Args:
            T: T array (see EM)
            min_d: minimum distance allowable to consensus
        """
        baseweights = self.get_weight_base_array(T)
        ststar = [c for c in self.consensus]
        for bi in self.ds.VALID_INDICES:
            bw = baseweights[bi]
            maxi = max([j for j in range(4) if len(self.rd.V_INDEX[bi][j]) > self.min_cov], key=lambda x:bw[x])
            if sum(bw) > 0:
                ststar[bi] = maxi
            else:
                ststar[bi] = self.consensus[bi]
        diff = ham(ststar,self.consensus) - min_d
        if diff >= 0:
            return ststar
        else:
            return self.regularize_st(ststar, baseweights, diff)

    def recalc_st_refs(self, T, refs, curr_ri):
        """ Calculate the string s that maximizes Q such that s in refs.

        Args:
            T: T array as in EM
            refs: RefPanel object with permissible refs
            curr_ri: current reference

        Returns:
            maxind: index of max ref
            rh: header of max ref
            rseq: sequence of max ref
        """
        W = self.get_weight_base_array(T)
        assert refs.size() > 0
        # Calculate scores that maximize Q
        refscores = np.zeros(refs.size())
        conscores = [W[bi, self.consensus[bi]] for bi in self.ds.VALID_INDICES]
        conscore = sum(conscores)
        # Calculate ref scores quickly by relying on differences to consensus
        # And updating that, instead of recomputing the whole thing
        for ri in range(refs.size()):
            refh, refstr = refs.get_ref(ri)
            refscores[ri] = conscore
            for bi in refs.get_diff_inds(ri):
                refscores[ri] -= conscores[bi]
                refscores[ri] += W[bi, refstr[bi]]

        maxind = np.argmax(refscores)
        rh, rseq = refs.get_ref(maxind)
        return maxind, rh, rseq
 
    def recalc_pi(self,T):
        return sum([T[i,0] * self.rd.X[i].count for i in range(len(T))]) / self.n_reads

    def init_st_random(self, M):
        st = [c for c in self.consensus]
        for vi, vt in enumerate(M):
            st[vi] = np.random.choice([j for j in range(4)], p=vt)
        return st

    def init_st(self, M):
        """ Initiallize st by chosing second most frequent base at each position. """
        st = [c for c in self.consensus]
        second_best = []
        for vi in self.ds.VALID_INDICES:
            vt = M[vi]
            stups = sorted([j for j in range(4)],key=lambda j:vt[j])
            sb = stups[-2]
            if vt[sb] > 0.0 and len(self.rd.V_INDEX[vi][sb]) > self.min_cov:
                second_best.append((vt[sb],vi,sb))
        second_best = sorted(second_best,key=lambda x:x[0])
        c = 0
        for val, vi, sb in second_best[::-1]:
            if c > self.min_d:
                break
            c += 1
            st[vi] = sb
        return st

    def check_st(self, st):
        # Big assertion
        assert len(st) == len(self.consensus), (len(st), len(self.consensus))
        for i in range(len(st)):
            if i not in self.ds.VALID_INDICES:
                assert st[i] == self.consensus[i]

    def calc_L0(self):
        """ Calculate the likelihood under the null. """
        g = self.recalc_gamma(np.array([[1,0] for j in range(len(self.ds.X))]))
        return self.calc_log_likelihood(self.ds.get_consensus(),g,g,1)

    def estimate(self, ref_panel=None, n_its=None, random_init=False, debug=False,
             debug_minor=None, max_pi=1.0, min_pi=0.5, fixed_st=None,
             one_gamma=True):
        """ Estimate parameters by expectation-maximization.

        Optional args:
            ref_panel: RefPanel object for reference-based estimation.
            n_its: number of iterations.
            random_init: whether to initialize st by randomization.
            debug: whether to enable debugging plotting mode.
            debug_minor: true st for plotting comparison.
            max_pi: maximum value of pi.
            min_pi: minimum value of pi.
            fixed_st: fix st to a given string.
            one_gamma: set gamma1=gamma2

        Returns:
            trace: list of values of [t, Lt, pit, g1t, g2t, st] for each
                   iteration.
        """
        # Abitrary initialization
        pit = 0.5
        g1t = 0.01
        g2t = 0.01
        
        # TODO: could move to a function
        # Initialization of st
        if fixed_st is None:
            if random_init:
                st = self.init_st_random(self.rd.M)
            else:
                if ref_panel is None:
                    st = self.init_st(self.rd.M)
                else:
                    curr_ri, refht, st = ref_panel.get_random_ref()
        else:
            st = fixed_st

        # Some safety checks
        assert len(st) == len(self.consensus), (len(st), len(self.consensus))
        for row in self.rd.M:
            for v in row:
                assert not np.isnan(v)
        assert len(self.rd.X) > 0
        for i, Xi in enumerate(self.rd.X):
            for pos, bk in Xi.get_aln():
                assert i in self.rd.V_INDEX[pos][bk]
                assert self.rd.M[pos, bk] > 0
        for m in self.rd.M:
            if sum([q for q in m]) > 0:
                assert sum(m) > 0.98, m
        assert len(self.rd.V_INDEX) == len(self.rd.M)
        assert 0 <= gt <= 1,gt
        assert ham(st, self.consensus) >= self.min_d, ham(st, self.consensus)

        trace = []
        t = 0
        Lt = self.calc_log_likelihood(st, g1t, g2t, pit)
        changed_inds = []
        old_st = st
        while True: 

            assert pit <= max_pi
            assert ham(st, self.consensus) >= self.min_d

            # Check breaking conditions
            if n_its is not None:
                if t > n_its: 
                    break

            trace.append([t, Lt, pit, g1t, g2t, st])

            if ref_panel is None:
                refht = "NA"
            # TODO: replace with logging
            sys.stderr.write("Iteration:%d" % t + str([Lt, refht, pit, g1t, 
                                g2t, ham_nogaps(st, self.consensus)]) + "\n")

            Ltold = Lt
            Tt, Lt = self.recalc_T(pit, g1t, st, g2t, old_st, changed_inds)

            if debug:
                plotter.plot_genome(self.ds, Tt, st, debug_minor)

            # Store variables
            # TODO: why are these stored elsewhere too? duplication.
            self.st = st
            self.Tt = Tt
            self.gt = gt
            self.pit = pit
        
            # If probability has become 1; nothing should theoretically occur after this
            if sum(Tt[:, 1]) == 0:
                break

            # Recalculate scalars
            old_pi = pit
            pit = self.recalc_pi(Tt)
            pit = max(min_pi,min(max_pi,pit))

            g1t = self.recalc_gamma(Tt, st, 0)
            g1t = min(max(gt, 0.0001), 0.05)

            g2t = gt
            if not one_gamma:
                g2t = self.recalc_gamma(Tt, st, 1)
                g2t = min(max(g2t, 0.0001), 0.05)
    
            # Recalculate string
            old_st = st
            if ref_panel is not None:
                curr_ri, refht, st = self.recalc_st_refs(Tt, ref_panel, curr_ri)
            elif fixed_st is None:
                st = self.recalc_st(Tt, self.min_d)     
            else:
                st = fixed_st
            changed_inds = [sti for sti in range(len(st)) if st[sti] != old_st[sti]]

            if np.abs(Ltold-Lt) < 0.001 and np.abs(old_pi-pit) < 0.001 and old_st == st:
                break

            t += 1

        trace.append([t, Lt, pit, g1t, g2t, st])
        assert pit <= max_pi
        sys.stderr.write("Iteration:%d" % t + str([Lt, pit, gt, mut, ham_nogaps(st, self.consensus)]) + "\n")

        if debug:
            plotter.plot_genome(self.ds, Tt, st, debug_minor)

        return trace
