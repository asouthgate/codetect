import numpy as np
import sys
import pycodetect.plotter as plotter
from pycodetect.utils import ham, ham_nogaps
from pycodetect.likelihood_calculator import LikelihoodCalculator
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
        self.min_cov = 0
        self.consensus = rd.get_consensus()
        self.min_d = min_d
        self.llc = None

    def recalc_T(self, pi, g0, st, g1, prev_st, st_changed_bases=None):
        """ Recalculate the T array given new parameters.

        Args:
            pi: mixing proportion
            g0: gamma of cluster 0
            g1: gamma of cluster 1
            st: proposed st
            prev_st: previous st
            changed_inds: indices such that st[i] != prev_st[i]        

        Returns:
            Tt: T array at current iteration
            Lt: Likelihood at current iteration            
        """

        for j, prevb in st_changed_bases: assert prev_st[j] == prevb
        res = []
        # Also calculate the log likelihood while we're at it
        Lt = 0
        for i, Xi in enumerate(self.rd.X):
            pairT, logL = self.llc.cal_P_clusters_given_read(i, Xi, pi, g0, g1, st, self.consensus, st_changed_bases)
            res.append(pairT)
            Lt += logL
        Tt = np.array(res)
        return Tt, Lt

    def recalc_gk(self, T, S, k):
        # TODO: replace cal_ham with a call to a cache
        """ Recalculate gamma for the kth cluster. """
        numo = sum([T[i,k] * Xi.count * Xi.cal_ham(S) for i, Xi in enumerate(self.rd.X)])
        deno = sum([T[i,k] * Xi.count * len(Xi.get_aln_tuples()) for i, Xi in enumerate(self.rd.X)])
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
        for k in self.rd.VALID_INDICES:
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
        for k in self.rd.VALID_INDICES:
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
        for bi in self.rd.VALID_INDICES:
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
        conscores = [W[bi, self.consensus[bi]] for bi in self.rd.VALID_INDICES]
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
        return sum([T[i,0] * self.rd.X[i].count for i in range(len(T))]) / self.rd.n_reads

    def init_st_random(self, M):
        st = [c for c in self.consensus]
        for vi, vt in enumerate(M):
            st[vi] = np.random.choice([j for j in range(4)], p=vt)
        return st

    def init_st(self, M):
        """ Initiallize st by chosing second most frequent base at each position. """
        st = [c for c in self.consensus]
        second_best = []
        for vi in self.rd.VALID_INDICES:
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
            if i not in self.rd.VALID_INDICES:
                assert st[i] == self.consensus[i]

    def calc_L0(self):
        """ Calculate the likelihood under the null. """
        g = self.recalc_gk(np.array([[1,0] for j in range(len(self.rd.X))]), self.consensus(), 0)
        return self.llc.calc_data_log_likelihood(self.rd, self.consensus, g, g, 1, self.consensus, None)

    def estimate(self, ref_panel=None, n_its=None, random_init=False, debug=False,
             debug_minor=None, max_pi=1.0, min_pi=0.5, fixed_st=None,
             one_gamma=True, pit_init=0.5):
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
            trace: list of values of [t, Lt, pit, g0t, g1t, st] for each
                   iteration.
        """
        # Abitrary initialization
        pit = pit_init
        g0t = 0.01
        g1t = 0.01
        
        # TODO: could move to a function
        # Initialization of st
        if fixed_st is None:
            if random_init:
                st = self.init_st_random(self.rd.M)
            else:
                if ref_panel is None:
                    st = self.init_st(self.rd.M)
                else:
                    curr_ri, refht, st = ref_panel.get_ref_maximizing_second(self.rd.M)
        else:
            st = fixed_st

        # Some safety checks
        assert len(st) == len(self.consensus), (len(st), len(self.consensus))
        for row in self.rd.M:
            for v in row:
                assert not np.isnan(v)
        assert len(self.rd.X) > 0
        for i, Xi in enumerate(self.rd.X):
            for pos, bk in Xi.get_aln_tuples():
                assert i in self.rd.V_INDEX[pos][bk]
                assert self.rd.M[pos, bk] > 0
        for m in self.rd.M:
            if sum([q for q in m]) > 0:
                assert sum(m) > 0.98, m
        assert len(self.rd.V_INDEX) == len(self.rd.M)
        assert 0 <= g0t <= 1, g0t
        assert ham(st, self.consensus) >= self.min_d, ham(st, self.consensus)

        # Initialize the llc with st
        # TODO: find a solution that allows llc to be updated
        # with changed bases, but safely. Some form of
        # safety check or validation
        self.llc = LikelihoodCalculator(self.rd, st, g0t)


        trace = []
        t = 0
        Lt = self.llc.calc_data_log_likelihood(self.rd, st, g0t, g1t, pit, self.consensus, [])
        st_changed_bases = []
        old_st = st
        while True: 

            assert pit <= max_pi
            assert ham(st, self.consensus) >= self.min_d

            # Check breaking conditions
            if n_its is not None:
                if t > n_its: 
                    break

            trace.append([t, Lt, pit, g0t, g1t, st])

            if ref_panel is None:
                refht = "NA"
            # TODO: replace with logging
            sys.stderr.write("Iteration:%d" % t + str([Lt, refht, pit, g0t, 
                                g1t, ham_nogaps(st, self.consensus)]) + "\n")

            Ltold = Lt
            for bi, b in st_changed_bases: assert old_st[bi] == b
            Tt, Lt = self.recalc_T(pit, g0t, st, g1t, old_st, st_changed_bases)

            if debug:
                plotter.plot_genome(self.rd, Tt, st, debug_minor)
        
            # If probability has become 1; nothing should theoretically occur after this
            if sum(Tt[:, 1]) == 0:
                break

            # Recalculate scalars
            old_pi = pit
            pit = self.recalc_pi(Tt)
            pit = max(min_pi,min(max_pi,pit))

            g0t = self.recalc_gk(Tt, self.rd.get_consensus(), 0)
            g0t = min(max(g0t, 0.0001), 0.05)

            g1t = g0t
            if not one_gamma:
                g1t = self.recalc_gk(Tt, st, 1)
                g1t = min(max(g1t, 0.0001), 0.05)
    
            # Recalculate string
            old_st = st
            if ref_panel is not None:
                curr_ri, refht, st = self.recalc_st_refs(Tt, ref_panel, curr_ri)
            elif fixed_st is None:
                st = self.recalc_st(Tt, self.min_d)     
            else:
                st = fixed_st
            st_changed_bases = [(sti, old_st[sti]) for sti in range(len(st)) if st[sti] != old_st[sti]]

            if np.abs(Ltold-Lt) < 0.001 and np.abs(old_pi-pit) < 0.001 and old_st == st:
                break

            t += 1

        Tt, Lt = self.recalc_T(pit, g0t, st, g1t, old_st, st_changed_bases)
        trace.append([t, Lt, pit, g0t, g1t, st])
        assert pit <= max_pi
        sys.stderr.write("Iteration:%d" % t + str([Lt, refht, pit, g0t, g1t, ham_nogaps(st, self.consensus)]) + "\n")

        if debug:
            plotter.plot_genome(self.rd, Tt, st, debug_minor)

        return trace, refht
