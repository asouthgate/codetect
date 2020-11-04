import numpy as np
from pycodetect.utils import logsumexp

def cal_read_logP_simple(read, gamma, st):
    sumo = 0
    for pos, c in read.get_aln_tuples():
        if c == st[pos]:
            sumo += np.log(1-gamma)
        else:
            sumo += np.log(gamma)
    return sumo

class NMCache():
    """ Cache for storing and optimized updating the number of mismatches
        between reads and dynamic reference.

        Args:
            read_aln_data: ReadAlnData object
            inistr: initial s1    
    """
    def __init__(self, read_aln_data, initstr):
        self.nmarr = np.zeros((2, len(read_aln_data.X)))
        self.init_nmarr(read_aln_data, initstr)

    def __getitem__(self, args):
        return self.nmarr[args[0], args[1]]

    def init_nmarr(self, ds, altstr):
        """ Calculate the number of mismatches between reads and a string. """
        nmarr = np.zeros((2, len(ds.X)))
        for i, xi in enumerate(ds.X):
            nmarr[0, i] = xi.cal_ham(ds.get_consensus())
            nmarr[1, i] = xi.cal_ham(altstr)
            assert nmarr[0, i] >= 0
            assert nmarr[1, i] >= 0
        self.nmarr = nmarr

    def set(self, ci, ri, val):
        self.nmarr[ci, ri] = val

    def update(self, ci, ri, inc):
        """ Update the number of mismatches between reads and a string.

        Args:
            ri: ReadData object
            inc: +1 or -1 
         """
        assert inc in [-1,1]
        self.nmarr[ci, ri] += inc

class LikelihoodCalculator():
    """ 
    Optimized likelihood calculator using caching.

    Args:
        read_aln_data: ReadAlnData object
        init_st: string for first likelihood calculation
    """
    def __init__(self, read_aln_data, init_st, gamma):
        self.nm_cache = NMCache(read_aln_data, init_st)
        self.llcache = np.zeros((2, len(read_aln_data.X)))
        for ri, read in enumerate(read_aln_data.X):
            self.cal_logP_read(ri, read, 0, gamma, read_aln_data.get_consensus())
            self.cal_logP_read(ri, read, 1, gamma, init_st)

    def cal_logP_read(self, ri, read, ci, gamma, st, st_changed_bases=None):
        # TODO: should index of read be put inside read if they are being passed around like ids?
        """ Calculate log P(Aln | g,st) with caching.

        Args:
            ri: read index
            read: ReadAln object
            ci: cluster index.
            gamma: gamma parameter.
            st: cluster string.
            changed_bases: (bi,b) tuples where st differs from the st_prev of the last call.
        """
        nothing_changed_for_read = True
        if st_changed_bases is not None:
            for bp, prevb in st_changed_bases:
                if bp in read.map:
                    c = read.map[bp]
                    if c == st[bp]:
                        assert c != prevb
                        # Previously it didnt match, now it does, so decrement the mismatch cache
                        nothing_changed_for_read = False
                        self.nm_cache.update(ci, ri, -1)
                    elif c == prevb:
                        nothing_changed_for_read = False
                        self.nm_cache.update(ci, ri, 1)
                    else:
                        # Previously it didnt match, still doesnt, do nothing
                        pass
        else:
            nm = 0
            for bp, c in read.get_aln_tuples():
                if c == st[bp]:
                    pass
                else:
                    nm += 1
            nothing_changed_for_read = False
            self.nm_cache.set(ci, ri, nm)

        #! TODO: reconsider. Something always changes for read in EM
        #        gamma! and if not, gamma needs to be stored too to know this
        nothing_changed_for_read = False
        if not nothing_changed_for_read:
            matches = len(read.map) - self.nm_cache[ci, ri]
            assert matches + self.nm_cache[ci, ri] > 0
            self.llcache[ci, ri] = ( np.log(gamma) * self.nm_cache[ci, ri] ) + ( np.log(1-gamma) * matches )
            assert self.llcache[ci, ri] != 0
        assert self.llcache[ci, ri] != 0
        return self.llcache[ci, ri]

    def calc_data_log_likelihood(self, rd, st, g0, g1, pi, consensus, st_changed_bases):
        """
        Calculate the log likelihood of data given parameters.

        Args:
            rd: ReadAlnData object
            st: current string.
            g0: cluster 0 gamma.
            g1: cluster 1 gamma.
            pi: cluster proportion.
        """
        # We now seek the log likelihood 
        # = log(P(X_i | Zi=1,theta)pi + P(Xi | Zi=2,theta)(1-pi))
        # TODO: if we add an index to denote whether a read likelihood has changed
        #       we can increment and decrement a score
        sumo = 0
        for i, Xi in enumerate(rd.X):
            # TODO: replace read calculating its own LL with cache?
            a = self.cal_logP_read(i, Xi, 0, g0, consensus, [])
            b = self.cal_logP_read(i, Xi, 1, g1, st, st_changed_bases)
            lw1 = np.log(pi)
            lw2 = np.log(1 - pi)
            # TODO: Xi.count accessed -- breaking encapsulation for optimization. Better way?
            sumo += ( logsumexp([a + lw1, b + lw2]) * Xi.count ) 
        return sumo

    def cal_P_clusters_given_read(self, ri, Xi, pi, g0, g1, st, consensus, st_changed_bases):
        """ Calculate the ith membership conditional probability array element.
        
        Args:
            Xi: ith ReadAln object
            pi: mixture model proportion
            g0: gamma parameter for cluster 0
            st: cluster 1 string
            g1: gamma parameter for cluster 1
            consensus: consensus used for cluster 0 string
            st_changed_bases: ind, base tuples such that st[ind] != base

        Returns:
            tp: T array: P(Z_i = k | X_i)
            lp: Likelihood of Xi given cluster j
        """

        l1 = self.cal_logP_read(ri, Xi, 0, g0, consensus, [])
        l2 = self.cal_logP_read(ri, Xi, 1, g1, st, st_changed_bases)
#        l1val = cal_read_logP_simple(Xi, g0, consensus) 
#        assert l1val == l1, (l1val, l1)
#        assert cal_read_logP_simple(Xi, g1, st) == l2

        lw1 = np.log(pi)
        lw2 = np.log(1-pi)

        exp1 = np.exp(l1 + lw1)
        exp2 = np.exp(l2 + lw2)

        c = exp1 + exp2
        assert 0 < c <= 1.01

        t1i = exp1/c
        t2i = exp2/c

        tp = np.array([t1i,t2i])
        assert sum(tp) > 0.999, sum(tp)
        return tp, np.log(c)
 

