import sys
import random
import numpy as np
import functools
from pycodetect.utils import ham

class ReadAlnData():
    """ Class that stores read alignments.

    Attributes:
        X: data array of alignments.
        V_INDEX: mapping of position,base -> reads with that combination.
        M: frequency distribution of bases across the alignment.
        reference: reference to which alignments are mapped.
    """
    def __init__(self, X, reference):        
        self._reference = reference
        assert self._reference[0] in [0,1,2,3]
        self.X = X
        # Establish valid indices for iteration
        self.VALID_INDICES = [i for i in range(len(self._reference))]
        # Build V index
        sys.stderr.write("Building V index\n")
        self.V_INDEX = self.build_Vindex()

        X = self.subsample_stratified(X, 5000)

        sys.stderr.write("%d reads survived\n" % len(self.X))
        # Rebuild V index
        sys.stderr.write("Rebuilding V index\n")
        self.V_INDEX = self.build_Vindex()
        sys.stderr.write("Counting duplicate reads\n")
        # Deduplicate/count datapoints in X
        self.X = self.deduplicate(self.X)
        sys.stderr.write("Rebuilding V index\n")
        # Rebuild V index
        self.V_INDEX = self.build_Vindex()
        sys.stderr.write("Generating starting matrix M\n")
        # Build M matrix
        self.C, self.M = self.reads2mats()
        # Recompute consensus to be actual consensus
        self._CONSENSUS = tuple(np.argmax(v) for v in self.M)
        assert 4 not in self._CONSENSUS
        # Calculate the number of mismatches
        self.test_v_array()
        self.n_reads = sum([Xi.count for Xi in self.X])

    def filter(self, n, mode="window"):
        """ Mask low-variance positions. """
        # Mask low variance positions
        sys.stderr.write("Masking low variance positions\n")
        if mode == "window":
            self.VALID_INDICES = self.get_indices_max_window(n=n)
        elif mode == "rank":
            self.VALID_INDICES = self.get_indices_max(n=n)            
        self.test_v_array() 

    def pos2reads(self,i):
        return functools.reduce(lambda a, b : a + b, self.V_INDEX[i])

    def get_consensus(self):
        return self._CONSENSUS

    def test_v_array(self):
        for i, Xi in enumerate(self.X):
            for pos, b in Xi.get_aln_tuples():
                assert i in self.V_INDEX[pos][b], (i, self.V_INDEX[pos])

    def simple_subsample(self, N_SAMPLES=500):
        """
        Subsample by choosing alignments randomly.
        """
        return np.random.choice(self.X, N_SAMPLES, replace=False)

    def subsample(self, N_SAMPLES=2000):
        self.X = self.subsample_stratified(self.X, N_SAMPLES=N_SAMPLES)

    def subsample_stratified(self, X, N_SAMPLES=2000):
        """
        Subsample across the reference to correct for depth imbalance.
        A form of stratified sampling.

        Args:
            X: list of ReadAln objects.
        """
        #TODO: check math for legitimacy
        pos_start_arr = [[] for i in range(len(self.get_consensus()))]
        for i, Xi in enumerate(X):
            pos_start_arr[Xi.pos].append(i)
        subsample = []
        while len(subsample) < N_SAMPLES:
            for k in range(len(pos_start_arr)):
                l = pos_start_arr[k]
                if len(l) > 0:
                    choiceli = random.randint(0,len(l)-1)
                    choicexi = l[choiceli]
                    del l[choiceli]
                    subsample.append(X[choicexi])
        return subsample        

    def deduplicate(self, X):
        """ Get unique reads and update their counts. """
        seqcountd = {}
        for i, Xi in enumerate(X):
            if str(Xi) in seqcountd:
                seqcountd[str(Xi)].count += 1
            else:
                seqcountd[str(Xi)] = Xi
        return [Xi for s,Xi in seqcountd.items()]

    def build_Vindex(self): 
        """ Build a reference pos with c -> reads mapping to pos index. """
        Vindex = [[[] for c in range(4)] for i in range(len(self._reference))]
        for i, Xi in enumerate(self.X):
            for pos, c in Xi.get_aln_tuples():
                assert pos < len(self._reference), (pos, len(self._reference))
                Vindex[pos][c].append(i)
        return Vindex
       
    def reads2mats(self):
        """ Build a per position base frequency dist matrix """
        mat = np.zeros(shape=(len(self._reference), 4))
        Cmat = np.zeros(shape=(len(self._reference), 4))
        for i, Xi in enumerate(self.X):
            for pos, c in Xi.get_aln_tuples():
                mat[pos,c] += Xi.count
                Cmat[pos,c] += Xi.count
        for ri in range(len(mat)):
            if sum(mat[ri]) > 0:
                mat[ri] /= sum(mat[ri])
        return Cmat, mat

    def get_indices_max(self, n=100, mindepth=30):
        """ Mask uninteresting positions of the matrix if not in top n. """
        max_indices = []
        scores = {}
        for ri in range(0, len(self.M)):
            row = self.M[ri]
            if sum([len(k) for k in self.V_INDEX[ri]]) > mindepth:
                scores[ri] = sorted(row)[-2]
        sort = sorted([(i,q) for i,q in scores.items()],key=lambda x:x[1])
        wmaxs = [(i,q) for i,q in sort[-n:]]
        max_indices = [i for i,q in wmaxs]
        for mi in max_indices:
            assert sum([len(k) for k in self.V_INDEX[mi]]) > mindepth
        return np.array(max_indices)

    def get_indices_max_window(self, n=100, windowsize=200, mindepth=20):
        """ Mask uninteresting positions of the matrix if not in top n  

        """
        n_windows = int(len(self.V_INDEX) / windowsize)
        n_per_window = max(int(n / n_windows), 2)
        assert n_per_window > 1, ("n:%d,n_windows:%d,n_per_window:%d,len(vindex):%d,windowsize:%d" % (n,n_windows,n_per_window,len(self.V_INDEX),windowsize))
        max_indices = []
        for winl in range(0, len(self.V_INDEX) - windowsize, windowsize):
            scores = {}
            winu = winl + windowsize
            for ri in range(winl,winu):
                row = self.M[ri]
                if sum([len(k) for k in self.V_INDEX[ri]]) > mindepth:
                    scores[ri] = sorted(row)[-2]
            sort = sorted([(i,q) for i,q in scores.items()],key=lambda x:x[1])
            wmaxs = [(i,q) for i,q in sort[-n_per_window:]]
            wmaxinds = [i for i,q in wmaxs]
            max_indices += wmaxinds
        return np.array(max_indices)
