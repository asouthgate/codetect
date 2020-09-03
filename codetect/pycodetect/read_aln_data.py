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
    def __init__(self,X,reference):        
        self._reference = reference
        assert self._reference[0] in [0,1,2,3]
        self.X = X
        # Establish valid indices for iteration
        self.VALID_INDICES = [i for i in range(len(self._reference))]
        # Build V index
        sys.stderr.write("Building V index\n")
        self.V_INDEX = self.build_Vindex()
#        # Subsample
#        sys.stderr.write("Subsampling across the reference\n")
#        self.X = self.subsample()
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
        # Calculate the number of mismatches
        [Xi.calc_nm_major(self._CONSENSUS) for Xi in self.X]
        self.test_v_array()

    def filter(self,n):
        """ Mask low-variance positions. """
        # Mask low variance positions
        sys.stderr.write("Masking low variance positions\n")
        self.VALID_INDICES = self.get_indices_max_window(n=n)
        self.test_v_array() 

    def pos2reads(self,i):
        return functools.reduce(lambda a,b : a+b,self.V_INDEX[i])

    def get_consensus(self):
        return self._CONSENSUS

    def test_v_array(self):
        for i,Xi in enumerate(self.X):
            for pos,b in Xi.get_aln():
                assert i in self.V_INDEX[pos][b], (i, self.V_INDEX[pos])

    def simple_subsample(self, N_SAMPLES=500):
        """
        Subsample by choosing alignments randomly.
        """
        return np.random.choice(self.X, N_SAMPLES, replace=False)

    def subsample(self, N_SAMPLES=2000):
        """
        Subsample across the reference to correct for depth imbalance.
        """
        #TODO: check math for legitimacy
        pos_start_arr = [[] for i in range(len(self.get_consensus()))]
        for i,Xi in enumerate(self.X):
            pos_start_arr[Xi.pos].append(i)
        subsample = []
        while len(subsample) < N_SAMPLES:
            for k in range(len(pos_start_arr)):
                l = pos_start_arr[k]
                if len(l) > 0:
                    choiceli = random.randint(0,len(l)-1)
                    choicexi = l[choiceli]
                    del l[choiceli]
                    subsample.append(self.X[choicexi])
        return subsample        

    def deduplicate(self, X):
        """ Get unique reads and update their counts. """
        seqcountd = {}
        for i,Xi in enumerate(X):
            if str(Xi) in seqcountd:
                seqcountd[str(Xi)].count += 1
            else:
                seqcountd[str(Xi)] = Xi
        return [Xi for s,Xi in seqcountd.items()]

    def build_Vindex(self): 
        """ Build a reference pos with c -> reads mapping to pos index. """
        Vindex = [[[] for c in range(5)] for i in range(len(self._reference))]
        for i,Xi in enumerate(self.X):
            for pos, c in Xi.get_aln():
                Vindex[pos][c].append(i)
        return Vindex
       
    def reads2mats(self):
        """ Build a per position base frequency dist matrix """
        mat = np.zeros(shape=(len(self._reference),4))
        Cmat = np.zeros(shape=(len(self._reference),4))
        for i, Xi in enumerate(self.X):
            for pos,c in Xi.get_aln():
                mat[pos,c] += Xi.count
                Cmat[pos,c] += Xi.count
        for ri in range(len(mat)):
            if sum(mat[ri]) > 0:
                mat[ri] /= sum(mat[ri])
        return Cmat, mat

    def get_indices_max_window(self, n=100, windowsize=200, mindepth=20):
        """ Mask uninteresting positions of the matrix if not in top n  """
        n_windows = int(len(self.V_INDEX)/windowsize)
        n_per_window = int(n/n_windows)
        assert n_per_window > 1
        max_indices = []
        for winl in range(0,len(self.V_INDEX)-windowsize,windowsize):
            scores = {}
            winu = winl+windowsize
            for ri in range(winl,winu):
                row = self.M[ri]
                if sum([len(k) for k in self.V_INDEX[ri]]) > mindepth:
                    scores[ri] = sorted(row)[-2]
            sort = sorted([(i,q) for i,q in scores.items()],key=lambda x:x[1])
            wmaxs = [(i,q) for i,q in sort[-n_per_window:]]
            print(wmaxs)
            wmaxinds = [i for i,q in wmaxs]
            max_indices += wmaxinds
        return np.array(max_indices)

    def get_indices(self,t=0.97,mindepth=20):
        """ Mask uninteresting positions of the matrix. """
        delinds = set()
        for ri,row in enumerate(self.M):
            if max(row) > t or max(row) == 0:
                delinds.add(ri)
            if sum([len(k) for k in self.V_INDEX[ri]]) < mindepth:
                delinds.add(ri)
        sys.stderr.write("Deleting %d positions\n"% len(delinds))
        if len(delinds) == 0:
            return [j for j in range(len(self._reference))]
        if len(delinds) == len(self._reference):
            raise ValueError("no sites remaining")
        #TODO: figure out if and when it is legitimate to delete these bases from the reads entirely
        return np.array([i for i in range(len(self._reference)) if i not in delinds])
