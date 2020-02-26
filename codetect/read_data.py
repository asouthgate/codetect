import sys
import random
import numpy as np

def ham(s1, s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])

class ReadData():
    """ Class that stores read alignments.

    Attributes:
        X: data array of alignments.
        CONSENSUS: consensus constant for major population.
    """
    def __init__(self,X,consensus):        
        self.CONSENSUS = consensus
        assert self.CONSENSUS[0] in [0,1,2,3]
        self.X = X
        # Build V index
        sys.stderr.write("Building V index\n")
        self.V_INDEX = self.build_Vindex()
#        # Subsample
        sys.stderr.write("Subsampling across the reference\n")
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
        self.M = self.reads2mat()
        # Mask low variance positions
        sys.stderr.write("Masking low variance positions\n")
        self.mask_low_variance_positions()
        # Rebuild V index
        sys.stderr.write("Rebuilding V index\n")
        self.V_INDEX = self.build_Vindex()
        sys.stderr.write("Recalculating matrix M\n")
        # Build M matrix
        self.M = self.reads2mat()
        # Calculate the number of mismatches
        [Xi.calc_nm(self.CONSENSUS) for Xi in self.X]
        self.test_v_array()

    def test_v_array(self):
        for i,Xi in enumerate(self.X):
            for pos,b in Xi.get_aln():
                assert i in self.V_INDEX[pos][b], (i, self.V_INDEX[pos])

    def simple_subsample(self, N_SAMPLES=500):
        return np.random.choice(self.X, N_SAMPLES, replace=False)

    def subsample(self, N_SAMPLES=2000):
        pos_start_arr = [[] for i in range(len(self.CONSENSUS))]
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
        Vindex = [[[] for c in range(4)] for i in range(len(self.CONSENSUS))]
        for i,Xi in enumerate(self.X):
            for pos, c in Xi.get_aln():
                Vindex[pos][c].append(i)
        return Vindex
       
    def reads2mat(self):
        """ Build a per position base frequency dist matrix """
        mat = np.zeros(shape=(len(self.CONSENSUS),4))
        for i, Xi in enumerate(self.X):
            for pos,c in Xi.get_aln():
                mat[pos,c] += Xi.count
        for ri in range(len(mat)):
            if sum(mat[ri]) > 0:
                mat[ri] /= sum(mat[ri])
        return mat

    def mask_low_variance_positions(self,t=1.0,mindepth=20):
        """ Mask uninteresting positions of the matrix. """
        def gen_index_remap(L,delinds):
            shift = [0 for i in range(L)]
            for si in range(len(shift)):
                shift[si] = shift[si-1]
                if si-1 in delinds:
                    shift[si] += 1
            return {si:si-s for si,s in enumerate(shift)}
        delinds = set()
        for ri,row in enumerate(self.M):
            if max(row) > t or max(row) == 0:
                delinds.add(ri)
            if sum([len(k) for k in self.V_INDEX[ri]]) < mindepth:
                delinds.add(ri)
        sys.stderr.write("Deleting %d positions\n"% len(delinds))
        if len(delinds) == 0:
            return True
        if len(delinds) == len(self.CONSENSUS):
            raise ValueError("no sites remaining")
        newCONS = [c for ci,c in enumerate(self.CONSENSUS) if ci not in delinds]
        Msub = [row for ri,row in enumerate(self.M) if ri not in delinds]
        delindsl = sorted(delinds)
        # Get rid of any empty strings after deletion
        newX = []
        remap = gen_index_remap(len(self.CONSENSUS),delinds)
        for Xi in self.X:
            #SAFEGUARD
            prepos = Xi.pos
            prestr = Xi.get_string()
            prestr2 = "".join([c for ci,c in enumerate(prestr) if Xi.pos+ci not in delinds])
            res = Xi.del_inds(delindsl,remap)
            poststr = Xi.get_string()        
            pushback = 0
            for ci,c in enumerate(prestr2):
                assert poststr[ci] == c
            if res:
                newX.append(Xi)
        self.X = newX
        self.CONSENSUS = newCONS
        self.M = np.array(Msub)
        self.minor = [c for ci,c in enumerate(self.minor) if ci not in delinds]
        assert len(self.M) == len(self.CONSENSUS) 
