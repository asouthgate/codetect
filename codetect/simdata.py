import numpy as np
import random
import matplotlib.pyplot as plt
from  em import *
from aln import ReadAln

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
#        print(self.CONSENSUS)
        assert self.CONSENSUS[0] in [0,1,2,3]
        self.X = X
        # Build V index
        sys.stderr.write("Building V index\n")
        self.V_INDEX = self.build_Vindex()
        # Subsample
        sys.stderr.write("Subsampling across the reference\n")
        self.X = self.simple_subsample()
        # Rebuild V index
        sys.stderr.write("Rebuilding V index\n")
        self.V_INDEX = self.build_Vindex()
#        sys.stderr.write("Counting duplicate reads\n")
        # Deduplicate/count datapoints in X
#        self.X = self.deduplicate(self.X)
#        sys.stderr.write("Rebuilding V index\n")
        # Rebuild V index
#        self.V_INDEX = self.build_Vindex()
#        sys.stderr.write("Generating starting matrix M\n")
        # Build M matrix
        self.M = self.reads2mat()
        # Mask low variance positions
#        sys.stderr.write("Masking low variance positions\n")
#        self.mask_low_variance_positions()
        # Rebuild V index
#        sys.stderr.write("Rebuilding V index\n")
#        self.V_INDEX = self.build_Vindex()

    def simple_subsample(self, N_SAMPLES=1000):
        return np.random.choice(self.X, N_SAMPLES, replace=False)

    def subsample(self, N_SAMPLES=1000):
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
            mat[ri] /= sum(mat[ri])
        return mat

    def mask_low_variance_positions(self,t=1.0,mindepth=0):
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
            if max(row) > t:
                delinds.add(ri)
        if len(delinds) == len(self.CONSENSUS):
            raise ValueError("no sites remaining")
        newCONS = [c for ci,c in enumerate(self.CONSENSUS) if ci not in delinds]
        Msub = [row for ri,row in enumerate(self.M) if ri not in delinds]
        delindsl = sorted(delinds)
        # Get rid of any empty strings after deletion
        newX = []
        for Xi in self.X:
            res = Xi.del_inds(delindsl,gen_index_remap(len(self.CONSENSUS),delinds))
            if res:
                newX.append(Xi)
        self.X = newX
        self.CONSENSUS = newCONS
        self.M = np.array(Msub)
        assert len(self.M) == len(self.CONSENSUS) 

class DataSimulator(ReadData):
    def __init__(self, N_READS, READ_LENGTH, GENOME_LENGTH, GAMMA, PI, D):
        self.N_READS = N_READS
        self.READ_LENGTH = READ_LENGTH
        self.GENOME_LENGTH = GENOME_LENGTH
        self.GAMMA = GAMMA
        self.PI = PI
        self.D = D    
        sys.stderr.write("Generating population\n")
        self.major, self.minor = self.gen_pop(GENOME_LENGTH, D)
        self.CONSENSUS = self.major
        self.true_ham = ham(self.major,self.minor)
        assert self.true_ham > 0.0
        self.true_pid = self.true_ham/len(self.major)
        self.POPULATION = [self.major,self.minor]
        sys.stderr.write("Simulating reads\n")
        self.CONSENSUS = [c2i[c] for c in self.CONSENSUS]
        self.X = self.sample_reads()
        super(DataSimulator,self).__init__(self.X,self.CONSENSUS)

        self.N_READS = sum([Xi.count for Xi in self.X])
        # OPTIONAL
        self.true_pi = 0
        self.true_gamma = []
        for i,Xi in enumerate(self.X):
            if Xi.z == 0:
                self.true_pi += Xi.count
                self.true_gamma.append(Xi.calc_nm(self.CONSENSUS)/len(Xi.base_pos_pairs))
            else:
                self.true_gamma.append(Xi.calc_nm([c2i[c] for c in self.minor])/len(Xi.base_pos_pairs))
        self.true_gamma = sum(self.true_gamma)/len(self.true_gamma)
        self.true_pi /= len(self.X)
        assert self.true_pi > 0
#        print("COUNTD")
#        plt.hist([Xi.count for Xi in self.X])
#        plt.show()

        nms = np.array([Xi.calc_nm(self.CONSENSUS) for Xi in self.X])
        assert len(self.V_INDEX) == len(self.M)
        readdists = [Xi.nm for Xi in self.X if Xi.z == 0]
        plt.hist(readdists,bins=100)
        readdists = [Xi.nm for Xi in self.X if Xi.z == 1]
        plt.hist(readdists,bins=100)
        plt.show()
#        assert self.CONSENSUS == self.major

    def gen_pop(self,L,D):
        major = [random.choice("ATCG") for i in range(self.GENOME_LENGTH)]
        minor = [c for c in major]
        mutpos = np.random.choice(len(major), D, replace=False)
        for j in mutpos:
            minor[j] = random.choice([c for c in "ATCG" if c != major[j]])
        return "".join(major),"".join(minor)

    def sample_reads(self):
        w = [self.PI, 1-self.PI]
        assert w[0] == max(w), "first sequence should be the major var"
        X = []
        for i in range(self.N_READS):
            seqi = np.random.choice([0,1],p=w)
            assert len(self.POPULATION) > 1
            seq = self.POPULATION[seqi]
            randpos = np.random.randint(0,len(seq)-self.READ_LENGTH+1)
            sampinds = [randpos+l for l in range(self.READ_LENGTH)]
            aln = ReadAln(i)
            for si in sampinds:
                roll = random.uniform(0,1)
                c = seq[si]
                if roll < self.GAMMA:
                    alt = random.choice([z for z in "ATCG" if z != c])
                    aln.append_mapped_base(si,aln.c2i(alt))
                else:
                    aln.append_mapped_base(si,aln.c2i(c))                    
            aln.z = seqi
            X.append(aln)
        return X

if __name__ == "__main__":
#    def __init__(self, N_READS, READ_LENGTH, GENOME_LENGTH, GAMMA, PI, D):
    for h in [1, 2, 3, 4, 5, 10, 15, 20, 25, 30][::-1]:
        PI = 0.8
#        D = 0.02
        D = h
        GAMMA = 0.01
        READLEN = 200
        L = 2000
        NREADS = 1000
        ds = DataSimulator(NREADS,READLEN,L,GAMMA,PI,D) 
        print("  truepi, truegamma, trueham")
        print(" ",ds.true_pi,ds.true_gamma,ds.true_ham)
        print("*********PASSING TO EM*********")
        NITS=10
        EPS = int(0.01*L)
        assert len(ds.X) == NREADS
    #    posreads = []
    #    for Xi in ds.X:
    #        posreads.append((Xi.pos, Xi.get_string()))
    #    import mixtest_sub as mts
    #    mts.run(posreads, NREADS, READLEN, GAMMA, D, L, PI, ds.CONSENSUS, ds.true_pi, ds.minor,NITS)
        em = EM(ds.X, ds.M, ds.V_INDEX, ds.CONSENSUS, EPS)
        em.do2(NITS)


