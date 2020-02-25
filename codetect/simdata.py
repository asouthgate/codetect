import numpy as np
import random
import matplotlib.pyplot as plt
from  em import *
from aln import ReadAln
import copy
from read_data import *

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

        cov0 = np.zeros(len(self.CONSENSUS))
        cov1 = np.zeros(len(self.CONSENSUS))
        for Xi in self.X:
            if Xi.z == 0:
                cov0[Xi.pos:Xi.pos+len(Xi.base_pos_pairs)] += 1
            else:
                assert Xi.z == 1
                cov1[Xi.pos:Xi.pos+len(Xi.base_pos_pairs)] += 1
        self.COV = cov0+cov1
        for k in range(len(self.COV)):
            assert self.COV[k] == sum([len(l) for l in self.V_INDEX[k]])
        hamarr = np.zeros(len(self.CONSENSUS))
        for i,hi in enumerate(self.CONSENSUS):
            if self.CONSENSUS[i] != c2i[self.minor[i]]:
                hamarr[i] = 1
        mc0 = max(cov0)
        cov0 /= mc0
        cov1 /= mc0
        plt.plot(hamarr, color='red')
        plt.plot(cov0)
        plt.plot(cov1)
        plt.show()

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
        sys.stderr.write("Mutating positions" + str(sorted(mutpos)) + "\n")
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
    for h in range(18,25):
#        h = 4
        PI = 0.95
    #        D = 0.02
        D = h
        GAMMA = 0.03
        READLEN = 200
        L = 2000
        NREADS = 5000
        ds = DataSimulator(NREADS,READLEN,L,GAMMA,PI,D) 
        print("  truepi, truegamma, trueham")
        print(" ",ds.true_pi,ds.true_gamma,ds.true_ham)
        print("*********PASSING TO EM WITH %d MUTATIONS*********" % h)
        NITS=10
        EPS = 20
#        assert len(ds.X) == NREADS
    #    posreads = []
    #    for Xi in ds.X:
    #        posreads.append((Xi.pos, Xi.get_string()))
    #    import mixtest_sub as mts
    #    mts.run(posreads, NREADS, READLEN, GAMMA, D, L, PI, ds.CONSENSUS, ds.true_pi, ds.minor,NITS)
        em = EM(ds.X, ds.M, ds.V_INDEX, ds.CONSENSUS, EPS, ds.COV)
        em.do2(NITS,  [c2i[c] for c in ds.minor])
        print()


