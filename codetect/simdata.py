import numpy as np
import random
import matplotlib.pyplot as plt
from  em import *
from aln import ReadAln
import copy
from read_data import *
import math

class DataSimulator(ReadData):
    def __init__(self, N_READS, READ_LENGTH, GENOME_LENGTH, GAMMA, PI, D, MU, COVQ):
        """ Initialize data simulator class with parameters
        
        Args:
            N_READS: int number of reads
            READ_LENGTH: int read length
            GENOME_LENGTH: int genome length
            GAMMA: float gamma error/mutation parameter
            PI: mixture proportion of major species
            D: hamming distance between major and minor species
            MU: stick breaking per-base mutation probability within a population
            COVQ: maximum coverage difference for coverage random walk
        """
        # Initialize constants
        self.N_READS = N_READS
        self.READ_LENGTH = READ_LENGTH
        self.GENOME_LENGTH = GENOME_LENGTH
        self.GAMMA = GAMMA
        self.MU = MU
        self.PI = PI
        self.D = D
#        assert D > 0
        # Simulate a population
        sys.stderr.write("Generating population\n")
        self.major, self.minor = self.gen_pair(GENOME_LENGTH, D)
        assert ham(self.major,self.minor) == self.D
        self.majorpop = self.gen_population(self.major, self.MU)
        self.minorpop = self.gen_population(self.minor, self.MU)
        minorhams = [ham(self.minor, s) for s in self.minorpop[0]]
        majorhams = [ham(self.major, s) for s in self.majorpop[0]]
        plt.bar(x=majorhams, height=self.majorpop[1])
        plt.title("majorhams")
        plt.show()                   
        plt.bar(x=minorhams, height=self.minorpop[1])
        plt.title("minorhams")
        plt.show()

        self.CONSENSUS = self.major
        # Simulate reads
        sys.stderr.write("Simulating reads\n")
        self.COVWALK = self.random_coverage_walk(COVQ)
        self.CONSENSUS = [c2i[c] for c in self.CONSENSUS]
        self.X = self.sample_reads()
        # Parse data into a ReadData object
        super(DataSimulator,self).__init__(self.X,self.CONSENSUS)
        self.N_READS = sum([Xi.count for Xi in self.X])
        # Guarantee that the consensus has the highest
        for ci, c in enumerate(self.CONSENSUS):
            assert max(self.M[ci]) == self.M[ci][c], (self.M[ci], c)


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
   
    def debug_interval(self, emObj):
        pi = emObj.pit
        g = emObj.gt
        T = emObj.Tt
        st = emObj.st
        # Calculate est pi

        bw = self.get_weight_base_array(T)
        for i in range(len(self.CONSENSUS)):
            if st[i] == self.CONSENSUS[i] and self.CONSENSUS[i] != c2i[self.minor[i]]:
                print(i, st[i], self.CONSENSUS[i], c2i[self.minor[i]], self.M[i], bw[i])
        z1reads = [Xi for Xi in self.X if Xi.z == 1]
        z1M = np.zeros((len(self.CONSENSUS), 4))
        for Xi in z1reads:
            for pos,b in Xi.get_aln():
                z1M[pos,b] += 1
        wba = self.get_weight_base_array(T)
        pbweights = []
        for i in range(len(self.V_INDEX)):
            tmp = []
            for ri in self.V_INDEX[i]:
                tmp.append(T[ri,1])
            pbweights.append(tmp)
        for i,c in enumerate(st):
            if st[i] != c2i[ds.minor[i]]:
                print(i,sum([len(l) for l in ds.V_INDEX[i]]),[len(l) for l in ds.V_INDEX[i]],ds.CONSENSUS[i], c2i[ds.minor[i]], ds.M[i], st[i], z1M[i], wba[i])
                for pbw in pbweights[i]:
                    print(pbw)

    def plot_genome(self,T,st):
        cov0 = np.zeros(len(self.CONSENSUS))
        cov1 = np.zeros(len(self.CONSENSUS))
        for Xi in self.X:
            if Xi.z == 0:
                for p,b in Xi.get_aln():
                    cov0[p] += 1
            else:
                assert Xi.z == 1
                for p,b in Xi.get_aln():
                    cov1[p] += 1
        self.COV = cov0+cov1

        estcov0 = np.zeros(len(self.CONSENSUS))
        estcov1 = np.zeros(len(self.CONSENSUS))
        hamarr = np.zeros(len(self.CONSENSUS))
        hamarr2 = np.zeros(len(self.CONSENSUS))
        for i,hi in enumerate(self.CONSENSUS):
            if self.CONSENSUS[i] != c2i[self.minor[i]]:
                hamarr[i] = len(self.V_INDEX[i][c2i[self.minor[i]]])
            if st[i] != self.CONSENSUS[i]:
                hamarr2[i] = len(self.V_INDEX[i][st[i]])

        for i,Xi in enumerate(self.X):
            Ti = T[i]
#            if Ti[0] > Ti[1]:
 #               for pos,base in Xi.get_aln():
            for pos,base in Xi.get_aln():
                estcov0[pos] += Ti[0]
    #            else:
     #               for pos,base in Xi.get_aln():
                estcov1[pos] += Ti[1]
        plt.plot(hamarr, color='red', alpha=0.5)
        plt.plot(hamarr2, color='blue', alpha=0.5)
        plt.plot(cov0,color='blue')
        plt.plot(cov1,color='orange')
        plt.plot(estcov0,color='purple')
        plt.plot(estcov1,color='pink')
        plt.show()


    def debug_plot(self,emObj):
        """ Plot simulated data statistics for debugging.
    
        Args:
            st: estimted string for minor species
            T: T matrix from EM
        """
        T = emObj.Tt
        st = emObj.st

        for k in range(len(self.COV)):
            assert self.COV[k] == sum([len(l) for l in self.V_INDEX[k]])
#        mc0 = max(cov0)
#        hamarr *= mc0
#        hamarr2 *= mc0
        nms = np.array([Xi.nm_major for Xi in self.X])
        plt.plot(self.COVWALK)
        plt.title("coverage_walk")
        plt.show()
        self.plot_genome()
        inp = [int(l) for l in input("Specify interval").split()]
        l,r = inp
        plt.plot(hamarr[l:r], color='red', alpha=0.5)
        plt.plot(hamarr2[l:r], color='green', alpha=0.5)
        plt.plot(cov0[l:r],color='blue')
        plt.plot(cov1[l:r],color='orange')
        plt.plot(estcov0[l:r],color='purple')
        plt.plot(estcov1[l:r],color='pink')
        plt.show()
        self.debug_interval(emObj)
        readdists = [Xi.nm_major for Xi in self.X if Xi.z == 0]
        plt.hist(readdists,bins=100)
        readdists = [Xi.nm_major for Xi in self.X if Xi.z == 1]
        plt.hist(readdists,bins=100)
        plt.show()        
        plt.hist([Xi.z for Xi in  self.X])
        plt.title("True z")
        plt.show()
        plt.hist([r[0] for r in T])
        plt.title("T0")
        plt.show()
#        print(T)


    def random_coverage_walk(self,covq):
        xs = np.zeros(self.GENOME_LENGTH-self.READ_LENGTH+1)
        x = 0
        for i in range(len(xs)):
            step = random.choice([1,-1])
            x += step
            x = min(covq,max(1,x))
            xs[i] = x
        xs /= sum(xs)
        return xs

    def mutate_perbase(self, seq, mutp):
        res = [c for c in seq]
        for i in range(len(res)):
            roll = random.uniform(0,1)
            if roll < mutp:
                res[i] = random.choice([c for c in "ATCG" if res[i] != c])
        return "".join(res)

    def break_sticks(self, nsticks=100, alpha=5, min_middle=0.9):
#        props = sorted(np.random.dirichlet([ for i in range(nsticks)]))
        agg = min_middle
        props = [min_middle]
        for n in range(nsticks):
            p = np.random.beta(1,alpha)
            props.append(p*(1-agg))
            agg += (p*(1-agg))
        assert max(props) >= min_middle
        props = sorted(props)
        assert sum(props) <= 1.0
        props[-1] += (1-sum(props))
        assert max(props) >= min_middle, max(props)
        return props

    def gen_population(self, ref, mutp):
        # Guarantee the max pop has at least 50% of the mass
        props = sorted(self.break_sticks())
        assert sum(props) == 1, sum(props)
        minorseqs = [self.mutate_perbase(ref, mutp) for i in props[1:]]
        hams = [ham(ms, ref) for ms in minorseqs]
        return minorseqs + [ref], props

    def gen_pair(self,L,D):
        # Mutates by a fixed number
        major = [random.choice("ATCG") for i in range(self.GENOME_LENGTH)]
        minor = [c for c in major]
        mutpos = np.random.choice(len(major), D, replace=False)
        for j in mutpos:
            minor[j] = random.choice([c for c in "ATCG" if c != major[j]])
        return "".join(major),"".join(minor)

    def gen_aln(self,i,randpos, seq):
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
        return aln

    def sample_reads_uniform(self,min_cov=20):
        w = [self.PI, 1-self.PI]
        assert w[0] == max(w), "first sequence should be the major var"
        X = []
        pops = [self.majorpop, self.minorpop]
        # Generate reads across the genome for minimal coverage
        for randpos in range(self.GENOME_LENGTH):
            for j in range(min_cov):
                seqi = np.random.choice([0,1],p=w)
             #   assert len(self.POPULATION) > 1
                popseqs, popfreqs = pops[seqi]
                seq = np.random.choice(popseqs, p=popfreqs)
      #          randpos = np.random.randint(0,len(seq)-self.READ_LENGTH+1)
    #            randpos = np.random.choice(range(len(seq)-self.READ_LENGTH+1), p=self.COVWALK)
                aln = self.gen_aln(str(randpos)+str(j),randpos,seq)
                aln.z = seqi
                X.append(aln)
        return X

    def sample_reads(self):
        w = [self.PI, 1-self.PI]
        assert w[0] == max(w), "first sequence should be the major var"
        X = []
        pops = [self.majorpop, self.minorpop]   
        # Generate random coverage
        for i in range(self.N_READS):
            seqi = np.random.choice([0,1],p=w)
         #   assert len(self.POPULATION) > 1
            popseqs, popfreqs = pops[seqi]
            seq = np.random.choice(popseqs, p=popfreqs)
  #          randpos = np.random.randint(0,len(seq)-self.READ_LENGTH+1)
            randpos = np.random.choice(range(len(seq)-self.READ_LENGTH+1), p=self.COVWALK)
            aln = self.gen_aln(i,randpos,seq)
            aln.z = seqi
            X.append(aln)
        return X

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(description="Detect Coinfection!")
    parser.add_argument("--pi", required=True, type=float)
    parser.add_argument("--D", required=True, type=int)
    parser.add_argument("--gamma", required=True, type=float)
    parser.add_argument("--n_reads", required=True, type=int)
    parser.add_argument("--covq", required=True, type=int)
    parser.add_argument("--read_length", required=True, type=int)
    parser.add_argument("--genome_length", required=True, type=int)
    parser.add_argument("--n_iterations", required=True, type=int)
    parser.add_argument("--min_mixture_distance", required=True, type=int)
    parser.add_argument("--mu", required=True, type=float)
    parser.add_argument("--debug_plot", required=False, action="store_true", default=False)
    args = parser.parse_args()

    PI = args.pi
    D = args.D
    GAMMA = args.gamma
    READLEN = args.read_length
    L = args.genome_length
    NREADS = args.n_reads
    NITS=args.n_iterations
    EPS = args.min_mixture_distance
    MU = args.mu
    COVQ = args.covq

    ds = DataSimulator(NREADS,READLEN,L,GAMMA,PI,D,MU,COVQ) 
    sys.stderr.write("Simulating dataset with parameters:\n")
    sys.stderr.write("GENOME_LENGTH=%d\n" % L)
    sys.stderr.write("PI=%f\n" % PI)
    sys.stderr.write("GAMMA=%f\n" % GAMMA)
    sys.stderr.write("MU=%f\n" % MU)
    sys.stderr.write("READ_LENGTH=%d\n" % READLEN)
    sys.stderr.write("NREADS=%d\n" % NREADS)
    sys.stderr.write("N_ITERATIONS=%d\n" % NITS)
    sys.stderr.write("D=%d\n" % D)
    sys.stderr.write("EPS=%d\n" % EPS)

#    em = EM(ds.X, ds.M, ds.V_INDEX, ds.CONSENSUS, EPS)
    ems = []
    for l in range(1):
        em = EM(ds, EPS)
        ll,b,st,Tt = em.do2(NITS, random_init=False, debug=args.debug_plot)
        llo,bo,sto,Tto = em.do_one_cluster(5,  debug=args.debug_plot)
        AIC_cluster = 6*len(ds.CONSENSUS) - 2*ll
        AIC_one = -2*llo 
        print("LIKELIHOODS", ll, llo)
        print("AICS", AIC_cluster, AIC_one, AIC_cluster-AIC_one)
        if AIC_cluster < AIC_one:
            print("AIC SAYS coinfection!")
        else:
            print("AIC SAYS NO COINFECTION!")
