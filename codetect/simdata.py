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
        # Simulate a population
        sys.stderr.write("Generating population\n")
        self.major, self.minor = self.gen_pair(GENOME_LENGTH, D)
        assert ham(self.major,self.minor) == self.D
        self.majorpop = self.gen_population(self.major, self.MU)
        self.minorpop = self.gen_population(self.minor, self.MU)
        minorhams = [ham(self.minor, s) for s in self.minorpop[0]]
        majorhams = [ham(self.major, s) for s in self.majorpop[0]]
        self._MAJOR = self.major
        self._MINOR = self.minor
        # Simulate reads
        sys.stderr.write("Simulating reads\n")
        self.COVWALK = self.random_coverage_walk(COVQ)
        self.X = self.sample_reads()
        # Parse data into a ReadData object
        super(DataSimulator,self).__init__(self.X,self._MAJOR)
        # In case the number of reads has changed (some have been deleted:)
        self.N_READS = sum([Xi.count for Xi in self.X])
        # Guarantee that the consensus has the highest for each
        for ci, c in enumerate(self.get_consensus()):
            assert max(self.M[ci]) == self.M[ci][c], (self.M[ci], c)

    def get_minor(self):
        return self._MINOR

    def get_major(self):
        return self._MAJOR

    def get_weight_base_array(self, T):
        """
        Get a weighted (by membership probability, or similar) frequency matrix

        Args:
            T: N_reads x 2 array of membership probabilites for 2 clusters
        """
        baseweights = np.zeros((len(self.get_consensus()), 4))
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
   
    def plot_genome(self,T,st):
        """
        Plot estimated coverage for each cluster across the genome,
        respective posititions, estimated mutations,
        and true values for each.

        Args:
            T: membership probability array per read
            st: estimated sequence for alternative cluster
        """
        cov0 = np.zeros(len(self.get_consensus()))
        cov1 = np.zeros(len(self.get_consensus()))
        for Xi in self.X:
            if Xi.z == 0:
                for p,b in Xi.map.items():
                    cov0[p] += 1
            else:
                assert Xi.z == 1
                for p,b in Xi.map.items():
                    cov1[p] += 1
        self.COV = cov0+cov1
        estcov0 = np.zeros(len(self.get_consensus()))
        estcov1 = np.zeros(len(self.get_consensus()))
        hamarr = np.zeros(len(self.get_consensus()))
        hamarr2 = np.zeros(len(self.get_consensus()))
        for i,hi in enumerate(self.get_consensus()):
            if self.get_consensus()[i] != c2i[self.minor[i]]:
                hamarr[i] = len(self.V_INDEX[i][c2i[self.minor[i]]])
            if st[i] != self.get_consensus()[i]:
                hamarr2[i] = len(self.V_INDEX[i][st[i]])

        for i,Xi in enumerate(self.X):
            Ti = T[i]
            for pos,base in Xi.map.items():
                estcov0[pos] += Ti[0]
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
            emObj: an EM object used for parameter estimation
        """
        T = emObj.Tt
        st = emObj.st
        plt.bar(x=majorhams, height=self.majorpop[1])
        plt.title("majorhams")
        plt.show()                   
        plt.bar(x=minorhams, height=self.minorpop[1])
        plt.title("minorhams")
        plt.show()
        for k in range(len(self.COV)):
            assert self.COV[k] == sum([len(l) for l in self.V_INDEX[k]])
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

    def random_coverage_walk(self,covq):
        """
        Produce a random walk for simulating uneven coverage.
        
        Args:
            covq: maximum fold coverage difference between minimum and maximum.
        """
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
        """
        Mutate the bases of a sequence independently.

        Args:
            seq: sequence to mutate.
            mutp: probability a base will mutate.
        """
        res = [c for c in seq]
        for i in range(len(res)):
            roll = random.uniform(0,1)
            if roll < mutp:
                res[i] = random.choice([c for c in range(4) if res[i] != c])
        return res

    def break_sticks(self, nsticks=100, alpha=5, q1=0.9):
        """
        Simulate a frequency distribution by stick-breaking.
        
        Optional args:
            nsticks: dimension
            alpha: beta distribution parameter
            q1: probability of first stick
        """
        agg = q1
        props = [q1]
        for n in range(nsticks):
            p = np.random.beta(1,alpha)
            props.append(p*(1-agg))
            agg += (p*(1-agg))
        assert max(props) >= q1
        props = sorted(props)
        assert sum(props) <= 1.0
        props[-1] += (1-sum(props))
        assert max(props) >= q1, max(props)
        return props

    def gen_population(self, ref, mutp):
        """
        Generate a tight population of sequences by mutation and stick breaking.

        Args:
            ref: base string to mutate
            mutp: per-base mutational probability
        """
        # Guarantee the max pop has at least 50% of the mass
        props = sorted(self.break_sticks())
        assert sum(props) == 1, sum(props)
        minorseqs = [self.mutate_perbase(ref, mutp) for i in props[1:]]
        hams = [ham(ms, ref) for ms in minorseqs]
        return minorseqs + [ref], props

    def gen_pair(self,L,D):
        """
        Generate a pair of sequences representing the center of two clusters.

        Args:
            L: sequence length
            D: Hamming distance between the two
        """
        # Mutates by a fixed number
        major = [random.choice(range(4)) for i in range(self.GENOME_LENGTH)]
        minor = [c for c in major]
            mutpos = np.random.choice(len(major), D, replace=False)
        for j in mutpos:
            minor[j] = random.choice([c for c in range(4) if c != major[j]])
        return major,minor

    def gen_aln(self,i,randpos, seq):
        """ 
        Generate a random read alignment.
    
        Args:
            i: number.
            randpos: random position of the alignment.
            seq: base sequence.

        Returns:
            A ReadAln object.
        """
        sampinds = [randpos+l for l in range(self.READ_LENGTH)]
        aln = ReadAln(i)
        for si in sampinds:
            roll = random.uniform(0,1)
            c = seq[si]
            if roll < self.GAMMA:
                alt = random.choice([z for z in range(4) if z != c])
                aln.append_mapped_base(si,alt)
            else:
                aln.append_mapped_base(si,c)                    
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
                popseqs, popfreqs = pops[seqi]
                seq = np.random.choice(popseqs, p=popfreqs)
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
            seqi = np.random.choice(range(len(popseqs)), p=popfreqs)
            seq = popseqs[seqi]
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

#    em = EM(ds.X, ds.M, ds.V_INDEX, ds.get_consensus(), EPS)
    ems = []
    for l in range(1):
        em = EM(ds, EPS)
        ll,b,st,Tt = em.do2(NITS, random_init=False, debug=args.debug_plot)
        llo,bo,sto,Tto = em.do_one_cluster(5,  debug=args.debug_plot)
        AIC_cluster = 2.5*len(ds.VALID_INDICES) - 2*ll
        AIC_one = -2*llo 
        print("LIKELIHOODS", ll, llo)
        print("AICS", AIC_cluster, AIC_one, AIC_cluster-AIC_one)
        if AIC_cluster < AIC_one:
            print("AIC SAYS coinfection!")
        else:
            print("AIC SAYS NO COINFECTION!")
