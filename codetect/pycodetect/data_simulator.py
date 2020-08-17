import numpy as np
import random
import matplotlib.pyplot as plt
from pycodetect.aln import ReadAln
import copy
from pycodetect.read_aln_data import *
import math
from pycodetect.utils import *

class DataSimulator(ReadAlnData):
    def __init__(self, n_reads, read_length, gamma, pi, covq=None, mu=None,  d=None,paired_end=False,template_sequences=None, dmat=None, genome_length=None, min_d=None, max_d=None):
        """ Initialize data simulator class with parameters
        
        Args:
            n_reads: int number of reads
            read_length: int read length
            genome_length: int genome length
            gamma: float gamma error/mutation parameter
            pi: mixture proportion of major species
            D: hamming distance between major and minor species
            mu: stick breaking per-base mutation probability within a population
            covq: maximum coverage difference for coverage random walk
        """
        # Initialize constants
        self.paired_end = paired_end
        self.n_reads = n_reads
        self.read_length = read_length
        self.gamma = gamma
        self.mu = mu
        self.pi = pi
        # Simulate a population
        sys.stderr.write("Generating population\n")
        if template_sequences == None:
            assert genome_length is not None
            assert d is not None
            self.D = d
            self.genome_length = genome_length
            self.major, self.minor = self.gen_pair_random(genome_length, d)
            assert ham(self.major,self.minor) == self.D
        else:
            sys.stderr.write("Picking a reference\n")
            self.major, self.minor = self.pick_references(template_sequences, dmat, min_d, max_d)
            sys.stderr.write("References chosen with distance: %f\n" % ham(self.major, self.minor))
            self.genome_length = len(self.major)
#            assert ham(self.major,self.minor) >= self.D
        if self.mu is not None:
            self.majorpop = self.gen_population(self.major, self.mu)
            self.minorpop = self.gen_population(self.minor, self.mu)
        else:
            self.majorpop = ([self.major], [1.0])
            self.minorpop = ([self.minor], [1.0])
        self._minorhams = [ham(self.minor, s) for s in self.minorpop[0]]
        self._majorhams = [ham(self.major, s) for s in self.majorpop[0]]

#        plt.bar(x=majorhams, height=self.majorpop[1])
#        plt.title("majorhams")
#        plt.show()                   
#        plt.bar(x=minorhams, height=self.minorpop[1])
#        plt.title("minorhams")
#        plt.show()
        self._major = self.major
        self._minor = self.minor
        # Simulate reads
        sys.stderr.write("Simulating reads\n")
        if covq is not None:
            self._covwalk = self.random_coverage_walk(covq)
        else:
            self._covwalk = np.ones(self.genome_length-self.read_length+1)
            self._covwalk /= sum(self._covwalk)
        
        self.X = self.sample_reads(paired_end)
        # Parse data into a ReadData object
        super(DataSimulator,self).__init__(self.X,self.major)
        # In case the number of reads has changed (some have been deleted:)
#        self.n_reads = sum([Xi.count for Xi in self.X])
        # Guarantee that the consensus has the highest for each
#        for ci, c in enumerate(self.get_consensus()):
#            assert max(self.M[ci]) == self.M[ci][c], (self.M[ci], c)

    def get_minor(self):
        return self._minor

    def get_major(self):
        return self._major

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

    def random_coverage_walk(self,covq):
        """
        Produce a random walk for simulating uneven coverage.
        
        Args:
            covq: maximum fold coverage difference between minimum and maximum.
        """
        xs = np.zeros(self.genome_length-self.read_length+1)
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

    def pick_references(self,refs, dmat, D, max_d):
        """Generate a pair of sequence representing the center of two clusters using references.
    
        Args:
            refs: reference strings to pick from.
            dmat: distance matrix specifying distances between references.
            D: minimum distance between them.
        """
        mj = None
        while (mj == None) or (mj == mi):
            mi = random.randint(0,len(refs)-1)
            row = dmat[mi]
            possinds = np.where((row >= D) & (row <= max_d))[0]
            if len(possinds) > 0:
                mj = random.choice(possinds)
        return self.mutate_n(refs[mi],2), self.mutate_n(refs[mj],2)

    def mutate_n(self,seq,n):
        newseq = [c for c in seq]
        mutpos = np.random.choice(len(seq), n, replace=False)
        for j in mutpos:
            newseq[j] = random.choice([c for c in range(4) if c != seq[j]])
        return newseq

    def gen_pair_random(self,L,D):
        """
        Generate a pair of sequences representing the center of two clusters.

        Args:
            L: sequence length
            D: Hamming distance between the two
        """
        # Mutates by a fixed number
        major = [random.choice(range(4)) for i in range(self.genome_length)]
        minor = self.mutate_n(major,D)
        return major,minor

    def gen_aln(self, label, seq, paired_end, insert_size=350):
        """ 
        Generate a random read alignment.
    
        Args:
            i: number.
            randpos: random position of the alignment.
            seq: base sequence.
            paired_end: whether to generate a second alignment.

        Returns:
            A ReadAln object.
        """
        if paired_end:
            maxind = len(seq)-(2*self.read_length+insert_size)+1
            prob_covwalk = self._covwalk[:maxind]/sum(self.covwalk[:maxind])
            randpos = np.random.choice(range(maxind), p=prob_covwalk)
            sampinds = [randpos+l for l in range(self.read_length)]
            sampinds += [randpos+self.read_length+insert_size+l for l in range(self.read_length)]
        else:
            randpos = np.random.choice(range(len(seq)-self.read_length+1), p=self._covwalk)
            sampinds = [randpos+l for l in range(self.read_length)]
        aln = ReadAln(label)
        for si in sampinds:
            roll = random.uniform(0,1)
            c = seq[si]
            if roll < self.gamma:
                alt = random.choice([z for z in range(4) if z != c])
                aln.append_mapped_base(si,alt)
            else:
                aln.append_mapped_base(si,c)                    
        return aln

    def sample_reads(self, paired_end):
        """
        Sample a set of reads, currently using internal state (member variables).

        Return:
            X: a list of ReadAln objects sampled from the simulated population.
        """
        w = [self.pi, 1-self.pi]
        assert w[0] == max(w), "first sequence should be the major var"
        X = []
        pops = [self.majorpop, self.minorpop]   
        # Generate random coverage
        for i in range(self.n_reads):
            seqi = np.random.choice([0,1],p=w)
            popseqs, popfreqs = pops[seqi]
            subseqi = np.random.choice(range(len(popseqs)), p=popfreqs)
            seq = popseqs[subseqi]
            indicstr = "MAJOR" if seqi == 0 else "MINOR"
            lab = str(i)+"_"+indicstr
            aln = self.gen_aln(lab,seq,paired_end)
            aln.z = seqi
            # Discard any reads that cover regions not covered
            if 4 not in aln.get_ints():
                X.append(aln)
        return X

def write_reads(ds, opref):
    if ds.paired_end:
        pairs = [aln.get_fq_entry_pair() for aln in ds.X]
        fwd,rev = zip(*pairs)
        with open(opref + ".1.fq", "w") as of1:
            for l in fwd:
                of1.write(l)
        with open(opref + ".2.fq", "w") as of2:
            for l in rev:
                of2.write(l)        
    else:
        reads = [aln.get_fq_entry_single() for aln in ds.X]
        with open(opref + ".fq", "w") as of1:
            for l in reads:
                of1.write(l)

def write_refs(ds, opref):
    with open(opref + ".major.fa", "w") as of1:
        of1.write(">major\n"+"".join(["ACGT"[c] for c in ds.major if c != 4]))
    with open(opref + ".minor.fa", "w") as of2:
        of2.write(">minor\n"+"".join(["ACGT"[c] for c in ds.minor if c != 4]))

if __name__ == "__main__":
    import argparse 
    from Bio import SeqIO
    # Usage e.g.
    # python3.7 
    parser = argparse.ArgumentParser(description="Detect Coinfection!")
    parser.add_argument("--pi", required=True, type=float)
    parser.add_argument("--D", required=True, type=int)
    parser.add_argument("--gamma", required=True, type=float)
    parser.add_argument("--n_reads", required=True, type=int)
    parser.add_argument("--covq", required=True, type=int)
    parser.add_argument("--read_length", required=True, type=int)
    parser.add_argument("--mu", required=True, type=float)
    parser.add_argument("--refs", required=True)
    parser.add_argument("--dmat", required=True)
    parser.add_argument("--paired_end", required=False, action="store_true", default=False)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # TODO: record headers as well
    refs = [[c2i[c] for c in str(r.seq).upper()] for r in SeqIO.parse(args.refs, "fasta")]
    dmat = np.load(args.dmat)

    ds = DataSimulator(args.n_reads,args.read_length,args.gamma,args.pi,args.D,args.mu,args.covq,paired_end=args.paired_end,template_sequences=refs, dmat=dmat) 
    write_reads(ds,args.out)
    write_refs(ds,args.out)

