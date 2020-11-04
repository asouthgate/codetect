import unittest
import random
import numpy as np
import math
import sys
from pycodetect.likelihood_calculator import NMCache, LikelihoodCalculator
from pycodetect.read_aln_data import ReadAlnData
from pycodetect.aln import ReadAln
from pycodetect.utils import ham

class TestNMCache(unittest.TestCase):
    def test_init(self):
        sys.stderr.write("Testing NMCache\n")
        genome_length = 100
        readlen = 10
        n_reads = 2000
        n_subsamples = 1000
        X = [ReadAln(j) for j in range(n_reads)]
        # Generate the reads
        for ra in X:
            pos = random.randint(0, genome_length - readlen)
            for l in range(readlen):
                assert pos + l < genome_length
                ra.append_mapped_base(pos + l, random.randint(0,3))            
        
        rad = ReadAlnData(X, [0 for j in range(genome_length)])
        alt_genome = [random.choice([0,1,2,3]) for j in range(genome_length)]
        nm_cache = NMCache(rad, alt_genome)
        consensus = rad.get_consensus()
        for i, Xi in enumerate(rad.X):
            h0 = 0
            h1 = 0
            for pos, c in Xi.get_aln_tuples():
                if c != consensus[pos]: h0 += 1
                if c != alt_genome[pos]: h1 += 1
            self.assertEqual(h0, nm_cache[0, i])
            self.assertEqual(h1, nm_cache[1, i])        

class TestLogLikelihoodCalculator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        sys.stderr.write("Testing LLC init\n")
        super(TestLogLikelihoodCalculator, self).__init__(*args, **kwargs)
        genome_length = 100
        readlen = 10
        n_reads = 2000
        n_subsamples = 1000
        X = [ReadAln(j) for j in range(n_reads)]
        # Generate the reads
        for ra in X:
            pos = random.randint(0, genome_length-readlen)
            for l in range(readlen):
                assert pos + l < genome_length, (pos+l, genome_length)
                ra.append_mapped_base(pos + l, random.randint(0,3))            
        
        self.rad = ReadAlnData(X, [0 for j in range(genome_length)])
        self.st0 = [random.choice([0,1,2,3]) for j in range(genome_length)]
        self.g0 = 0.01
        self.consensus = self.rad.get_consensus()
        self.llc = LikelihoodCalculator(self.rad, self.st0, self.g0)

    def cal_read_logP_simple(self, read, gamma, st):
        sumo = 0
        for pos, c in read.get_aln_tuples():
            if c == st[pos]:
                sumo += np.log(1-gamma)
            else:
                sumo += np.log(gamma)
        return sumo

    def test_cal_logP_read(self):
        gamma = self.g0
        st = [j for j in self.st0]
        # Validate the initialization
        for ri, read in enumerate(self.rad.X):
            ll0_val = self.cal_read_logP_simple(read, gamma, self.consensus)
            ll1_val = self.cal_read_logP_simple(read, gamma, st)
            self.assertAlmostEqual(self.llc.llcache[0, ri], ll0_val, places=7)
            self.assertAlmostEqual(self.llc.llcache[1, ri], ll1_val, places=7)

        # Validate updates:
        for it in range(100):
            gamma = random.uniform(0,0.1)
            # Change some positions randomly
            st_changed_bases = []
            randposs = np.random.choice(range(len(st)), size=5, replace=False)
            for randpos in randposs:
                st_changed_bases.append((randpos, st[randpos]))
                st[randpos] = random.choice([z for z in [0,1,2,3] if st[randpos] != z])
            for p, b in st_changed_bases:
                assert b != st[p] 
            for ri, read in enumerate(self.rad.X):
                ll0 = self.llc.cal_logP_read(ri, read, 0, gamma, self.consensus, [])
                ll0_val = self.cal_read_logP_simple(read, gamma, self.consensus)
                ll1 = self.llc.cal_logP_read(ri, read, 1, gamma, st, st_changed_bases)
                ll1_val = self.cal_read_logP_simple(read, gamma, st)
                self.assertAlmostEqual(ll0, ll0_val, places=7)
                self.assertAlmostEqual(ll1, ll1_val, places=7)

    
    def test_cal_loglikelihood_pi_update(self):
        pass

    def test_cal_full_loglikelihood(self):
        pass

    def test_update_loglikelihood(self):
        pass

