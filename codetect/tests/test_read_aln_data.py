import unittest
import numpy as np
from pycodetect.aln import ReadAln
from pycodetect.read_aln_data import ReadAlnData

class TestReadAlnData(unittest.TestCase):

    def test_subsample_stratified(self):
        genome_length = 100
        readlen = 10
        n_reads = 2000
        n_subsamples = 1000
        X = [ReadAln(j) for j in range(n_reads)]

        # Generate the reads
        for ra in X:
            pos = random.randint(0, genome_length)
            for l in range(readlen):
                ra.append_mapped_base(pos + l, random.randint(0,3))            
        
        rad = ReadAlnData(X, [0 for j in range(genome_length)])
        subX = rad.subsample_stratified(X, n_subsamples)

        # There must be N, and each read must be unique
        self.assertTrue( len([Xi.label for Xi in subX]) == n_subsamples  )
        # The depths must be such that depth is at least expected
        # if depth >= n_subsamples % genome_length
        d_low = (n_subsamples % genome_length)
        depth = np.zeros(genome_length)
        subdepth = np.zeros(genome_length)
        for Xi in subX:
            pos = Xi.get_aln_segments()[0][0]
            subdepth[pos] += 1
        for Xi in X:
            pos = Xi.get_aln_segments()[0][0]
            subdepth[pos] += 1            
        maxd = max(depth)
        for di, d in enumerate(depth):
            if d >= d_low:
                self.assertTrue( subdepth[di] >= d_low )
            else:
                self.assertTrue( subdepth[di] == d )
        

    def test_deduplicate(self):
        pass
    def test_build_Vindex(self):
        pass
    def test_reads2mat(self):
        pass
    def test_get_indices_max(self):
        pass
    def test_get_indices_max_window(self):
        pass
    def test_pos2reads(self):
        pasebuilding
