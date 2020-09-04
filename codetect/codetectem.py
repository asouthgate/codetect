#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
usage: codetect.py [-h] -bam file.bam -fa file2.fa -mind 5

required arguments:
    -bam: file.bam      Input bam file (sorted and indexed)
    -fa: ref.fa         Input reference fasta
    -mind: int          Minimum number of substitutions between cluster strings
"""

import sys
from Bio import SeqIO
import numpy as np
from pycodetect.bam_importer import collect_alns
from pycodetect.em import EM
from pycodetect.read_aln_data import ReadAlnData
from pycodetect.utils import str_c2i, str_i2c
import sys
import argparse

if __name__ == "__main__":
    #//*** Parse args ***
    parser = argparse.ArgumentParser()
    parser.add_argument("-bam", type=str, required=True)
    parser.add_argument("-ref", type=str, required=True)
    parser.add_argument("-out", type=str, required=True)
    parser.add_argument("-mind", type=int, required=True)
    parser.add_argument("-debug_minor", type=str, required=False, default=None)
    args = parser.parse_args()
    alns = collect_alns(args.bam)
    ref = [str_c2i(str(r.seq)) for r in SeqIO.parse(args.ref, "fasta")][0]
    rad = ReadAlnData(alns, ref)
    rad.filter(150)

    #//*** EM ***
    em = EM(rad,args.mind)
#[t, self.calc_log_likelihood(st,gt,mut,pit), pit, gt, mut
    if not args.debug_minor:
        trace = em.do2()
    else:
        dbm = [str_c2i(str(r.seq)) for r in SeqIO.parse(args.debug_minor, "fasta")][0] 
        trace = em.do2(debug=True,debug_minor=dbm)
#    L0 = em.calc_L0()
    sys.stderr.write("Calculating H0\n")
    alt_trace = em.do2(min_pi=0.98,fixed_st=trace[-1][-1])
    nsites = len(em.ds.VALID_INDICES)
    with open(args.out+".summary.csv", "w") as f:
        f.write("nsites\n%d" % (nsites))
    with open(args.out+".trace.csv", "w") as f:
        for line in trace:
            f.write(",".join([str(s) for s in line][:-1])+"\n")
    with open(args.out+".alt_trace.csv", "w") as f:
        for line in alt_trace:
            f.write(",".join([str(s) for s in line][:-1])+"\n")
    with open(args.out+".est.fa", "w") as f:
       f.write(">1\n%s" % str_i2c(trace[-1][-1]) )


