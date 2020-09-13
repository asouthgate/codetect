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

def preprocess_refs(ref_fname, s0_h, min_d=None):
    from Bio import FastaIO
    # Pull out header for s0
    s0_msa_seq = ""
    recs = FastaIO.SimpleFastaParser(ref_fname)
    for h,s in recs: 
        if h == s0_h:
            s0_msa_seq = s
    # Pull out valid indices
    valinds = [j for j,c in enumerate(s0_msa_seq) if c =! "-"]
    # Remove indels relative to s0
    recs2 = []
    for h,s in recs:
        d = 0
        s2 = ""
        for i in valinds:
            s2 += s[i]
        recs2.append((h,s2))
    return recs2

if __name__ == "__main__":
    #//*** Parse args ***
    parser = argparse.ArgumentParser()
    parser.add_argument("-bam", type=str, required=True)
    parser.add_argument("-ref", type=str, required=True)
    parser.add_argument("-out", type=str, required=True)
    parser.add_argument("-mind", type=int, required=True)
    parser.add_argument("-ref_msa", type=str, required=False, default=None)
    parser.add_argument("-debug_minor", type=str, required=False, default=None)
    args = parser.parse_args()
    alns = collect_alns(args.bam)
    ref_rec = [r for r in SeqIO.parse(args.ref, "fasta")][0]
    ref = str(ref_rec.seq)
    rad = ReadAlnData(alns, ref)
    rad.filter(100)

    #*** Using a fixed reference panel
    if args.ref_msa is not None:
        ref_panel = preprocess_refs(args.ref_msa, ref_rec.description)

    #//*** EM ***
    em = EM(rad,args.mind)
#[t, self.calc_log_likelihood(st,gt,mut,pit), pit, gt, mut
    if not args.debug_minor:
        trace = em.do2()
    else:
        dbm = [str_c2i(str(r.seq)) for r in SeqIO.parse(args.debug_minor, "fasta")][0] 
        trace = em.do2(debug=True,debug_minor=dbm)
    L0 = em.calc_L0()
    nsites = len(em.ds.VALID_INDICES)
    with open(args.out+".summary.csv", "w") as f:
        f.write("L0,nsites\n%f,%d" % (L0,nsites))
    with open(args.out+".trace.csv", "w") as f:
        for line in trace:
            f.write(",".join([str(s) for s in line][:-1])+"\n")
    with open(args.out+".est.fa", "w") as f:
       f.write(">1\n%s" % str_i2c(trace[-1][-1]) )


