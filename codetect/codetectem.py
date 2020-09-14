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
from pycodetect.utils import str_c2i, str_i2c, ham
import sys
import argparse

def preprocess_refs(ref_fname, s0_h, min_d=None):
    from Bio.SeqIO import FastaIO
    # Pull out header for s0
    s0_msa_seq = ""
    with open(ref_fname) as f:
        recs = [r for r in FastaIO.SimpleFastaParser(f)]
    for h,s in recs: 
        if h == s0_h:
            s0_msa_seq = s
            break
    assert len(s0_msa_seq) > 0, "Reference %s not found in msa" % s0_h
    # Pull out valid indices
    print(s0_msa_seq)
    valinds = [j for j,c in enumerate(s0_msa_seq) if c != "-"]
    print("Valid inds:", valinds)
    # Remove indels relative to s0
    recs2 = []
    for h,s in recs:
        d = 0
        s2 = ""
        for i in valinds:
            s2 += s[i]
        d = ham(s0_msa_seq,s2) 
        if min_d is not None:
            if d > min_d:
                recs2.append((h,str_c2i(s2)))
        else:
            recs2.append((h,str_c2i(s2)))
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
    ref = str_c2i(str(ref_rec.seq))
    rad = ReadAlnData(alns, ref)
    rad.filter(100)

    #*** Using a fixed reference panel
    if args.ref_msa is not None:
        ref_panel = preprocess_refs(args.ref_msa, ref_rec.description, min_d=args.mind)

    #//*** EM ***
    em = EM(rad,args.mind)
#[t, self.calc_log_likelihood(st,gt,mut,pit), pit, gt, mut
    if not args.debug_minor:
        if args.ref_msa is None:
            sys.stderr.write("Running without ref panel\n")
            trace = em.do2()
        else:
            sys.stderr.write("running with ref panel\n")
            assert len(ref_panel) > 0
            trace = em.do2(ref_panel=ref_panel)
    else:
        dbm = [str_c2i(str(r.seq)) for r in SeqIO.parse(args.debug_minor, "fasta")][0] 
        trace = em.do2(debug=True,debug_minor=dbm)
<<<<<<< HEAD
            
=======
#    L0 = em.calc_L0()
    sys.stderr.write("Calculating H0\n")
>>>>>>> 36ee9e94ec57ed8df45d86113f6be04b7fdd924c
    L0 = em.calc_L0()
    sys.stderr.write("L0: %f\n" % L0)
    alt_trace = em.do2(min_pi=1.0,fixed_st=trace[-1][-1])
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


