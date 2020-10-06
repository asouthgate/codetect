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
from pycodetect.bam_importer import collect_alns
from pycodetect.em import EM
from pycodetect.read_aln_data import ReadAlnData
from pycodetect.utils import str_c2i, str_i2c, str_only_ACGT
from pycodetect.plotter import plot_mask
from pycodetect.ref_panel import RefPanel
import argparse


if __name__ == "__main__":
    #//*** Parse args ***
    parser = argparse.ArgumentParser()
    parser.add_argument("-bam", type=str, required=True)
    parser.add_argument("-ref", type=str, required=True)
    parser.add_argument("-out", type=str, required=True)
    parser.add_argument("-mind", type=int, required=True)
    parser.add_argument("-ref_msa", type=str, required=False, default=None)
    parser.add_argument("-debug_minor", type=str, required=False, default=None)
    parser.add_argument("--filter", type=str, required=False, default="winestimatew")
    args = parser.parse_args()
    alns = collect_alns(args.bam)
    ref_rec = [r for r in SeqIO.parse(args.ref, "fasta")][0]
    ref = str_c2i(str(ref_rec.seq))
    rad = ReadAlnData(alns, ref)

    #*** Mask the alignment if we are not using references
    if args.ref_msa is None:
        rad.filter(20, args.filter)
        if args.debug_minor is not None:
            dbm = [str_c2i(str_only_ACGT(str(r.seq))) for r in SeqIO.parse(args.debug_minor, "fasta")][0] 
            for ci,c in enumerate(rad.get_consensus()):
                if c == 4: dbm[ci] = 4
            plot_mask(rad, rad.get_consensus(), dbm)

    #//*** EM ***
    em = EM(rad,args.mind)
#[t, self.calc_log_likelihood(st,gt,mut,pit), pit, gt, mut
    if not args.debug_minor:
        if args.ref_msa is None:
            trace = em.estimate()
        else:
            rp = RefPanel(em.consensus, args.ref_msa, ref_rec.description, min_d=args.mind)
            trace = em.estimate(ref_panel=rp)
    else:
        dbm = [str_c2i(str_only_ACGT(str(r.seq))) for r in SeqIO.parse(args.debug_minor, "fasta")][0] 
        if args.ref_msa is None:
            sys.stderr.write("Running without ref panel\n")
            trace = em.estimate(debug_minor=dbm,debug=True)
        else:
            sys.stderr.write("running with ref panel\n")
            rp = RefPanel(em.consensus, args.ref_msa, ref_rec.description, min_d=args.mind)
            trace = em.estimate(ref_panel=rp,debug_minor=dbm,debug=True)

#    L0 = em.calc_L0()
    sys.stderr.write("Calculating H0\n")
    L0 = em.calc_L0()
    sys.stderr.write("L0: %f\n" % L0)
    alt_trace = em.estimate(min_pi=1.0,fixed_st=trace[-1][-1])
    nsites = len(em.rd.VALID_INDICES)
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


