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
from pycodetect.sampler import MixtureModelSampler
from pycodetect.read_aln_data import ReadAlnData
from pycodetect.utils import str_c2i
import logging
import sys
import argparse
import random

if __name__ == "__main__":
    #//*** Parse args ***
    parser = argparse.ArgumentParser()
    parser.add_argument("-bam", type=str, required=True)
    parser.add_argument("-ref", type=str, required=True)
    parser.add_argument("-mind", type=int, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    alns = collect_alns(args.bam)
    ref = [str_c2i(str(r.seq)) for r in SeqIO.parse(args.ref, "fasta")][0]
    rad = ReadAlnData(alns, ref)
    rad.filter(0.98)

    #//*** Debugging
    if not args.debug:
        logging.disable(logging.CRITICAL)

    #//*** Get permitted states and positions ***
    valid_inds = rad.VALID_INDICES
#    valid_inds = [j for j in range(len(rad.get_consensus()))]
    valid_states = []
    for j in range(len(rad.get_consensus())):
        row = rad.M[j]
        vs = np.argsort(row)[2:]
        valid_states.append(vs)

    sys.stderr.write("%d valid indices identified" % len(valid_inds))
    #//*** Initialize sampler ***
#      randy = [random.randint(0,3) for j in range(len(rad.get_consensus()))]
    randy = [c for c in rad.get_consensus()]
    mms = MixtureModelSampler(rad, rad.get_consensus(), allowed_states=valid_states, allowed_positions=valid_inds, initstring=randy, min_d = args.mind)

    #//*** Sample ***//
    strings,params,Ls = mms.sample(rad,nits=5)

    #//*** Collect results ***//
    params = np.array(params)

    print("L,pi,g1,g2,seq")
    for i in range(len(strings)):
        print("%d,%f,%f,%f,%f,%s" % (i, Ls[i], params[i,0], params[i,1], params[i,2], strings[i]))

