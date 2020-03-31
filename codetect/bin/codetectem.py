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
from pycodetect.utils import str_c2i
import sys
import argparse

if __name__ == "__main__":
    #//*** Parse args ***
    parser = argparse.ArgumentParser()
    parser.add_argument("-bam", type=str, required=True)
    parser.add_argument("-ref", type=str, required=True)
    parser.add_argument("-mind", type=int, required=True)
    args = parser.parse_args()
    alns = collect_alns(args.bam)
    ref = [str_c2i(str(r.seq)) for r in SeqIO.parse(args.ref, "fasta")][0]
    rad = ReadAlnData(alns, ref)

    #//*** EM ***
    em = EM(rad,args.mind)
    print(em.do2(20))
