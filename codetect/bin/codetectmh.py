#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
usage: codetect.py [-h] -bam file.bam -msa refs.fa

required arguments:
    -bam: file.bam      Input bam file (sorted and indexed)
    -fa: ref.fa         Input ref (bam maps to)
    -msa: refs.m.fa     Input reference msa (.fasta)
    -dmat: dmat.npy     Input reference distance matrix (.npy file)
    -mind: int          Minimum distance required between consensus and minor cluster sequence
"""

from Bio import SeqIO
import numpy as np
from pycodetect.bam_importer import collect_alns
from pycodetect.sampler import MixtureModelSampler
from pycodetect.sampler import del_close_to_fixed_point
from pycodetect.utils import str_c2i
from pycodetect.read_aln_data import ReadAlnData
import logging
import sys
import argparse

if __name__ == "__main__":
    #//*** Parse args ***
    parser = argparse.ArgumentParser()
    parser.add_argument("-bam", type=str, required=True)
    parser.add_argument("-ref", type=str, required=True)
    parser.add_argument("-msa", type=str, required=True)
    parser.add_argument("-dmat", type=str, required=True)
    parser.add_argument("-mind", type=int, required=True)
    args = parser.parse_args()
    
    #//*** Parse alignment***
    alns = collect_alns(args.bam)
    refrec = [r for r in SeqIO.parse(args.ref, "fasta")][0]
    fixed_point_id = refrec.split()[0]    
    ref = str_c2i(str(refreq.seq))
    rad = ReadAlnData(alns, ref)

    #//*** Parse MSA ***
    if not args.debug:
        logging.disable(logging.CRITICAL)
    ref_msa = []
    fixed_point = None
    for r in SeqIO.parse(args.ref, "fasta"):
        seq = str_c2i(str(r.seq).upper())
        ref_msa.append(seq)
        if fixed_poind_id in r.description:
            fixed_point = seq
    assert fixed_point is not None, "%s is not a reference in the references" % args.fixed_point

    # Delete any gaps; get indices in reverse order
    fixed_point_delinds = [i for i in range(len(fixed_point)) if fixed_point[i] == 4][::-1]
    for ind in fixed_point_delinds:
        for seq in ref_msa: del seq[ind]
    refs = ref_msa

    #//*** Parse distance matrix
    dmat = np.load()
    # TODO: SLOW: FIX FOR IMPORT
    for i in range(len(dmat)-1):
        for j in range(i+1,len(dmat)):
            dmat[j,i] = dmat[i,j]

    # Delete references too close to the consensus (taken as fixed point)
    sys.stderr.write("Removing refs too close to fixed point\n")
#    fixed_point = rad.get_consensus()
    refs,dmat = del_close_to_fixed_point(fixed_point,args.mind,refs,dmat)
    sys.stderr.write("Initializing mixture model")
    #//*** Initialize mixture model ***//
    mms = MixtureModelSampler(rad,fixed_point,refs=refs,dmat=dmat)

    #//*** Sample ***//
    strings,params,Ls = mms.sample(rad,nits=1000)

    #//*** Collect results ***//
    params = np.array(params)
    meanpi = np.mean(params[:,0])
    meang0 = np.mean(params[:,1])
    meang1 = np.mean(params[:,2])
    print("mean pi=%f, mean g0=%f, mean g1=%f" % (meanpi, meang0, meang1))
