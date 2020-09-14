#!/usr/bin/env python
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    import argparse 
    from Bio import SeqIO
    import sys
    sys.path.append("./")
    from pycodetect.data_simulator import DataSimulator, write_reads, write_refs
    from pycodetect.utils import c2i
    import numpy as np
    import subprocess as sp
    import pickle

    parser = argparse.ArgumentParser(description="Detect Coinfection!")
    parser.add_argument("--pi", required=True, type=float)
    parser.add_argument("--d", required=False, type=int)
    parser.add_argument("--min_d", required=False, type=int)
    parser.add_argument("--max_d", required=False, type=int)    
    parser.add_argument("--gamma", required=True, type=float)
    parser.add_argument("--n_reads", required=True, type=int)
    parser.add_argument("--genome_length", default=None, required=False, type=int)
    parser.add_argument("--covq", required=False, type=int, default=None)
    parser.add_argument("--read_length", required=True, type=int)
    parser.add_argument("--mu", required=False, type=float)
    parser.add_argument("--refs", default=None, required=False)
    parser.add_argument("--dmat", default=None, required=False)
    parser.add_argument("--paired_end", required=False, action="store_true", default=False)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    # TODO: record headers as well
    if args.refs is not None:
        assert args.dmat is not None
        refs = [[c2i[c] for c in str(r.seq).upper().replace("-","")] for r in SeqIO.parse(args.refs, "fasta")]
        dmat = np.load(args.dmat)
        ds = DataSimulator(args.n_reads,args.read_length,args.gamma,args.pi,args.covq,paired_end=args.paired_end,template_sequences=refs, dmat=dmat, min_d=args.min_d, max_d=args.max_d, mu=args.mu) 
    else:
        assert args.genome_length is not None
        ds = DataSimulator(args.n_reads,args.read_length,args.gamma,args.pi,args.covq,paired_end=args.paired_end,genome_length=args.genome_length, mu=args.mu, d=args.d) 

    sp.call("mkdir %s" % args.out_prefix, shell=True)
    ofilepref = args.out_prefix + "/" + args.out_prefix.split("/")[-1]
    print(ofilepref)
    write_reads(ds,ofilepref)
    write_refs(ds,ofilepref)
    if args.paired_end:
        sp.call("minimap2 -ax sr {ref} {fwd} {rev} | samtools view -b | samtools sort > {bam}".format(ref=ofilepref + ".major.fa",fwd=opfilepref+".1.fq",rev=ofilepref+".2.fq",bam=ofilepref+".bam"), shell=True)
    else:
        sp.call("minimap2 -ax sr {ref} {fwd} | samtools view -b | samtools sort > {bam}".format(ref=ofilepref + ".major.fa",fwd=ofilepref+".fq",bam=ofilepref+".bam"), shell=True)
    sp.call("samtools index %s" % (ofilepref+".bam"), shell=True)
    sp.call("echo '{paramstr}' > {outcsv}".format(paramstr=" ".join(sys.argv),outcsv=ofilepref+".csv"),shell=True)
    pickleof= open(ofilepref+".pckl",'wb')
    pickle.dump(ds,pickleof)
    pickleof.close()
