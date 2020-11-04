import pysam
from pycodetect.read_aln_data import ReadAlnData
from pycodetect.aln import ReadAln
from pycodetect.utils import c2i,ham
 
def collect_alns(fname):
    alns = {}
    samfile = pysam.AlignmentFile(fname, "rb")
    for ci, aln in enumerate(samfile.fetch()):
        pairs = [p for p in aln.get_aligned_pairs() if None not in p]
        refpositions = [p[1] for p in pairs]
        qseq = aln.query_sequence
        seqbases = [c2i(qseq[p[0]]) for p in pairs]
        if not aln.query_name in alns:
            ra = ReadAln(aln.query_name)
            alns[aln.query_name] = ra
        for ri,i in enumerate(refpositions):
            if seqbases[ri] is not None and i is not None:
                # Don't keep any ambiguous bases
                if seqbases[ri] < 4:
                    alns[aln.query_name].append_mapped_base(i, seqbases[ri])
    return [q for i,q in alns.items() if q.get_length() > 0] 

if __name__ == "__main__":
    import sys
    from Bio import SeqIO
    import numpy as np
    np.set_printoptions(threshold=sys.maxsize)
    alns = collect_alns(sys.argv[1])
    majorref = [[c2i[c] for c in str(r.seq)] for r in SeqIO.parse(sys.argv[2], "fasta")][0]
    minorref = [[c2i[c] for c in str(r.seq)] for r in SeqIO.parse(sys.argv[3], "fasta")][0]
    rad = ReadAlnData(alns, majorref)
    from em import *
    em = EM(rad,0.001)
    em.do2(20,debug=True,debug_minor=minorref)
