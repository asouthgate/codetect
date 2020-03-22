from Bio import SeqIO
import sys
import random
import numpy as np

def ham_nogaps(s1,s2):
    h = 0
    for i,c1 in enumerate(s1):
        c2 = s2[i]
        if c1 in "ACGT" and c2 in "ACGT":
            if c2 != c1:
                h += 1
    return h

# Import a MSA
print("Parsing")
seqs = [str(r.seq).upper() for r in SeqIO.parse(sys.argv[1], format="fasta")]
seqset = set(seqs)
assert len(seqs) == len(seqset), (len(seqs),len(seqset))
seqs = [list(s) for s in seqs]

# Check no duplicates
## Delete any constant sites
#    # Create a spectrum
print("Getting constant sites")
c2i = {"A":0, "C":1, "G":2, "T":3}
C = np.zeros((len(seqs[0]),4))
for seq in seqs:
    for ci, c in enumerate(seq):
        if c in "ACGT":
            C[ci,c2i[c]] += 1       
delinds = set()
print("Getting sites to delete")
for rowi,row in enumerate(C):
    if sum(row) == max(row):
        delinds.add(rowi)
print("Deleting %d constant sites:" % len(delinds))
assert len(delinds) < len(seqs[0]), "No sites left"
if len(delinds) > 0:
    for i in range(len(seqs)):
        tmp = seqs[i]
        tmp2 = [tmp[j] for j in range(len(tmp)) if j not in delinds]
        seqs[i] = tmp2
        assert len(seqs[i]) > 0

print("Computing distance matrix")
# Get a distance matrix
dmat = np.zeros((len(seqs), len(seqs)),dtype=np.uint16)
for i in range(len(seqs)-1):
    print(i)
    for j in range(i+1,len(seqs)):
        dmat[i,j] = ham_nogaps(seqs[i], seqs[j])

# Save the distance matrix
np.save(sys.argv[2]+".npy",dmat)

# Validate with random sample
#seqs = [str(r.seq).upper() for r in SeqIO.parse(sys.argv[1], format="fasta")]
#for i, seqi in enumerate(seqs):
#    for n in range(5):
#        j = random.randint(0,len(seqs)-1)
#        seqj = seqs[j]
#        h = ham_nogaps(seqi,seqj) 
#        if i < j:
#            assert h == dmat[i,j], (h, dmat[i,j])
#        else:
#            assert h == dmat[j,i], (h, dmat[j,i])
#



