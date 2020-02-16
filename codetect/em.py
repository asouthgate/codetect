#import pymc3 as pm
from io import StringIO
import math
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#import theano.tensor as tt
import sys
import random
np.set_printoptions(threshold=sys.maxsize)

def ham(s1, s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])

def gen_pop(L,D):
    major = [random.choice("ATCG") for i in range(L)]
    minor = [c for c in major]
    for j in range(len(minor)):
        roll = random.uniform(0,1)
        if roll < D:  
            minor[j] = random.choice([c for c in "ATCG" if c != minor[j]])
    return "".join(major),"".join(minor)

def sample_reads(N,l,popseqs,minorpropr,mu):
    w = [1-minorpropr, minorpropr]
    assert w[0] == max(w), "first sequence should be the major var"
    major = popseqs[0]
    X = []
    truen = 0
    for i in range(N):
        nm = 0
        seqi = np.random.choice([0,1],p=w)
        truen += seqi
        seq = popseqs[seqi]
        randpos = np.random.randint(0,len(seq)-l+1)
        sampseq = list(seq[randpos:randpos+l])
        for ci,c in enumerate(sampseq):
            roll = random.uniform(0,1)
            if roll < mu:
                alt = random.choice([z for z in "ATCG" if z != c])
                sampseq[ci] = alt
        print(seqi, randpos)
        print( "".join(sampseq))
        X.append((randpos,sampseq))
    return (1-(truen/N),X)

c2i = {"A":0, "C":1, "G":2, "T":3}
def reads2mat(ref,readpospairs):
    mat = np.zeros(shape=(len(ref),4))
    for rstart, read in readpospairs:
        for j,c in enumerate(read):
            i = rstart+j
            cind = c2i[c]
            mat[i,cind] += 1
    for ri in range(len(mat)):
        mat[ri] /= sum(mat[ri])
    return mat

NCOMPS = 2
NREADS = 5000
READLEN = 200
GAMMA = 0.02
D = 0.01
L = 2000
PI = 0.91
major,minor = gen_pop(L, D)
CONSENSUS = major
truepi, posreads = sample_reads(NREADS, READLEN, [major,minor], 1-PI, GAMMA)
readdists = [ham(major[pos:pos+READLEN], read) for pos, read in posreads]
plt.hist(readdists,bins=100)
plt.show()
print("TRUEPI", truepi)
print("HAM", ham(major,minor), ham(major,minor)/len(major))

# This index
Vindex = [[[] for c in range(4)] for i in range(len(CONSENSUS))]
for i,Xi in enumerate(posreads):
    pos,read = Xi
    for ri,r in enumerate(read):
        k = pos+ri
#        if r != CONSENSUS[k]:
        bind = c2i[r]
        Vindex[k][bind].append(i)
Vindex = np.array(Vindex, dtype=object)
#for vi,v in enumerate(Vindex):
#    print(vi,v)
#    assert sum([len(o) for o in v]) > 0


DATA = posreads
M = reads2mat(CONSENSUS, posreads)

def mask_low_variance_positions(M,X,Vindex,t=0.90,mindepth=20):
    delinds = []
    for ri,row in enumerate(M):
        if max(row) > t or sum([len(v) for v in Vindex[ri]]) < mindepth:
            delinds.append(ri)
    newCONS = "".join([c for ci,c in enumerate(CONSENSUS) if ci not in delinds])
    Msub = [row for ri,row in enumerate(M) if ri not in delinds]
    Xsub = [(pos,read) for pos,read in X]
    for di in delinds:
        for c in range(4):
            readswith = Vindex[di][c]
            for i in readswith:
                pos, read = Xsub[i]
                read2 = [c for c in read]
                read2[di-pos] = "X"
                Xsub[i] = (pos,read2)
#    for ri,tup in enumerate(Xsub):
#        print(ri,tup)
        
    Vindexsub = [row for ri,row in enumerate(Vindex) if ri not in delinds]
    for i in range(len(Xsub)):
        pos,read = Xsub[i]
        ndbefore = sum([1 for k in delinds if k < pos])
        Xsub[i] = (pos-ndbefore,[z for z in read if z != 'X'])
    return newCONS,Xsub, np.array(Msub), np.array(Vindexsub)

CONSENSUS, DATA, M, Vindex = mask_low_variance_positions(M,DATA,Vindex)
nmcache = []
for pos,read in DATA:
    nm = 0
    for k,c in enumerate(read):
        if CONSENSUS[k+pos] != c:
            nm += 1
    nmcache.append(nm)   

print("Cut down to", len(M), "from", L)

def PXi_condZi(xi,zi,g,v):
    logp = np.float64(0)
    pos,read = xi
    if zi == 0:
        for ki, c in enumerate(read):
            if c == CONSENSUS[ki+pos]:
    #            if zi == 0:
                logp += np.log((1-g))
    #            elif zi == 1:
    #                logp += np.log(1-v[ki+pos])
            else:
    #            if zi == 0:
                logp += np.log(g)
    #            elif zi == 1:
    #                logp += np.log(v[ki+pos])
    elif zi == 1:
        for ki, c in enumerate(read):
            bind = c2i[c]
#            print(ki,v[pos+ki],bind,c)
            assert v[pos+ki,bind] != 0.0
            logp += np.log(v[pos+ki,bind])
#            print(logp, v[ki])
#        assert np.exp(logp) != 0
#    print("??",zi,logp,g,np.log(g),np.log(1-g))
#    print("???",g)
    return np.exp(logp)

def calTi_pair(xi,pi,g,v):
    a = PXi_condZi(xi,0,g,v)
#    assert a > 0
    b = PXi_condZi(xi,1,g,v)
#    assert b > 0
    c = pi*a + (1-pi)*b
    t1i = (pi * a) / c
    t2i = ((1-pi) * b) / c
#    print("".join(xi[1]),a,b,pi*a,(1-pi)*b,c)
    return np.array([t1i,t2i])

def recalc_T(X,pi,g,v):
    res = []
    for i in range(len(X)):
        pair = calTi_pair(X[i],pi,g,v)
#        print(i,Xi,pi,g,pair)
        res.append(pair)
    return np.array(res)

def recalc_gamma(T,X,nmcache):
    # sum over reads, calculate the number of mismatches
    numos = [T[i,0]*nmcache[i] for i,Xi in enumerate(X)]
    denos = [T[i,0]*len(Xi[1]) for i,Xi in enumerate(X)]
    lens = [len(Xi[1]) for i,Xi in enumerate(X)]
#    print(numos,denos,lens)
    newgt = sum(numos)/sum(denos)
    assert 0 <= newgt <= 1
    return newgt

def recalc_V(T,X,Vindex,MIN_THRESHOLD=0.001):
    # Regularize by claiming that the probability of a mismatch can never be less than MIN_THRESHOLD
#    res = []
    newv = np.zeros((len(Vindex),4))
    for k in range(len(Vindex)):
        for c in range(4):
            # recalc Vi
            sumo = 0
            # Iterate over reads that mismatch at position k
            # THIS IS THE PROBABILITY THAT THEY ARE NOT THE SAME
            for ri in Vindex[k,c]:
#                print("ri")
                sumo += T[ri,1]
            assert sum(T[:,1]) > 0
    #        print(sumo, "???")
            assert np.isfinite(sumo), sumo
            newv[k,c] = sumo
#            res.append(max(sumo,MIN_THRESHOLD))
        newv[k] += MIN_THRESHOLD
        assert sum(newv[k]) != 0,(k,newv[k])
        newv[k] /= sum(newv[k])
        assert sum(newv[k]) > 0.99999, (newv[k], sum(newv[k]))
    return newv

def recalc_pi(T):
    newpi = sum([T[i,0] for i in range(len(T))])/len(T)
    new1mpi = sum([T[i,1] for i in range(len(T))])/len(T)
#    print(newpi,new1mpi)
    return sum([T[i,0] for i in range(len(T))])/len(T)
        
#vt = np.zeros((L,4))
#for ci in range(len((CONSENSUS))):
#    for c in range(4):
#        vt[ci,c] = M[ci,c]/sum(M[ci])
vt = M
pit = 0.99
gt = 0.01

for xi, tup in enumerate(DATA):
    pos,read = tup
#    print(xi,pos,read)
    for k,c in enumerate(read):
        assert xi in Vindex[k+pos][c2i[c]]
        assert M[pos+k,c2i[c]] > 0, (M[pos+k], pos, k,c)

assert truepi < 1.0
#print("vindexme", Vindex)
#print()
#for vi,v in enumerate(Vindex):
#    print(vi,v)
#    assert sum([len(o) for o in v]) > 0

for m in M:
    if sum([np.isnan(q) for q in m]) == 0:
        assert sum(m) > 0.98, m

assert ham(major,minor) > 0
#print(vt)

assert len(Vindex) == len(M)
assert 0 <= gt <= 1,gt

def expected_d(v,CONSENSUS):
    sumo = 0
    for ci, c in enumerate(CONSENSUS):
        alts = [v[ci,j] for j in range(4) if j != c2i[c]]
        sumo += sum(alts)
    return sumo/L

for t in range(200):
    print()
    print("******ITERATION %d" %t)
    print("TRUEPI", truepi)
    print("EST",pit,gt,expected_d(vt, CONSENSUS))
    print("HAM", ham(major,minor), ham(major,minor)/len(major))
#    print(vt)
    Tt = recalc_T(DATA,pit,gt,vt)
#    print(Tt[:,0])
    assert sum(Tt[:,1]) > 0
    assert sum(Tt[:,0]) > 0
#    pit = max(0.5,recalc_pi(Tt))
    pit = recalc_pi(Tt)
#    pitprop = recalc_pi(Tt)
#    if pitprop < 1.0:
#        pit = pitprop
#    gt = min(recalc_gamma(Tt,DATA,nmcache), 0.02)
    gt = recalc_gamma(Tt,DATA,nmcache)
    vt = recalc_V(Tt,DATA,Vindex)
    
print("EST",pit,gt,len([(CONSENSUS[j],minor[j],p) for j,p in enumerate(vt) if vt[j,c2i[CONSENSUS[j]]] < 0.03]))
print("TRUEPI", truepi)
print("NREADS", len(DATA))
print("HAM", ham(major,minor), ham(major,minor)/len(major))
   


    

    
    
        

