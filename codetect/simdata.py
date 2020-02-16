import numpy as np
import random
import matplotlib.pyplot as plt
from  em import *

def ham(s1, s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])

class DataSimulator():
    def __init__(self, N_READS, READ_LENGTH, GENOME_LENGTH, GAMMA, PI, D):
        self.N_READS = N_READS
        self.READ_LENGTH = READ_LENGTH
        self.GENOME_LENGTH = GENOME_LENGTH
        self.GAMMA = GAMMA
        self.PI = PI
        self.D = D    
        major, minor = self.gen_pop(GENOME_LENGTH, D)
        self.true_ham = ham(major,minor)
        assert self.true_ham > 0.0
        self.true_pid = self.true_ham/len(major)
        self.POPULATION = [major,minor]
        self.CONSENSUS = major
        self.X,self.Z,self.C,self.DEPTH = self.sample_reads()
        self.true_pi = 0
        for zi,z in enumerate(self.Z):
            if z == 0:
                self.true_pi += self.C[zi]
        self.true_pi /= sum(self.C)
        # NEED COUNT
        self.M = self.reads2mat()
        # OPTIONAL
        readdists = [ham(major[pos:pos+self.READ_LENGTH], read) for pos, read in self.X]
        plt.hist(readdists,bins=100)
        plt.show()
        assert sum(self.C) == self.N_READS
        self.V_INDEX = self.build_Vindex()
        self.mask_low_variance_positions()
        self.V_INDEX = self.build_Vindex()
        self.NM_CACHE = self.build_nm_cache()
        assert len(self.V_INDEX) == len(self.M)
        assert len(self.NM_CACHE) == len(self.X)


    def build_Vindex(self):
        Vindex = [[[] for c in range(4)] for i in range(len(self.CONSENSUS))]
        for i,Xi in enumerate(self.X):
            pos,read = Xi
            for ri,r in enumerate(read):
                k = pos+ri
                bind = self.c2i(r)
                Vindex[k][bind].append(i)
        Vindex = np.array(Vindex, dtype=object)
        return Vindex
       
    def c2i(self,c):
        c2id = {"A":0, "C":1, "G":2, "T":3}
        return c2id[c]        

    def gen_pop(self,L,D):
        major = [random.choice("ATCG") for i in range(self.GENOME_LENGTH)]
        minor = [c for c in major]
        for j in range(len(minor)):
            roll = random.uniform(0,1)
            if roll < D:  
                minor[j] = random.choice([c for c in "ATCG" if c != minor[j]])
        return "".join(major),"".join(minor)

    def sample_reads(self):
        w = [self.PI, 1-self.PI]
        assert w[0] == max(w), "first sequence should be the major var"
        major = self.CONSENSUS
        X = []
        Z = []
        countd = {}
        C = []
        DEPTH = np.zeros(self.GENOME_LENGTH)
        truen = 0
        for i in range(self.N_READS):
            nm = 0
            seqi = np.random.choice([0,1],p=w)
            truen += seqi
            seq = self.POPULATION[seqi]
            randpos = np.random.randint(0,len(seq)-self.READ_LENGTH+1)
            sampseq = list(seq[randpos:randpos+self.READ_LENGTH])
            for ci,c in enumerate(sampseq):
                roll = random.uniform(0,1)
                if roll < self.GAMMA:
                    alt = random.choice([z for z in "ATCG" if z != c])
                    sampseq[ci] = alt
            print("???",seqi, randpos)
            print( "".join(sampseq))
            pair = (randpos,"".join(sampseq))
            if pair in countd:
                countd[pair] += 1
            else:
                countd[pair] = 1
                X.append(pair)
                Z.append(seqi)
        for pair in X:
            count = countd[pair]
            C.append(count)
            DEPTH[pair[0]:pair[0]+len(pair[1])] += count
        assert sum(C) == self.N_READS
        return X, Z, C, DEPTH

    def reads2mat(self):
        mat = np.zeros(shape=(len(self.CONSENSUS),4))
        for ri, pair in enumerate(self.X):
            rstart,read = pair
            for j,c in enumerate(read):
                i = rstart+j
                cind = self.c2i(c)
                mat[i,cind] += self.C[ri]
        for ri in range(len(mat)):
            mat[ri] /= sum(mat[ri])
        return mat

    def mask_low_variance_positions(self,t=0.95,mindepth=0):
        delinds = []
        for ri,row in enumerate(self.M):
            if max(row) > t:
                delinds.append(ri)
        newCONS = "".join([c for ci,c in enumerate(self.CONSENSUS) if ci not in delinds])
#        Csub = [c for ci, c in enumerate(self.C) if ci not in delinds]
#        Zsub = [c for ci, c in enumerate(self.Z) if ci not in delinds]
        Msub = [row for ri,row in enumerate(self.M) if ri not in delinds]
        Xsub = [(pos,read) for pos,read in self.X]
        for di in delinds:
            for c in range(4):
                readswith = self.V_INDEX[di][c]
                for i in readswith:
                    pos, read = Xsub[i]
                    read2 = [c for c in read]
                    read2[di-pos] = "X"
                    Xsub[i] = (pos,read2) 
        delinds2 = {}
        for i in range(len(Xsub)):
            pos,read = Xsub[i]
            ndbefore = sum([1 for k in delinds if k < pos])
            Xsub[i] = (pos-ndbefore,[z for z in read if z != 'X'])
            if len(Xsub[i][1]) == 0:
                delinds2[i] = 1
#        self.Z = Zsub
#        self.C = Csub
        self.CONSENSUS = newCONS
        self.X = [Xi for i,Xi in enumerate(Xsub) if i not in delinds2]
        self.C = [Ci for i,Ci in enumerate(self.C) if i not in delinds2]
        self.Z = [Zi for i,Zi in enumerate(self.Z) if i not in delinds2]
        self.M = np.array(Msub)
        assert len(self.Z) == len(self.C) == len(self.X)
        assert len(self.M) == len(self.CONSENSUS) 

    def build_nm_cache(self):
        nmcache = []
        for pos,read in self.X:
            nm = 0
            for k,c in enumerate(read):
                if self.CONSENSUS[k+pos] != c:
                    nm += 1
            nmcache.append(nm)   
        return nmcache

if __name__ == "__main__":
#    def __init__(self, N_READS, READ_LENGTH, GENOME_LENGTH, GAMMA, PI, D):
    ds = DataSimulator(5000,200,2000,0.02,0.7,0.02)
    print(ds.X)
    print(ds.C)
    print(ds.Z)
    print(ds.M)
    print(ds.V_INDEX)
    print("*********PASSING TO EM*********")
    em = EM(ds.X, ds.C, ds.M, ds.V_INDEX, ds.CONSENSUS, ds.NM_CACHE)
    em.do(ds.Z,ds.true_pi,ds.true_ham,ds.true_pid)


