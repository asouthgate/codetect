import random
import numpy as np

mu = 0.49
L = 300
N = 100
def mut(s):
    s2 = [c for c in s]
    for i in range(len(s2)):
        roll = random.uniform(0,1)
        if roll < mu:
            s2[i] = random.choice([j for j in range(4) if j != s[i]])
    return s2

def gendata():
    S = [random.choice([0,1,2,3]) for i in range(L)]
    muts = [mut(S) for j in range(N)]
    return S, muts

class LogLikelihoodCache():
    def __init__(self):
        self.Larr = np.zeros((N,L))
        self.Lsums = np.zeros(N)    
        self.initialized = False
    def getlogp(self,a,b):
        if a == b:
            return np.log(1-mu)
        else:
            return np.log(mu)
    def cal_full_loglikelihood(self, X,st):
        sumo = 0
        for ri,read in enumerate(X):
            risumo = 0
            for j in range(len(read)):
                logp = self.getlogp(read[j],st[j])
                self.Larr[ri,j] = logp
                risumo += self.Larr[ri,j]
            self.Lsums[ri] = risumo
            sumo += risumo
        self.initialized = True
        return sumo
    def update_loglikelihood(self,X,i,b):
        sumo = 0
        for ri, read in enumerate(X):
            logp = self.getlogp(read[i],b)
            self.Lsums[ri] -= self.Larr[ri,i]
            self.Larr[ri,i] = logp
            self.Lsums[ri] += logp               
            sumo += self.Lsums[ri]
        return sumo             
    def cal_loglikelihood(self, X,st,i,b):
        if not self.initialized:
            return self.cal_full_loglikelihood(X,st)
        else:
            return self.update_loglikelihood(X,i,b)        

def logsumexp(logls):
    m = max(logls)
    sumo = 0
    for l in logls:
        sumo += np.exp(l-m)
    return m + np.log(sumo)        

llc = LogLikelihoodCache()
def sample_si(X,i,st0):
    logpmf = np.zeros(4)
    for b in range(4):
        tmp = [c for c in st0]
        tmp[i] = b
        ll = llc.cal_loglikelihood(X,tmp,i,b)
        logpmf[b] = ll
    deno = logsumexp(logpmf)
    logpmf -= deno
    pmf = np.exp(logpmf)
    return np.random.choice(range(4),p=pmf)        

def gibbs_sample(X,init,NITS=100):    
    strings = []
    st0 = [c for c in init]
    for nit in range(NITS):
        print(nit)
        for i,si in enumerate(st0):
            newsi = sample_si(X,i,st0)
            st0[i] = newsi
        strings.append([c for c in st0])      
    return strings

def gen_array(strings):
    C = np.zeros((L,4))
    for s in strings:
        for ci,c in enumerate(s):
            C[ci,c] += 1
    return C

Strue,X = gendata()
randy = [random.choice([0,1,2,3]) for i in range(L)]
assert randy != Strue
samps = gibbs_sample(X,randy)
assert randy != Strue
C = gen_array(samps)

for ci, c in enumerate(C):
    print("samparr=",ci,c,"true=",Strue[ci])
    
print("total error=",sum([1 for j in range(len(Strue)) if Strue[j] != np.argmax(C[j])]))
