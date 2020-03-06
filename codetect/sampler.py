import random
import numpy as np

def mut(s, mu):
    s2 = [c for c in s]
    for i in range(len(s2)):
        roll = random.uniform(0,1)
#        print(mu)
        if roll < mu:
            s2[i] = random.choice([j for j in range(4) if j != s[i]])
    return s2

def gendata(N,L,mu):
    S = [random.choice([0,1,2,3]) for i in range(L)]
    muts = [mut(S,mu) for j in range(N)]
    return S, muts

def logsumexp(logls):
    m = max(logls)
    sumo = 0
    for l in logls:
        sumo += np.exp(l-m)
    return m + np.log(sumo)        

class LogLikelihoodCache():
    def __init__(self, N, L, MU, CONSENSUS, PI):
        self.Larr = np.zeros((2,N,L))
        self.Lsums0 = np.zeros(N)
        self.Lsums1 = np.zeros(N)
        self.Lsums = np.zeros(N)    
        self.mu = MU
        self.pi = PI
        self.initialized = False
        self.consensus = CONSENSUS
    def getlogp(self,a,b):
        if a == b:
            return np.log(1-self.mu)
        else:
            return np.log(self.mu)
    def cal_read_loglikelihood(self, read, st, ci):
        sumo = 0
        for j in range(len(read)):
            logp = self.getlogp(read[j],st[j])
            self.Larr[ci,ri,j] = logp
            sumo += self.Larr[ci,ri,j]
        return sumo
    def cal_full_loglikelihood(self, X, st):
        sumo = 0
        for ri,read in enumerate(X):
            Lsums0[ri] = cal_read_loglikelihood(self,read,self.consensus,0)
            Lsums1[ri] = cal_read_loglikelihood(self,read,st,1)
            self.Lsums[ri] = logsumexp([Lsums0[ri] + np.log(1-self.pi), Lsums1[ri] + np.log(self.pi)])
            sumo += self.Lsums[ri]
        self.initialized = True
        return sumo
    def update_loglikelihood(self,X,i,b):
        sumo = 0
        for ri, read in enumerate(X):
            #** TODO: likelihood should not be recomputed for both here unless gamma has changed
            # 2X slower to do so
            logp0 = self.getlogp(read[i],self.consensus[i])
            self.Lsums0[ri] -= self.Larr0[ri,i]
            self.Larr0[ri,i] = logp
            self.Lsums0[ri] += logp               
            #*****
            logp1 = self.getlogp(read[i],b)
            self.Lsums1[ri] -= self.Larr1[ri,i]
            self.Larr1[ri,i] = logp
            self.Lsums1[ri] += logp               
            sumo += logsumexp([Lsums0[ri] + np.log(1-self.pi), Lsums1[ri] + np.log(self.pi)])
        return sumo             
    def cal_loglikelihood(self, X,st,i,b):
        if not self.initialized:
            return self.cal_full_loglikelihood(X,st)
        else:
            return self.update_loglikelihood(X,i,b)        


def sample_si(X,i,st0):
    logpmf = np.zeros(4)
    for b in range(4):
        tmp = [c for c in st0]
        tmp[i] = b
        ll = llc.cal_loglikelihood(X,tmp,i,b)
        logpmf[b] = ll
#        print(ll)
    deno = logsumexp(logpmf)
#    print(logpmf,deno)
    logpmf -= deno
#    print(logpmf)
    pmf = np.exp(logpmf)
#    print(pmf)
#    assert False
    return np.random.choice(range(4),p=pmf)        

def gibbs_sample(X,init,NITS=300):    
    strings = []
    st0 = [c for c in init]
    for nit in range(NITS):
        print(nit)
        for i,si in enumerate(st0):
            newsi = sample_si(X,i,st0)
            st0[i] = newsi
        strings.append([c for c in st0])      
    return strings

def gen_array(strings, L):
    C = np.zeros((L,4))
    for s in strings:
        for ci,c in enumerate(s):
            C[ci,c] += 1
    return C

NREADS =  10
GENOME_LENGTH = 30
MU = 0.49
Strue,X = gendata(NREADS,GENOME_LENGTH,MU)
print(Strue)
print(X[0])
llc = LogLikelihoodCache(NREADS, GENOME_LENGTH, MU)
randy = [random.choice([0,1,2,3]) for i in range(GENOME_LENGTH)]
assert randy != Strue
samps = gibbs_sample(X,randy)
assert randy != Strue
Cog = gen_array(X, GENOME_LENGTH)
C = gen_array(samps, GENOME_LENGTH)

for ci, c in enumerate(C):
    print(ci,"realarr=",Cog[ci],"samparr=",c,"true=",Strue[ci])
    
print("total error=",sum([1 for j in range(len(Strue)) if Strue[j] != np.argmax(C[j])]))
