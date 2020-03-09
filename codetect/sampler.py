import random
import numpy as np
import math

def ham(s1,s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])

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
    def __init__(self, N, GL, MU, CONSENSUS, PI):
        self.N = N
        self.GL = GL
        self.wipe_memory()
        self.mu = MU
        self.pi = PI
        self.initialized = False
        self.consensus = tuple([c for c in CONSENSUS])
    def wipe_memory(self):
        self.initialized = False
        self.Larr = np.zeros((2,self.N,self.GL))
        self.Lsums0 = np.zeros(self.N)
        self.Lsums1 = np.zeros(self.N)
        self.Lsums = np.zeros(self.N)    
        self.L = None
    def getlogp(self,a,b):
        if a == b:
            return np.log(1-self.mu)
        else:
            return np.log(self.mu)
    def cal_read_loglikelihood(self, ri, read, st, ci):
        sumo = 0
        for j in range(len(read)):
            logp = self.getlogp(read[j],st[j])
            self.Larr[ci,ri,j] = logp
            sumo += self.Larr[ci,ri,j]
        return sumo
    def cal_full_loglikelihood(self, X, st):
        sumo = 0
        for ri,read in enumerate(X):
            self.Lsums0[ri] = self.cal_read_loglikelihood(ri,read,self.consensus,0)
            self.Lsums1[ri] = self.cal_read_loglikelihood(ri,read,st,1)
            self.Lsums[ri] = logsumexp([self.Lsums0[ri] + np.log(1-self.pi), self.Lsums1[ri] + np.log(self.pi)])
#            print("read=",ri)
#            print("pi=",self.pi)
#            print("hams=",ham(self.consensus,st), ham(read, self.consensus), ham(read,st))
#            print("logs=",self.Lsums0[ri], self.Lsums1[ri], self.Lsums[ri])
#            print("condlike=",np.exp(self.Lsums0[ri])*(1-self.pi), np.exp(self.Lsums1[ri])*self.pi, self.Lsums[ri])
#            print("likelihood=",np.exp(self.Lsums[ri]), np.exp(self.Lsums0[ri])*(1-self.pi) + np.exp(self.Lsums1[ri])*self.pi)
            sumo += self.Lsums[ri]
        self.initialized = True
        self.L = sumo
        return sumo
    def update_loglikelihood(self,X,i,b):
        sumo = 0
#        assert False
        for ri, read in enumerate(X):
            #** TODO: likelihood should not be recomputed for both here unless gamma has changed
            # 2X slower to do so
            logp0 = self.getlogp(read[i],self.consensus[i])
            self.Lsums0[ri] -= self.Larr[0,ri,i]
            self.Larr[0,ri,i] = logp0
            self.Lsums0[ri] += logp0             
            #*****
            logp1 = self.getlogp(read[i],b)
            self.Lsums1[ri] -= self.Larr[1,ri,i]
            self.Larr[1,ri,i] = logp1
            self.Lsums1[ri] += logp1             
            self.Lsums[ri] = logsumexp([self.Lsums0[ri] + np.log(1-self.pi), self.Lsums1[ri] + np.log(self.pi)])
            sumo += self.Lsums[ri]
        self.L = sumo
        return sumo             
    def cal_PZ0s(self):
        # Computes the log conditional probability for each datapoint
        PZs = []
        for ri, Li in enumerate(self.Lsums):       
            numo = self.Lsums0[ri] + np.log(1-self.pi)
            deno = self.Lsums[ri]
            PZs.append(np.exp(numo - deno))
        return PZs
    def set_mu(self,newmu):
        self.mu = newmu
        self.initialized = False
    def set_pi(self,newpi):
        self.pi = newpi
        self.wipe_memory()
    def cal_loglikelihood(self, X,st,i=None,b=None,newpi=None):
        if newpi != None:
            self.set_pi(newpi)
        if not self.initialized:
            print("recomputing fully")
            return self.cal_full_loglikelihood(X,st)
        else:
            return self.update_loglikelihood(X,i,b)

def mh_sample_pi(X,st,pi):
    # Independence sampler
    proppi = random.uniform(0.05,0.95)
    u = random.uniform(0,1)
    deno = llc.L
    numo = llc.cal_loglikelihood(X,st,newpi=proppi)
    print(pi,proppi,u,numo/deno)
    if np.log(u) <= numo-deno:
        #accept
        # TODO: turn off recomputing full likelihood after!
        return proppi
    else:
        #reject
        return pi 

def point_estimate_pi():
    probZ0s = llc.cal_PZ0s()
    return sum(probZ0s)/len(probZ0s)

def gibbs_sample_si(X,i,st0,allowed_states):
#    print("begin sampling")
    logpmf = []
    for b in allowed_states:
#        print("base", b)
        tmp = [c for c in st0]
        tmp[i] = b
        ll = llc.cal_loglikelihood(X,tmp,i,b)
        logpmf.append(ll)
    logpmf = np.array(logpmf)
#    print(logpmf)
    deno = logsumexp(logpmf)
#    print(logpmf, deno)
    logpmf -= deno
#    print(logpmf)
    pmf = np.exp(logpmf)
#    print(pmf)
#    print("ref:", Strue[i])
#    print()
#    assert False
    return np.random.choice(allowed_states,p=pmf)        

def sample(X,init,allowed,NITS=100):    
    strings = []
    pis = np.array([])
    st0 = [c for c in init]
    pi = 0.5
    for nit in range(NITS):
        print("iteration=",nit,"currham=",ham(st0,Strue),"currpi=%f" % pi, "meanpi=%f"%np.mean(pis))
        for i,si in enumerate(st0):
            newsi = gibbs_sample_si(X,i,st0,allowed[i])
#            if newsi != st0[i]:
#                print("******* NEW STATE")
#            if i > 10:
#                assert False
            st0[i] = newsi
#        pi = mh_sample_pi(X,st0,pi)
        pis = np.append(pis,pi)
        print(point_estimate_pi())
        strings.append([c for c in st0])      
    return strings

def gen_array(strings, L):
    C = np.zeros((L,4))
    for s in strings:
        for ci,c in enumerate(s):
            C[ci,c] += 1
    return C

NMAJOR = 200
NMINOR = 200
NREADS =  NMAJOR+NMINOR
GENOME_LENGTH = 50
MU = 0.3
Strue,X1 = gendata(NMAJOR,GENOME_LENGTH,MU)
Salt,X2 = gendata(NMINOR,GENOME_LENGTH,MU)
X = X1 + X2
Cog = gen_array(X, GENOME_LENGTH)
allowed=[]
states_per_site = 4
for c in Cog:
    allowed.append(np.argsort(c)[-states_per_site:])
print(allowed)
print(Strue)
print(X[0])
CONSENSUS = Strue
llc = LogLikelihoodCache(NREADS, GENOME_LENGTH, MU, CONSENSUS, 0.5)
randy = [random.choice([0,1,2,3]) for i in range(GENOME_LENGTH)]
assert randy != Strue
samps = sample(X,randy,allowed)
assert randy != Strue
C = gen_array(samps, GENOME_LENGTH)
SaltC = gen_array(X2, GENOME_LENGTH)

for ci, c in enumerate(C):
    errstr = "ERROR"
    if Salt[ci] == np.argmax(c):
        errstr = ""
    print(ci,"saltarr=",SaltC[ci],"samparr=",c,"true=",Strue[ci], "salt=", Salt[ci], errstr)
    
print("total error=",sum([1 for j in range(len(Strue)) if Salt[j] != np.argmax(C[j])]))
