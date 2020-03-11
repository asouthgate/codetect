import random
import numpy as np
import math
from scipy.special import beta
import matplotlib.pyplot as plt
from aln import ReadAln

def ham(s1,s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])

def X2Aln(X):
    alns = []
    for i,xi in enumerate(X):
        aln = ReadAln(i)
        for j in range(len(xi)):
            aln.append_mapped_base(j,xi[j])
        alns.append(aln)
    return alns

def mut(s, gamma):
    s2 = [c for c in s]
    for i in range(len(s2)):
        roll = random.uniform(0,1)
#        print(mu)
        if roll < gamma:
            s2[i] = random.choice([j for j in range(4) if j != s[i]])
    return s2

def gendata(N,L,gamma):
    S = [random.choice([0,1,2,3]) for i in range(L)]
    muts = [mut(S,gamma) for j in range(N)]
    return S, muts

def logsumexp(logls):
    m = max(logls)
    sumo = 0
    for l in logls:
        sumo += np.exp(l-m)
    return m + np.log(sumo)        

class LogLikelihoodCache():
    def __init__(self, N, GL, GAMMA0, GAMMA1, CONSENSUS, PI):
        self.N = N
        self.GL = GL
        self.wipe_memory()
        self.g1 = GAMMA1
        self.g0 = GAMMA0
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
    def getlogp(self,ci,a,b):
#        assert a ==b,(a,b)
        q = self.g0
        if ci == 1:
            q = self.g1
        if a == b:
            return np.log(1-q)
        else:
            return np.log(q)

    def cal_read_loglikelihood(self, ri, read, st, ci):
        sumo = 0
        for j in range(len(self.consensus)):
            logp = self.getlogp(ci,read.map[j],st[j])
            self.Larr[ci,ri,j] = logp
            sumo += self.Larr[ci,ri,j]
        return sumo
    def cal_full_loglikelihood(self, X, st):
        sumo = 0
        for ri,read in enumerate(X):
            self.Lsums0[ri] = self.cal_read_loglikelihood(ri,read,self.consensus,0)
            self.Lsums1[ri] = self.cal_read_loglikelihood(ri,read,st,1)
            self.Lsums[ri] = logsumexp([self.Lsums0[ri] + np.log(self.pi), self.Lsums1[ri] + np.log(1-self.pi)])
#            print("read=",ri)
#            print("pi=",self.pi)
#            print("hams=",ham(self.consensus,st), ham(read, self.consensus), ham(read,st))
#            print("logs=",self.Lsums0[ri], self.Lsums1[ri], self.Lsums[ri])
#            print("condlike=",np.exp(self.Lsums0[ri])*(1-self.pi), np.exp(self.Lsums1[ri])*self.pi, self.Lsums[ri])
#            print("likelihood=",np.exp(self.Lsums[ri]), np.exp(self.Lsums0[ri])*(1-self.pi) + np.exp(self.Lsums1[ri])*self.pi)
            sumo += self.Lsums[ri]*X[ri].count
        self.initialized = True
        self.L = sumo
        return sumo
    def update_loglikelihood(self,X,i,b):
        sumo = 0
#        assert False
        for ri, read in enumerate(X):
            #** TODO: likelihood should not be recomputed for both here unless gamma has changed
            # 2X slower to do so
            logp0 = self.getlogp(0,read.map[i],self.consensus[i])
            self.Lsums0[ri] -= self.Larr[0,ri,i]
            self.Larr[0,ri,i] = logp0
            self.Lsums0[ri] += logp0             
            #*****
            logp1 = self.getlogp(1,read.map[i],b)
            self.Lsums1[ri] -= self.Larr[1,ri,i]
            self.Larr[1,ri,i] = logp1
            self.Lsums1[ri] += logp1             
            self.Lsums[ri] = logsumexp([self.Lsums0[ri] + np.log(self.pi), self.Lsums1[ri] + np.log(1-self.pi)])
            sumo += self.Lsums[ri]*X[ri].count
        self.L = sumo
        return sumo             
    def cal_PZ0s(self):
        # Computes the log conditional probability for each datapoint
        PZs = []
        for ri, Li in enumerate(self.Lsums):       
            numo = self.Lsums0[ri] + np.log(self.pi)
            deno = self.Lsums[ri]
#            print(numo,deno, np.exp(numo-deno))
#            print(self.Lsums0[ri], self.Lsums1[ri])
            PZs.append(np.exp(numo - deno))
        return PZs
    def point_estimate_pi(self):
        probZ0s = self.cal_PZ0s()
#        print(probZ0s, len(probZ0s))
        newpi = sum(probZ0s)/len(probZ0s)
        self.set_pi(newpi)
        return newpi
    def point_estimate_gamma(self,X):
        probZ0s = self.cal_PZ0s()
        spzs = sum(probZ0s)
        numo = 0
        for ri,r in enumerate(X):
            numo += (probZ0s[ri] * r.cal_ham(self.consensus))
#            print(probZ0s[ri] * ham(r, self.consensus))
        newgam = numo/(spzs*GENOME_LENGTH)
#        print(numo,spzs,newgam)
        self.set_gamma(newgam)
        return newgam
    def set_g1(self,newg1):
        self.g1 = newg1
        self.wipe_memory()
    def set_g0(self,newg0):
        self.g0 = newg0
        self.wipe_memory()
    def set_pi(self,newpi):
        self.pi = newpi
        self.wipe_memory()
    def cal_loglikelihood(self, X,st,i=None,b=None,newpi=None,newg0=None,newg1=None):
        if newg1 != None:
            self.set_g1(newg1)
        if newg0 != None:
            self.set_g0(newg0)
        if newpi != None:
            self.set_pi(newpi)
#        self.initialized=False
        if not self.initialized:
#            print("recomputing fully")
            return self.cal_full_loglikelihood(X,st)
        else:
            return self.update_loglikelihood(X,i,b)

def mh_sample_normal(X,st,pi,g0,g1,sigma_pi=0.03,sigma_g=0.015):
    u = random.uniform(0,1)
    proppi = np.random.normal(pi,sigma_pi)
    propg1 = np.random.normal(g1,sigma_g)
    propg0 = np.random.normal(g0,sigma_g)
    deno = llc.L
    assert pi == llc.pi, (pi,llc.pi)
    assert g0 == llc.g0
    assert g1 == llc.g1
#    print("proposing:",proppi,propgamma,propmu)
    numo = llc.cal_loglikelihood(X,st,newpi=proppi,newg0=propg0,newg1=propg1)
#    print(numo,deno)
#    print(np.exp(numo),np.exp(deno))
    if 0 <= proppi <= 1 and 0 <= propg0 <= GAMMA_UPPER and 0 <= propg1 <= GAMMA_UPPER:
        if np.log(u) <= numo-deno:
            return proppi,propg0,propg1
#    else:
#        print("out of bounds")
    #else reject
#    print("reject")
#    print("rejecting",proppi,propgamma,propmu)
    llc.set_pi(pi)
    llc.set_g0(g0)
    llc.set_g1(g1)
    llc.cal_loglikelihood(X,st,newpi=pi,newg0=g0,newg1=g1)
    return pi,g0,g1

def gibbs_sample_si(X,i,st0,allowed_states):
#    print("begin sampling")
    logpmf = []
    for b in allowed_states:
#        print("base", b)
        tmp = [c for c in st0]
        tmp[i] = b
        # TODO: speed up by 1/Nstates by not recomputing for the one it already is
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
    print(pmf)
    choice = np.random.choice(allowed_states,p=pmf)        
    if choice != st0[i]:
        print("accepted new!")
    llc.cal_loglikelihood(X,tmp,i,choice)
    return choice

def sample(ds,init,allowed,NITS=100):    
    X = ds.X
    strings = []
    params = []
    st0 = [c for c in init]
    pi = llc.pi
    g0 = llc.g0
    g1 = llc.g1
    trueminor = ds.minor

    for nit in range(NITS):
        print("iteration=",nit,"currham=",ham(st0,trueminor),"currpi=%f" % pi, "currgam=%f" % g0, "currmu=%f" % g1)
#        for i,si in enumerate(st0):
        for i in ds.VALID_INDICES:
            si = st0[i]
            newsi = gibbs_sample_si(X,i,st0,allowed[i])
#            if newsi != st0[i]:
#                print("******* NEW STATE")
#            if i > 10:
#                assert False
            st0[i] = newsi
#        pi = mh_sample_pi(X,st0,pi)
#        pi = llc.point_estimate_pi()
#        gamma = mh_sample_gamma(X,st0,gamma)
#        mu = mh_sample_mu(X,st0,mu)
#        g0 = llc.point_estimate_gamma(X)
#        g1 = g0
        for k in range(5):
            pi,g0,g1 = mh_sample_normal(X,st0,pi,g0,g1)
            params.append([pi,g0,g1])
        strings.append([c for c in st0])      
#    plt.hist(params[100:,0])
#    print(np.mean(params[100:,0]))
#    plt.show()
#    plt.hist(params[100:,1])
#    print(np.mean(params[100:,1]))
#    plt.show()
#    plt.hist(params[100:,2])
#    print(np.mean(params[100:,2]))
#    plt.show()
    return strings

def gen_array(strings, L):
    C = np.zeros((L,4))
    for s in strings:
        for ci,c in enumerate(s):
            C[ci,c] += 1
    return C

if __name__ == "__main__":
    from simdata import DataSimulator
    GENOME_LENGTH = 200
    READ_LENGTH = 200
    N_READS = 50
    PI = 0.67
    GAMMA = 0.01
    D = 0
    MU = 0.01
    GAMMA_UPPER = 0.4
    ds = DataSimulator(N_READS,READ_LENGTH,GENOME_LENGTH,GAMMA,PI,D,MU,1)
    for xi in ds.X:
        print("?",xi.get_string())
    allowed=[]
    states_per_site = 2
    for m in ds.M:
        allowed.append(sorted(np.argsort(m)[-states_per_site:]))
#        print(allowed[-1])
        assert len(allowed[-1]) > 0
    llc = LogLikelihoodCache(ds.N_READS, GENOME_LENGTH, ds.GAMMA, ds.GAMMA, ds.get_consensus(), ds.PI)
    randy = [random.choice([0,1,2,3]) for i in range(GENOME_LENGTH)]
    samps = sample(ds,ds.get_minor(),allowed)
    C = gen_array(samps, GENOME_LENGTH)
    print("total error=",sum([1 for j in range(len(ds.get_minor())) if ds.get_minor()[j] != np.argmax(C[j])]))
