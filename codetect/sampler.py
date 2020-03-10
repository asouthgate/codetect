import random
import numpy as np
import math
from scipy.special import beta
import matplotlib.pyplot as plt
from aln import *

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
    def __init__(self, N, GL, GAMMA, MU, CONSENSUS, PI):
        self.N = N
        self.GL = GL
        self.wipe_memory()
        self.mu = MU
        self.gamma = GAMMA
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
        q = self.gamma
        if ci == 1:
            q = self.mu
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
            logp0 = self.getlogp(0,read.map[i],self.consensus[i])
            self.Lsums0[ri] -= self.Larr[0,ri,i]
            self.Larr[0,ri,i] = logp0
            self.Lsums0[ri] += logp0             
            #*****
            logp1 = self.getlogp(1,read.map[i],b)
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
            numo += (probZ0s[ri] * ham(r, self.consensus))
#            print(probZ0s[ri] * ham(r, self.consensus))
        newgam = numo/(spzs*GENOME_LENGTH)
#        print(numo,spzs,newgam)
        self.set_gamma(newgam)
        return newgam
    def set_mu(self,newmu):
        self.mu = newmu
        self.wipe_memory()
    def set_gamma(self,newgam):
        self.gamma = newgam
        self.wipe_memory()
    def set_pi(self,newpi):
        self.pi = newpi
        self.wipe_memory()
    def cal_loglikelihood(self, X,st,i=None,b=None,newpi=None,newgam=None,newmu=None):
        if newmu != None:
            self.set_mu(newmu)
        if newgam != None:
            self.set_gamma(newgam)
        if newpi != None:
            self.set_pi(newpi)
#        self.initialized=False
        if not self.initialized:
#            print("recomputing fully")
            return self.cal_full_loglikelihood(X,st)
        else:
            return self.update_loglikelihood(X,i,b)

def mh_sample_mu(X,st,mu):
    # Independence sampler
    propmu = random.uniform(0.0,0.1)
    u = random.uniform(0,1)
    deno = llc.cal_loglikelihood(X,st,newgam=mu)
    numo = llc.cal_loglikelihood(X,st,newmu=propmu)
#    print(numo,deno)
#    print(pi,proppi,u,numo/deno)
    if u <= np.exp(numo-deno):
        #accept
        # TODO: turn off recomputing full likelihood after!
        return propmu
    else:
        #reject
        llc.cal_loglikelihood(X,st,newmu=mu)
        return mu


def mh_sample_gamma(X,st,gamma):
    # Independence sampler
    propgam = random.uniform(0.0,0.3)
    u = random.uniform(0,1)
    deno = llc.cal_loglikelihood(X,st,newgam=gamma)
    numo = llc.cal_loglikelihood(X,st,newgam=propgam)
#    print(numo,deno)
#    print(pi,proppi,u,numo/deno)
    if u <= np.exp(numo-deno):
        #accept
        # TODO: turn off recomputing full likelihood after!
        return propgam
    else:
        #reject
        llc.cal_loglikelihood(X,st,newgam=gamma)
        return gamma

def logbetapdf(x,a,b):
#    print(x,1-x,a,b)
    assert a > 1
    assert b > 1
    return (a-1)*np.log(x) + (b-1)*np.log(1-x) - np.log(beta(a+1,b+1))

def mh_sample_normal(X,st,pi,gamma,mu,sigma_pi=0.03,sigma_gamma=0.015,sigma_mu=0.01):
    u = random.uniform(0,1)
    proppi = np.random.normal(pi,sigma_pi)
    propgamma = np.random.normal(gamma,sigma_gamma)
    propmu = np.random.normal(mu,sigma_mu)
    deno = llc.L
    assert pi == llc.pi, (pi,llc.pi)
    assert gamma == llc.gamma
    assert mu == llc.mu
#    print("proposing:",proppi,propgamma,propmu)
    numo = llc.cal_loglikelihood(X,st,newpi=proppi,newgam=propgamma,newmu=propmu)
#    print(numo,deno)
#    print(np.exp(numo),np.exp(deno))
    if 0 <= proppi <= 1 and 0 <= propgamma <= GAMMA_UPPER and 0 <= propmu <= MU_UPPER:
        if np.log(u) <= numo-deno:
            return proppi,propgamma,propmu
#    else:
#        print("out of bounds")
    #else reject
#    print("reject")
#    print("rejecting",proppi,propgamma,propmu)
    llc.set_pi(pi)
    llc.set_gamma(gamma)
    llc.set_mu(mu)
    llc.cal_loglikelihood(X,st,newpi=pi,newgam=gamma,newmu=mu)
    return pi,gamma,mu


def mh_sample_three(X,st,pi,gamma,mu,MF=100):
    # 3d Beta proposal
    proppi = np.random.beta(pi*MF,(1-pi)*MF)
    propgamma = np.random.beta(gamma*MF,(1-gamma)*MF)
    propmu = np.random.beta(mu*MF,(1-mu)*MF)
    # TODO: fast acceptance ratio by working out the ratio
    xs = [i/1000 for i in range(1,1000)]
    proppisy = [np.exp(logbetapdf(i,pi*MF,(1-pi)*MF)) for i in xs]
    pisy = [np.exp(logbetapdf(i,proppi*MF,(1-proppi)*MF)) for i in xs]
    plt.scatter(x=xs,y=proppisy,color='blue',alpha=0.5)
    plt.scatter(x=xs,y=pisy,color='purple',alpha=0.5)
    plt.show()
    propgamy = [np.exp(logbetapdf(i,gamma*MF,(1-gamma)*MF)) for i in xs]
    gamy = [np.exp(logbetapdf(i,propgamma*MF,(1-propgamma)*MF)) for i in xs]
    plt.scatter(x=xs,y=propgamy,color='green',alpha=0.5)
    plt.scatter(x=xs,y=gamy,color='brown',alpha=0.5)
    plt.show()
    propmusy = [np.exp(logbetapdf(i,mu*MF,(1-mu)*MF)) for i in xs]
    musy = [np.exp(logbetapdf(i,propmu*MF,(1-propmu)*MF)) for i in xs]
    plt.scatter(x=xs,y=propmusy,color='red',alpha=0.5)
    plt.scatter(x=xs,y=musy,color='pink',alpha=0.5)
    plt.show()
    plt.show()
    assert proppi > 0, proppi
    assert propgamma > 0, propgamma
    assert propmu > 0, propmu
    # numo2 = log g(x | x')
    numo2  = logbetapdf(pi,proppi*MF,(1-proppi)*MF)
    numo2 += logbetapdf(mu,propmu*MF,(1-propmu)*MF)
    numo2 += logbetapdf(gamma,propgamma*MF,(1-propgamma)*MF)
    
    deno2  = logbetapdf(proppi,pi*MF,(1-pi)*MF)
    deno2 += logbetapdf(propmu,mu*MF,(1-mu)*MF)
    deno2 += logbetapdf(propgamma,gamma*MF,(1-gamma)*MF)
    u = random.uniform(0,1)
    deno = llc.L
    assert pi == llc.pi, (pi,llc.pi)
    assert gamma == llc.gamma
    assert mu == llc.mu
#    assert math.fabs(tmp-deno) < 0.000001
#    deno = llc.L
    numo = llc.cal_loglikelihood(X,st,newpi=proppi,newgam=propgamma,newmu=propmu)
    print(numo,deno)
    print(numo2,deno2)
    print()
#    print(pi,proppi,u,numo/deno)
    if np.log(u) <= numo-deno+numo2-deno2:
        return proppi,propgamma,propmu
    else:
        #reject
        print("reject")
        llc.set_pi(pi)
        llc.set_gamma(gamma)
        llc.set_mu(mu)
        return pi,gamma,mu

def mh_sample_pi(X,st,pi):
    # Independence sampler
    proppi = random.uniform(0.0,0.3)
    u = random.uniform(0,1)
    deno = llc.L
    assert pi == llc.pi, (pi,llc.pi)
#    assert math.fabs(tmp-deno) < 0.000001
#    deno = llc.L
    numo = llc.cal_loglikelihood(X,st,newpi=proppi)
#    print(pi,proppi,u,numo/deno)
    if np.log(u) <= numo-deno:
        #accept
        # TODO: turn off recomputing full likelihood after!
        return proppi
    else:
        #reject
        llc.set_pi(pi)
        return pi 

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
    choice = np.random.choice(allowed_states,p=pmf)        
    llc.cal_loglikelihood(X,tmp,i,choice)
    return choice
def sample(X,init,allowed,NITS=500):    
    strings = []
    params = []
    st0 = [c for c in init]
    pi = llc.pi
    gamma = llc.gamma
    mu = llc.mu
    for nit in range(NITS):
        print("iteration=",nit,"currham=",ham(st0,Salt),"currpi=%f" % pi, "currgam=%f" % gamma, "currmu=%f" % mu)
        for i,si in enumerate(st0):
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
#        gamma = llc.point_estimate_gamma(X)
        for k in range(10):
            pi,gamma,mu = mh_sample_normal(X,st0,pi,gamma,mu)
            params.append([pi,gamma,mu])
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
        for ci,c in s.get_aln():
            C[ci,c] += 1
    return C

NMAJOR = 40
NMINOR = 10
NREADS =  NMAJOR+NMINOR
GENOME_LENGTH = 20
GAMMA = 0.33
MU = 0.3
MU_UPPER = 0.4
GAMMA_UPPER = 0.4
Strue,X1 = gendata(NMAJOR,GENOME_LENGTH,GAMMA)
Salt,X2 = gendata(NMINOR,GENOME_LENGTH,MU)
X = X1 + X2
X = X2Aln(X)
Cog = gen_array(X, GENOME_LENGTH)
allowed=[]
states_per_site = 2
for c in Cog:
    allowed.append(np.argsort(c)[-states_per_site:])
print(allowed)
print(Strue)
print(X[0])
CONSENSUS = Strue
llc = LogLikelihoodCache(NREADS, GENOME_LENGTH, random.uniform(0,GAMMA_UPPER), random.uniform(0,MU_UPPER), CONSENSUS, random.uniform(0,1))
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
