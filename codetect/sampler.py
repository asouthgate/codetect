import random
import numpy as np
import math
from scipy.special import beta
import matplotlib.pyplot as plt
from aln import ReadAln
from likelihood_cache import *

def ham(s1,s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])

def mh_sample_normal(mm,ds,st,pi,g0,g1,sigma_pi=0.01,sigma_g=0.002):
    u = random.uniform(0,1)
    proppi = np.random.normal(pi,sigma_pi)
    propg1 = np.random.normal(g1,sigma_g)
    propg0 = np.random.normal(g0,sigma_g)
    deno =  mm.cal_loglikelihood(ds,newpi=pi,newg0=g0,newg1=g1)
    assert deno != None
    if 0 <= proppi <= 1 and 0 <= propg0 <= GAMMA_UPPER and 0 <= propg1 <= GAMMA_UPPER:
        numo = mm.cal_loglikelihood(ds,newpi=proppi,newg0=propg0,newg1=propg1)
        if np.log(u) <= numo-deno:
            print("accepting", proppi, propg0, propg1)
            return proppi, propg0, propg1
    print("rejecting",proppi,propg0,propg1)
    mm.cal_loglikelihood(ds,newpi=pi,newg0=g0,newg1=g1)
    return pi, g0, g1

def gibbs_sample_si(ds,i,allowed_states,mm):
#    print("begin gibbs single sampling with:")
#    print(llc.st)
    logpmf = []
    currL = mm.cal_loglikelihood(ds)
    assert currL != None
    oldb = mm.st[i]
    # Keep the last b calculated; nm array must be recomputed
    for b in allowed_states:
        # TODO: speed up by 1/Nstates by not recomputing for the one it already is; PUT THAT IN FIRST
#        print("new state",b,i)
        ll = mm.cal_loglikelihood(ds,i=i,newb=b)
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
#    assert False
#    print(pmf)
#    print()
    choice = np.random.choice(allowed_states,p=pmf)        
    ll = mm.cal_loglikelihood(ds,i=i,newb=choice)
    if choice != oldb:
        print("new b accepted", choice, oldb, ll)
    else:
        print("new b rejected", choice, oldb, ll)
    return choice

def sample(ds,init,allowed,NITS=100):    
    mm = MixtureModel(ds, ds.GAMMA, ds.GAMMA, ds.get_consensus(), ds.PI, init)
    X = ds.X
    strings = []
    params = []
    st0 = [c for c in init]
    pi = mm.pi
    g0 = mm.g0
    g1 = mm.g1
    trueminor = ds.get_minor()
    for nit in range(NITS):
        print("i=",nit,"L=", mm.cal_loglikelihood(ds),"currham=",ham(mm.st,trueminor),"currpi=%f" % pi, "currgam=%f" % g0, "currmu=%f" % g1)
        for i in ds.VALID_INDICES:
            newsi = gibbs_sample_si(ds,i,allowed[i],mm)
        pi,g0,g1 = mh_sample_normal(mm,ds,st0,pi,g0,g1)
        params.append([pi,g0,g1])
        strings.append([c for c in mm.st]) 
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
    from data_simulator import DataSimulator
    GENOME_LENGTH = 10
    READ_LENGTH = 10
    N_READS = 20
    PI = 0.7
    GAMMA = 0.0001
    D = 2
    MU = 0.000
    GAMMA_UPPER = 0.4
    ds = DataSimulator(N_READS,READ_LENGTH,GENOME_LENGTH,GAMMA,PI,D,MU,1)
    print(ds.M)
#    ds.filter()
#    print(ds.get_consensus())
#    print(ds.get_major())
#    assert ds.get_consensus() == ds.get_major(), ham(ds.get_consensus(),ds.get_major())
    for read in ds.X:
        print(read.get_string(),read.count)
    allowed=[]
    states_per_site = 2
    for m in ds.M:
        allowed.append(sorted(np.argsort(m)[-states_per_site:]))
#        print(allowed[-1])
        assert len(allowed[-1]) > 0
    randy = [random.choice([0,1,2,3]) for i in range(GENOME_LENGTH)]
    samps = sample(ds,[c for c in ds.get_minor()],allowed)
    C = gen_array(samps, GENOME_LENGTH)
    print("total error to minor=",sum([1 for j in range(len(ds.get_minor())) if ds.get_minor()[j] != np.argmax(C[j])]))
    print("total error to major=",sum([1 for j in range(len(ds.get_consensus())) if ds.get_minor()[j] != np.argmax(C[j])]))
