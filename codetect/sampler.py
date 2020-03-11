import random
import numpy as np
import math
from scipy.special import beta
import matplotlib.pyplot as plt
from aln import ReadAln
from likelihood_cache import *

def ham(s1,s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])

def mh_sample_normal(llc,ds,st,pi,g0,g1,sigma_pi=0.01,sigma_g=0.002):
    u = random.uniform(0,1)
    proppi = np.random.normal(pi,sigma_pi)
    propg1 = np.random.normal(g1,sigma_g)
    propg0 = np.random.normal(g0,sigma_g)
    deno =  llc.cal_loglikelihood(ds,st,newpi=pi,newg0=g0,newg1=g1)
    assert deno != None
    assert pi == llc.pi, (pi,llc.pi)
    assert g0 == llc.g0
    assert g1 == llc.g1
#    print("proposing:",proppi,propgamma,propmu)
    numo = llc.cal_loglikelihood(ds,st,newpi=proppi,newg0=propg0,newg1=propg1)
#    print(numo,deno)
#    print(np.exp(numo),np.exp(deno))
    if 0 <= proppi <= 1 and 0 <= propg0 <= GAMMA_UPPER and 0 <= propg1 <= GAMMA_UPPER:
        if np.log(u) <= numo-deno:
            print("accepting")
            return proppi,propg0,propg1
#    else:
#        print("out of bounds")
    #else reject
#   print("reject")
    print("rejecting",proppi,propg0,propg1)
#    print(numo,deno)
    llc.set_pi(pi)
    llc.set_g0(g0)
    llc.set_g1(g1)
    llc.cal_loglikelihood(ds,st,newpi=pi,newg0=g0,newg1=g1)
    return pi,g0,g1

def gibbs_sample_si(ds,i,st0,allowed_states,llc):
#    print("begin sampling")
    logpmf = []
    for b in allowed_states:
#        print("base", b)
        tmp = [c for c in st0]
        tmp[i] = b
        # TODO: speed up by 1/Nstates by not recomputing for the one it already is
        ll = llc.cal_loglikelihood(ds,tmp,i,b)
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
#    print(pmf)
    choice = np.random.choice(allowed_states,p=pmf)        
    llc.cal_loglikelihood(ds,tmp,i,choice)
    return choice

def sample(ds,init,allowed,NITS=100):    
    llc = LogLikelihoodCache(ds.X, ds.N_READS, GENOME_LENGTH, ds.GAMMA, ds.GAMMA, ds.get_consensus(), ds.PI, init)
    X = ds.X
    strings = []
    params = []
    st0 = [c for c in init]
    pi = llc.pi
    g0 = llc.g0
    g1 = llc.g1
    trueminor = ds.get_consensus()
    for nit in range(NITS):
        print("iteration=",nit,"currham=",ham(st0,trueminor),"currpi=%f" % pi, "currgam=%f" % g0, "currmu=%f" % g1)
#        for i,si in enumerate(st0):
        for i in ds.VALID_INDICES:
            si = st0[i]
            newsi = gibbs_sample_si(ds,i,st0,allowed[i],llc)
#            if newsi != st0[i]:
#                print("******* NEW STATE")
#            if i > 10:
#                assert False
            st0[i] = newsi
#            print(llc.L)
#        pi = mh_sample_pi(X,st0,pi)
#        pi = llc.point_estimate_pi()
#        gamma = mh_sample_gamma(X,st0,gamma)
#        mu = mh_sample_mu(X,st0,mu)
#        g0 = llc.point_estimate_gamma(X)
#        g1 = g0
        pi,g0,g1 = mh_sample_normal(llc,ds,st0,pi,g0,g1)
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
    from data_simulator import DataSimulator
    GENOME_LENGTH = 50
    READ_LENGTH = 20
    N_READS = 200
    PI = 0.98
    GAMMA = 0.03
    D = 10
    MU = 0.01
    GAMMA_UPPER = 0.4
    ds = DataSimulator(N_READS,READ_LENGTH,GENOME_LENGTH,GAMMA,PI,D,MU,1)
    ds.filter()
#    print(ds.get_consensus())
#    print(ds.get_major())
#    assert ds.get_consensus() == ds.get_major(), ham(ds.get_consensus(),ds.get_major())
    for xi in ds.X:
        print("?",xi.get_string())
    allowed=[]
    states_per_site = 2
    for m in ds.M:
        allowed.append(sorted(np.argsort(m)[-states_per_site:]))
#        print(allowed[-1])
        assert len(allowed[-1]) > 0
    randy = [random.choice([0,1,2,3]) for i in range(GENOME_LENGTH)]
    samps = sample(ds,ds.get_minor(),allowed)
    C = gen_array(samps, GENOME_LENGTH)
    print("total error=",sum([1 for j in range(len(ds.get_minor())) if ds.get_minor()[j] != np.argmax(C[j])]))
