import random
import numpy as np
import math
from scipy.special import beta
import matplotlib.pyplot as plt
from aln import ReadAln
from likelihood_cache import *
from log import logger
import logging

def ham(s1,s2):
    return sum([1 for i in range(len(s1)) if s1[i] != s2[i]])

class MixtureModelSampler():
    """ Samples from posterior of mixture model parameters. """
    def __init__(self,initstring,allowed):
        """
        Args:
            initstring: initialization string.
            allowed: allowed states for each position
        """
        self.mm = MixtureModel(ds, random.uniform(0.0001,0.02), random.uniform(0.0001, 0.02), ds.get_consensus(), random.uniform(0.5,1.0), initstring)
        self.sample_strings = []
        self.sample_params = []
        self.allowed = allowed

    def mh_sample_normal(self,rd,sigma_pi=0.01,sigma_g=0.002):
        """ Sample triple pi, gamma0, gamma1 using Metropolis-Hastings. 
        
        Args:
            rd: ReadData object
        Optional Args:
            sigma_pi: variance for pi proposals.
            sigma_g: variance for gamma proposals.
        """
        u = random.uniform(0,1)
        curr_pi = self.mm.pi
        curr_g0 = self.mm.g0
        curr_g1 = self.mm.g1
        proppi = np.random.normal(curr_pi,sigma_pi)
        propg1 = np.random.normal(curr_g1,sigma_g)
        propg0 = np.random.normal(curr_g0,sigma_g)
        deno =  self.mm.cal_loglikelihood(rd,newpi=curr_pi,newg0=curr_g0,newg1=curr_g1)
        assert deno != None
        if 0 <= proppi <= 1 and 0 <= propg0 <= GAMMA_UPPER and 0 <= propg1 <= GAMMA_UPPER:
            numo = self.mm.cal_loglikelihood(rd,newpi=proppi,newg0=propg0,newg1=propg1)
            if np.log(u) <= numo-deno:
                logger.warning("accepting %f %f %f" % (proppi, propg0, propg1))
                return proppi, propg0, propg1
        logger.warning("rejecting %f %f %f" % (proppi,propg0,propg1))
        self.mm.cal_loglikelihood(rd,newpi=curr_pi,newg0=curr_g0,newg1=curr_g1)
        return curr_pi, curr_g0, curr_g1
    def gibbs_sample_si(self,ds,i):
        """ Sample a new base at position i using full conditional probability.

        Args:
            i: position for which to sample a new base.
        """
        # TODO: if there is regularization for a maximum distance and minimum distance, sampling can be skipped whenever
        # the alternatives are impossible
        logpmf = []
        currL = self.mm.cal_loglikelihood(ds)
        assert currL != None
        oldb = self.mm.st[i]
        # Keep the last b calculated; nm array must be recomputed
        for b in self.allowed[i]:
            # TODO: speed up by 1/Nstates by not recomputing for the one it already is; PUT THAT IN FIRST
            ll = self.mm.cal_loglikelihood(ds,i=i,newb=b)
            logpmf.append(ll)
        logpmf = np.array(logpmf)
        deno = logsumexp(logpmf)
        logpmf -= deno
        pmf = np.exp(logpmf)
        choice = np.random.choice(self.allowed[i],p=pmf)        
        ll = self.mm.cal_loglikelihood(ds,i=i,newb=choice)
        if choice != oldb:
            logger.warning("new b accepted", choice, oldb, ll)
        else:
            logger.warning("new b rejected", choice, oldb, ll)
        return choice
    def sample(self,rd,nits=300):    
        """ Sample from the full posterior of the mixture model.
        
        Args:
            rd: ReadData object.
        Optional Args:
            nits: number of samples.
        """
        consensus = ds.get_consensus()
        pi = self.mm.pi
        g0 = self.mm.g0
        g1 = self.mm.g1
        for nit in range(nits):
            sys.stderr.write("i=%d,L=%f,currham=%d,currpi=%f,currgam=%f,currmu=%f\n" % (nit,self.mm.cal_loglikelihood(ds), ham(self.mm.st,consensus), pi, g0,  g1))
            for i in ds.VALID_INDICES:
                newsi = self.gibbs_sample_si(ds,i)
            pi,g0,g1 = self.mh_sample_normal(ds)
            self.sample_params.append([pi,g0,g1])
            self.sample_strings.append([c for c in self.mm.st]) 
        return self.sample_strings, self.sample_params

def gen_array(strings):
    """ Generate a count matrix."""
    C = np.zeros((len(strings[0]),4))
    for s in strings:
        assert len(s) == len(strings[0])
        for ci,c in enumerate(s):
            C[ci,c] += 1
    return C

if __name__ == "__main__":
    from data_simulator import DataSimulator
    import sys
    if "--debug" not in sys.argv:
        logging.disable(logging.CRITICAL)

    #//*** Simulate dataset ***//
    GENOME_LENGTH = 30
    READ_LENGTH = 30
    N_READS = 10
    PI = 0.8
    GAMMA = 0.03
    D = 10
    MU = 0.000
    GAMMA_UPPER = 0.4
    ds = DataSimulator(N_READS,READ_LENGTH,GENOME_LENGTH,GAMMA,PI,D,MU,1)
    assert (ham(ds.get_consensus(), ds.get_major())) < ham(ds.get_minor(), ds.get_consensus())
    assert (ham(ds.get_major(), ds.get_minor()) <= ham(ds.get_consensus(), ds.get_major())) + ham(ds.get_minor(), ds.get_consensus())
    print(ds.M)
#    ds.filter(0.90)

    #//*** Preprocess dataset ***//
    allowed=[]
    states_per_site = 2
    for m in ds.M:
        allowed.append(sorted(np.argsort(m)[-states_per_site:]))
        assert len(allowed[-1]) > 0

    #//*** Initialize mixture model ***//
    randy = [random.choice([0,1,2,3]) for i in range(GENOME_LENGTH)]
    mms = MixtureModelSampler(ds.get_consensus(),allowed)

    #//*** Sample ***//
    strings,params = mms.sample(ds,nits=1000)

    #//*** Collect results ***//
    params = np.array(params)
    meanpi = np.mean(params[:,0])
    meang0 = np.mean(params[:,1])
    meang1 = np.mean(params[:,2])
    print("mean pi=%f, mean g0=%f, mean g1=%f" % (meanpi, meang0, meang1))
    C = gen_array(strings)
    assert ham(ds.get_minor(), ds.get_major()) == D
    print("total error to minor=",sum([1 for j in range(len(ds.get_minor())) if ds.get_minor()[j] != np.argmax(C[j])]))
    print("total error to major=",sum([1 for j in range(len(ds.get_consensus())) if ds.get_major()[j] != np.argmax(C[j])]))
    print("total error to consensus=",sum([1 for j in range(len(ds.get_consensus())) if ds.get_consensus()[j] != np.argmax(C[j])]))
    assert (ham(ds.get_consensus(), ds.get_major())) < ham(ds.get_minor(), ds.get_consensus())
