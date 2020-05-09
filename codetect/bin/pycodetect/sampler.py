import random
import numpy as np
import matplotlib.pyplot as plt
from pycodetect.likelihood_cache import MixtureModel
from pycodetect.log import logger
import logging
from pycodetect.utils import ham, logsumexp, ham_early_bail
import sys

class RefPropDist():
    def __init__(self, dmat):
        self.dmat = dmat
        assert dmat.shape[0] == dmat.shape[1]
        self.pmf = np.zeros(dmat.shape)
        self.cmf = np.zeros(dmat.shape)
        self.rankmat = np.zeros(dmat.shape).astype(int)
        for i in range(len(dmat)):
            for k in range(len(dmat[i])):
                self.pmf[i,k] = 1/(1+dmat[i,k]**3)
            # Should not sample itself!!!
            self.pmf[i,i] = 0
            self.pmf[i] /= sum(self.pmf[i])
            # Now arg sort it; keep track of which index has which probability
            self.rankmat[i] = np.argsort(self.pmf[i])
    def sample(self,i):        
        # TO DO: a bit slow
        j = np.random.choice([k for k in range(len(self.pmf[i]))], p=self.pmf[i])
        assert i != j
        return j
    def logpdf(self,i,j):
        return np.log(self.pmf[i,j])

class MixtureModelSampler():
    """ Samples from posterior of mixture model parameters."""
    def __init__(self,rd,fixed_point,initstring=None,allowed_states=None,allowed_positions=None,dmat=None,refs=None,sample_gammas=False, min_d=None):
        # TODO: split class; too many responsibilities; args should not have so many mutually exclusive groups
        """
        Args:
            rd: ReadData object
            fixed_point: specified sequence for zeroth cluster 
        Optional Args:
            initstring: initialization string
            allowed_positions: positions where the string can vary from any reference (used for Gibbs string sampling)
            allowed_states: allowed states for each position (used for Gibbs string sampling)
            dmat: distance matrix for references (used for reference sampling)
            refs: references (used for reference sampling)
            fixed_gamma: a fixed gamma (used when estimation of gamma is not performed)
        """
        self.sample_gammas = sample_gammas
        self.sample_refs = False
        self.sample_strings_gibbs = False

        # Initialize references used for reference sampling
        if dmat is not None:
            # Estimate given references
            assert refs is not None
            self.sample_refs = True
            self.refs = refs
            self.ref_prop_dist = RefPropDist(dmat)
            assert len(refs) == len(dmat)
            assert initstring is None, "Do not provide an initstring when reference sampling"
            self.init_refi = random.randint(0,len(refs)-1)
            initstring = refs[self.init_refi]

        # Initialized states used for Gibbs sampling
        if allowed_states is not None:
            assert not self.sample_refs, "Currently ref MH+gibbs is disabled. It may not be mathematically valid. Fixed+gibbs is allowed, however."
            # Also perform Gibbs sampling        
            self.allowed_states = allowed_states
            assert len(allowed_states) == len(fixed_point), "Allowed states is not like allowed sites; it is indexed relative to the alignment"
            self.allowed_positions = allowed_positions
            self.sample_strings_gibbs = True
            self.min_d = min_d
            assert initstring is not None, "You must provide an initialization string. Try a random one."

        # Initialize gamma
        if sample_gammas:
            self.sample_gammas = True
            init_gamma0 = random.uniform(0.0001, 0.02)
            init_gamma1 = random.uniform(0.0001, 0.02)
        else:
            est_gamma = self.point_estimate_gamma(rd)
            init_gamma0 = init_gamma1 = est_gamma
            sys.stderr.write("Point estimated gamma as %f\n" % est_gamma)

        # Initialize pi; always estimate pi
        self.estimate_pi = True
        init_pi = random.uniform(0.5,1.0)

        # Build model
        self.mm = MixtureModel(rd, init_gamma0, init_gamma1, fixed_point, init_pi, initstring)
        self.mm.cal_loglikelihood(rd)
        self.sample_strings = []
        self.sample_params = []
        self.sample_Ls = []

    def point_estimate_gamma(self,rd,t=0.95):
        """ Point estimate gamma 
        
        Assumes to take the positions that probably aren't variant positions.
        """
        nmuts = 0
        ntotal = 0
        for ri,row in enumerate(rd.C):
            if max(rd.M[ri]) > t:
                nmuts += (sum(row)-max(row))
                ntotal += sum(row)
        return nmuts/ntotal               

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
#        propg1 = np.random.normal(curr_g1,sigma_g)
        propg0 = np.random.normal(curr_g0,sigma_g)
        propg1 = propg0
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

    def mh_sample_normal_onlypi(self,rd,sigma_pi=0.01):
        """ Sample pi using Metropolis-Hastings. 
        
        Args:
            rd: ReadData object
        Optional Args:
            sigma_pi: variance for pi proposals.
        """
        u = random.uniform(0,1)
        curr_pi = self.mm.pi
        proppi = np.random.normal(curr_pi,sigma_pi)
        deno =  self.mm.cal_loglikelihood(rd,newpi=curr_pi)
        assert deno != None
        if 0 <= proppi <= 1:
            numo = self.mm.cal_loglikelihood(rd,newpi=proppi)
            if np.log(u) <= numo-deno:
                logger.warning("accepting %f" % (proppi))
                return proppi
        logger.warning("rejecting %f" % (proppi))
        self.mm.cal_loglikelihood(rd,newpi=curr_pi)
        return curr_pi

    def get_difference_positions(self, currs, props):
        # curris = propis! always
        iss = []
        currbs = []
        propbs = []
        for i in range(len(currs)):
            if currs[i] != props[i]:
                iss.append(i)
                currbs.append(currs[i])
                propbs.append(props[i])
        return iss,currbs,iss,propbs   

    def propose_ref_st(self,curr_st,curr_refi):
        # TODO: can cache proposal changes
        # TODO: better: pre-calculate positions and differences between pairs in a numpy array
        # That could be a thousand times bigger but, we could use a proposal walk that perhaps only allows stepping between
        # A few differences at a time; maybe even stepping between nearest neighbors
        prop_refj = self.ref_prop_dist.sample(curr_refi)
        prop_st = self.refs[prop_refj]
        lprob_ij = self.ref_prop_dist.logpdf(curr_refi,prop_refj)
        lprob_ji = self.ref_prop_dist.logpdf(prop_refj,curr_refi)
        factor = lprob_ji-lprob_ij
        curris,currbs,propis,propbs = self.get_difference_positions(self.mm.st,prop_st)
        return prop_refj,curris,currbs,propis,propbs,factor
        
    def mh_reference_sample(self,rd,curr_refi):
        u = random.uniform(0,1)
        # TODO: SLOW: FIX
        # TODO: add prior
        assert ham(self.mm.st, self.refs[curr_refi]) == 0, ham(self.mm.st, self.refs[curr_refi])
        # Factor is the log probability of the hastings ratio (proposal coefficient)
        deno = self.mm.cal_loglikelihood(rd)
        prop_refi,curris,currbs,propis,propbs,factor = self.propose_ref_st(self.mm.st,curr_refi)
        numo = self.mm.cal_loglikelihood(rd,newis=propis,newbs=propbs)
#        print(numo,deno,factor)
        if np.log(u) <= numo-deno+factor:
            return prop_refi
        else:
            self.mm.cal_loglikelihood(rd,newis=curris,newbs=currbs)
            return curr_refi

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
        for b in self.allowed_states[i]:
            # TODO: speed up by 1/Nstates by not recomputing for the one it already is; PUT THAT IN FIRST
            ll = self.mm.cal_loglikelihood(ds,newis=[i],newbs=[b])
            logpmf.append(ll)
        logpmf = np.array(logpmf)
        deno = logsumexp(logpmf)
        logpmf -= deno
        pmf = np.exp(logpmf)
        choice = np.random.choice(self.allowed_states[i],p=pmf)        
        ll = self.mm.cal_loglikelihood(ds,newis=[i],newbs=[choice])
        if choice != oldb:
            logger.warning("new b accepted=%s,old=%s,ll=%f" % (choice, oldb, ll))
        else:
            logger.warning("new b rejected=%s,old=%s,ll=%f" % (choice, oldb, ll))
        return choice

    def constrained_sample_string(self,ds, refstr, curr_ref_diffs, max_dist=100):
        for j in self.allowed_positions:
            if j in curr_ref_diffs:
                # Even if we at the maximum dist, we can revert
                assert refstr[j] != self.mm.st[j]
                choice = self.gibbs_sample_si(ds,j)
                if choice == refstr[j]:
                    # If we have reverted, record that
                    curr_ref_diffs.remove(j)
            elif len(curr_ref_diffs) >= max_dist:
                assert refstr[j] == self.mm.st[j]
                # Cannot accept a new state (zero prior outside self.min_d)
                # Do nothing
                pass
            else:
                assert refstr[j] == self.mm.st[j]
                # We are within a radius of max_ref_dist to our refstr
                choice = self.gibbs_sample_si(ds,j)
                if choice != refstr[j]:
                    curr_ref_diffs.add(j)
        return self.mm.st, curr_ref_diffs

    def sample(self,rd,nits=300):
        """ Sample from the full posterior of the mixture model using references only.
        
        Args:
            rd: ReadData object.
        Optional Args:
            nits: number of samples.
        """
        pi = self.mm.pi
        g0 = self.mm.g0
        g1 = self.mm.g1
        st0 = self.mm.st
        refstr = tuple([c for c in st0])
        curr_ref_diffs = set()
        sampreflog = ""
        if self.sample_refs:
            refi = self.init_refi
        for nit in range(nits):
            currham = ham(rd.get_consensus(), st0)
            currL = self.mm.cal_loglikelihood(rd)
            sys.stderr.write("i=%d,L=%f,%scurrpi=%f,currgam=%f,currmu=%f,currham=%d\n" % (nit,currL, sampreflog, pi, g0, g1, currham))
            if self.sample_refs:
                refi = self.mh_reference_sample(rd, refi)
                refstr = self.refs[refi]
                st0 = self.mm.st
                sampreflog = "sampref=%d," % refi
            if self.sample_strings_gibbs:
                assert not self.sample_refs, "Only ref sampling OR gibbs is allowed currently."
#                st0,curr_ref_diffs = self.constrained_sample_string(rd, refstr, curr_ref_diffs) 
                for j in rd.VALID_INDICES:
                    st0[j] = self.gibbs_sample_si(rd, j)
            if self.sample_gammas:
                pi,g0,g1 = self.mh_sample_normal(rd)
            else:
                pi = self.mh_sample_normal_onlypi(rd)
            self.sample_params.append([pi,g0,g1])
            self.sample_strings.append(st0) 
            self.sample_Ls.append(currL)
        return self.sample_strings, self.sample_params, self.sample_Ls

    def sample_full(self,rd,nits=300):    
        """ Sample from the full posterior of the mixture model, allowing any string (gibbs sample).
        
        Args:
            rd: ReadData object.
        Optional Args:
            nits: number of samples.
        """
        consensus = rd.get_consensus()
        pi = self.mm.pi
        g0 = self.mm.g0
        g1 = self.mm.g1
        for nit in range(nits):
            sys.stderr.write("i=%d,L=%f,currham=%d,currpi=%f,currgam=%f,currmu=%f\n" % (nit,currL, ham(self.mm.st,consensus), pi, g0,  g1))
            for i in rd.VALID_INDICES:
                newsi = self.gibbs_sample_si(rd,i)
            pi,g0,g1 = self.mh_sample_normal(rd)
            currL = self.mm.cal_loglikelihood(rd)
            self.sample_params.append([pi,g0,g1])
            self.sample_strings.append([c for c in self.mm.st]) 
            self.sample_Ls.append(currL)
        return self.sample_strings, self.sample_params, self.sample_Ls

def gen_array(strings):
    """ Generate a count matrix."""
    C = np.zeros((len(strings[0]),5))
    for s in strings:
        assert len(s) == len(strings[0])
        for ci,c in enumerate(s):
            C[ci,c] += 1
    return C

def del_close_to_fixed_point(fixed_point,mind,refs,dmat):
    """ Remove references too close to a fixed point. """
    new_refs = []
    inds2del = []
    for ri,ref in enumerate(refs):
        if ham_early_bail(ref, fixed_point, mind) >= mind:
            new_refs.append(ref)
        else:
            inds2del.append(ri)
    inds2del = np.array(inds2del)
    if len(inds2del) > 0:
        dmat = np.delete(dmat,inds2del,0)
        dmat = np.delete(dmat,inds2del,1)
    assert len(new_refs) == dmat.shape[0] == dmat.shape[1], (len(new_refs), dmat.shape, len(inds2del))
    return new_refs, dmat

if __name__ == "__main__":
    from data_simulator import DataSimulator
    from Bio import SeqIO
    import sys
    if "--debug" not in sys.argv:
        logging.disable(logging.CRITICAL)
    refs = [tuple([c2i[c.upper()] for c in str(r.seq)]) for r in SeqIO.parse(sys.argv[1], "fasta")]
    dmat = np.load(sys.argv[2])
    # TODO: SLOW: FIX FOR IMPORT
    for i in range(len(dmat)-1):
        for j in range(i+1,len(dmat)):
            dmat[j,i] = dmat[i,j]

    #//*** Simulate dataset ***//
#    GENOME_LENGTH = 30
    READ_LENGTH = 200
    N_READS = 5000
    PI = 0.8
    GAMMA = 0.02
    D = 0
    MU = 0.000
    GAMMA_UPPER = 0.04
    MIN_D = 0
    ds = DataSimulator(N_READS,READ_LENGTH,GAMMA,PI,D,MU,1,TEMPLATE_SEQUENCES=refs,DMAT=dmat)
    assert len(ds.get_major()) == len(refs[0])
    assert len(ds.get_minor()) == len(refs[0])
    assert len(ds.get_consensus()) == len(refs[0])
    assert (ham(ds.get_consensus(), ds.get_major())) < ham(ds.get_minor(), ds.get_consensus())
    assert (ham(ds.get_major(), ds.get_minor()) <= ham(ds.get_consensus(), ds.get_major())) + ham(ds.get_minor(), ds.get_consensus())
#    print(ds.M)
#    ds.filter(0.90)

    #//*** Preprocess dataset ***//
    # Get allowed sites and allowed bases at each site
    allowed_bases=[]
    allowed_sites=ds.VALID_INDICES
    states_per_site = 2
    for m in ds.M:
        allowed_bases.append(sorted(np.argsort(m)[-states_per_site:]))
        assert len(allowed_bases[-1]) > 0

    # Delete references too close to the consensus (taken as fixed point)
    sys.stderr.write("Removing those too close to fixed point\n")
    fixed_point = ds.get_consensus()
    refs,dmat = del_close_to_fixed_point(fixed_point,MIN_D,refs,dmat)

    #//*** Initialize mixture model ***//
    mms = MixtureModelSampler(ds,ds.get_consensus(),refs=refs,dmat=dmat)

    #//*** Sample ***//
    strings,params,Ls = mms.sample(ds,nits=1000)
    plt.plot(Ls)
    plt.show()

    #//*** Collect results ***//
    params = np.array(params)
    meanpi = np.mean(params[:,0])
    meang0 = np.mean(params[:,1])
    meang1 = np.mean(params[:,2])
    print("mean pi=%f, mean g0=%f, mean g1=%f" % (meanpi, meang0, meang1))
    C = gen_array(strings)
    hams = []
    BURNIN = int(input("Specify Burnin:"))
    for s in strings[BURNIN:]:
        hams.append(ham(s,ds.get_minor()))
    hams = np.array(hams)
    print("mean error:", np.mean(hams))
    assert ham(ds.get_minor(), ds.get_major()) >= D
    assert (ham(ds.get_consensus(), ds.get_major())) < ham(ds.get_minor(), ds.get_consensus())
