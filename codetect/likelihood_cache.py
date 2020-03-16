import numpy as np
import math      
from log import logger

def logsumexp(logls):
    m = max(logls)
    sumo = 0
    for l in logls:
        sumo += np.exp(l-m)
    return m + np.log(sumo) 

def approx(a,b,eps=0.000001):
    if math.fabs(a-b) < eps:
        return True
    return False

class MixtureModel():
    def __init__(self, rd, gamma0, gamma1, consensus, pi, initstr):
        self.set_g1(gamma1)
        self.set_g0(gamma0)
        self.set_pi(pi)
        self.st = initstr
        self.llc = LogLikelihoodCalculator(rd,initstr)    
        self.consensus = tuple([c for c in consensus])
        self.initialized = False
        self.cal_loglikelihood(rd)
    def set_g1(self,newg1):
        self.g1 = newg1
        self.logg1 = math.log(self.g1)
        self.log1mg1 = math.log(1-self.g1)
        self.initialized = False
    def set_g0(self,newg0):
        self.g0 = newg0
        self.logg0 = math.log(self.g0)
        self.log1mg0 = math.log(1-self.g0)
        self.initialized = False
    def set_pi(self,newpi):
        self.pi = newpi
        self.logpi = math.log(self.pi)
        self.log1mpi = math.log(1-self.pi)
        self.initialized = False
    def test_state(self,rd):
        self.llc.test_caches(rd,rd.get_consensus(), self.st) 
        currL = self.llc.L
        assert self.cal_loglikelihood(cache=False) == currL
    def cal_loglikelihood(self,ds,i=None,newb=None,newpi=None,newg0=None,newg1=None,cache=True):
        """ Calculate log likelihood. 

        Interface to likelihood computation machinery. Takes a number of optional arguments,
        and performs computation based on several cases.

        Args:
            rd: ReadData object

        Optional Args:
            i: position for new base
            newb: new base at position i
            cache: if false, recomputes the entire likelihood, otherwise makes use of speedups
    
        Returns:
            Log likelihood of data given parameters.
        """
        # Logging
        newst = [c for c in self.st]
        if i  != None:
            newst[i] = newb
        logger.warning("Updating loglikelihood. Was: %s" % self.llc.L)
        logger.warning("consensus  : %s" % str(self.consensus))
        logger.warning("prev string: %s" % str(self.st))
        logger.warning("new string: %s" % str(newst))
        logger.warning("new parameters: %s %s %s %s %s" % (i, newb, newpi, newg0, newg1))
        self.llc.nmcache.test_nmarr(ds,ds.get_consensus(),self.st)

        #//*** Cases ***//
        # Case 1: no caching, full recalculation required
        if not cache:
            self.initialized = False

        # Case 2: new states have been proposed; any trigger full recalculation. Set state.
        if newg1 != None:
            self.set_g1(newg1)
        if newg0 != None:
            self.set_g0(newg0)
        if newpi != None:
            self.set_pi(newpi)

        # Case 3: no new parameters have been proposed that trigger full recalculation
        if self.initialized:
            # Case 3.a: no new parameters have been proposed at all; just return 
            if None == newb == i:
                assert self.llc.L != None, "State has been wiped and likelihood requested without recalculation!"
                return self.llc.L
            # Case 3.b: new base and old base are the same
            else:
                oldb = self.st[i]
                if newb == oldb:
                    return self.llc.L                
            # Case 3.c: new base at position i has been proposed; fast recalculation performed
                else:
                    # Case 3.c.1: new base at position i has been proposed but it is the same as the old one; don't recompute
                    ll = self.llc.update_loglikelihood(ds,newst,i,newb,oldb,self.logg1, self.log1mg1, self.logpi, self.log1mpi)
                    self.st[i] = newb
                    return ll

        # Now perform a full recalculation
        assert not self.initialized, "State is initialized but a full recomputation is requested! Bug."
        assert newb == None, "Full recomputation requested but new base also requested. Bug."
        res = self.llc.cal_full_loglikelihood(ds, self.logg0, self.log1mg0, self.logg1, self.log1mg1, self.logpi, self.log1mpi)
        assert self.llc.L != None
        self.initialized = True
        return res


class NmCache():
    def __init__(self, ds, initstr):
        self.nmarr = np.zeros((2,len(ds.X)))
        self.init_nmarr(ds,initstr)
    def __getitem__(self,args):
        return self.nmarr[args[0],args[1]]
    def init_nmarr(self,ds,altstr):
        """ Calculate the number of mismatches between reads and a string. """
        nmarr = np.zeros((2,len(ds.X)))
        for i,xi in enumerate(ds.X):
            nmarr[0,i] = xi.cal_ham(ds.get_consensus())
            nmarr[1,i] = xi.cal_ham(altstr)
            assert nmarr[0,i] >= 0
            assert nmarr[1,i] >= 0
        self.nmarr = nmarr
    def update(self,ci,ri,inc):
        """ Update the number of mismatches between reads and a string.

        Args:
            ri: ReadData object
            inc: +1 or -1 
         """
        assert inc in [-1,1]
        self.nmarr[ci,ri] += inc
    def test_read_nm(self,ri, read,consensus,st):
        h1 = read.cal_ham(consensus)
        h2 = read.cal_ham(st)
        assert self.nmarr[0,ri] == h1, ((self.nmarr[0,ri], h1), "Bad nmarr value for read %d." % ri)
        assert self.nmarr[1,ri] == h2, ((self.nmarr[1,ri], h2), "Bad nmarr value for read %d." % ri)
    def test_nmarr(self,rd,consensus,st):
        """ Test that the number of mismatches in nmarr is correct. 

        Args:
            rd: ReadData object
        """
        for ri,ra in enumerate(rd.X):
            self.test_read_nm(ri, ra,consensus,st)

class LogLikelihoodCalculator():
    def __init__(self,rd,initstr):
        self.condL = np.zeros((2,len(rd.X)))
        self.margL = np.zeros(len(rd.X))    
        self.L = None
        self.nmcache = NmCache(rd,initstr)
    def test_caches(self,rd,consensus,st):
        nmcache.test_nmarr(rd,consensus,st)
        for ri, read in enumerate(rd.X):
            assert self.condL[0,ri] == self.cal_read_loglikelihood(ri,read,0, logg0, log1mg0), "Bad conditional L value."
            assert self.condL[1,ri] == self.cal_read_loglikelihood(ri,read,1, logg1, log1mg1), "Bad conditional L value."
    def cal_read_loglikelihood(self, ri, read, ci, logg, log1mg):
        """ Calculate the loglikelihood of a read given a cluster.

        Args:
            ri: index of the read
            read: the ReadAln object itself
            ci: the cluster index for which to calculate the likelihood
            logg: log(g), where g is the gamma parameter for ci
            logq: log(1-g), where g is the gamma parameter for ci
        """
        return log1mg*(read.get_length()-self.nmcache[ci,ri]) + logg*(self.nmcache[ci,ri])
    def cal_full_loglikelihood(self, rd, logg0, log1mg0, logg1, log1mg1, logpi, log1mpi):
        """ Calculate the full log likelihood of all reads.

        Args:
            rd: ReadData object containing read data.
        """
        for ri,read in enumerate(rd.X):
            self.condL[0,ri] = self.cal_read_loglikelihood(ri,read,0, logg0, log1mg0)
            self.condL[1,ri] = self.cal_read_loglikelihood(ri,read,1, logg1, log1mg1)
            self.margL[ri] = logsumexp([self.condL[0,ri] + logpi, self.condL[1,ri] + log1mpi]) * read.count
            assert self.margL[ri] < 0.0, "Full likelihood calculation resulted in improper likelihood (>0):%f" % self.margL[ri]
        self.L = sum(self.margL)
        assert self.L != None
        return self.L
    def update_loglikelihood(self,rd,newst,newi,newb,oldb,logg1,log1mg1,logpi,log1mpi):
        """ Update the log likelihood given a new base.

        Args:
            rd: ReadData object.
            newst: specified for debugging purposes
            newi: index of new base.
            newb: new base.
        """
        logger.warning("\tupdating ll %d %d %d %f %f %f %f" % (newi, newb, oldb, logg1, log1mg1, logpi, log1mpi))
        assert oldb != newb, "Loglikelihood update requested but no base was changed! Wasteful or a bug."
        read_list = rd.pos2reads(newi)
        for ri in read_list: 
            read = rd.X[ri]
            logger.warning("\tupdating read %d, curr likelihood %f" % (ri, self.margL[ri]))
            logger.warning("\tread str: %s" % str(read.get_ints()))
            logger.warning("\tnew  str: %s" % str(newst))
            if read.map[newi] == newb:
                logger.warning("\tnew base match; decrease mismatches, increase likelihood")
                # TODO: compute log gamma once per round; dont need to log loads of times. slow.
                self.condL[1,ri] -= logg1
                self.condL[1,ri] += log1mg1
                self.nmcache.update(1,ri,-1)
                self.margL[ri] = logsumexp([self.condL[0,ri] + logpi, self.condL[1,ri] + log1mpi]) * read.count
            elif read.map[newi] == oldb:
                logger.warning("\told base match; increase mismatches, decrease likelihood")
                self.condL[1,ri] -= log1mg1
                self.condL[1,ri] += logg1
                self.nmcache.update(1,ri,1)
                self.margL[ri] = logsumexp([self.condL[0,ri] + logpi, self.condL[1,ri] + log1mpi]) * read.count
            # TODO: remove this slow check
            logger.warning("\tnew condlikelihood %f" % (self.condL[1,ri] * read.count))
            logger.warning("\tnew marglikelihood %f" % self.margL[ri])
            self.nmcache.test_read_nm(ri,read,rd.get_consensus(),newst)
            slowcheck = self.cal_read_loglikelihood(ri,read,1,logg1, log1mg1) * read.count
            assert approx(self.condL[1,ri]*read.count,slowcheck), "Likelihood computed incorrectly; %f should be %f." % (self.condL[1,ri] * read.count, slowcheck)
            assert self.condL[1,ri] <= 0, "Bad value %f, read likelihood should not exceed 0" % self.condL[1,ri]
            assert self.margL[ri] <= 0, "Bad value %f, read likelihood should not exceed 0" % self.margL[ri]
        self.L = sum(self.margL)
        return self.L

