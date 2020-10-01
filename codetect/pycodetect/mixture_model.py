import numpy as np
import math      
from pycodetect.utils import *
from pycodetect.log import logger

# TODO: ambiguous bases or gaps (5) are currently just counted as mismatches; throw error if you find them

class MixtureModel():
    def __init__(self, rd, gamma0, gamma1, consensus, pi, initstr):
        # FORMULATE AS A FINITE STATE MACHINE
        self.set_g1(gamma1)
        self.set_g0(gamma0)
        self.set_pi(pi)
        self.st = [c for c in initstr]
        self.llc = LogLikelihoodCalculator(rd, self.st)    
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

    def cal_loglikelihood(self, ds, newis=None, newbs=None, newpi=None,
                                    newg0=None, newg1=None, cache=True):
        # TODO: responsibility should be given to the likelihood calculator?
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
        if newis != None:
            for j,newi in enumerate(newis):
                newst[newi] = newbs[j]

        # TODO: implement logger warnings
        #logger.warning("Updating loglikelihood. Was: %s" % self.llc.L)
        #logger.warning("consensus  : %s" % str(self.consensus))
        #logger.warning("prev string: %s" % str(self.st))
        #logger.warning("new string: %s" % str(newst))
        #logger.warning("new parameters: %s %s %s %s %s" % (i, newb, newpi, newg0, newg1))

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
            if None == newbs == newis:
                assert self.llc.L != None, "State has been wiped and likelihood requested without recalculation!"
                return self.llc.L
            # Case 3.b: new base and old base are the same
            else:
                oldbs = [self.st[i] for i in newis]
                if newbs == oldbs:
                    return self.llc.L                
            # Case 3.c: new base at position i has been proposed; fast recalculation performed
                else:
                    # Case 3.c.1: new base at position i has been proposed but it is the same as the old one; don't recompute
                    ll = self.llc.update_loglikelihood(ds,newst,newis,newbs,oldbs,self.logg1, self.log1mg1, self.logpi, self.log1mpi)
                    for j,newi in enumerate(newis):
                        self.st[newi] = newbs[j]
                    return ll

        # Now perform a full recalculation
        assert newbs == None, "Full recomputation requested but new base also requested. Bug."
        # Case 4.a: only pi changed; partial recalculation required
        if newg0 == newg1 == None and newpi != None:
#            assert self.initialized, "Attempting to calculate likelihood without fully calculating read likelihoods first!"
            res = self.llc.cal_pi_loglikelihood(ds, self.logpi, self.log1mpi)
        else:
            assert not self.initialized, "State is initialized but a full recomputation is requested! Bug."
        # Case 4.b: pi, g0, g1 changes; full recalculation required
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

class LogLikelihoodCalculator():
    def __init__(self,rd,initstr):
        self.condL = np.zeros((2,len(rd.X)))
        self.margL = np.zeros(len(rd.X))    
        self.L = None
        self.nmcache = NmCache(rd,initstr)

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

    def cal_pi_loglikelihood(self, rd, logpi, log1mpi):
        """ Calculate the full log likelihood of all reads with only pi changing.

        Args:
            rd: ReadData object containing read data.
        """
        # TODO: if only pi changes, read condL likelihoods do not need to be recomputed
        for ri,read in enumerate(rd.X):
            self.margL[ri] = logsumexp([self.condL[0,ri] + logpi, self.condL[1,ri] + log1mpi]) * read.count
            assert self.margL[ri] < 0.0, "Full likelihood calculation resulted in improper likelihood (>0):%f" % self.margL[ri]
        self.L = sum(self.margL)
        assert self.L != None
        return self.L

    def cal_full_loglikelihood(self, rd, logg0, log1mg0, logg1, log1mg1, logpi, log1mpi):
        """ Calculate the full log likelihood of all reads.

        Args:
            rd: ReadData object containing read data.
        """
        # TODO: if only pi changes, read condL likelihoods do not need to be recomputed
        for ri,read in enumerate(rd.X):
            self.condL[0,ri] = self.cal_read_loglikelihood(ri,read,0, logg0, log1mg0)
            self.condL[1,ri] = self.cal_read_loglikelihood(ri,read,1, logg1, log1mg1)
            self.margL[ri] = logsumexp([self.condL[0,ri] + logpi, self.condL[1,ri] + log1mpi]) * read.count
            assert self.margL[ri] < 0.0, "Full likelihood calculation resulted in improper likelihood (>0):%f" % self.margL[ri]
        self.L = sum(self.margL)
        assert self.L != None
        return self.L

    def update_loglikelihood(self,rd,newst,newis,newbs,oldbs,logg1,log1mg1,logpi,log1mpi):
        # TODO: separate out somehow.
        """ Update the log likelihood given a new base.

        Args:
            rd: ReadData object.
            newst: specified for debugging purposes
            newi: index of new base.
            newb: new base.
        """
        #logger.warning("\tupdating ll %d %d %d %f %f %f %f" % (newi, newb, oldb, logg1, log1mg1, logpi, log1mpi))
        assert oldbs != newbs, "Loglikelihood update requested but no base was changed! Wasteful or a bug."
        for j, newi in enumerate(newis):
            newb = newbs[j]
            oldb = oldbs[j]
            read_list = rd.pos2reads(newi)
            for ri in read_list: 
                read = rd.X[ri]
                #logger.warning("\tupdating read %d, curr likelihood %f" % (ri, self.margL[ri]))
                #logger.warning("\tread str: %s" % str(read.get_ints()))
                #logger.warning("\tnew  str: %s" % str(newst))
                if read.map[newi] == newb:
                    #logger.warning("\tnew base match; decrease mismatches, increase likelihood")
                    self.condL[1,ri] -= logg1
                    self.condL[1,ri] += log1mg1
                    self.nmcache.update(1,ri,-1)
                    self.margL[ri] = logsumexp([self.condL[0,ri] + logpi, self.condL[1,ri] + log1mpi]) * read.count
                elif read.map[newi] == oldb:
                    #logger.warning("\told base match; increase mismatches, decrease likelihood")
                    self.condL[1,ri] -= log1mg1
                    self.condL[1,ri] += logg1
                    self.nmcache.update(1,ri,1)
                    self.margL[ri] = logsumexp([self.condL[0,ri] + logpi, self.condL[1,ri] + log1mpi]) * read.count
                #logger.warning("\tnew condlikelihood %f" % (self.condL[1,ri] * read.count))
                #logger.warning("\tnew marglikelihood %f" % self.margL[ri])
        self.L = sum(self.margL)
        return self.L

