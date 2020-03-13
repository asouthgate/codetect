import numpy as np

def logsumexp(logls):
    m = max(logls)
    sumo = 0
    for l in logls:
        sumo += np.exp(l-m)
    return m + np.log(sumo)        

class LogLikelihoodCache():
    def __init__(self, ds, N, GL, GAMMA0, GAMMA1, CONSENSUS, PI, initstr):
        self.N = N
        self.GL = GL
        self.wipe_memory()
        self.g1 = GAMMA1
        self.g0 = GAMMA0
        self.pi = PI
        self.st = initstr
        self.initialized = False
        self.consensus = tuple([c for c in CONSENSUS])
        self.nmarr = None
        self.nmarrFlag = False
        self.init_nmarr(ds.X,initstr)
        self.cal_full_loglikelihood(ds)
    def wipe_memory(self):
        self.initialized = False
        self.Lsums0 = np.zeros(self.N)
        self.Lsums1 = np.zeros(self.N)
        self.Lsums = np.zeros(self.N)    
        self.L = None
    def init_nmarr(self,X,altstr):
        nmarr = np.zeros((2,self.N))
        for i,xi in enumerate(X):
            nmarr[0,i] = xi.cal_ham(self.consensus)
            nmarr[1,i] = xi.cal_ham(altstr)
            assert nmarr[0,i] >= 0
            assert nmarr[1,i] >= 0
        self.nmarr = nmarr
        self.nmarrFlag = True
    def update_nmarr(self,ds,j,bnew):
        assert bnew < 4
        bold = self.st[j]
        self.nmarrFlag = True
        if bold == bnew:
            return None
        assert bold != bnew
        for i in ds.V_INDEX[j][bold]: 
            self.nmarr[1,i] += 1
        for i in ds.V_INDEX[j][bnew]:
            self.nmarr[1,i] -= 1
            assert self.nmarr[1,i] >= 0, (bold, bnew)
        return None
    def cal_read_loglikelihood(self, ri, read, ci):
        q = self.g0
        if ci == 1:
            q = self.g1
        assert self.nmarr[ci,ri] >= 0, self.nmarr[ci,ri]
        return np.log(1-q)*(read.get_length()-self.nmarr[ci,ri]) + np.log(q)*(self.nmarr[ci,ri])
    def cal_full_loglikelihood(self, ds):
        assert self.nmarrFlag
        for ri,read in enumerate(ds.X):
            self.Lsums0[ri] = self.cal_read_loglikelihood(ri,read,0)
            self.Lsums1[ri] = self.cal_read_loglikelihood(ri,read,1)
            self.Lsums[ri] = logsumexp([self.Lsums0[ri] + np.log(self.pi), self.Lsums1[ri] + np.log(1-self.pi)]) * read.count
        self.initialized = True
        self.L = sum(self.Lsums)
        assert self.L != None
        return self.L
    def update_loglikelihood(self,ds,i,b):
#        assert False
#        assert b != self.st[i]
        assert self.nmarrFlag
        for ri in ds.pos2reads(i):
            read = ds.X[ri]
            #** TODO: likelihood should not be recomputed for both here unless gamma has changed
            # 2X slower to do so
            if read.map[i] == self.st[i]:
                # TODO: compute log gamma once per round; dont need to log loads of times. slow.
                self.Lsums0[ri] -= np.log(1-self.g0)
                self.Lsums0[ri] += np.log(self.g0)
                self.Lsums[ri] = logsumexp([self.Lsums0[ri] + np.log(self.pi), self.Lsums1[ri] + np.log(1-self.pi)]) * read.count
            elif read.map[i] == b:
                self.Lsums0[ri] -= np.log(self.g0)
                self.Lsums0[ri] += np.log(1-self.g0) 
                self.Lsums[ri] = logsumexp([self.Lsums0[ri] + np.log(self.pi), self.Lsums1[ri] + np.log(1-self.pi)]) * read.count
        self.L = sum(self.Lsums)
        self.st[i] = b
        return self.L
    def set_newb(self,ds,b,i):
        assert b < 4
        if self.st[i] != b:
            self.nmarrFlag = False
            self.st[i] = b
            self.update_nmarr(ds,i,b)
    def set_g1(self,newg1):
        self.g1 = newg1
        self.wipe_memory()
    def set_g0(self,newg0):
        self.g0 = newg0
        self.wipe_memory()
    def set_pi(self,newpi):
        self.pi = newpi
        self.wipe_memory()
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

        #//*** Cases ***//
        # Case 1: no caching, full recalculation required
        if not cache:
            self.wipe_memory()

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
                assert self.L != None, "State has been wiped and likelihood requested without recalculation!"
                return self.L
            # Case 3.b: new base at position i has been proposed; fast recalculation performed
            else:
                # Case 3.b.1: new base at position i has been proposed but it is the same as the old one; don't recompute
                if newb == self.st[i]:
                    return self.L
                self.set_newb(ds,newb,i)
#                print("UPDATING FAST LIKELIHOOD")
                return self.update_loglikelihood(ds,i,newb)

        # Now perform a full recalculation
        assert not self.initialized, "State is initialized but a full recomputation is requested! Bug."
        assert newb == None
#        print("CALCULATING FULL LIKELIHOOD")
        res = self.cal_full_loglikelihood(ds)
        assert self.L != None
        return res
