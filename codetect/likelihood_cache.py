import numpy as np

def logsumexp(logls):
    m = max(logls)
    sumo = 0
    for l in logls:
        sumo += np.exp(l-m)
    return m + np.log(sumo)        

class LogLikelihoodCache():
    def __init__(self, X, N, GL, GAMMA0, GAMMA1, CONSENSUS, PI, initstr):
        self.N = N
        self.GL = GL
        self.wipe_memory()
        self.g1 = GAMMA1
        self.g0 = GAMMA0
        self.pi = PI
        self.initialized = False
        self.consensus = tuple([c for c in CONSENSUS])
        self.nmarr = self.init_nmarr(X,initstr)
    def wipe_memory(self):
        self.initialized = False
#        self.Larr = np.zeros((2,self.N,self.GL))
        self.Lsums0 = np.zeros(self.N)
        self.Lsums1 = np.zeros(self.N)
        self.Lsums = np.zeros(self.N)    
        self.L = None
    def init_nmarr(self,X,altstr):
        nmarr = np.zeros((2,self.N))
        for i,xi in enumerate(X):
            nmarr[0,i] = xi.cal_ham(self.consensus)
            nmarr[1,i] = xi.cal_ham(altstr)
        return nmarr
    def update_nmarr(self,ds,j,bold,bnew):
        assert bold != bnew
        for i in ds.pos2reads(j): 
            xi = ds.X[i]
            if xi.map[j] == bold:
                self.nmarr[1,i] += 1
            elif xi.map[j] == bnew:
                self.nmarr[1,i] -= 1
    def cal_read_loglikelihood(self, ri, read, ci):
        q = self.g0
        if ci == 1:
            q = self.g1
        return np.log(1-q)*(read.get_length()-self.nmarr[ci,ri]) + np.log(q)*(self.nmarr[ci,ri])
    def cal_full_loglikelihood(self, X, st):
        for ri,read in enumerate(X):
            self.Lsums0[ri] = self.cal_read_loglikelihood(ri,read,0)
            self.Lsums1[ri] = self.cal_read_loglikelihood(ri,read,1)
            self.Lsums[ri] = logsumexp([self.Lsums0[ri] + np.log(self.pi), self.Lsums1[ri] + np.log(1-self.pi)]) * X[ri].count
#            print("read=",ri)
#            print("pi=",self.pi)
#            print("hams=",ham(self.consensus,st), ham(read, self.consensus), ham(read,st))
#            print("logs=",self.Lsums0[ri], self.Lsums1[ri], self.Lsums[ri])
#            print("condlike=",np.exp(self.Lsums0[ri])*(1-self.pi), np.exp(self.Lsums1[ri])*self.pi, self.Lsums[ri])
#            print("likelihood=",np.exp(self.Lsums[ri]), np.exp(self.Lsums0[ri])*(1-self.pi) + np.exp(self.Lsums1[ri])*self.pi)
        self.initialized = True
        self.L = sum(self.Lsums)
        return self.L
    def update_loglikelihood(self,X,i,b):
        sumo = 0
#        assert False
        for ri, read in enumerate(X):
            #** TODO: likelihood should not be recomputed for both here unless gamma has changed
            # 2X slower to do so
            if i in read.map:
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
    def cal_loglikelihood(self,ds,st,i=None,bnew=None,bold=None,newpi=None,newg0=None,newg1=None):
        if newg1 != None:
            self.set_g1(newg1)
        if newg0 != None:
            self.set_g0(newg0)
        if newpi != None:
            self.set_pi(newpi)
        if bnew != None:
            self.update_nmarr(ds,i,bold,bnew)
        self.initialized=False
        if not self.initialized:
            return self.cal_full_loglikelihood(ds.X,st)
        else:
            return self.update_loglikelihood(ds.X,i,b)

