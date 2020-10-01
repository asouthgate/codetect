import numpy as np
from pycodetect.utils import rev_comp
import sys

class ReadAln():
    """Read alignment class.

    Attributes:
        name: identifier for a given read.

    """
    def __init__(self,name):
        self.name = str(name)
        self.pos = None
#        self.base_pos_pairs = []
        self.map = {}
        self.unmasked_map = {}
        self.string = ""
        self.nm_major = None
        self.count = 1
        self.z = None
        self.aln = None
        #TODO: remove somehow; this is for only the em.py
#        self.prev_logPminor2 = None
        self.nm_minor = None

    def __repr__(self):
        s = ""
        prevpos = 0
        for pos, base in self.get_aln():
            diff = pos-prevpos
            if diff > 1:
                s += "-"*(diff-1)
            s += self.i2c(base)
            prevpos = pos
        return "ReadAln@pos=%d@count=%d@str=%s" % (self.get_aln()[0][0],self.count,s)

    def get_aln_segments(self):
        s = ""
        prevpos = 0
        for pos, base in self.get_aln():
            diff = pos-prevpos
            if diff > 1:
                s += "X"
            s += self.i2c(base)
            prevpos = pos
        segs = [l for l in s.split("X") if l != ""]
        return segs
 
    def get_length(self):
        return len(self.map)

    def get_string(self):
        return "".join([self.i2c(b) for p,b in self.get_aln()])

    def get_ints(self):
        return [b for p,b in self.get_aln()]

    def get_fq_entry_pair(self):
        raise NotImplementedError
        firstseq, secondseq = self.get_aln_segments()                
        first = "@"+str(self.name)+"/1\n"+firstseq+"\n+\n"+"I"*len(firstseq)+"\n"
        second = "@"+str(self.name)+"/2\n"+rev_comp(secondseq)+"\n+\n"+"I"*len(secondseq)+"\n"
        return first, second

    def get_fq_entry_single(self):
        firstseq = self.get_aln_segments()[0]
        first = "@"+str(self.name)+"/1\n"+firstseq+"\n+\n"+"I"*len(firstseq)+"\n"
        return first

    def cal_ham(self,S): 
        assert False, "Function depreciated; to be moved."
        # DEPRECIATED, SOMEHOW
        h = 0
        for p,b in self.get_aln():
            if b != S[p]:
                h += 1
        return h

    def get_aln(self):
        assert False, "Function to be renamed"
        if self.aln != None:
            return self.aln
        else:
            aln = [(p,b) for p,b in sorted(self.map.items(), key = lambda x:x[0])]
            return aln
 
    def i2c(self,i):
        # TODO: replace with util
        assert False, "Duplication; should be common util"
        return list("ACGT-")[i]

    def c2i(self,c):
        # TODO: replace with util
        assert False, "Duplication; should be common util"
        return {"A":0, "C":1, "G":2, "T":3}[c]

    def del_inds(self, delinds):
        # TODO: delete totally
        """ Remove all bases mapping to positions in inds, and shift indices accordingly
            
        Args:
            inds: indices of deleted bases
        """
        assert False, "Depreciated. we don't delete indices currently"
        for di in delinds:
            if di in self.map:
                del self.map[di]

        if len(self.map) == 0:
            self.pos = None
            return False

        self.nm_major = None
        return True

    def append_mapped_base(self, pos, c):
        """ Add a base c at pos to the end of the alignment.

        Args:
            pos: position in reference.
            c: base that maps at position pos (encoded as {0,1,2,3}).
        """
        if c is not None:
            self.map[pos] = c

    def calc_nm_major(self, consensus):
        assert False, "depreciated"
        # TODO: MOVE TO A CALCULATOR CACHE
        # TODO: THIS IS SLOW. SHOULD BE CACHED 
        """ Calculate the number of mismatches to the reference. """
        if self.nm_major != None:
            assert False, "Refusing to recalculate constant nm"
        self.nm_major = 0
        for bp,c in self.get_aln():
#            if self.i2c(c) != consensus[p+self.pos]:
            if c != consensus[bp]:
                self.nm_major += 1            
        return self.nm_major

    def calc_nm_minor(self, minorcon):
        assert False, "depreciated"
        # TODO: MOVE TO A CALCULATOR CACHE
        # TODO: THIS IS SLOW. SHOULD BE CACHED 
        """ Calculate the number of mismatches to the reference. """
        if self.nm_minor != None:
            assert False, "Refusing to recalculate constant nm"
        self.nm_minor = 0
        for bp,c in self.get_aln():
#            if self.i2c(c) != consensus[p+self.pos]:
            if c != consensus[bp]:
                self.nm_minor += 1 
        return self.nm_minor
            
    def logPmajor(self, gamma):
        assert False, "depreciated"
        # TODO: MOVE TO A CALCULATOR
        """ Calculate Pr aln given that it belongs to the major group.
    
        Args:
            gamma: probability of a mismatch through mutation and error.
            consensus: consensus sequence that generates the population.
        """
#        print(len(self.base_pos_pairs))
        logp = self.nm_major * np.log(gamma) + (len(self.map)-self.nm_major)*np.log(1-gamma)
        return logp

    def Pminor(self, vt):
        assert False, "depreciated"
        # TODO: MOVE TO A CALCULATOR
        """ Calculate Pr aln given that it belongs to minor group.

        Args:
            M: Lx4 categorical marginal distributions (sum(M[i]) = 1)
        """
        logp = 0
        assert False, "TODO:FIX, SLOW"
        for pos,c in self.get_aln():
            logp += np.log(vt[pos,c])
        res = np.exp(sumo)
        if np.isnan(res) and sumo < -100:
            res = 0
        return res

    def logPminor2(self, mu, st, prevst=None, changed_inds=None):
        assert False, "depreciated"
        # TODO: MOVE TO A CALCULATOR
        """ Calculate Pr aln given that it belongs to minor group.

        Args:
            M: Lx4 categorical marginal distributions (sum(M[i]) = 1)
        """
        if changed_inds is not None:
#            sys.stderr.write("Not None ci\n")
#            print(changed_inds)
#            assert self.prev_logPminor2 is not None
#            sumo = self.prev_logPminor2
            for bp in changed_inds:
    #            for bp,c in self.get_aln():
                if bp in self.map:
                    c = self.map[bp]
                    if c == st[bp]:
                        assert c != prevst[bp]
                        # previously it didnt match, now it does
#                        sumo -= np.log(mu)
#                        sumo += np.log(1-mu)
                        self.nm_minor -= 1
                    else:
                        assert c == prevst[bp]
                        # previously it did match, now it does not
#                        sumo -= np.log(1-mu)
#                        sumo += np.log(mu)
                        self.nm_minor += 1
        else:
            self.nm_minor = 0
            for bp,c in self.get_aln():
                if c == st[bp]:
                    pass
                else:
                    self.nm_minor += 1
        matches = len(self.map)-self.nm_minor
        self.prev_logPminor2 = np.log(mu)*self.nm_minor + np.log(1-mu)*matches
        return self.prev_logPminor2

if __name__ == "__main__":
    import random
    def gen_index_remap(L,delinds):
        shift = [0 for i in range(L)]
        for si in range(len(shift)):
            shift[si] = shift[si-1]
            if si-1 in delinds:
                shift[si] += 1
        return {si:si-s for si,s in enumerate(shift)}

    def test_index_deletions():
        """ TODO: Test if index deletion code is working """
        L = 21
        seq = np.array([random.choice("ACGT") for i in range(L)])
        for nread in range(10):
            baseinds = sorted(list(set([random.randint(0,20) for i in range(10)])))
            print("".join(seq))
            a = ReadAln(nread)
            for bi in baseinds:
                a.append_mapped_base(bi,a.c2i(seq[bi]))
            print(" "*a.pos + str(a))
            delinds = sorted(list(set([random.randint(0,20) for i in range(5)])))
            shiftmap = gen_index_remap(L,delinds)
            print("PLEASE DELETE",  delinds)
            nodelinds = [i for i in range(len(seq)) if i not in delinds]
            seq2 = seq[nodelinds]
            a.del_inds(delinds,shiftmap)
            print("".join(seq2))
            print(" "*a.pos + str(a))
            print()

    test_index_deletions()
            
        
    
