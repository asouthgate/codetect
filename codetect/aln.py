import numpy as np

class ReadAln():
    """Read alignment class.

    Attributes:
        name: identifier for a given read.

    """
    def __init__(self,name):
        self.name = name
        self.pos = None
        self.base_pos_pairs = []
        self.string = ""
        self.nm = False
        self.count = 1
        self.z = None

    def __repr__(self):
        s = ""
        prevpos = 0
        for pos,base in self.base_pos_pairs:
            diff = pos-prevpos
            if diff > 1:
                s += "-"*(diff-1)
            s += self.i2c(base)
            prevpos = pos
        return s

    def get_aln(self):
        return [[p[0]+self.pos,p[1]] for p in self.base_pos_pairs]

    def i2c(self,i):
        return list("ACGT")[i]

    def c2i(self,c):
        return {"A":0, "C":1, "G":2, "T":3}[c]

    def del_inds(self,inds):
        """ Remove all bases mapping to positions in inds, and shift indices accordingly

        Args:
            inds: sorted positions to delete. """
#        print(self.get_aln())
        print("***DELETING***:")
        print("\t delinds ", inds)
        print("\t start   ", self.get_aln())
        print("\t startpos", self.pos)
        print("\t startbpp", self.base_pos_pairs)

        dmini = inds[0]
        dmaxi = inds[-1]
        rmini = self.pos
        rmaxi = self.base_pos_pairs[-1][0] + self.pos
        if dmini > rmaxi:
            # the deletion indices are after the last base, do nothing
            print("\t all indices beyond last base, do nothing")
            return True
        if dmaxi < rmini:
            # all deletion indices are before the first base; do nothing
            print("\t all indices before first base, shift pos")
            self.pos -= len(inds)
            return True

        print("\t dels before and after")
        # first delete any we wish to remove
        dels_in_read = [j for j in inds if rmini <= j <= rmaxi]

        if len(dels_in_read) == 0:
            print("\t dels before and after only, shift pos")
            self.pos -= len(dels_in_read)
            return True

        print("\t remove offending positions")
        dels_in_read_set = set(dels_in_read)   
        self.base_pos_pairs = [pb for pb in self.base_pos_pairs if pb[0]+self.pos not in dels_in_read_set]
        print("\t newaln:", self.get_aln())
        print("\t newpos:", self.pos)
        print("\t newbpp:", self.base_pos_pairs)

        print("\t build shift array")
        currdel = dels_in_read.pop(0)
        shift_arr = [0 for i in range(len(self.base_pos_pairs))]
        for ri, pb in enumerate(self.base_pos_pairs):
            p = pb[0]
            shift_arr[ri] = shift_arr[ri-1]
            if currdel < p+self.pos:
                shift_arr[ri] += 1
                if len(dels_in_read) == 0:
                    for rj in range(ri,len(shift_arr)):
                        shift_arr[rj] = shift_arr[ri]
                    break
                else:
                    currdel = dels_in_read.pop(0)
#                    print(currdel)

        print("\t shiftarr:", shift_arr)
        for ai in range(len(self.base_pos_pairs)):
            pb = self.base_pos_pairs[ai]
            shift = shift_arr[ai]
            self.base_pos_pairs[ai][0] -= shift
        print("\t shiftbpp:", self.base_pos_pairs)

        # shift for any before the start
        self.pos -= len([j for j in inds if j < self.pos])
        if self.base_pos_pairs[0][0] > 0:
            minus = self.base_pos_pairs[0][0]
            self.pos += minus
            for i in range(len(self.base_pos_pairs)):
                self.base_pos_pairs[i][0] -= minus


            

    def append_mapped_base(self, pos, c):
        """ Add a base c at pos to the end of the alignment.

        Args:
            pos: position in reference.
            c: base that mapps at position pos (encoded as {0,1,2,3}).
        """
        assert c in [0,1,2,3]
        if self.pos == None:
            self.pos = pos
        self.base_pos_pairs.append([pos-self.pos,c])
        last_pos = self.base_pos_pairs[-1][0]
        nins = pos-self.pos-last_pos
        assert self.base_pos_pairs[0][0] == 0, (self.base_pos_pairs, pos, self.pos)

    def calc_nm(self, consensus):
        """ Calculate the number of mismatches to the reference. """
        self.nm = len([pos for pos,c in self.base_pos_pairs if consensus[pos] != c])
            
    def Pmajor(self, gamma, consensus):
        """ Calculate Pr aln given that it belongs to the major group.
    
        Args:
            gamma: probability of a mismatch through mutation and error.
            consensus: consensus sequence that generates the population.
        """
        return (gamma ** (self.nm)) * ((1-gamma) ** (len(self.base_pos_pairs) - self.nm))

    def Pminor(self, M):
        """ Calculate Pr aln given that it belongs to minor group.

        Args:
            M: Lx4 categorical marginal distributions (sum(M[i]) = 1)
        """
        sumo = 0
        for pos,c in self.base_pos_pairs():
            sumo += np.log(M[c])
        return sumo

if __name__ == "__main__":
    import random
    def test_index_deletions():
        """ TODO: Test if index deletion code is working """
        seq = np.array([random.choice("ACGT") for i in range(21)])
        for nread in range(10):
            baseinds = sorted(list(set([random.randint(0,20) for i in range(10)])))
            print("".join(seq))
            a = ReadAln(nread)
            for bi in baseinds:
                a.append_mapped_base(bi,a.c2i(seq[bi]))
            print(" "*a.pos + str(a))
            delinds = sorted(list(set([random.randint(0,20) for i in range(5)])))
            print("PLEASE DELETE",  delinds)
            nodelinds = [i for i in range(len(seq)) if i not in delinds]
            seq2 = seq[nodelinds]
            a.del_inds(delinds)
            print("".join(seq2))
            print(" "*a.pos + str(a))
            print()

    test_index_deletions()
            
        
    
