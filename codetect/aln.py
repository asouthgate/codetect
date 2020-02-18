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

    def del_inds(self, delinds, new_index_map):
        """ Remove all bases mapping to positions in inds, and shift indices accordingly
            
        Args:
            inds: indices of deleted bases
            shiftmap: mapping old positions to new ones
        """
        # Delete specified positions
        self.base_pos_pairs = [bpp for bpp in self.base_pos_pairs if bpp[0]+self.pos not in delinds]        
        # Remap other positions
        self.base_pos_pairs = [[new_index_map[bpp[0]+self.pos]-self.pos,bpp[1]] for bpp in self.base_pos_pairs]
        # Shift the position back to zero-first base mapping
        mini = self.base_pos_pairs[0][0] 
        self.pos += mini
        self.base_pos_pairs = [[bpp[0]-mini,bpp[1]] for bpp in self.base_pos_pairs]       

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
            
        
    
