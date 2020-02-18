class ReadAln():
    """Read alignment class.

    Attributes:
        name: identifier for a given read.
        pos: left-most starting position of the alignment.

    """
    def __init__(self,name):
        self.name = name
        self.base_pos_pairs = []
        self.string = ""
        self.nm = False

    def __repr__(self):
        return self.string

    def append_mapped_base(self, pos, c):
        """ Add a base c at pos to the end of the alignment.

        Args:
            pos: position in reference.
            c: base that mapps at position pos (encoded as {0,1,2,3}).
        """
        assert c in [0,1,2,3]
        self.base_pos_pairs.append((pos,c))
        last_pos = self.base_pos_pairs[-1][0]
        nins = pos-last_pos
        self.string += "-"*nins + c

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

