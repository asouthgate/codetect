from pycodetect.utils import i2c

class ReadAln():
    """Read alignment class.

    Attributes:
        name: identifier for a given read.

    """
    def __init__(self, name):
        self.name = str(name)
        self.pos = None
        self.map = {}
        self.unmasked_map = {}
        self.string = ""
        self.count = 1
        self.aln = None

    def __repr__(self):
        s = ""
        prevpos = 0
        for pos, base in self.get_aln_tuples():
            diff = pos-prevpos
            if diff > 1:
                s += "-"*(diff-1)
            s += i2c(base)
            prevpos = pos
        return "ReadAln@pos=%d@count=%d@str=%s" % (self.get_aln_tuples()[0][0],self.count,s)

    def cal_ham(self, S): 
        """ Calculate the hamming distance to S. """
        h = 0
        for p, b in self.get_aln_tuples():
            if b != S[p]:
                h += 1
        return h

    def get_aln_segments(self):
        s = ""
        prevpos = 0
        for pos, base in self.get_aln_tuples():
            diff = pos-prevpos
            if diff > 1:
                s += "X"
            s += i2c(base)
            prevpos = pos
        segs = [l for l in s.split("X") if l != ""]
        return segs
 
    def get_length(self):
        return len(self.map)

    def get_string(self):
        return "".join([i2c(b) for p,b in self.get_aln_tuples()])

    def get_ints(self):
        return [b for p,b in self.get_aln_tuples()]

    def get_fq_entry_single(self):
        firstseq = self.get_aln_segments()[0]
        first = "@"+str(self.name)+"/1\n"+firstseq+"\n+\n"+"I"*len(firstseq)+"\n"
        return first

    def get_aln_tuples(self):
        if self.aln != None:
            return self.aln
        else:
            aln = [(p,b) for p,b in sorted(self.map.items(), key = lambda x:x[0])]
            return aln
 
    def del_inds(self, delinds):
        # TODO: delete totally? When is this needed?
        raise NotImplementedError
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

        return True

    def append_mapped_base(self, pos, c):
        """ Add a base c at pos to the end of the alignment.

        Args:
            pos: position in reference.
            c: base that maps at position pos (encoded as {0,1,2,3}).
        """
        assert c != 4, "mapped bases should only be ACGT (0123)"
        if c is not None:
            self.map[pos] = c
