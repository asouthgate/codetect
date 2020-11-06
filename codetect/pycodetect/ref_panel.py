from pycodetect.utils import str_c2i, ham_nogaps_str, str_only_ACGT, ham_nogaps, str_only_ACGTgap
from Bio.SeqIO import FastaIO 
import numpy as np
import random
import sys

class RefPanel():
    def __init__(self, cons, ref_msa, s0_h, min_d):
        assert type(cons[0]) == np.int64, type(cons[0])
        assert 4 not in cons
        self._s0_h = s0_h
        self._s0_s, self._ref_panel = self.preprocess_msa_refs(ref_msa, s0_h, cons, min_d=min_d)
        self._ref_diff_inds = []
        self._min_d = min_d

        # All references should be length of consensus.
        for s, ref in self._ref_panel: 
            assert(len(ref) == len(cons))

        # Consensus should never have a 4 in it; convert any 4s to cons seq
        for ri in range(len(self._ref_panel)):
            h,s = self._ref_panel[ri]
            s = list(s)
            for ci, c in enumerate(s):
                if s[ci] == 4: s[ci] = cons[ci]
            self._ref_panel[ri] = (h,tuple(s))
        sys.stderr.write("WARNING: SLOW TEST\n")
        for h,r in self._ref_panel: assert 4 not in r
        self.cal_diff_inds(cons)
        assert len(self._ref_panel) > 0
        assert len(self._ref_panel) == len(self._ref_diff_inds), (len(self._ref_panel), len(self._ref_diff_inds))
        for h,s in self._ref_panel: assert len(s) == len(cons)

    def get_random_ref(self):
        ri = random.randint(0,len(self._ref_panel)-1)
        rh, rseq = self._ref_panel[ri]
        return ri, rh, rseq

    def get_ref_maximizing_second(self, M):
        secondbest = []
        for pos, v in enumerate(M):
            sortind = np.argsort(v)
            second = sortind[2]
            secondbest.append(second)
        hams = [ham_nogaps(secondbest, s) for h, s in self._ref_panel]
        chosen_ri = np.argmin(hams)
        chosen = self._ref_panel[chosen_ri]
        return chosen_ri, chosen[0], chosen[1]

    def cal_diff_inds(self, cons):
        """ Calculate an index for each record that gives the
            bases that are different to the consensus.

        Args:
            cons: consensus sequence to calculate diffs relative to.
        """
        for ri, rec in enumerate(self._ref_panel):
            cdiff = [ci for ci,c in enumerate(rec[1]) if c != cons[ci]]
            self._ref_diff_inds.append(cdiff)

    def get_diff_inds(self, ri):
        return self._ref_diff_inds[ri]

    def size(self):
        return len(self._ref_diff_inds)
    
    def get_ref(self, ri):
        return self._ref_panel[ri]

    def get_diff_inds_pair(self, ri, rj):
        """ Get the diff inds between ref ri and rj. """
        ridiff = set(self._ref_diff_inds[ri])
        rjdiff = set(self._ref_diff_inds[rj])
        union = ridiff.union(rjdiff)
        inter = ridiff.intersection(rjdiff)
        riseq = self._ref_panel[ri][1]
        rjseq = self._ref_panel[rj][1]
        resdiffs = []
        # Iterate through all dis
        for di in union:
            # If both have di
            if di in inter:
                if riseq[di] != rjseq[di]: resdiffs.append(di)
            else:
                # Otherwise one is equal to ref, one is not
                assert "N" not in [riseq[di], rjseq[di]]
                resdiffs.append(di)
        sys.stderr.write("WARNING: SLOW CHECK! DISABLE FOR NON DEBUGGING MODE\n")
        rds = set(resdiffs)
        for ci in range(len(riseq)):
            if riseq[ci] != rjseq[ci]:
                assert ci in rds
            else:
                assert ci not in rds
        return resdiffs

    def preprocess_msa_refs(self, ref_fname, s0_h, cons, min_d=None):
        """ Preprocess MSA references by cutting out any
            insertions relative to s0.

        Args:
            ref_fname: MSA fasta file name.
            s0_h: name of header for s0.
            min_d: minimum distance to s0 refs can be.

        Return:
            A list of references aligned to s0.
        """
        # Parse refs
        with open(ref_fname) as f:
            recs = [(h,str_only_ACGTgap(s.upper())) for h,s in FastaIO.SimpleFastaParser(f)]

        # Pull out s0 msa sequence
        s0_msa_seq = ""
        for h,s in recs: 
            if h == s0_h:
                s0_msa_seq = s
                break

        # Cut out any gaps from s0 to get s0_seq
        s0_seq = s0_msa_seq.replace("-","")
        assert len(s0_msa_seq) > 0, "Reference %s not found in msa" % s0_h

        assert len(s0_seq) == len(cons), (len(s0_seq), len(cons))

        # Pull out indices that are not gaps in s0
        nongapinds = [j for j,c in enumerate(s0_msa_seq) if c != "-"]

        # Remove indels relative to s0
        recs2 = []
        for h,s in recs:
            assert len(s) == len(s0_msa_seq)
            s2 = ""
            for i in nongapinds:
                # If at one of these indices, it is an ambiguous base, coerce to be the same as ref
                if s[i] in "ACGT":
                    s2 += s[i]
                else: 
                    s2 += s0_msa_seq[i]
            assert len(s2) == len(s0_seq)
            recs2.append((h, str_c2i(s2)))

        # Check, for all indicecs that are "N", s0 is also "N"
        for h, s in recs2:
            for ci, c in enumerate(s):
                if c == 4: assert s0_seq[ci] == "N"

        # If this is the required distance from cons
        recs2_final = []
        if min_d is not None:
            for h, s in recs2:
                d = ham_nogaps(cons, s) 
                if d >= min_d:
                    recs2_final.append((h, s))
        else:
            recs2_final = recs2

        return s0_seq, recs2_final



