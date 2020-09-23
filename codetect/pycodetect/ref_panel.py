class RefPanel():
    def __init__(self, ref_msa, s0_h, min_d):
        self._s0_h = s0_h
        self._s0_s, self._ref_panel = preprocess_msa_refs(args.ref_msa,
                                                  ref_rec.description, 
                                                  min_d=args.mind)
        self._ref_diff_inds = []
        self._min_d = min_d

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
        """ Get the diff inds for ref ri. """
        return self._ref_diff_inds[ri]

    def preprocess_msa_refs(self, ref_fname, s0_h, min_d=None):
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
            recs = [(h,str_only_ACGT(s.upper())) for h,s in FastaIO.SimpleFastaParser(f)]

        # Pull out s0 msa sequence
        s0_msa_seq = ""
        for h,s in recs: 
            if h == s0_h:
                s0_msa_seq = s
                break

        # Cut out any gaps from s0 to get s0_seq
        s0_seq = s0_msa_seq.replace("-","")
        assert len(s0_msa_seq) > 0, "Reference %s not found in msa" % s0_h

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
            # If this is the required distance from s0, append
            d = ham_nogaps_str(s0_seq, s2) 
            dmsa = ham_nogaps_str(s0_msa_seq, s)
            assert d == dmsa, (d, dmsa) 
            if min_d is not None:
                if d > min_d:
                    recs2.append((h, str_c2i(s2)))
            else:
                recs2.append((h, str_c2i(s2)))

        # Check, for all indicecs that are "N", s0 is also "N"
        for h, s in recs2:
            for ci, c in enumerate(s):
                if c == 4: assert s0_seq[ci] == "N"
        return s0_seq, recs2



