if __name__ == "__main__":
    import sys
    from Bio import SeqIO
    import numpy as np
    from codetect.bam_importer import *
    from codetect.sampler import *
    from codetect.em import *
    import sys
    np.set_printoptions(threshold=sys.maxsize)

    #//*** Get data and args ***
    alns = collect_alns(sys.argv[1])
    majorref = [[c2i[c] for c in str(r.seq)] for r in SeqIO.parse(sys.argv[2], "fasta")][0]
    minorref = [[c2i[c] for c in str(r.seq)] for r in SeqIO.parse(sys.argv[3], "fasta")][0]
    rad = ReadAlnData(alns, majorref)

    #//*** EM ***
    em = EM(rad,0.001)
#    em.do2(20,debug=True,debug_minor=minorref)
    em.do2(1)

    #//*** MCMC ***
    MIN_D = 5
    if "--debug" not in sys.argv:
        logging.disable(logging.CRITICAL)
    ref_msa = []
    fixed_point = None
    for r in SeqIO.parse(sys.argv[4], "fasta"):
        seq = [c2i[c.upper()] for c in str(r.seq)]
        ref_msa.append(seq)
        if sys.argv[6] in r.description:
            fixed_point = seq

    assert fixed_point is not None, "%s is not a reference in the references" % sys.argv[6]

    # Delete any gaps; get indices in reverse order
    fixed_point_delinds = [i for i in range(len(fixed_point)) if fixed_point[i] == 4][::-1]
    for ind in fixed_point_delinds:
        for seq in ref_msa: del seq[ind]
    refs = ref_msa

    dmat = np.load(sys.argv[5])
    # TODO: SLOW: FIX FOR IMPORT
    for i in range(len(dmat)-1):
        for j in range(i+1,len(dmat)):
            dmat[j,i] = dmat[i,j]

    #//*** Preprocess dataset ***//
    # Get allowed sites and allowed bases at each site
    allowed_bases=[]
    allowed_sites=rad.VALID_INDICES
    states_per_site = 2
    for m in rad.M:
        allowed_bases.append(sorted(np.argsort(m)[-states_per_site:]))
        assert len(allowed_bases[-1]) > 0

    # Delete references too close to the consensus (taken as fixed point)
    sys.stderr.write("Removing those too close to fixed point\n")
#    fixed_point = rad.get_consensus()
    refs,dmat = del_close_to_fixed_point(fixed_point,MIN_D,refs,dmat)

    sys.stderr.write("Initializing mixture model")
    #//*** Initialize mixture model ***//
    mms = MixtureModelSampler(rad,fixed_point,refs=refs,dmat=dmat)

    #//*** Sample ***//
    strings,params,Ls = mms.sample(rad,nits=1000)
    plt.plot(Ls)
    plt.show()

    #//*** Collect results ***//
    params = np.array(params)
    meanpi = np.mean(params[:,0])
    meang0 = np.mean(params[:,1])
    meang1 = np.mean(params[:,2])
    print("mean pi=%f, mean g0=%f, mean g1=%f" % (meanpi, meang0, meang1))
    C = gen_array(strings)
    hams = []
    BURNIN = int(input("Specify Burnin:"))
    for s in strings[BURNIN:]:
        hams.append(ham(s,minorref))
    hams = np.array(hams)
    print("mean error:", np.mean(hams))
