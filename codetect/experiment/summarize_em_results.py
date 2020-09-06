import sys
import pickle
import numpy as np
sys.path.append("./")
import pycodetect

print("run,L,L0,nsites,piest,g1est,g2est,pi,g1,h,wald,z")
for fn in sys.argv[1:]:
    try:
        # Get results vars
        pref = fn.rstrip("/") + "/" + fn.rstrip("/").split("/")[-1]
        tracefn = pref + ".emout.trace.csv"
        alttracefn = pref + ".emout.alt_trace.csv"
        summfn = pref + ".emout.summary.csv"
        estfa = pref + ".emout.est.fa"
        with open(tracefn) as f:
            lines = [l.strip() for l in f]
            L,piest,g1est,g2est = lines[-1].split(",")[1:]
        with open(alttracefn) as f:
            lines = [l.strip() for l in f]
            L0 = lines[-1].split(",")[1]
        with open(summfn) as f:
            lines = [l.strip() for l in f]
            nsites = lines[-1]
        with open(estfa) as f:
            lines = [l.strip() for l in f]
            estseq = lines[-1]

        # Get simulation statistics
        minorfa = pref + ".minor.fa"
        with open(minorfa) as f:
            lines = [l.strip() for l in f]
            minorseq = lines[-1]
        pickles = pref + ".pckl"
        with open(pickles, "rb") as f:
            ds = pickle.load(f)
            pi = ds.pi
            g1 = ds.gamma
            nreads = ds.n_reads

        L = float(L)
        L0 = float(L0)
        ham = lambda x,y: sum([1 for j in range(min(len(x),len(y))) if x[j] != y[j]])
        h = ham(estseq,minorseq)
        wald = -2*(L0-L)
        print(fn.rstrip("/").split("/")[-1],L,L0,nsites,piest,g1est,g2est,pi,g1,h,wald,sep=',')
    except:
        sys.stderr.write("WARNING: Failed to get %s\n" %fn)

        

        

        
