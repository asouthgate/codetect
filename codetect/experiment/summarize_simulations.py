import numpy as np
import sys
sys.path.append("./")
import matplotlib.pyplot as plt
import pickle

def ham(s1,s2):
    return len([j for j in range(len(s1)) if s1[j] != s2[j]])

mean_snps = []
ranked_proportions = []
pair_ds = []
print("paird,majorexp,minorexp,majorvar,minorvar,majormax,minormax,covmean,covvar,readmean,readvar")
for folder in sys.argv[1:]:
    opref = folder.rstrip("/").split("/")[-1]
    pckfn = folder+"/"+opref+".pckl"
    with open(pckfn, 'rb') as f:
        ds = pickle.load(f)
        pair_d = ham(ds.major,ds.minor)
        majorprops = ds.majorpop[1]
        minorprops = ds.minorpop[1]
        minorhams = ds._minorhams
        majorhams = ds._majorhams
        minorexp = sum([minorhams[j]*minorprops[j] for j in range(len(minorhams))])
        majorexp = sum([majorhams[j]*majorprops[j] for j in range(len(minorhams))])
        minorvar = sum([(minorhams[j]**2)*minorprops[j] for j in range(len(minorhams))]) - (minorexp**2)
        majorvar = sum([(majorhams[j]**2)*majorprops[j] for j in range(len(minorhams))]) - (majorexp**2)
        minormax = max(minorhams)
        majormax = max(majorhams)

        coverage = ds._covwalk 
        cov0 = np.zeros(len(ds.get_consensus()))
        cov1 = np.zeros(len(ds.get_consensus()))
        for Xi in ds.X:
            if "MAJOR" in Xi.name:
                for p,b in Xi.map.items():
                    cov0[p] += 1
            else:
                for p,b in Xi.map.items():
                    cov1[p] += 1
        nms = np.array([Xi.nm_major for Xi in ds.X])
        print(pair_d,majorexp,minorexp,majorvar,minorvar,majormax,minormax,np.mean(cov0+cov1),np.var(cov0+cov1),np.mean(nms), np.var(nms),sep=",")
