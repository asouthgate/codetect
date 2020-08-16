import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle


mean_snps = []
ranked_proportions = []
pair_ds = []
for folder in sys.argv[1:]:
    opref = folder.rstrip("/").split("/")[-1]
    pckfn = folder+"/"+opref+".pckl"
    with open(pckfn, 'rb') as f:
        ds = pickle.load(f)
        pair_ds.append(ds)
        majorprops = ds.majorpop[1]
        minorprops = ds.minorpop[1]
        plt.bar(x=[j for j in range(len(majorprops))],height=majorprops)
        plt.show()
        plt.bar(x=[j for j in range(len(majorprops))],height=minorprops)
        plt.show()
        minorhams = ds._minorhams
        majorhams = ds._majorhams
        print(minorhams)
        plt.hist(minorhams)
        plt.show()
        plt.hist(majorhams)
        plt.show()
        minorexp = sum([minorhams[j]*minorprops[j] for j in range(len(minorhams))])
        majorexp = sum([majorhams[j]*majorprops[j] for j in range(len(minorhams))])
        print(minorexp)
        print(majorexp)
        minormax = max(minorhams)
        majormax = max(majorhams)
        print(minormax)
        print(majormax)
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
        plt.plot([j for j in range(len(cov0))], cov0)
        plt.plot([j for j in range(len(cov0))], cov1)
        plt.show()
        nms = np.array([Xi.nm_major for Xi in ds.X])
        plt.hist(nms)
        plt.show()
        print(np.mean(nms), np.var(nms))  
