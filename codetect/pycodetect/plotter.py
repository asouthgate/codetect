import matplotlib.pyplot as plt
import numpy as np
from pycodetect.utils import c2i, ham_nogaps

def plot_m_mat_vs_seqs(rd,major,minor):
    major_arr = [[0,0,0,0] for i in range(len(major))]
    for i, c in enumerate(major): major_arr[i][c] = 1.0
    minor_arr = [[0,0,0,0] for i in range(len(minor))]
    for i, c in enumerate(minor): minor_arr[i][c] = 1.0
    for i in range(len(rd.M)):
        if major_arr[i] != minor_arr[i]:
            print(i, rd.M[i], major_arr[i], minor_arr[i])    

def plot_mask(rad,debug_major,debug_minor):
    # Plot basic array
    y = []
    for row in rad.M:
        y.append(sorted(row)[-2])
    xarr = [j for j in range(len(y))]
    plt.plot(xarr, y, color='blue',alpha=0.3)
    # Plot valid inds
    v = [y[vi] for vi in rad.VALID_INDICES]
    plt.scatter(x=rad.VALID_INDICES,y=v,c="grey")
    # Plot anany false negatives and false positives
    fp = []
    fn = []
    for j in range(len(debug_major)):
        if debug_major[j] != debug_minor[j] and "N" not in [debug_major[j], debug_minor[j]]:
            if j in rad.VALID_INDICES:
                fp.append(j)
            else:
                fn.append(j)
    plt.scatter(x=fp, y=[y[fpi] for fpi in fp], c="green")
    plt.scatter(x=fn, y=[y[fni] for fni in fn], c="red")
    plt.show()
    

def plot_genome(rd,T,st,minor):
    """
    Plot estimated coverage for each cluster across the genome,
    respective posititions, estimated mutations,
    and true values for each.

    Args:
        T: membership probability array per read
        st: estimated sequence for alternative cluster
    """
    # First calculate actual coverage
    cov0 = np.zeros(len(rd.get_consensus()))
    cov1 = np.zeros(len(rd.get_consensus()))
    for Xi in rd.X:
        if "MAJOR" in Xi.name:
            for p,b in Xi.map.items():
                cov0[p] += 1
        else:
            for p,b in Xi.map.items():
                cov1[p] += 1
    rd.COV = cov0 + cov1

    # Calculate indicators for mismatches etc.
    hamarr = np.zeros(len(rd.get_consensus()))
    hamarr2 = np.zeros(len(rd.get_consensus()))
    for i,hi in enumerate(rd.get_consensus()):
        if st[i] != minor[i]:
            hamarr[i] = len(rd.V_INDEX[i][minor[i]])
        elif st[i] == minor[i] and st[i] != rd.get_consensus()[i]:
            hamarr2[i] = len(rd.V_INDEX[i][st[i]])

    # Next calculate estimated coverage
    estcov0 = np.zeros(len(rd.get_consensus()))
    estcov1 = np.zeros(len(rd.get_consensus()))
    for i,Xi in enumerate(rd.X):
        Ti = T[i]
        for pos,base in Xi.map.items():
            estcov0[pos] += Ti[0]
            estcov1[pos] += Ti[1]
    MAX_WIDTH=len(cov0)
    plt.plot(hamarr[:MAX_WIDTH], color='red', alpha=0.5, label="Negatives")
    plt.plot(hamarr2[:MAX_WIDTH], color='green', alpha=0.5, label="Positives")
    plt.plot(cov0[:MAX_WIDTH],color='blue', label="Cov0")
    plt.plot(cov1[:MAX_WIDTH],color='orange', label="Cov1")
    plt.plot(estcov0[:MAX_WIDTH],color='purple', label="EstCov0")
    plt.plot(estcov1[:MAX_WIDTH],color='pink', label="EstCov1")
    plt.legend()
    plt.show()

def debug_plot(rd,emObj):
    """ Plot simulated data statistics for debugging.

    Args:
        emObj: an EM object used for parameter estimation
    """
    T = emObj.Tt
    st = emObj.st
    for k in range(len(rd.COV)):
        assert rd.COV[k] == sum([len(l) for l in rd.V_INDEX[k]])
    nms = np.array([Xi.nm_major for Xi in rd.X])
    plt.plot(rd.COVWALK)
    plt.title("coverage_walk")
    plt.show()
    rd.plot_genome()
    inp = [int(l) for l in input("Specify interval").split()]
    l,r = inp
    plt.plot(hamarr[l:r], color='red', alpha=0.5)
    plt.plot(hamarr2[l:r], color='green', alpha=0.5)
    plt.plot(cov0[l:r],color='blue')
    plt.plot(cov1[l:r],color='orange')
    plt.plot(estcov0[l:r],color='purple')
    plt.plot(estcov1[l:r],color='pink')
    plt.show()
    rd.debug_interval(emObj)
    readdists = [Xi.nm_major for Xi in rd.X if Xi.z == 0]
    plt.hist(readdists,bins=100)
    readdists = [Xi.nm_major for Xi in rd.X if Xi.z == 1]
    plt.hist(readdists,bins=100)
    plt.show()        
    plt.hist([Xi.z for Xi in  rd.X])
    plt.title("True z")
    plt.show()
    plt.hist([r[0] for r in T])
    plt.title("T0")
    plt.show()

