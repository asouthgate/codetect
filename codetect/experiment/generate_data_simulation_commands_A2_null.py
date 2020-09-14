import numpy as np

if __name__ == "__main__":
    # Let's be explicit instead of implicit:
    i = 0
    nreads = 5000
    read_length = 200
    mu = 0.0005
    gamma = 0.030
    nreps = 1000
    pi = 1.0
    for nrep in range(nreps):
        string = "python3 experiment/simulate_dataset.py --n_reads {nreads} --refs res/H3N2_HA_nfu.m.fa --dmat res/H3N2_HA_nfu.dmat.npy --min_d 1 --max_d 1000 --pi {pi} --read_length {read_length} --gamma {gamma} --mu {mu} --out_prefix tmp/runA2_null_%d".format(pi=pi,gamma=gamma,nreads=nreads,read_length=read_length,mu=mu) % i
        i += 1
        print(string)
