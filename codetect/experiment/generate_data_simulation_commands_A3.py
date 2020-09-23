import numpy as np

if __name__ == "__main__":
    # Let's be explicit instead of implicit:
    i = 0
    nreads = 5000
    read_length = 200
    mu = 0.0005
    nreps = 5
    gamma = 0.03
    for nrep in range(nreps):
        for pi in np.linspace(0.7,1.0,num=50,endpoint=False):
            string = "python3 experiment/simulate_dataset.py --n_reads {nreads} --refs res/cog_2020-09-07.tr.p09.rand1000.fa --dmat res/cog_2020-09-07.tr.p09.rand1000.dmat.npy --min_d 1 --max_d 1000 --pi {pi} --read_length {read_length} --gamma {gamma} --mu {mu} --out_prefix tmp/runA3_%d".format(pi=pi,gamma=gamma,nreads=nreads,read_length=read_length,mu=mu) % i
            i += 1
            print(string)
