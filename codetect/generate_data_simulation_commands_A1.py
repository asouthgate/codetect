if __name__ == "__main__":
    # Let's be explicit instead of implicit:
    i = 0
    for pi in [0.7, 0.8, 0.9, 0.95, 0.98]:
        for d in [0,5,10,15,20]:
            for gamma in [0.005, 0.01, 0.02, 0.03]:
                for nreads in [1000,2000,3000]:
                    string = "python3 simulate_dataset.py --n_reads {nreads} --pi {pi} --d {d} --read_length 200 --gamma {gamma} --genome_length 2000 --covq 3 --out_prefix tmp/runA1_%d".format(pi=pi,d=d,gamma=gamma,nreads=nreads) % i
                    i += 1
                    print(string)
