import sys

mind = 5
for fname in sys.argv[1:]:
    
    pref = fname.rstrip("/").split("/")[-1]
    bamf = fname + "/" + pref + ".bam"
    reff = fname + "/" + pref + ".major.fa"
    outf = fname + "/" + pref + ".emout"
    print("python3 codetectem.py -mind {mind} -bam {bamf} -ref {reff} -out {outf}".format(bamf=bamf,reff=reff,mind=mind,outf=outf))
