import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy import stats
from scipy import signal
import seaborn as sns; sns.set()
from matplotlib.ticker import MultipleLocator

# Configuration options
sns.set_style("whitegrid", {"xtick.bottom":True, "ytick.left":True, "grid.color":"grey", "axes.edgecolor":"black", "xtick.color":"black"})
pd.set_option('display.max_rows', None)

df = pd.read_csv(sys.argv[1])
w = -2*(df['L0'].values-df['L'].values)
w = w[np.where(w >= 0)[0]]
df_null = pd.read_csv(sys.argv[2])
w_null = -2*(df_null['L0'].values-df_null['L'].values)
w_null = w_null[np.where(w_null >= 0)[0]]
dof = 1
w = w[:len(w_null)]
assert len(w_null) == len(w), (len(w_null), len(w))
null_prob_w = chi2.pdf(w,1)/2.0
null_prob_w += ((w == 0).astype(int)/2.0)
null_prob_w_null = chi2.pdf(w_null,1)/2.0
null_prob_w_null += ((w_null == 0).astype(int)/2.0)
plt.hist(np.log(null_prob_w+1),alpha=0.3,bins=100,density=True)
plt.hist(np.log(null_prob_w_null+1),alpha=0.3,bins=100,density=True)
plt.show()
assert len(null_prob_w) == len(null_prob_w_null)
print(null_prob_w)
print(null_prob_w_null)
assert not all(null_prob_w == null_prob_w_null)

TPRs = []
FPRs = []
AUROC = 0
delta = (1-min(null_prob_w_null)/2)/5000
for p in np.linspace(min(null_prob_w_null)/2,1.0,5000,endpoint=False):
    class_w = null_prob_w < p
    class_w_null = null_prob_w_null < p
    TPR = sum(class_w)/len(class_w)
    FPR = sum(class_w_null)/len(class_w_null)
    print(p,TPR,FPR, null_prob_w[0], null_prob_w_null[0])
    TPRs.append(TPR)
    FPRs.append(FPR)   
    AUROC += TPR*delta

print("AUROC:", AUROC)
fig = plt.figure(figsize=(10,5))
plt.step(x=FPRs,y=TPRs,color="#f09a32")
plt.axes().yaxis.set_major_locator(MultipleLocator(0.005))
plt.axes().yaxis.set_minor_locator(MultipleLocator(0.0025))
plt.axes().xaxis.set_major_locator(MultipleLocator(0.2))
plt.axes().xaxis.set_minor_locator(MultipleLocator(0.1))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.gca().grid(True, which='major',axis='y',lw=1,color='grey',alpha=0.5)
plt.gca().grid(True, which='minor',axis='y',lw=0.5,color='grey',alpha=0.5,linestyle='--')
plt.gca().grid(True, which='major',axis='x',lw=1,color='grey',alpha=0.5)
plt.gca().grid(True, which='minor',axis='x',lw=0.5,color='grey',alpha=0.5,linestyle='--')

plt.savefig("sim_roc.pdf", format="pdf")
plt.show()

