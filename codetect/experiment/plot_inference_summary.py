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
#w = -2*(df['L0']-df['L'])
#df['LRS'] = w
#plt.scatter(x=df['pi'],y=(w<2))
#plt.show()

#print(w)
#plt.hist(w,alpha=0.3,color='black',density=True)
#plt.show()
#H0_ws = w[np.where(df['pi'] > 0.98)[0]]
#print(H0_ws)
#H0_ws = H0_ws[np.where(H0_ws < 10)[0]]
#plt.hist(H0_ws,density=True,bins=100)
#plt.hist(w,density=True,bins=100)
#plt.show()
#dof = 1
#x = np.linspace(0, 10, 500)
#print(x)
#print(chi2.pdf(0,1))
#chi0 = chi2.pdf(0.5*x, 0)
#chi1 = chi2.pdf(0.5*x, 1)
#mix = chi2.pdf(x, 1)
#mix[0] += sum(mix)
#mix /= 2
#print(mix)
#plt.plot(x, mix, 'r-', lw=5, alpha=0.6, label='chi2 pdf')
#plt.xlim([0,10])
#plt.show()


plt.hist(df['pi']-df['piest'],bins=25)
print("pierr",np.mean(np.abs(df['pi']-df['piest'])))
plt.savefig("pierr_hist.pdf", format="pdf")
plt.clf()

fig = plt.figure(figsize=(10,5))
plt.scatter(x=df['pi'], y=df['piest'])
#slope, intercept, r_value, p_value, std_err = stats.linregress(df['pi'], df['piest'])

#print(slope, intercept)
plt.plot(df['pi'], df['pi'], c='black', label='fitted line')
plt.axes().yaxis.set_major_locator(MultipleLocator(0.05))
# For the minor ticks, use no labels; default NullFormatter.
plt.axes().yaxis.set_minor_locator(MultipleLocator(0.01))
plt.xlabel("$\pi$")
plt.ylabel("$\hat{\pi}$")
plt.gca().grid(True, which='major',axis='y',lw=1,color='grey',alpha=0.5)
plt.gca().grid(True, which='minor',axis='y',lw=0.5,color='grey',alpha=0.5,linestyle='--')

plt.savefig("pi_vs_piest.pdf", format="pdf")
plt.clf()

plt.hist(df['g1']-df['g1est'],bins=25)
print("g1err",np.mean(np.abs(df['g1']-df['g1est'])))
plt.savefig("gerr_hist.pdf", format="pdf")
plt.clf()



#plt.hist(np.logical_and(df['piest'] < z, df['pi'] < z).astype(int))
#plt.show()
#plt.hist(np.logical_and(df['piest'] < z, df['pi'] > z).astype(int))
#plt.show()


