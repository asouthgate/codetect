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

df_null = pd.read_csv(sys.argv[1])
w_null = -2*(df_null['L0']-df_null['L'])
w_null = w_null[np.where(w_null >= 0)[0]]

#plt.hist(np.log(w+1),density=True,bins=100,alpha=0.5)
#plt.hist(np.log(w_null+1),density=True,bins=100,alpha=0.5)
#plt.show()

#plt.scatter(x=df['pi'],y=(w<2))
#plt.show()
#print(w)
#plt.hist(w,alpha=0.3,color='black',density=True)
#plt.show()
#H0_ws = w[np.where(df['pi'] > 0.98)[0]]
#print(H0_ws)
#H0_ws = H0_ws[np.where(H0_ws < 10)[0]]
#plt.hist(w,density=True,bins=100)
#plt.show()
dof = 1
#xmax = 5
#xmin = 0.001
#xmin = chi2.ppf(0.1,dof)
#x = np.linspace(xmin, xmax, 1000)
#print(x)
#plt.plot(x, chi2.pdf(x,1))
#plt.show()
#mix = chi2.pdf(x, 1)
#mix /= 2.0
# with one DOF, the dist is already unbounded at 0, dont need to add to it at all
#mix[0] += 0.5
#print(min(x),max(x))
#plt.plot(x, mix, 'r-', lw=2, alpha=0.8, label='chi2 pdf')
#plt.hist(w[np.where(w < xmax)[0]],density=True,bins=200)
#plt.xlim([0, xmax])
#plt.ylim([0, 1.0])
#plt.show()


xmax = max(w_null)
xmin = min(w_null)
#xmin = chi2.ppf(0.1,dof)
x1 = np.linspace(xmin, xmax, 10000)
x2 = np.linspace(xmax, max(w_null), 100)
x = np.concatenate((x1, x2), axis=0)
logx = np.log(x+1)
print("logxmin",min(logx))
#plt.plot(x, chi2.pdf(x,1))
#plt.show()
mix = chi2.pdf(x,1)
mix /= 2.0
#mix[0] += 0.5
#logmix = chi2.pdf(np.exp(logx)-1, 1)
#logmix /= 2.0
# with one DOF, the dist is already unbounded at 0, dont need to add to it at all
#logmix[0] += 0.5
print(min(x),max(x))
#logw = np.log(w+1)
#logw_null = np.log(w_null+1)
#print(logmix)
fig = plt.figure(figsize=(10,5))
plt.plot(x, mix, 'r-', lw=2, alpha=0.8, label='chi2 pdf')
plt.axvline(x=np.log(chi2.ppf(0.95,1)+1))
#plt.plot(x, mix)
#plt.hist(logw,bins=100,alpha=0.7,density=True)
plt.hist(w_null,bins=100,alpha=0.7,density=True)
#plt.hist(w_null[np.where(w_null < 5)[0]],bins=1000,density=True,alpha=0.5)
#print(max(logw_null), max(logw))
#plt.xlim([min(logw_null), max(max(logw),max(logw_null))])
plt.ylim([0, 3])
#plt.gca().set_xscale('log')
plt.axes().yaxis.set_major_locator(MultipleLocator(0.5))
# For the minor ticks, use no labels; default NullFormatter.
plt.axes().yaxis.set_minor_locator(MultipleLocator(0.1))
plt.xlabel("$\lambda$")
plt.gca().grid(True, which='major',axis='y',lw=1,color='grey',alpha=0.5)
plt.gca().grid(True, which='minor',axis='y',lw=0.5,color='grey',alpha=0.5,linestyle='--')
plt.xlim([0,max(w_null)])
plt.savefig("chi2_null.pdf", format="pdf")
plt.show()
