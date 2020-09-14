import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns; sns.set()
from matplotlib.ticker import MultipleLocator

# Configuration options
sns.set_style("whitegrid", {"xtick.bottom":True, "ytick.left":True, "grid.color":"grey", "axes.edgecolor":"black", "xtick.color":"black"})
pd.set_option('display.max_rows', None)


"paird,majorexp,minorexp,majorvar,minorvar,majormax,minormax,covmean,covvar,readmean,readvar"
# First should be paird, (majorexp, minorexp), (majorvar,minorvar), (covmean, covvar), (readmean,readvar)
df = pd.read_csv(sys.argv[1])
fig, axs = plt.subplots(3,2,figsize=(10,8))

axs[0,0].hist(df['paird'].values,bins=50,color='#544d7d')
plt.sca(axs[0,0])
axs[0,0].yaxis.set_major_locator(MultipleLocator(10))
# For the minor ticks, use no labels; default NullFormatter.
axs[0,0].yaxis.set_minor_locator(MultipleLocator(5))

plt.gca().grid(True, which='major',axis='y',lw=1,color='grey',alpha=0.5)
plt.gca().grid(True, which='minor',axis='y',lw=0.5,color='grey',alpha=0.5,linestyle='--')
print("paird mean:",np.mean(df['paird'].values), "stdev:", np.sqrt(np.var(df['paird'].values)))
axs[0,0].set_xlabel('d')
axs[0,1].boxplot(np.vstack((df['majorexp'].values,df['minorexp'].values)).T)
print("majorexp:",np.mean(df['majorexp'].values), "stdev:", np.sqrt(np.var(df['majorexp'].values)))
print("minorexp:",np.mean(df['minorexp'].values), "stdev:", np.sqrt(np.var(df['majorexp'].values)))

plt.sca(axs[0, 1])
axs[0,1].yaxis.set_major_locator(MultipleLocator(0.1))
# For the minor ticks, use no labels; default NullFormatter.
axs[0,1].yaxis.set_minor_locator(MultipleLocator(0.05))

plt.gca().grid(True, which='major',axis='y',lw=1,color='grey',alpha=0.5)
plt.gca().grid(True, which='minor',axis='y',lw=0.5,color='grey',alpha=0.5,linestyle='--')
plt.xticks([1,2],['$E[h]$ (major)', '$E[h]$ (minor)'])
#axs[0,1].set_title('Axis [0,1]')
axs[1,0].boxplot(np.vstack((np.sqrt(df['majorvar'].values),np.sqrt(df['minorvar'].values))).T)

plt.sca(axs[1, 0])
plt.gca().grid(True, which='major',axis='y',lw=1,color='grey',alpha=0.5)
plt.gca().grid(True, which='minor',axis='y',lw=0.5,color='grey',alpha=0.5,linestyle='--')
axs[1,0].yaxis.set_major_locator(MultipleLocator(0.4))
# For the minor ticks, use no labels; default NullFormatter.
axs[1,0].yaxis.set_minor_locator(MultipleLocator(0.2))

plt.xticks([1,2],['$stdev [h]$ (major)', '$stdev [h]$ (minor)'])
axs[1,1].boxplot(np.vstack((df['majormax'].values,df['minormax'].values)).T)

plt.sca(axs[1, 1])
plt.gca().grid(True, which='major',axis='y',lw=1,color='grey',alpha=0.5)
plt.gca().grid(True, which='minor',axis='y',lw=0.5,color='grey',alpha=0.5,linestyle='--')
axs[1,1].yaxis.set_major_locator(MultipleLocator(2))
# For the minor ticks, use no labels; default NullFormatter.
axs[1,1].yaxis.set_minor_locator(MultipleLocator(1))

plt.xticks([1,2],['$max [h]$ (major)', '$max [h]$ (minor)'])
axs[2,0].boxplot(np.vstack((df['covmean'].values,np.sqrt(df['covvar'].values))).T)
print("covmean:",np.mean(df['covmean'].values), "stdev:", np.sqrt(np.var(df['covmean'].values)))
axs[2,0].yaxis.set_major_locator(MultipleLocator(200))
# For the minor ticks, use no labels; default NullFormatter.
axs[2,0].yaxis.set_minor_locator(MultipleLocator(100))



plt.sca(axs[2, 0])
plt.gca().grid(True, which='major',axis='y',lw=1,color='grey',alpha=0.5)
plt.gca().grid(True, which='minor',axis='y',lw=0.5,color='grey',alpha=0.5,linestyle='--')
plt.xticks([1,2],['mean coverage', 'stdev coverage'])
axs[2,1].boxplot(np.vstack((df['readmean'].values,np.sqrt(df['readvar'].values))).T)
print("readmean:",np.mean(df['readmean'].values), "stdev:", np.sqrt(np.var(df['readmean'].values)))
plt.sca(axs[2, 1])
plt.gca().grid(True, which='major',axis='y',lw=1,color='grey',alpha=0.5)
plt.gca().grid(True, which='minor',axis='y',lw=0.5,color='grey',alpha=0.5,linestyle='--')
axs[2,1].yaxis.set_major_locator(MultipleLocator(4))
# For the minor ticks, use no labels; default NullFormatter.
axs[2,1].yaxis.set_minor_locator(MultipleLocator(2))



plt.xticks([1,2],['mean read $h$', 'stdev read $h$'])
#axs[1,0].set_title('Axis [1,0]')
#axs[1,1].plot(x, -y, 'tab:red')
#axs[1,1].set_title('Axis [1,1]')
plt.subplots_adjust(hspace=0.25)
plt.savefig(sys.argv[1]+".pdf", format="pdf")
plt.show()


