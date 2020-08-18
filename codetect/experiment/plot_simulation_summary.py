import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns; sns.set()
sns.set_style("whitegrid", {"xtick.bottom":True, "ytick.left":True})
#sns.set_style("ticks")
sns.despine()

"paird,majorexp,minorexp,majorvar,minorvar,majormax,minormax,covmean,covvar,readmean,readvar"

# First should be paird, (majorexp, minorexp), (majorvar,minorvar), (covmean, covvar), (readmean,readvar)
df = pd.read_csv(sys.argv[1])
fig, axs = plt.subplots(3,2,figsize=(8,10))

axs[0,0].hist(df['paird'].values,bins=50,color='#544d7d')
axs[0,0].set_xlabel('d')
axs[0,1].boxplot(np.vstack((df['majorexp'].values,df['minorexp'].values)).T)
plt.sca(axs[0, 1])
plt.xticks([1,2],['$E[h]$ (major)', '$E[h]$ (minor)'])
#axs[0,1].set_title('Axis [0,1]')
axs[1,0].boxplot(np.vstack((df['majorvar'].values,np.sqrt(df['minorvar'].values))).T)
plt.sca(axs[1, 0])
plt.xticks([1,2],['$\sigma [h]$ (major)', '$\sigma [h]$ (minor)'])
axs[1,1].boxplot(np.vstack((df['majormax'].values,df['minormax'].values)).T)
plt.sca(axs[1, 1])
plt.xticks([1,2],['$max [h]$ (major)', '$max [h]$ (minor)'])
axs[2,0].boxplot(np.vstack((df['covmean'].values,np.sqrt(df['covvar'].values))).T)
plt.sca(axs[2, 0])
plt.xticks([1,2],['mean coverage', 'stdev coverage'])
axs[2,1].boxplot(np.vstack((df['readmean'].values,np.sqrt(df['readvar'].values))).T)
plt.sca(axs[2, 1])
plt.xticks([1,2],['mean read $h$', 'stdev read $h$'])
#axs[1,0].set_title('Axis [1,0]')
#axs[1,1].plot(x, -y, 'tab:red')
#axs[1,1].set_title('Axis [1,1]')
plt.subplots_adjust(hspace=0.5)
plt.show()


