import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
sns.set_style("whitegrid")
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np



samplers=['GWG','DMALA','ACS','PT+DMALA','HiSS']
#samplers=['HiSS-noMH','HiSS']
for s in samplers:
    mean=np.load('bernoulli_sample_data/lmae_'+s+'.npy')
    se = np.load('bernoulli_sample_data/lmae_se_' + s + '.npy')
    N=range(0,len(mean))
    N= [t * 1 for t in N]
    plt.plot(N, mean,marker='', label=s)
    plt.fill_between(N, mean- se, mean+ se, alpha=0.3)

    # Add labels and title
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Average logMAE', fontsize=14)
plt.title('Convergence Analysis', fontsize=16)
plt.legend(loc='best', fontsize=11, frameon=False)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig(f"Beroulli1.png", dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

for s in samplers:
    mean=np.load('bernoulli_sample_data/lmae_'+s+'.npy')
    se = np.load('bernoulli_sample_data/lmae_se_' + s + '.npy')
    time=np.load('bernoulli_sample_data/times_' + s + '.npy')
    time = time - time[0]
    N=time
    plt.plot(N, mean,marker='', label=s)
    plt.fill_between(N, mean- se, mean+ se, alpha=0.3)

    # Add labels and title
plt.xlabel('log adjusted runtime',fontsize=14)
plt.ylabel('Average logMAE', fontsize=14)
plt.title('Relative Convergence Analysis', fontsize=16)
plt.legend(loc='best', fontsize=11, frameon=False)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xscale('log')
plt.savefig(f"Beroulli2.png", dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

for s in samplers:
    mean=np.load('bernoulli_sample_data/coverage_'+s+'.npy')
    se = np.load('bernoulli_sample_data/coverage_se_' + s + '.npy')
    N=range(0,len(mean))
    N= [t * 1 for t in N]
    plt.plot(N, mean,marker='', label=s)
    plt.fill_between(N, mean- se, mean+ se, alpha=0.3)

    # Add labels and title
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Average Coverage', fontsize=14)
plt.title('Coverage Analysis', fontsize=16)
plt.legend(loc='best', fontsize=11, frameon=False)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig(f"Beroulli3.png", dpi=300, bbox_inches='tight')
plt.show()
plt.clf()
