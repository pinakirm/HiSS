import matplotlib.pyplot as plt
import numpy as np

# Values of eta and corresponding acceptance probabilities and standard deviations
eta_values = [0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 0.8, 1, 2, 5, 8, 10]
acceptance_probs = [1.0, 1.0, 1.0, 0.99, 0.965, 0.643, 0.216, 0.162, 0.154, 0.143, 0.136, 0.135, 0.134]
std_devs = [0.0, 0.0, 0.007, 0.032, 0.058, 0.152, 0.13, 0.116, 0.114, 0.11, 0.109, 0.109, 0.108]

# Create the plot
plt.figure(figsize=(10, 6))
plt.errorbar(eta_values, acceptance_probs, yerr=std_devs, fmt='o-', capsize=5)
plt.xlabel("log(η)", fontsize=16)
plt.xscale('log')
plt.ylabel("Average MwG Acceptance Probability", fontsize=16)
plt.title("Impact of scaling on Mixing", fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()


lMAE=[]
lMAE_std=[]
for eta in eta_values:
    data=np.load(('bernoulli_sample_data/lmae_HiSS_'+str(eta)+'.npy'))
    mean=np.mean(data)
    se = np.std(data)
    lMAE.append(mean)
    lMAE_std.append(se)


plt.figure(figsize=(10, 6))
plt.errorbar(eta_values, lMAE, yerr=lMAE_std, fmt='o-', capsize=5)
plt.xlabel("log(η)", fontsize=16)
plt.xscale('log')
plt.ylabel("Average logMAE", fontsize=16)
plt.title("Impact of scaling on Convergence", fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()

lMAE = []
lMAE_std = []
for eta in eta_values:
    data = np.load(('bernoulli_sample_data/coverage_HiSS_' + str(eta) + '.npy'))
    mean = np.mean(data)
    se = np.std(data)
    lMAE.append(mean)
    lMAE_std.append(se)

plt.figure(figsize=(10, 6))
plt.errorbar(eta_values, lMAE, yerr=lMAE_std, fmt='o-', capsize=5)
plt.xlabel("log(η)", fontsize=16)
plt.xscale('log')
plt.ylabel("Average Coverage", fontsize=16)
plt.title("Impact of scaling on Coverage", fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()
