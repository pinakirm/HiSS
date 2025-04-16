import argparse
import math
import seaborn as sns
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision
import bernoulli_distribution, samplers, block_samplers
import pandas as pd
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import tensorflow_probability as tfp
from matplotlib.colors import Normalize
import time
import itertools
import random
from matplotlib.colors import ListedColormap

#from asbs_code.GBS.sampling.globally import AnyscaleBalancedSampler

def hamming_distance_batch(tensors, reference):
    return (tensors != reference).sum(dim=-1)
def local_entropy(theta_a, model, eta):
    binary_combinations = list(itertools.product([0, 1], repeat=model.data_dim))
    parameter_space = torch.tensor(binary_combinations, dtype=torch.float64)  # Shape: (16, 4)

    # Compute energy function for the entire parameter space
    energies = torch.stack([model(param.unsqueeze(0)).sum() for param in parameter_space])  # Shape: (16,)

    # Convert theta_a to tensor if not already
    theta_a_tensor = torch.stack(theta_a)  # Shape: (num_samples, 1, 4)

    # Compute the differences and the second part
    param_diff = parameter_space.unsqueeze(0) - theta_a_tensor  # Shape: (num_samples, 16, 4)
    second_part = torch.sum(param_diff ** 2, dim=-1) / (2 * eta)

    exp_local_entropy = torch.exp(energies.unsqueeze(0) - second_part)
    internal_sum=exp_local_entropy.sum(dim=1)# Shape: (num_samples, 16)
    local_entropy = torch.log(internal_sum)
    # Shape: (num_samples,)

    return local_entropy.numpy()

def auxillary_dist(theta_a, model, eta):
    le=local_entropy(theta_a,model,eta)
    prob=np.exp(le)
    #Z=np.sum(prob)
    Z=1

    return prob/Z

def generate_transition_heatmap(transitions, n_steps, thinning, state_labels, t,MAE):
    norm = Normalize(vmin=np.min(MAE), vmax=np.max(MAE))
    normalized_weights = norm(MAE)

    len=int(n_steps / thinning) + 1


    plt.figure(figsize=(12, 8))
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    weighted_cmap = ListedColormap(cmap(normalized_weights))
    sns.heatmap(transitions, cmap=weighted_cmap, cbar=True, xticklabels=state_labels, annot=True,
                yticklabels=range(len), linewidths=.5, linecolor='black', fmt='.3f')
    plt.xlabel('States')
    plt.ylabel('Iterations')
    plt.yticks(fontsize=8)
    plt.title('State Transitions Heatmap for ' + t)
    plt.show()


def generate_sample_distribution(chain_a, t):
    theta_a_components = []
    for i in range(4):
        theta_a_components.append(np.array([tensor[0, i].item() for tensor in chain_a]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    b = int(len(theta_a_components[0]) ** 0.5)

    # Titles for each subplot
    titles = ['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4']
    for i in range(4):
        axes[i].hist(theta_a_components[i], bins=b, edgecolor='black', density=True)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Empirical Density')

    # Adjust layout
    plt.tight_layout()
    fig.suptitle('Sample Distribution for ' + t)

    # Show the plot
    plt.show()


def generate_target_distribution(num_samples, model, eta, t):
    samples = generate_random_4d_samples(num_samples)
    stationary_probabilities = auxillary_dist(samples, model, eta)
    # print(stationary_probabilities)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    b = int(num_samples ** 0.5)

    # Titles for each subplot
    titles = ['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4']
    for i in range(4):
        axes[i].hist(np.array([tensor[i] for tensor in samples]), bins=b, edgecolor='black', density=True,
                     color='green', weights=stationary_probabilities)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Stationary Density')

    plt.tight_layout()
    fig.suptitle('Stationary Distribution for ' + t)
    plt.show()


def generate_target_distribution_glu(n, model, eta, t):
    cov_matrix = eta * np.eye(4)

    # Generate the initial x_a
    x = torch.randint(low=0, high=2, size=(4,), dtype=torch.float64).unsqueeze(0)
    x_a = x

    # Generate all binary combinations and create parameter space tensor
    binary_combinations = list(itertools.product([0, 1], repeat=4))
    parameter_space = torch.tensor(binary_combinations, dtype=torch.float64)  # Shape: (16, 4)

    samples = []
    energies = torch.stack([model(param.unsqueeze(0)).sum() for param in parameter_space])

    for _ in range(n):
        # Compute energy functions for the entire parameter space

        # Compute the second part for all parameter space combinations
        param_diff = parameter_space.unsqueeze(1) - x_a.unsqueeze(0)  # Shape: (16, 1, 4)
        second_part = torch.sum(param_diff ** 2, dim=-1) / (2 * eta)  # Shape: (16, 1)

        # Compute probabilities
        probs = torch.exp(energies.unsqueeze(1) - second_part).squeeze()  # Shape: (16,)
        total_sum = probs.sum().item()
        probs = probs / total_sum

        # Convert probabilities and parameters to numpy for random.choices
        probs_np = probs.numpy()
        params_np = parameter_space.numpy()

        # Sample one parameter based on computed probabilities
        chosen_param = params_np[random.choices(range(len(probs_np)), probs_np, k=1)[0]]

        # Sample from the multivariate normal distribution
        sample = np.random.multivariate_normal(chosen_param, cov_matrix, 1)
        samples.append(sample)

        # Update x_a
        x_a = torch.tensor(sample, dtype=torch.float64)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    b = int(n ** 0.5)

    # Titles for each subplot
    titles = ['Dimension 1', 'Dimension 2', 'Dimension 3', 'Dimension 4']
    for i in range(4):
        axes[i].hist(np.array([tensor[0, i] for tensor in samples]), bins=b, edgecolor='black', density=True,
                     color='green')
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Stationary Density for ' + t)

    plt.tight_layout()
    fig.suptitle('Stationary Distribution(GLU) for ' + t)
    plt.show()


def generate_random_4d_samples(num_samples, lower_bound=-5, upper_bound=5):
    # Generate random samples within the specified bounds
    samples = np.random.uniform(lower_bound, upper_bound, size=(num_samples, 4))
    return torch.tensor(samples, dtype=torch.float64)

def get_LMAE(GT, ET, nchain):
    absolute_errors = torch.abs(GT- ET)

        # Calculate the average absolute error
    lmae = np.log(torch.mean(absolute_errors, dim=1))
    ste = torch.std(lmae) / np.sqrt(nchain)
    lmae = torch.mean(lmae)

    return lmae.item(), ste.item()

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n=args.nchains

    dim = 4
    tensor_states = list(itertools.product([0, 1], repeat=4))
    state_labels = [''.join(map(str, state)) for state in tensor_states]
    state_map = {"".join(map(str, state)): idx for idx, state in enumerate(tensor_states)}
    p = [ 0.588204 , 5.882e-6, 5.882e-6, 5.882e-6, 5.882e-6,
         5.882e-6, 5.882e-6, 5.882e-6,5.882e-6, 5.882e-6,
         5.882e-6, 5.882e-6, 5.882e-6, 5.882e-6,  0.294102 ,
          0.117641]

    plt.figure(figsize=(10, 6))
    plt.bar(state_labels, p, color='skyblue', edgecolor='black')
    # Customize plot
    plt.xlabel('States', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.title('Probability Mass Function (PMF)', fontsize=18)
    plt.ylim(0, max(p) + 0.05)
    plt.savefig(str('target_hist.png'), dpi=300, bbox_inches='tight')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()



    y = torch.randint(low=0, high=2, size=(n,dim), dtype=torch.float)
    #y = torch.tensor([[1.00, 0.00, 1.00, 0.00]])
    model = bernoulli_distribution.JointBernoulli(p, dim=dim)
    model.to(device)
    t = args.sampler
    j = 0
    transitions = np.zeros((int(args.n_steps / args.thinning) + 1, 2 ** dim))
    state_counts = {"".join(map(str, state)): 0 for state in tensor_states}
    chain = []
    chain_state=[]
    chain_a = []
    MAE = []
    Coverage=[]
    Coverage_se=[]
    TIME = []
    SE=[]
    if t == 'dim-gibbs':
        sampler = samplers.PerDimGibbsSampler(model.data_dim)
    elif t == "rand-gibbs":
        sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=True)
    elif "hb-" in t:
        block_size, hamming_dist = [int(v) for v in t.split('-')[1:]]
        sampler = block_samplers.HammingBallSampler(model.data_dim, block_size, hamming_dist)

    elif t == "dula":
        sampler = samplers.LangevinSampler(model.data_dim, 10,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                           step_size=0.1, mh=False)
    elif t == "DMALA":
        sampler = samplers.LangevinSampler(model.data_dim, 10,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                           step_size=0.2, mh=True)


    elif t == 'DiGS':
        sampler = samplers.DGLangevinSampler(model.data_dim, 5,
                                             fixed_proposal=False, approx=True, multi_hop=False, temp=2.
                                             , mh=True, alpha=np.sqrt(1-args.eta),step_size=0.2, eta=args.eta, score_sweeps=2)

    elif t == 'HiSS':
        sampler = samplers.ModifiedDiGsSampler(model.data_dim, 5,
                                             fixed_proposal=False, approx=True, multi_hop=False, temp=2.
                                             , mh=True, step_size=0.2, eta=args.eta, score_sweeps=2,K=args.K )

    elif t == 'cauchy-digs':
        sampler = samplers.ModifiedDiGsSampler2(model.data_dim, 5,
                                             fixed_proposal=False, approx=True, multi_hop=False, temp=2.
                                             , mh=True, step_size=0.2, eta=args.eta, score_sweeps=2,K=args.K )
    elif t == "GWG":
        sampler = samplers.DiffSampler(model.data_dim, 5*2,
                                       fixed_proposal=False, approx=True, multi_hop=False, temp=2.)


    elif t == 'PT+DMALA':
        sampler = samplers.ParallelTemperingLangevinSampler(model.data_dim, n_steps=5 * 2,
                                                            n_chains=5, step_size=0.2, swap_interval=4,
                                                            mh=True)

    elif t == 'ACS':
        sampler = samplers.AutomaticCyclicalSampler(dim=model.data_dim,
                                                    max_val=model.data_dim,
                                                    n_steps=5 * 2,
                                                    mh=True,
                                                    multi_hop=False,
                                                    fixed_proposal=False,
                                                    approx=True,
                                                    num_cycles=args.num_cycles,
                                                    num_iters=args.num_iters,
                                                    mean_stepsize=0.2,
                                                    initial_balancing_constant=args.initial_balancing_constant,
                                                    burnin_adaptive=args.burnin_adaptive,
                                                    burnin_budget=args.burnin_budget,
                                                    burnin_lr=args.burnin_lr,
                                                    sbc=args.use_manual_EE,
                                                    big_step=5,
                                                    big_bal=args.big_bal,
                                                    small_step=0.05,
                                                    small_bal=args.small_bal,
                                                    min_lr=args.min_lr,
                                                    device=device)





    else:
        assert False, 'Not implemented'
    probabilities=torch.zeros(n, 16)

    start_time = time.time()
    x = y
    x_a = x
    weights = torch.tensor([8, 4, 2, 1], dtype=torch.int64).to(x.device)
    for i in range(args.n_steps + 1):

        if i % args.thinning == 0:

            chain.append(x)
            if sampler.entropy == True:
                chain_a.append(x_a)


            indices = torch.sum(x * weights, dim=1, dtype=torch.int64)

            one_hot = torch.nn.functional.one_hot(indices, num_classes=16).type(torch.float32)
            probabilities += one_hot  # Sum across samples for each chain




        if (sampler.entropy == False):
            if args.sampler != 'ACS' and args.sampler != 'PT+DMALA':
                xhat = sampler.step(x.detach(), model).detach()
            else:
                xhat = sampler.step(x.detach(), model,i).detach()

        else:
            sample_tuple = sampler.step(x.detach(), x_a.detach(), model)
            xhat = sample_tuple[0].detach()
            xahat = sample_tuple[1].detach()

        x = xhat
        if (sampler.entropy == True):
            x_a = xahat

        if i % args.interval == 0:
            EP=probabilities/(i+1)
            lmae,se=get_LMAE(torch.tensor(p),EP,args.nchains)

            coverage = (probabilities > 0).sum(dim=1).float() / probabilities.size(1) *1

            # Calculate mean coverage and standard error
            mean_coverage = coverage.mean().item()
            std_error = coverage.std(unbiased=True).item() / torch.sqrt(
                torch.tensor(probabilities.size(0), dtype=torch.float32)).item()

            print("Mean Coverage:", mean_coverage)
            print("Standard Error:", std_error)


            print(t + " iter " + str(i) + " logMAE: " + str(lmae)+" "+str(se))
            print("\n")
            TIME.append(time.time() - start_time)
            MAE.append(lmae)
            SE.append(se)
            Coverage.append(mean_coverage)
            Coverage_se.append(std_error)

    np.save("{}/lmae_{}_{}.npy".format(args.save_dir, args.sampler, args.eta), MAE)
    np.save("{}/times_{}_{}.npy".format(args.save_dir, args.sampler, args.eta), TIME)
    np.save("{}/lmae_se_{}_{}.npy".format(args.save_dir, args.sampler, args.eta), SE)
    np.save("{}/coverage_{}_{}.npy".format(args.save_dir, args.sampler, args.eta),Coverage )
    np.save("{}/coverage_se_{}_{}.npy".format(args.save_dir, args.sampler, args.eta), Coverage_se)

    if  args.sampler=='dmala' or args.sampler=='cauchy-digs' or args.sampler=='DiGS' or args.sampler=='HiSS':
        print("Average Acceptance Probability: "+str(np.round(np.mean(sampler.a_s), decimals=3)))
        print("Standard Deviation: "+str(np.round(np.std(sampler.a_s),decimals=3)))





    #generate_transition_heatmap(transitions, args.n_steps, args.thinning,state_labels,t,MAE)

    """
    if sampler.entropy==True :
        generate_sample_distribution(chain_a,t)
        num_samples = 10 * len(chain_a)
        if sampler.gibbs_like_update==False:
            generate_target_distribution(num_samples,model,sampler.eta,t)
        else:
            generate_target_distribution_glu(num_samples,model,sampler.eta,t)
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int, default=1000) #50,000
    parser.add_argument('--thinning', type=int, default=1)   #1
    parser.add_argument('--interval', type=int, default=1) #1000
    parser.add_argument('--viz', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--nchains', type=int, default=10)
    parser.add_argument('--sampler', type=str, default='HiSS')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--eta', type=float, default=10)

    parser.add_argument('--K', type=int, default=1)

    parser.add_argument("--num_cycles", type=int, default=250)
    parser.add_argument("--burnin_budget", type=int, default=200)
    parser.add_argument("--burnin_big_bal", type=float, default=.95)
    parser.add_argument("--burnin_small_bal", type=float, default=.5)
    parser.add_argument("--a_s_cut", type=float, default=0.5)
    parser.add_argument("--burnin_lr", type=float, default=0.5)
    parser.add_argument("--bal_resolution", type=int, default=10)
    parser.add_argument("--initial_balancing_constant", type=float, default=.5)
    parser.add_argument("--big_step", type=float, default=2.0)
    parser.add_argument("--use_manual_EE", action="store_true")
    parser.add_argument("--adapt_every", type=int, default=25)
    parser.add_argument("--big_step_sampling_steps", type=int, default=5)
    parser.add_argument("--small_step", type=float, default=0.2)
    parser.add_argument("--small_bal", type=float, default=0.5)
    parser.add_argument("--big_bal", type=float, default=1.0)
    parser.add_argument("--burnin_adaptive", action="store_true")
    parser.add_argument("--min_lr", default=.011, type=float)

    parser.add_argument("--num_iters", type=int, default=1000)


    parser.add_argument('--save_dir', type=str, default="./bernoulli_sample_data")
    args = parser.parse_args()
    print((args.seed))
    main(args)



