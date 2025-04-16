import argparse
import rbm
import torch
import numpy as np
import samplers
import matplotlib.pyplot as plt
import os
import torchvision

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import tensorflow_probability as tfp
import block_samplers
import time
import pickle
import itertools



def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def Local_Entropy(theta_a, model, eta):
    dim = args.dim ** 2
    A = model.J
    b = model.bias

    lst = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=dim))).to(device)
    f = lambda x: torch.exp((x @ A * x).sum(-1) + torch.sum(b * x, dim=-1))
    energies = f(lst)
    energies = energies / torch.sum(energies)


    # Convert theta_a to tensor if not already
    theta_a_tensor = torch.stack(theta_a)  # Shape: (num_samples, 1, 4)

    # Compute the differences and the second part
    param_diff = lst.unsqueeze(0) - theta_a_tensor  # Shape: (num_samples, 16, 4)
    second_part = torch.sum(param_diff ** 2, dim=-1) / (2 * eta)

    exp_local_entropy = torch.exp(energies.unsqueeze(0) - second_part)
    internal_sum=exp_local_entropy.sum(dim=1)# Shape: (num_samples, 16)
    local_entropy = torch.log(internal_sum)
    # Shape: (num_samples,)

    return local_entropy.numpy()
def get_ess(chain, burn_in):
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv

def get_coverage(chain) :
    chain = np.array(chain)

    # Extract shape details
    p, k, n = chain.shape

    # Total possible states (binary vectors of size n)
    state_space_size = 2 ** n

    # To store coverage for each chain
    coverages = []

    # Iterate over each chain
    for chain_idx in range(k):
        # Extract the chain of shape (p, n)
        single_chain = chain[:, chain_idx, :]

        # Convert each state vector to a tuple for uniqueness
        unique_states = set(tuple(state) for state in single_chain)

        # Calculate coverage as the fraction of unique states observed
        coverage = len(unique_states) / state_space_size
        coverages.append(coverage)

    # Convert to numpy array for calculations
    mean = np.mean(coverages)

    # Calculate standard error of the coverage
    std_error = np.std(coverages) / np.sqrt(len(coverages))

    return mean, std_error


def get_log_rmse(x, gt_mean):
    x = 2. * x - 1.
    residuals = x - gt_mean
    n = x.shape[0]
    c=torch.log(torch.sqrt((residuals ** 2).mean(dim=1)))
    log_rmse=torch.mean(c)
    se=torch.std(c)/np.sqrt(n)



    # Return the logRMSE and its standard error
    return log_rmse.cpu().detach().numpy(), se.cpu().detach().numpy()


def tv(samples):
    gt_probs = np.load("{}/gt_prob_{}_{}.npy".format(args.save_dir, args.dim, args.bias))
    arrs, uniq_cnt = np.unique(samples, axis=0, return_counts=True)
    sample_probs = np.zeros_like(gt_probs)

    for i in range(arrs.shape[0]):
        sample_probs[i] = (uniq_cnt[i] * (1.) - 1.) / samples.shape[0]
    l_dist = np.abs((gt_probs - sample_probs)).sum()


def get_gt_mean(args, model):
    dim = args.dim ** 2
    A = model.J
    b = model.bias
    lst = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=dim))).to(device)
    f = lambda x: torch.exp((x @ A * x).sum(-1) + torch.sum(b * x, dim=-1))
    flst = f(lst)
    plst = flst / torch.sum(flst)

    plt.figure(figsize=(10, 6))
    plt.hist(range(0,2**dim), bins=2**dim, density=True, edgecolor='black', color='pink', weights=plst)
    plt.xlabel('States', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.title('Probability Mass Function (PMF)', fontsize=18)
    plt.ylim(0, max(plst) + 0.005)
    plt.savefig(str('{}/target_hist.png').format(args.save_dir), dpi=300, bbox_inches='tight')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()









    gt_mean = torch.sum(lst * plst.unsqueeze(1).expand(-1, lst.size(1)), 0)
    torch.save(gt_mean.cpu(),
               "{}/gt_mean_dim{}_sigma{}_bias{}.pt".format(args.save_dir, args.dim, args.sigma, args.bias))

    # gt_mean = torch.load("{}/gt_mean_dim{}_sigma{}_bias{}.pt".format(args.save_dir,args.dim,args.sigma,args.bias)).to(device)
    return gt_mean


def main(args):

    makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    model = rbm.LatticeIsingModel(args.dim, args.sigma, args.bias)
    model.to(device)
    gt_mean = get_gt_mean(args, model)

    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.dim, args.dim),
                                                     p, normalize=False, nrow=int(x.size(0) ** .5))
    ess_samples = model.init_sample(args.n_samples).to(device)

    hops = {}
    ess = {}
    times = {}
    chains = {}
    means = {}

    SE=[]

    rmses = {}
    times_list={}
    x0 = model.init_dist.sample((args.n_test_samples,)).to(device)
    temp = args.sampler
    print(temp)

    if temp == 'dim-gibbs':
        sampler = samplers.PerDimGibbsSampler(model.data_dim,n=20)
    elif temp == "rand-gibbs":
        sampler = samplers.PerDimGibbsSampler(model.data_dim,n=20, rand=True)
    elif temp == "lb":
        sampler = samplers.PerDimLB(model.data_dim)
    elif "bg-" in temp:
        block_size = int(temp.split('-')[1])
        sampler = block_samplers.BlockGibbsSampler(model.data_dim, block_size)
    elif "hb-" in temp:
        block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
        sampler = block_samplers.HammingBallSampler(model.data_dim, block_size, hamming_dist)
    elif temp == "GWG":
        sampler = samplers.DiffSampler(model.data_dim, 20,
                                       fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
    elif "GWG-" in temp:
        n_hops = int(temp.split('-')[1])
        sampler = samplers.MultiDiffSampler(model.data_dim, args.sweeps*args.denoise,
                                            approx=True, temp=2., n_samples=n_hops)

    elif temp == "dula":
        sampler = samplers.LangevinSampler(model.data_dim, args.sweeps*args.denoise,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.1,
                                           mh=False)

    elif temp == "DMALA":
        sampler = samplers.LangevinSampler(model.data_dim, args.sweeps*args.denoise,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.2,
                                           mh=True)

    elif temp=='PT+DMALA':
        sampler= samplers.ParallelTemperingLangevinSampler(model.data_dim, n_steps=args.sweeps*args.denoise,
                                           n_chains=5, step_size=0.2, swap_interval=2,
                                           mh=True)

    elif temp == 'ACS':
        sampler = samplers.AutomaticCyclicalSampler(dim=model.data_dim,
            max_val=model.data_dim,
            n_steps=args.sweeps*args.denoise,
            mh=True,
            approx=True,
             multi_hop=False,
            fixed_proposal=False,
            num_cycles=args.num_cycles,
            num_iters=1,
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

    elif temp == 'HiSS':
        sampler = samplers.ModifiedDiGsSampler(model.data_dim, args.sweeps,
                                             fixed_proposal=False, approx=True, multi_hop=False, temp=2.
                                             , mh=True,  step_size=0.2, eta=args.eta, score_sweeps=args.denoise,K=args.K )
        #eta=4
    elif temp == 'digs':
        sampler = samplers.DGLangevinSampler(model.data_dim, args.sweeps,
                                             fixed_proposal=False, approx=True, multi_hop=False, temp=2.,
                                             alpha=np.sqrt(1-args.eta),step_size=0.2, eta=args.eta,mh=False, score_sweeps=args.denoise)

    else:
        raise ValueError("Invalid sampler...")







    x = x0.clone().detach()
    x_a = x
    times[temp] = []
    hops[temp] = []
    chain = []
    chain_a=[]
    Cov=[]
    Cov_Std=[]
    sample=[]
    cur_time = 0.
    mean = torch.zeros_like(x)
    times_list[temp] = []
    st=time.time()
    rmses[temp] = []
    for i in range(args.n_steps):
        # do sampling and time it

        if (sampler.entropy == False):
            if temp!='ACS' and temp!='PT+DMALA':
                xhat = sampler.step(x.detach(), model).detach()
            else:
                xhat = sampler.step(x.detach(), model,i).detach()

        else:
            sample_tuple = sampler.step(x.detach(), x_a.detach(), model)
            xhat = sample_tuple[0].detach()
            xahat = sample_tuple[1].detach()

        cur_time += time.time() - st

        # compute hamming dist
        cur_hops = (x != xhat).float().sum(-1).mean().item()

        # update trajectory
        x = xhat
        sample.append(x)
        if (sampler.entropy == True):
            x_a = xahat
            # print(x_a)

        mean = mean + x
        if i % args.subsample == 0:
            if args.ess_statistic == "dims":
                chain.append(x.cpu().numpy()[0][None])
                if sampler.entropy==True:
                    chain_a.append(x_a)
            else:
                xc = x
                h = (xc != ess_samples[0][None]).float().sum(-1)
                chain.append(h.detach().cpu().numpy()[None])
                if sampler.entropy == True:
                    chain_a.append(x_a)




        if i % args.viz_every == 0 and plot is not None:
            running_time = time.time() - st
            times_list[temp].append(running_time)
            rmse,se = get_log_rmse(mean / (i + 1), gt_mean)
            print(i)
            print(rmse)
            #print(cur_hops)
            rmses[temp].append(rmse)
            SE.append(se)
            cov, cov_se=get_coverage(sample)
            print(cov)
            Cov.append(cov)
            Cov_Std.append(cov_se)

        if i % args.print_every == 0:
            times[temp].append(cur_time)
            hops[temp].append(cur_hops)

    means[temp] = mean / args.n_steps
    chain = np.concatenate(chain, 0)
    chains[temp] = chain
    if not args.no_ess:
        ess[temp] = get_ess(chain, args.burn_in)
        print("ess = {} +/- {}".format(ess[temp].mean(), ess[temp].std()))
    np.save("{}/ising_sample_times_{}.npy".format(args.save_dir, temp), times_list[temp])
    np.save("{}/ising_sample_logrmses_{}.npy".format(args.save_dir, temp), rmses[temp])
    np.save("{}/ising_sample_logrmses_se_{}.npy".format(args.save_dir, temp), SE)
    np.save("{}/ising_sample_coverage_{}.npy".format(args.save_dir, temp), Cov)
    np.save("{}/ising_sample_coverage_se_{}.npy".format(args.save_dir, temp), Cov_Std)


















if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./figs/ising_sample")
    parser.add_argument('--n_steps', type=int, default=2500)
    parser.add_argument('--sampler', type=str, default='PT+DMALA')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--eta', type=float, default=4000000)
    # parser.add_argument('--n_steps', type=int, default=10)
    parser.add_argument('--show_sample', type=int, default=1)



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



    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--n_test_samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1234567)
    # model def
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--sweeps', type=int, default=10)
    parser.add_argument('--denoise', type=int, default=2)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--bias', type=float, default=0.1)

    # logging
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--viz_every', type=int, default=1)

    # for rbm training
    parser.add_argument('--rbm_lr', type=float, default=.001)
    parser.add_argument('--cd', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=100)
    # for ess
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--burn_in', type=float, default=.1)
    parser.add_argument('--ess_statistic', type=str, default="dims", choices=["hamming", "dims"])
    parser.add_argument('--no_ess', action="store_true")
    args = parser.parse_args()

    main(args)
