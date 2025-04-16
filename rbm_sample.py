import argparse
import rbm
import torch
import numpy as np
import samplers
import mmd
import matplotlib.pyplot as plt
import os
import multiprocessing as mp

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import utils
import tensorflow_probability as tfp
import block_samplers
import time
import pickle
import itertools

import math



def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_ess(chain, burn_in):
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv


def main(args):
    makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = rbm.BernoulliRBM(args.n_visible, args.n_hidden)
    model.to(device)

    if args.data != "random":
        assert args.n_visible == 784
        train_loader, test_loader, plot, viz = utils.get_data(args)

        init_data = []
        for x, _ in train_loader:
            init_data.append(x)
        init_data = torch.cat(init_data, 0)
        init_mean = init_data.mean(0).clamp(.01, .99)

        model = rbm.BernoulliRBM(args.n_visible, args.n_hidden, data_mean=init_mean)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.rbm_lr)

        # train!
        itr = 0
        for x, _ in train_loader:
            x = x.to(device)
            xhat = model.gibbs_sample(v=x, n_steps=args.cd)

            d = model.logp_v_unnorm(x)
            m = model.logp_v_unnorm(xhat)

            obj = d - m
            loss = -obj.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if itr % args.print_every == 0:
                print("{} | log p(data) = {:.4f}, log p(model) = {:.4f}, diff = {:.4f}".format(itr, d.mean(), m.mean(),
                                                                                               (d - m).mean()))

    else:
        model.W.data = torch.randn_like(model.W.data) * (.05 ** .5)
        model.b_v.data = torch.randn_like(model.b_v.data) * 1.0
        model.b_h.data = torch.randn_like(model.b_h.data) * 1.0
        viz = plot = None

    gt_samples = model.gibbs_sample(n_steps=args.gt_steps, n_samples=args.n_samples + args.n_test_samples, plot=True)
    kmmd = mmd.MMD(mmd.exp_avg_hamming, False)
    gt_samples, gt_samples2 = gt_samples[:args.n_samples], gt_samples[args.n_samples:]
    if plot is not None:
        plot("{}/ground_truth.png".format(args.save_dir), gt_samples2)
    opt_stat = kmmd.compute_mmd(gt_samples2, gt_samples)
    print("gt <--> gt log-mmd", opt_stat, opt_stat.log10())

    probabilities = gt_samples.mean(dim=0)

    # Step 3: Plot the probabilities

    """
    plt.figure(figsize=(15, 5))
    plt.bar(range(probabilities.size(0)), probabilities.numpy(),color='skyblue')
    plt.xlabel('Column Index')
    plt.ylabel('Probability of Success')
    plt.title('Probability of Success for Each Column')
    plt.grid(True)
    #plt.show()
    """


    new_samples = model.gibbs_sample(n_steps=0, n_samples=args.n_test_samples)

    log_mmds = {}
    log_mmds['gibbs'] = []
    ars = {}
    hops = {}
    ess = {}
    times = {}
    chains = {}
    chain = []
    x0 = model.init_dist.sample((args.n_test_samples,)).to(device)
    temp = args.sampler
    if temp == 'dim-gibbs':
        sampler = samplers.PerDimGibbsSampler(args.n_visible)
    elif temp == "rand-gibbs":
        sampler = samplers.PerDimGibbsSampler(args.n_visible, rand=True)
    elif "bg-" in temp:
        block_size = int(temp.split('-')[1])
        sampler = block_samplers.BlockGibbsSampler(args.n_visible, block_size)
    elif "hb-" in temp:
        block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
        sampler = block_samplers.HammingBallSampler(args.n_visible, block_size, hamming_dist)
    elif temp == "GWG":
        sampler = samplers.DiffSampler(args.n_visible, args.gsweeps*args.L,
                                       fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
    elif "gwg-" in temp:
        n_hops = int(temp.split('-')[1])
        sampler = samplers.MultiDiffSampler(args.n_visible, 10,
                                            approx=True, temp=2., n_samples=n_hops)

    elif temp == "DMALA":
        sampler = samplers.LangevinSampler(args.n_visible, args.gsweeps*args.L,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.1,
                                           mh=True)
    elif temp == 'HiSS':
        sampler = samplers.ModifiedDiGsSampler(model.data_dim, args.gsweeps ,
                                               fixed_proposal=False, approx=True, multi_hop=False, temp=2.
                                               , mh=True, step_size=0.1, eta=args.eta, score_sweeps=args.L, K=1)

    elif args.sampler == 'ACS':
        sampler = samplers.AutomaticCyclicalSampler(dim=model.data_dim,
                                                    max_val=model.data_dim,
                                                    n_steps=args.gsweeps * args.L,
                                                    mh=True,
                                                    approx=True,
                                                    multi_hop=False,
                                                    fixed_proposal=False,
                                                    num_cycles=args.num_cycles,
                                                    num_iters=1,
                                                    mean_stepsize=0.1,
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

    elif temp == "dula":
        sampler = samplers.LangevinSampler(args.n_visible, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.1,
                                           mh=False)


    elif temp == 'PT+DMALA':
        sampler = samplers.ParallelTemperingLangevinSampler(model.data_dim, n_steps=args.gsweeps * args.L,
                                                            n_chains=5, step_size=0.1, swap_interval=4,
                                                            mh=True)
    else:
        raise ValueError("Invalid sampler...")





    x = x0.clone().detach()
    x_a = x
    ars[temp] = []
    hops[temp] = []
    times[temp] = []
    tracking_time = []
    logmmd = []
    chain = []



    cur_time = 0.
    st = time.time()
    for i in range(args.n_steps):
        # do sampling and time it

        if (sampler.entropy == False):
            if args.sampler != 'ACS' and args.sampler != 'PT+DMALA':
                xhat = sampler.step(x.detach(), model).detach()
            else:
                xhat = sampler.step(x.detach(), model, i).detach()

        else:
            sample_tuple = sampler.step(x.detach(), x_a.detach(), model)
            xhat = sample_tuple[0].detach()
            xahat = sample_tuple[1].detach()
        cur_time += time.time() - st

        # compute hamming dist
        cur_hops = (x != xhat).float().sum(-1).mean().item()
        cur_hops_std = torch.std(torch.sum((x != xhat).float(),dim=1)).item()/np.sqrt(100)

        # update trajectory
        x = xhat
        if (sampler.entropy == True):
            x_a = xahat

        if i % args.subsample == 0:
            if args.ess_statistic == "dims":
                chain.append(x.cpu().numpy()[0][None])
            else:
                xc = x[0][None]
                h = (xc != gt_samples).float().sum(-1)
                chain.append(h.detach().cpu().numpy()[None])

        if i % args.viz_every == 0 and plot is not None:
            plot("{}/temp_{}_samples_{}_{}.png".format(args.save_dir,args.data, temp, i), x)

        if i % args.print_every == 0:
            hard_samples = x

            stat = kmmd.compute_mmd(hard_samples, gt_samples)
            log_stat = stat.log().item()
            running_time=time.time() - st


            tracking_time.append(running_time)
            logmmd.append(log_stat)
            times[temp].append(cur_time)
            hops[temp].append(cur_hops)

            print("temp {}, itr = {}, log-mmd = {:.4f}, hop-dist = {:.4f}".format(temp, i, log_stat, cur_hops))

    chain = np.concatenate(chain, 0)
    #ess[temp] = get_ess(chain, args.burn_in)
    chains[temp] = chain
    #print("ess = {} +/- {}".format(ess[temp].mean(), ess[temp].std()))
    np.save("{}/rbm_sample_times_{}_{}_{}.npy".format(args.save_dir,args.seed, args.data, temp), tracking_time)
    np.save("{}/rbm_sample_logmmd_{}_{}_{}.npy".format(args.save_dir,args.seed, args.data, temp), logmmd)
    #np.save("{}/rbm_sample_logmmd_se_{}.npy".format(args.save_dir, temp), SE)





    """
    plt.clf()
    for temp in temps:
        plt.plot(log_mmds[temp], label="{}".format(temp))
    plt.legend()
    plt.savefig("{}/logmmd.png".format(args.save_dir))
    plt.clf()
    
    
    
    for temp in temps:
        plt.plot(times[temp],log_mmds[temp], label="{}".format(temp))
    plt.legend()
    plt.savefig("{}/runtime.png".format(args.save_dir))
    """

if __name__ == "__main__":
    potential_datasets = [
        "mnist",
        "fashion",
        "emnist",
        "caltech",
        "omniglot",
        "kmnist",
        "random"
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./figs/rbm_sample")
    parser.add_argument('--data', choices=potential_datasets, type=str, default='mnist')

    parser.add_argument('--sampler', type=str, default='GWG')
    parser.add_argument('--alpha_a', type=float, default=0.001)
    parser.add_argument('--eta', type=float, default=0.4)

    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--burnin', type=int, default=2000)

    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--n_test_samples', type=int, default=100)
    parser.add_argument('--gt_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234567)
    # rbm def
    parser.add_argument('--n_hidden', type=int, default=500)
    parser.add_argument('--n_visible', type=int, default=784)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--viz_every', type=int, default=1000)
    # for rbm training
    parser.add_argument('--gsweeps', type=int, default=10)
    parser.add_argument('--L', type=int, default=4)
    parser.add_argument('--rbm_lr', type=float, default=.001)
    parser.add_argument('--cd', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    # for ess
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--burn_in', type=float, default=.1)
    parser.add_argument('--ess_statistic', type=str, default="dims", choices=["hamming", "dims"])

    parser.add_argument("--num_cycles", type=int, default=20)
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

    args = parser.parse_args()

    main(args)
