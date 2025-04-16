import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import random
import torch
torch.cuda.empty_cache()
from torch.optim import Adam, Adagrad, SGD
from torch.distributions import Normal
from torch.distributions.gamma import Gamma
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import hiv_loader as hiv
import compas_loader as cp
import aids_loader as aids
import bcancer_loader as bc
import mushroom_loader as mushroom
import blog_loader as blog
import spambase_loader as spam
import wine_loader as wine
import protein_loader as prot
import creditcard_loader as cc
import matplotlib.pyplot as plt


import argparse
from GWG_release import samplers



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

EPOCH = 1000+1
TEMP = 100.

parser = argparse.ArgumentParser()
parser.add_argument('--sampler', type=str, default='gwg')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--eta', type=float, default=1)
parser.add_argument('--dataset', type=str, default='creditcard')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gsweeps', type=int, default=10)
parser.add_argument('--L', type=float, default=5)
parser.add_argument('--batchsize', type=int, default=-1)
parser.add_argument('--nchains', type=int, default=50)

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

args = parser.parse_args()

setup_seed(args.seed)

log_dir = 'logs/%s/%s_%d_%d'%(args.dataset, args.sampler, args.batchsize, args.seed)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print(args.sampler)


class BayesianNN(nn.Module):
    def __init__(self, X_train, y_train, batch_size, num_particles, hidden_dim):
        super(BayesianNN, self).__init__()
        #self.lambda_prior = Gamma(torch.tensor(1., device=device), torch.tensor(1 / 0.1, device=device))
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.n_features = X_train.shape[1] 
        self.hidden_dim = hidden_dim

    def forward_data(self, inputs, theta):
        # Unpack theta

        w1 = theta[:, 0:self.n_features * self.hidden_dim].reshape(-1, self.n_features, self.hidden_dim)
        b1 = theta[:, self.n_features * self.hidden_dim:(self.n_features + 1) * self.hidden_dim].unsqueeze(1)
        w2 = theta[:, (self.n_features + 1) * self.hidden_dim:(self.n_features + 2) * self.hidden_dim].unsqueeze(2)
        b2 = theta[:, -1].reshape(-1, 1, 1)

        # num_particles times of forward
        inputs = inputs.unsqueeze(0).repeat(self.num_particles, 1, 1)
        inter = F.tanh(torch.bmm(inputs, w1) + b1)
        #print(inter.shape, w2.shape, b2.shape, self.hidden_dim, (self.n_features + 1) * self.hidden_dim)
        out_logit = torch.bmm(inter, w2) + b2
        out = out_logit.squeeze()
        out = torch.sigmoid(out)

        return out

    def forward(self, theta):
        theta = 2. * theta - 1.
        model_w = theta[:, :]
        # w_prior should be decided based on current lambda (not sure)
        w_prior = Normal(0., 1.)

        random_idx = random.sample([i for i in range(self.X_train.shape[0])], self.batch_size)
        X_batch = self.X_train[random_idx]
        y_batch = self.y_train[random_idx]

        outputs = self.forward_data(X_batch[:, :], theta)  # [num_particles, batch_size]
        y_batch_repeat = y_batch.unsqueeze(0).repeat(self.num_particles, 1)
        log_p_data = (outputs - y_batch_repeat).pow(2) 
        log_p_data = (-1.)*log_p_data.mean(dim=1)*TEMP

        #log_p0 = w_prior.log_prob(model_w.t()).sum(dim=0)
        #log_p = log_p0 + log_p_data  # (8) in paper
        log_p = log_p_data
        
        return log_p



def train_log(model, theta, X_test, y_test):
    with torch.no_grad():
        theta = 2. * theta - 1.
        model_w = theta[:, :]

        outputs = model.forward_data(X_test[:, :], theta)  # [num_particles, batch_size]
        y_batch_repeat = y_test.unsqueeze(0).repeat(model.num_particles, 1)
        log_p_data = (outputs - y_batch_repeat).pow(2) 
        log_p_data = (-1.)*log_p_data.mean(dim=1)

        #log_p0 = w_prior.log_prob(model_w.t()).sum(dim=0)
        #log_p = log_p0 + log_p_data / X_test.shape[0]  # (8) in paper
        log_p = log_p_data

        rmse = (outputs.mean(dim=0) - y_test).pow(2) 
        
        return log_p.mean().cpu().numpy(), rmse.mean().cpu().numpy()

def test_log(model, theta, X_test, y_test):
    with torch.no_grad():
        theta = 2. * theta - 1.
        model_w = theta[:, :]
        w_prior = Normal(0., 1.)

        outputs = model.forward_data(X_test[:, :], theta)  # [num_particles, batch_size]
        log_p_data = (outputs.mean(dim=0) - y_test).pow(2) 
        log_p_data = (-1.)*log_p_data.mean()

        log_p = log_p_data

        rmse = (outputs.mean(dim=0) - y_test).pow(2) 
        
        return log_p.mean().cpu().numpy(), np.sqrt(rmse.mean().cpu().numpy())



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if args.dataset == 'hiv':
        X_train, y_train, X_test, y_test = hiv.load_data(get_categorical_info=False)
    elif args.dataset == 'compas':
        X_train, y_train, X_test, y_test = cp.load_data(get_categorical_info=False)
    elif args.dataset == 'aids':
        X_train, y_train, X_test, y_test = aids.load_data(get_categorical_info=False)
    elif args.dataset == 'bcancer':
        X_train, y_train, X_test, y_test = bc.load_data()
    elif args.dataset == 'creditcard':
        X_train, y_train, X_test, y_test = cc.load_data(get_categorical_info=False)
    elif args.dataset == 'protein':
        X_train, y_train, X_test, y_test = prot.load_data()
    elif args.dataset == 'wine':
        X_train, y_train, X_test, y_test = wine.load_data()
    elif args.dataset == 'mushroom':
        X_train, y_train, X_test, y_test = mushroom.load_data(get_categorical_info=False)
    elif args.dataset == 'blog':
        X_train, y_train, X_test, y_test = blog.load_data(get_categorical_info=False)
    elif args.dataset == 'spam':
        X_train, y_train, X_test, y_test = spam.load_data(get_categorical_info=False)
    else:
        print('Not Available')
        assert False


    n = X_train.shape[0]
    n = int(0.99*n)
    X_val = X_train[n:, :]
    y_val = y_train[n:]
    X_train = X_train[:n, :]
    y_train = y_train[:n]

    feature_num = X_train.shape[1]
    X_train = torch.tensor(X_train).float().to(device)
    X_test = torch.tensor(X_test).float().to(device)
    X_val = torch.tensor(X_val).float().to(device)
    y_train = torch.tensor(y_train).float().to(device)
    y_test = torch.tensor(y_test).float().to(device)
    y_val = torch.tensor(y_val).float().to(device)

    if args.dataset != 'protein':
        X_train_mean, X_train_std = torch.mean(X_train[:, :], dim=0), torch.std(X_train[:, :], dim=0)
        X_train[:, :] = (X_train [:, :]- X_train_mean) / X_train_std
        X_test[:, :] = (X_test[:, :] - X_train_mean) / X_train_std


    if args.batchsize == -1:
        num_particles, batch_size, hidden_dim = args.nchains, X_train.shape[0], 100 # 500 for others, 100 for blog
    else:
        num_particles, batch_size, hidden_dim = args.nchains, args.batchsize, 100


    model = BayesianNN(X_train, y_train, batch_size, num_particles, hidden_dim)
    model = model.to(device)

    theta = torch.cat([torch.zeros([num_particles, (X_train.shape[1] + 2) * hidden_dim + 1], device=device).normal_(0,
                                                                                                                    math.sqrt(
                                                                                                                        0.01))])
    theta = torch.bernoulli(torch.ones_like(theta) * 0.5).to(device)
    theta_a = theta.to(device)
    #print(theta.shape)
    dim = theta.shape[1]
    
    if args.sampler == 'gibbs':
        sampler = samplers.PerDimGibbsSampler(dim, rand=True)
    elif args.sampler == 'gwg':
        sampler = samplers.DiffSampler(dim, args.gsweeps*args.L, fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
    elif args.sampler == 'dmala':
        sampler = samplers.LangevinSampler(dim, args.gsweeps*args.L,fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=args.alpha, mh=True)
    elif args.sampler == 'hiss':
        sampler = samplers.ModifiedDiGsSampler(dim, args.gsweeps,
                                               fixed_proposal=False, approx=True, multi_hop=False, temp=2., mh=True, step_size=args.alpha, eta=args.eta, score_sweeps=args.L, K=1)
    elif args.sampler == 'acs':
        sampler = samplers.AutomaticCyclicalSampler(dim=dim,
                                                    max_val=dim,
                                                    n_steps=args.gsweeps * args.L,
                                                    mh=True,
                                                    approx=True,
                                                    multi_hop=False,
                                                    fixed_proposal=False,
                                                    num_cycles=args.num_cycles,
                                                    num_iters=1,
                                                    mean_stepsize=args.alpha,
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
    elif args.sampler=='pt+dmala':
        sampler= samplers.ParallelTemperingLangevinSampler(dim, n_steps=args.gsweeps*args.L,
                                           n_chains=4, step_size=args.alpha, swap_interval=20,
                                           mh=True)
    else:
        print('Not Available')
        assert False
    
    training_ll_cllt = []
    training_rmse_cllt = []
    test_ll_cllt = []
    test_rmse_cllt = []


    for epoch in range(EPOCH):
        if (sampler.entropy == False):
            if args.sampler!='acs' and args.sampler!='pt+dmala':
                theta_hat = sampler.step(theta.detach(), model).detach()
                theta = theta_hat.data.detach().clone()
            else:
                theta_hat = sampler.step(theta.detach(), model,epoch).detach()
                theta = theta_hat.data.detach().clone()


        else:
            #print(theta)
            sample_tuple = sampler.step(theta.detach(), theta_a.detach(), model)
            theta = sample_tuple[0].data.detach().clone()
            theta_a = sample_tuple[1].data.detach().clone()

        if epoch % 5 == 0:
            training_ll, training_rmse = train_log(model, theta, X_train, y_train)
            training_ll_cllt.append(training_ll)
            training_rmse_cllt.append(training_rmse)

            test_ll, test_rmse = test_log(model, theta, X_test, y_test)
            test_ll_cllt.append(test_ll)
            test_rmse_cllt.append(test_rmse)
            if epoch % 100 == 0:
                print(epoch, 'Training LL:', training_ll, 'Test LL:', test_ll)
                print(epoch, 'Training RMSE:', training_rmse, 'Test RMSE:', test_rmse)
    a=np.array(test_ll_cllt)
    b=np.array(test_rmse_cllt)
    c=np.array(training_ll_cllt)
    d=np.array(training_rmse_cllt)
    #np.save('%s/test_ll.npy'%(log_dir), y)
    print('Test LL Mean +- std ' + str(np.mean(a)) + " " + str(np.std(a)))
    print('Test RMSE Mean +- std ' + str(np.mean(b)) + " " + str(np.std(b)))
    print('Training LL Mean +- std ' + str(np.mean(c)) + " " + str(np.std(c)))
    print('Training RMSE Mean +- std ' + str(np.mean(d)) + " " + str(np.std(d)))



    

if __name__ == '__main__':
    main()
