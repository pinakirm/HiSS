import math
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import random
import torch

torch.cuda.empty_cache()
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn

# Import your data loaders
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

import argparse
from GWG_release import samplers


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# =============================================================================
# ===                      PARAMETER SETUP                            ===
# =============================================================================
EPOCH = 1000 + 1
TEMP = 100  # FIX: Set to a safe, standard value.

parser = argparse.ArgumentParser()
parser.add_argument('--sampler', type=str, default='dmala')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--eta', type=float, default=1)
parser.add_argument('--dataset', type=str, default='hiv')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gsweeps', type=int, default=10)
parser.add_argument('--L', type=int, default=5)
parser.add_argument('--batchsize', type=int, default=-1)
parser.add_argument('--nchains', type=int, default=50)

# --- NEW: Arguments to control the landscape ---
parser.add_argument('--hidden_dim', type=int, default=100, help="Number of hidden units in the BNN.")
parser.add_argument('--sparsity_param', type=float, default=0.0,
                    help="Strength of the sparsity-inducing prior. 0.0 means no prior.")
parser.add_argument('--activation', type=str, default='tanh', choices=['tanh', 'hardtanh'],
                    help="Activation function to use.")

# ACS specific args (kept from your original script)
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

log_dir = f'logs/{args.dataset}/{args.sampler}_h{args.hidden_dim}_s{args.sparsity_param}_seed{args.seed}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print(f"Starting run for sampler: {args.sampler} on dataset: {args.dataset}")
print(f"Log directory: {log_dir}")


# =============================================================================
# ===         FINAL, STABLE & CONTROLLABLE BayesianNN CLASS           ===
# =============================================================================
class BayesianNN(nn.Module):
    def __init__(self, X_train, y_train, batch_size, num_particles, hidden_dim,dim, sparsity_param=0.0, activation='tanh'):
        super(BayesianNN, self).__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.n_features = X_train.shape[1]
        self.hidden_dim = hidden_dim
        self.sparsity_param = sparsity_param
        self.activation_fn = F.tanh if activation == 'tanh' else F.hardtanh
        # Add this in __init__
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.05, requires_grad=False)

    def get_logits(self, inputs, theta):
        """Helper function to get raw logits without the final sigmoid."""
        w1 = theta[:, 0:self.n_features * self.hidden_dim].reshape(-1, self.n_features, self.hidden_dim)
        b1 = theta[:, self.n_features * self.hidden_dim:(self.n_features + 1) * self.hidden_dim].unsqueeze(1)
        w2 = theta[:, (self.n_features + 1) * self.hidden_dim:(self.n_features + 2) * self.hidden_dim].unsqueeze(2)
        b2 = theta[:, -1].reshape(-1, 1, 1)

        inputs_repeated = inputs.unsqueeze(0).repeat(self.num_particles, 1, 1)
        inter = self.activation_fn(torch.bmm(inputs_repeated, w1) + b1)
        out_logit = torch.bmm(inter, w2) + b2
        return out_logit.squeeze(-1)

    def forward_data(self, inputs, theta):
        """This function gets sigmoid probabilities for making predictions."""
        out_logit = self.get_logits(inputs, theta)
        return torch.sigmoid(out_logit)

    def forward(self, theta):
        """The core energy function using a stable loss and optional sparsity prior."""
        theta_mapped = 2. * theta - 1.

        random_idx = random.sample(range(self.X_train.shape[0]), self.batch_size)
        X_batch = self.X_train[random_idx]
        y_batch = self.y_train[random_idx]

        out_logit = self.get_logits(X_batch, theta_mapped)
        y_batch_repeat = y_batch.unsqueeze(0).repeat(self.num_particles, 1)

        log_likelihood = -F.binary_cross_entropy_with_logits(out_logit, y_batch_repeat, reduction='none').mean(dim=1)

        interaction_energy = torch.einsum('bi,ij,bj->b', theta_mapped, self.W, theta_mapped)
        log_likelihood  += interaction_energy

        log_p = log_likelihood

        # --- ADD THE SPARSITY-INDUCING PRIOR ---
        if self.sparsity_param > 0:
            log_prior = -self.sparsity_param * ((theta_mapped + 1).sum(dim=1) / 2.0)
            log_p += log_prior

        return log_p* TEMP


# =============================================================================
# ===            CORRECTED EVALUATION FUNCTIONS                      ===
# =============================================================================
def train_log(model, theta, X_train, y_train):
    with torch.no_grad():
        theta_mapped = 2. * theta - 1.

        # Calculate Log Likelihood (same as forward pass but on full train set)
        logits = model.get_logits(X_train, theta_mapped)
        y_repeat = y_train.unsqueeze(0).repeat(model.num_particles, 1)
        log_p = -F.binary_cross_entropy_with_logits(logits, y_repeat, reduction='none').mean(dim=1)

        # Calculate RMSE
        outputs = torch.sigmoid(logits)
        rmse = (outputs.mean(dim=0) - y_train).pow(2)

        return log_p.mean().cpu().numpy(), rmse.mean().cpu().numpy()


def test_log(model, theta, X_test, y_test):
    with torch.no_grad():
        theta_mapped = 2. * theta - 1.

        # Calculate Log Likelihood
        logits = model.get_logits(X_test, theta_mapped)
        y_repeat = y_test.unsqueeze(0).repeat(model.num_particles, 1)
        log_p = -F.binary_cross_entropy_with_logits(logits, y_repeat)  # Mean over all particles and data points

        # Calculate RMSE
        outputs = torch.sigmoid(logits)
        rmse = (outputs.mean(dim=0) - y_test).pow(2)

        return log_p.mean().cpu().numpy(), np.sqrt(rmse.mean().cpu().numpy())


# =============================================================================
# ===                      MAIN EXECUTION BLOCK                       ===
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading ---
    loaders = {
        'hiv': hiv.load_data, 'compas': cp.load_data, 'aids': aids.load_data,
        'bcancer': bc.load_data, 'creditcard': cc.load_data, 'protein': prot.load_data,
        'wine': wine.load_data, 'mushroom': mushroom.load_data, 'blog': blog.load_data,
        'spam': spam.load_data
    }
    if args.dataset not in loaders:
        print('Dataset not available')
        assert False
    X_train, y_train, X_test, y_test = loaders[args.dataset](get_categorical_info=False)

    # Using 99% for training, 1% for validation (as in original script)
    n = int(0.99 * X_train.shape[0])
    X_val, y_val = X_train[n:], y_train[n:]
    X_train, y_train = X_train[:n], y_train[:n]

    X_train = torch.tensor(X_train).float().to(device)
    X_test = torch.tensor(X_test).float().to(device)
    y_train = torch.tensor(y_train).float().to(device)
    y_test = torch.tensor(y_test).float().to(device)

    # --- Safe Data Normalization ---
    if args.dataset not in ['protein', 'hiv', 'mushroom']:
        X_train_mean, X_train_std = torch.mean(X_train, dim=0), torch.std(X_train, dim=0)
        epsilon = 1e-8
        X_train = (X_train - X_train_mean) / (X_train_std + epsilon)
        X_test = (X_test - X_train_mean) / (X_train_std + epsilon)

    # --- Model and Sampler Initialization ---
    batch_size = X_train.shape[0] if args.batchsize == -1 else args.batchsize

    dim = (X_train.shape[1] + 2) * args.hidden_dim + 1
    theta = torch.bernoulli(torch.ones(args.nchains, dim) * 0.5).to(device)
    theta_a = theta.to(device)

    model = BayesianNN(X_train, y_train, batch_size, args.nchains, args.hidden_dim,dim, args.sparsity_param,
                       args.activation)
    model = model.to(device)


    # Sampler setup...
    # Ensure you have applied the gradient clipping fix in your utils.py!
    sampler_args = {'dim': dim, 'n_steps': args.gsweeps * args.L}
    if args.sampler == 'gwg':
        sampler = samplers.DiffSampler(dim, args.gsweeps*args.L, fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
    elif args.sampler == 'dmala':
        sampler = samplers.LangevinSampler(**sampler_args, approx=True, step_size=args.alpha, mh=True)
    elif args.sampler == 'hiss':
        sampler = samplers.ModifiedDiGsSampler(dim, args.gsweeps, score_sweeps=args.L, eta=args.eta,
                                               step_size=args.alpha)
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

    print("Starting training loop...")
    for epoch in range(EPOCH):
        if (sampler.entropy == False):
            if args.sampler != 'acs' and args.sampler != 'pt+dmala':

                theta_hat = sampler.step(theta.detach(), model).detach()
                theta = theta_hat.data.detach().clone()
            else:
                theta_hat = sampler.step(theta.detach(), model, epoch).detach()
                theta = theta_hat.data.detach().clone()

        else:
            # print(theta)
            sample_tuple = sampler.step(theta.detach(), theta_a.detach(), model)
            theta = sample_tuple[0].data.detach().clone()
            theta_a = sample_tuple[1].data.detach().clone()
        if epoch % 5 == 0:
            test_ll, test_rmse = test_log(model, theta, X_test, y_test)
            test_ll_cllt.append(test_ll)
            test_rmse_cllt.append(test_rmse)

            training_ll, training_rmse = train_log(model, theta, X_train, y_train)
            training_ll_cllt.append(training_ll)
            training_rmse_cllt.append(training_rmse)

            if epoch % 100 == 0:
                print(epoch, 'Training LL:', training_ll, 'Test LL:', test_ll)
                print(epoch, 'Training RMSE:', training_rmse, 'Test RMSE:', test_rmse)
    a = np.array(test_ll_cllt)
    b = np.array(test_rmse_cllt)
    c = np.array(training_ll_cllt)
    d = np.array(training_rmse_cllt)
            # np.save('%s/test_ll.npy'%(log_dir), y)
    print('Test LL Mean +- std ' + str(np.mean(a)) + " " + str(np.std(a)))
    print('Test RMSE Mean +- std ' + str(np.mean(b)) + " " + str(np.std(b)))
    print('Training LL Mean +- std ' + str(np.mean(c)) + " " + str(np.std(c)))
    print('Training RMSE Mean +- std ' + str(np.mean(d)) + " " + str(np.std(d)))


    # Final evaluation...


if __name__ == '__main__':
    main()

