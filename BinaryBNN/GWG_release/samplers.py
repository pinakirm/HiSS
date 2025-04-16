import torch
import torch.nn as nn
import torch.distributions as dists
import GWG_release.utils as utils
import numpy as np
import math


class LangevinSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=0.2, mh=True):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size  #rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp
        self.mh = mh
        self.entropy = False

    def step(self, x, model):
        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []
        
        EPS = 1e-10
        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            term2 = 1./(2*self.step_size) # for binary {0,1}, the L2 norm is always 1        
            flip_prob = torch.exp(forward_delta-term2)/(torch.exp(forward_delta-term2)+1)
            rr = torch.rand_like(x_cur)
            ind = (rr<flip_prob)*1
            x_delta = (1. - x_cur)*ind + x_cur * (1. - ind)
            if self.mh:
                probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs+EPS),dim=-1)
                reverse_delta = self.diff_fn(x_delta, model)
                flip_prob = torch.exp(reverse_delta-term2)/(torch.exp(reverse_delta-term2)+1)
                probs = flip_prob*ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs+EPS),dim=-1)
                
                m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            else:
                x_cur = x_delta
        return x_cur


class ModifiedDiGsScoreSamp(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., alpha=0.2,
                 mh=True, eta=10 ** 4):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.alpha = alpha
        self.eta = eta  # flatness
        self.entropy = True

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.agrad = lambda eta, x, x_a: utils.auxillary_gradient_function(eta, x, x_a)

        self.mh = mh
        self.a_s = []
        self.hops = []

    def step(self, x, x_a, model):

        x_cur = x
        x_a_cur = x_a

        EPS = 1e-10
        for i in range(self.n_steps):

            grad_x_cur = self.diff_fn(x_cur, model)

            grad_x_cur_modified = grad_x_cur + (
                        torch.tanh((x_a_cur - x_cur) / 2 * self.eta) / (self.eta * self.temp)) * -(2 * x_cur - 1)

            term2 = 1. / (2 * self.alpha)  # for binary {0,1}, the L2 norm is always 1
            flip_prob = torch.exp(grad_x_cur_modified - term2) / (torch.exp(grad_x_cur_modified - term2) + 1)  # softmax

            rr = torch.rand_like(x_cur)
            ind = (rr < flip_prob) * 1
            x_delta = (1. - x_cur) * ind + x_cur * (1. - ind)

            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                grad_x_delta = self.diff_fn(x_delta, model)

                grad_x_delta_modified = grad_x_delta + (
                            torch.tanh((x_a_cur - x_delta) / 2 * self.eta) / (self.eta * self.temp)) * -(
                            2 * x_delta - 1)

                flip_prob = torch.exp(grad_x_delta_modified - term2) / (torch.exp(grad_x_delta_modified - term2) + 1)
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)

                m_term = (model(x_delta).squeeze() - 2 * (
                    torch.log(torch.cosh((x_a_cur - x_delta) / 2 * self.eta)).sum(dim=1))) - (
                                 model(x_cur).squeeze() - 2 * (
                             torch.log(torch.cosh((x_a_cur - x_cur) / 2 * self.eta)).sum(dim=1)))

                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

            else:
                x_cur = x_delta

        return (x_cur, x_a_cur)  # Update x, not x_a


class ModifiedDiGsSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=0.2,
                 mh=True, K=1, step_size_a=0.5, eta=10 ** 4, score_sweeps=50, n=1):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.chains = n
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        self.eta = eta  # flatness
        self.entropy = True

        self.L = score_sweeps
        self.K = K
        self.mh = mh
        self.score_based_sampler = ModifiedDiGsScoreSamp(self.dim, self.L, fixed_proposal=False, approx=True,
                                                         multi_hop=False,
                                                         temp=2., alpha=self.step_size, eta=self.eta, mh=self.mh)


        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.agrad = lambda eta, x, x_a: utils.auxillary_gradient_function(eta, x, x_a)

        self.a_s = []
        self.hops = []

    def step(self, x, x_a, model):

        x_cur = x
        x_a_cur = x_a
        EPS = 1e-10

        for i in range(self.n_steps):

            for k in range(self.K):
                uniform = torch.rand_like(x_cur)
                logistic_noise = torch.log(uniform / (1 - uniform))
                x_a_cur = x_cur + self.eta * logistic_noise


                """
                probs = torch.sigmoid(x_a_cur)  # Probabilities for each dimension
                rand_vals = torch.rand_like(probs)
                x_delta = (rand_vals < probs).float()  #
                """

                # Compute stabilized diff values
                diff_0 = (x_a_cur - 0) / (2 * self.eta)  # Energy difference for 0
                diff_1 = (x_a_cur - 1) / (2 * self.eta)  # Energy difference for 1

                """
                if torch.isnan(diff_0).any() or torch.isnan(diff_1).any():
                    raise ValueError("NaN detected in diff_0 or diff_1!")
                if torch.isinf(diff_0).any() or torch.isinf(diff_1).any():
                    raise ValueError("Infinite values detected in diff_0 or diff_1!")
                """

                energy_0 = -2 * torch.log(torch.cosh(diff_0) + EPS)  # Energy for 0
                energy_1 = -2 * torch.log(torch.cosh(diff_1) + EPS)

                """
                if torch.isinf(energy_0).any() or torch.isinf(energy_1).any():
                    raise ValueError("Overflow detected in torch.cosh(diff)!")
                """
                # Stack and clamp energies
                energies = torch.stack([energy_0, energy_1], dim=2)  # Shape: (batch_size, dim, 2)
                energies = torch.clamp(energies, min=-1e10, max=1e10)

                # Compute probabilities using log-sum-exp trick
                probs = torch.softmax(energies, dim=2)

                # Clamp probabilities to avoid invalid values
                probs = torch.clamp(probs, min=EPS, max=1.0)

                # Validate probabilities
                if torch.isnan(probs).any() or not torch.isfinite(probs).all():
                    raise ValueError("Invalid probabilities detected in probs!")

                # Sample binary state for each coordinate
                categorical_dist = torch.distributions.Categorical(probs=probs)
                sampled_indices = categorical_dist.sample()  # Shape: (batch_size, dim)
                x_delta = sampled_indices.float()  # Final sampled binary states
                if k!=self.K-1:
                    x_cur=x_delta

                #cur_hops = (x_cur != x_delta).float().sum(-1).mean().item()
                #print(cur_hops)

            if self.mh:
                q_forward = (x_a_cur - x_cur) / (2 * self.eta)
                lp_forward = -2 * torch.log(torch.cosh(q_forward) + EPS).sum(dim=1)

                q_reverse = (x_a_cur - x_delta) / (2 * self.eta)
                lp_reverse = -2 * torch.log(torch.cosh(q_reverse) + EPS).sum(dim=1)

                m_term = (model(x_delta).squeeze() - 2 * torch.log(
                    torch.cosh((x_a_cur - x_delta) / (2 * self.eta))).sum(dim=1)) - (
                                     model(x_cur).squeeze() - 2 * torch.log(
                                 torch.cosh((x_a_cur - x_cur) / (2 * self.eta))).sum(
                                 dim=1))

                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

            else:
                x_cur = x_delta

            t = self.score_based_sampler.step(x_cur.detach(), x_a_cur.detach(), model)
            x_cur = t[0].detach()

        return (x_cur, x_a_cur)

# Gibbs-With-Gradients for binary data
class DiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=1.0):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        self.entropy=False
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp


    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        if self.multi_hop:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.Bernoulli(probs=delta.sigmoid() * self.step_size)
                for i in range(self.n_steps):
                    changes = cd.sample()
                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.Bernoulli(logits=(forward_delta * 2 / self.temp))
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes).sum(-1)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)


                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.Bernoulli(logits=(reverse_delta * 2 / self.temp))

                    lp_reverse = cd_reverse.log_prob(changes).sum(-1)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                    m_terms.append(m_term.mean().item())
                    prop_terms.append((lp_reverse - lp_forward).mean().item())
                self._ar = np.mean(a_s)
                self._mt = np.mean(m_terms)
                self._pt = np.mean(prop_terms)
        else:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.OneHotCategorical(logits=delta)
                for i in range(self.n_steps):
                    changes = cd.sample()

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.OneHotCategorical(logits=forward_delta)
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

                    lp_reverse = cd_reverse.log_prob(changes)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

        return x_cur


# Gibbs-With-Gradients variant which proposes multiple flips per step
class MultiDiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1., n_samples=1):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        self.n_samples = n_samples
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp


    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            cd_forward = dists.OneHotCategorical(logits=forward_delta)
            changes_all = cd_forward.sample((self.n_samples,))

            lp_forward = cd_forward.log_prob(changes_all).sum(0)

            changes = (changes_all.sum(0) > 0.).float()

            x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
            self._phops = (x_delta != x).float().sum(-1).mean().item()

            reverse_delta = self.diff_fn(x_delta, model)
            cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

            lp_reverse = cd_reverse.log_prob(changes_all).sum(0)

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).mean().item()
        return x_cur


class PerDimGibbsSampler(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 1.
        self.rand = rand
        self.entropy=False

    def step(self, x, model):
        sample = x.clone()
        lp_keep = model(sample).squeeze()
        if self.rand:
            changes = dists.OneHotCategorical(logits=torch.zeros((self.dim,))).sample((x.size(0),)).to(x.device)
        else:
            changes = torch.zeros((x.size(0), self.dim)).to(x.device)
            changes[:, self._i] = 1.

        sample_change = (1. - changes) * sample + changes * (1. - sample)

        lp_change = model(sample_change)#.squeeze()

        lp_update = lp_change - lp_keep
        update_dist = dists.Bernoulli(logits=lp_update)
        updates = update_dist.sample()
        sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
        self.changes[self._i] = updates.mean()
        self._i = (self._i + 1) % self.dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


class PerDimMetropolisSampler(nn.Module):
    def __init__(self, dim, n_out, rand=False):
        super().__init__()
        self.dim = dim
        self.n_out = n_out
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 0.
        self.rand = rand

    def step(self, x, model):
        if self.rand:
            i = np.random.randint(0, self.dim)
        else:
            i = self._i

        logits = []
        ndim = x.size(-1)

        for k in range(ndim):
            sample = x.clone()
            sample_i = torch.zeros((ndim,))
            sample_i[k] = 1.
            sample[:, i, :] = sample_i
            lp_k = model(sample).squeeze()
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        dist = dists.OneHotCategorical(logits=logits)
        updates = dist.sample()
        sample = x.clone()
        sample[:, i, :] = updates
        self._i = (self._i + 1) % self.dim
        self._hops = ((x != sample).float().sum(-1) / 2.).sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


# Gibbs-With-Gradients for categorical data
class DiffSamplerMultiDim(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1.):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function_multi_dim(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function_multi_dim(x, m) / self.temp

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []


        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            # make sure we dont choose to stay where we are!
            forward_logits = forward_delta - 1e9 * x_cur
            #print(forward_logits)
            cd_forward = dists.OneHotCategorical(logits=forward_logits.view(x_cur.size(0), -1))
            changes = cd_forward.sample()

            # compute probability of sampling this change
            lp_forward = cd_forward.log_prob(changes)
            # reshape to (bs, dim, nout)
            changes_r = changes.view(x_cur.size())
            # get binary indicator (bs, dim) indicating which dim was changed
            changed_ind = changes_r.sum(-1)
            # mask out cuanged dim and add in the change
            x_delta = x_cur.clone() * (1. - changed_ind[:, :, None]) + changes_r

            reverse_delta = self.diff_fn(x_delta, model)
            reverse_logits = reverse_delta - 1e9 * x_delta
            cd_reverse = dists.OneHotCategorical(logits=reverse_logits.view(x_delta.size(0), -1))
            reverse_changes = x_cur * changed_ind[:, :, None]

            lp_reverse = cd_reverse.log_prob(reverse_changes.view(x_delta.size(0), -1))

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None, None] + x_cur * (1. - a[:, None, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).sum(-1).mean().item()
        return x_cur


class GibbsSampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))

    def step(self, x, model):
        sample = x.clone()
        for i in range(self.dim):
            lp_keep = model(sample).squeeze()

            xi_keep = sample[:, i]
            xi_change = 1. - xi_keep
            sample_change = sample.clone()
            sample_change[:, i] = xi_change

            lp_change = model(sample_change).squeeze()

            lp_update = lp_change - lp_keep
            update_dist = dists.Bernoulli(logits=lp_update)
            updates = update_dist.sample()
            sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
            self.changes[i] = updates.mean()
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.

class AutomaticCyclicalSampler(nn.Module):
    def __init__(
            self,
            dim,
            device,
            num_cycles,
            num_iters,
            n_steps=10,
            approx=False,
            multi_hop=False,
            fixed_proposal=False,
            temp=1.0,
            mean_stepsize=0.2,
            mh=True,
            initial_balancing_constant=1,
            burnin_adaptive=False,
            burnin_budget=500,
            burnin_lr=0.5,
            sbc=False,
            big_step=None,
            big_bal=None,
            small_step=None,
            small_bal=None,
            iter_per_cycle=None,
            min_lr=None,
            a_s_cut=None,
            **kwargs
    ):
        super().__init__()
        self.device = device
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = mean_stepsize
        self.initial_step_size = mean_stepsize
        self.num_cycles = num_cycles
        self.num_iters = num_iters
        self.burnin_adaptive = burnin_adaptive
        self.burnin_budget = burnin_budget
        self.burnin_lr = burnin_lr
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m)
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m)
        self.mh = mh
        self.a_s = []
        self.hops = []
        self.initial_balancing_constant = initial_balancing_constant
        self.balancing_constant = initial_balancing_constant
        self.bal = initial_balancing_constant
        if iter_per_cycle and sbc:
            self.iter_per_cycle = iter_per_cycle
        else:
            print(self.num_iters)
            print(self.num_cycles)
            self.iter_per_cycle = math.ceil(self.num_iters / self.num_cycles)
        self.step_sizes = self.calc_stepsizes(self.step_size)
        self.diff_values = []
        self.flip_probs = []
        self.D = None
        self.balancing_constants = self.calc_balancing_constants(self.initial_balancing_constant)
        if sbc and big_step and big_bal and small_step and small_bal:
            self.sbc = sbc
            self.big_step = big_step
            self.small_step = small_step
            self.big_bal = big_bal
            self.small_bal = small_bal
            self.step_sizes = [big_step] + [small_step] * (self.iter_per_cycle - 1)
            self.balancing_constants = [big_bal] + [small_bal] * (
                    self.iter_per_cycle - 1
            )

        else:
            self.sbc = False
        self.min_lr = min_lr
        if self.min_lr:
            self.min_lr_cutoff()
        self.a_s_cut = a_s_cut
        self.track_terms = False
        self.t1 = None
        self.t2 = None
        self.counts = 0
        self.entropy=False



    def get_name(self):
        if self.mh:
            base = "acs"
        else:
            base = "acs_no_mh"
        if self.burnin_adaptive:
            name = f"{base}_cycles_{self.num_cycles}"
            name = (
                    name
                    + f"_budget_{self.burnin_budget}_lr_{self.burnin_lr}_a_s_cut_{self.a_s_cut}"
            )
        else:
            name = f"{base}_cycles_{self.num_cycles}"
            name = (
                    name
                    + f"_stepsize_{self.initial_step_size}_initbal_{self.balancing_constant}"
            )
        if self.sbc:
            if self.burnin_adaptive:
                name = base + f"_adaptive_lr_{self.burnin_lr}_a_s_cut_{self.a_s_cut}"
            else:
                name = (
                        base
                        + f"_cycle_length_{self.iter_per_cycle}_big_s_{self.big_step}_b_{self.big_bal}_small_s_{self.small_step}_b_{self.small_bal}"
                )
        if self.min_lr:
            name += f"_min_lr_{self.min_lr}"
        return name

    def calc_balancing_constants(self, init_bal):
        res = []
        total_iter = self.iter_per_cycle
        for k_iter in range(total_iter):
            inner = (np.pi * k_iter) / total_iter
            cur_balancing_constant = (init_bal - 0.5) / 2 * (np.cos(inner)) + (
                    init_bal + 0.5
            ) / 2
            res.append(cur_balancing_constant)
        res = torch.tensor(res, device=self.device)
        return res

    def min_lr_cutoff(self):
        if self.mh:
            min_step = 0.2
        else:
            min_step = 0.1
        for i in range(len(self.step_sizes)):
            if self.step_sizes[i] <= min_step:
                self.step_sizes[i] = min_step
                self.balancing_constants[i] = 0.5

    def calc_stepsizes(self, mean_step):
        res = []
        total_iter = self.iter_per_cycle
        for k_iter in range(total_iter):
            inner = (np.pi * k_iter) / total_iter
            cur_step_size = mean_step * (np.cos(inner) + 1)
            step_size = cur_step_size
            res.append(step_size)
        res = torch.tensor(res, device=self.device)
        return res

    def adapt_big_step(
            self,
            x_init,
            model,
            budget,
            init_big_step,
            init_big_bal,
            lr,
            test_steps,
            a_s_cut,
            use_dula,
            bdmala=None,
    ):
        if bdmala is None:
            bdmala = LangevinSampler(
                dim=self.dim,
                n_steps=self.n_steps,
                approx=self.approx,
                multi_hop=self.multi_hop,
                fixed_proposal=self.fixed_proposal,
                step_size=1,
                mh=True,
                bal=init_big_bal,
            )

        (
            x_cur,
            alpha_max,
            alpha_max_metrics,
            _,
        ) = estimate_alpha_max(
            model=model,
            bdmala=bdmala,
            a_s_cut=a_s_cut,
            init_bal=init_big_bal,
            test_steps=test_steps,
            budget=budget,
            init_step_size=init_big_step,
            x_init=x_init,
            use_dula=use_dula,
            lr=lr,
        )
        self.step_sizes[0] = alpha_max
        self.balancing_constants[0] = init_big_bal
        return x_cur, alpha_max, alpha_max_metrics

    def adapt_small_step(
            self,
            x_init,
            model,
            budget,
            init_small_step,
            init_small_bal,
            lr,
            test_steps,
            a_s_cut,
            use_dula,
            bdmala=None,
    ):
        if bdmala is None:
            bdmala = LangevinSampler(
                dim=self.dim,
                n_steps=self.n_steps,
                approx=self.approx,
                multi_hop=self.multi_hop,
                fixed_proposal=self.fixed_proposal,
                step_size=1,
                mh=True,
                bal=init_small_bal,
            )
        x_cur, alpha_min, alpha_min_metrics, _ = estimate_alpha_min(
            model=model,
            bdmala=bdmala,
            x_cur=x_init,
            budget=budget,
            init_step_size=init_small_step,
            test_steps=test_steps,
            lr=lr,
            a_s_cut=a_s_cut,
            init_bal=init_small_bal,
        )
        for i in range(1, len(self.step_sizes)):
            self.step_sizes[i] = alpha_min
            self.balancing_constants[i] = init_small_bal
        return x_cur, alpha_min, alpha_min_metrics

    def tuning_alg(
            self,
            dula_x_init,
            model,
            budget,
            init_big_step,
            init_small_step,
            init_big_bal=0.95,
            init_small_bal=0.5,
            a_s_cut=0.5,
            test_steps=10,
            lr=0.5,
            step_zoom_res=5,
            step_size_pair=None,
            x_init_to_use="bal",
            bal_resolution=3,
            use_bal_cyc=False,
            dula_burnin=50,
            acs_burnin=0
    ):

        bdmala = LangevinSampler(
            dim=self.dim,
            n_steps=self.n_steps,
            approx=self.approx,
            multi_hop=self.multi_hop,
            fixed_proposal=self.fixed_proposal,
            step_size=1,
            mh=True,
            bal=init_small_bal,
        )
        # pre burn in
        # if self.norm_mterm:
        #     bdmala.norm_mterm = self.norm_mterm

        bdmala.D = self.D
        bdmala.mh = False
        bdmala.step_size = 5
        bdmala.bal = init_big_bal
        bal_x_init = dula_x_init
        for i in range(dula_burnin):
            dula_x_init = bdmala.step(dula_x_init, model).detach()
        bdmala.mh = True
        self.mh = True
        for i in range(acs_burnin):
            dula_x_init = self.step(dula_x_init, model, i).detach()
        total_res = {}
        possible_x_inits = [dula_x_init]
        # estimating alpha min
        use_dula = not self.mh
        if bdmala.t1 is not None:
            t1 = bdmala.t1 / bdmala.counts
            t1 = t1.mean(axis=0)
            D = torch.diag(t1)
        alpha_min_x_init, alpha_min, alpha_min_metrics, itr = estimate_alpha_min(
            model=model,
            bdmala=bdmala,
            x_cur=dula_x_init,
            budget=budget // 2,
            init_step_size=init_small_step,
            test_steps=test_steps,
            lr=lr,
            a_s_cut=a_s_cut,
            init_bal=init_small_bal,
            use_dula=use_dula,
        )
        possible_x_inits.append(alpha_min_x_init)
        (alpha_max_x_init, alpha_max, alpha_max_metrics, itrr) = estimate_alpha_max(
            model=model,
            bdmala=bdmala,
            a_s_cut=a_s_cut,
            init_bal=init_big_bal,
            test_steps=test_steps,
            budget=budget // 2,
            init_step_size=init_big_step,
            x_init=alpha_min_x_init,
            use_dula=use_dula,
        )

        init_big_bal = bdmala.bal
        possible_x_inits.append(alpha_max_x_init)
        total_res["alpha_max_metrics"] = alpha_max_metrics
        total_res["alpha_min_metrics"] = alpha_min_metrics

        opt_steps = self.calc_stepsizes(alpha_max / 2)
        for i in range(len(opt_steps)):
            if opt_steps[i] < alpha_min:
                break
        bal_x_init, opt_bal, bal_metrics = estimate_opt_bal(
            model=model,
            bdmala=bdmala,
            x_init=dula_x_init,
            init_bal=init_big_bal,
            opt_steps=opt_steps[:i],
            est_resolution=bal_resolution,
            test_steps=test_steps,
            use_dula=use_dula,
            init_small_bal=init_small_bal
        )
        possible_x_inits.append(bal_x_init)

        while i < len(opt_steps):
            opt_steps[i] = alpha_min
            opt_bal.append(init_small_bal)
            i += 1
        self.balancing_constants = opt_bal
        self.step_sizes = opt_steps
        total_res["bal_metrics"] = bal_metrics
        print("step sizes: \n")
        print(self.step_sizes)
        print("\n")
        print("bal: \n")
        print(self.balancing_constants)
        return possible_x_inits[-1], total_res

    def step(self, x, model, k_iter, return_diff=False):
        x_cur = x

        m_terms = []
        prop_terms = []

        EPS = 1e-10
        step_size = self.step_sizes[k_iter % self.iter_per_cycle]
        balancing_constant = self.balancing_constants[k_iter % self.iter_per_cycle]

        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            forward_delta_bal = forward_delta * balancing_constant
            if self.D is not None:
                term2 = self.D * (1.0 / (2 * step_size))
            else:
                term2 = 1.0 / (2 * step_size)  # for binary {0,1}, the L2 norm is always 1
            if self.track_terms:
                if self.t1 is None:
                    self.t1 = forward_delta_bal
                else:
                    self.t1 += forward_delta_bal
                self.counts += 1
            flip_prob = torch.exp(
                forward_delta_bal - term2
            ) / (
                                torch.exp(forward_delta_bal - term2)
                                + 1
                        )
            self.flip_probs.append(flip_prob.detach().sum(axis=-1).mean().item())
            rr = torch.rand_like(x_cur)
            ind = (rr < flip_prob) * 1
            x_delta = (1.0 - x_cur) * ind + x_cur * (1.0 - ind)
            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1.0 - ind)
                lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                reverse_delta = self.diff_fn(x_delta, model) * balancing_constant
                flip_prob = torch.exp(reverse_delta - term2) / (
                        torch.exp(reverse_delta - term2)
                        + 1
                )
                probs = flip_prob * ind + (1 - flip_prob) * (1.0 - ind)
                lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)
                m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
                probs_tmp = torch.minimum(
                    torch.ones_like(la, device=la.device), la.exp()
                )
                self.a_s.append(probs_tmp.detach().mean().item())
            else:
                x_cur = x_delta
        if return_diff:
            probs = torch.minimum(torch.ones_like(la, device=la.device), la.exp())
            return x_cur.detach(), forward_delta, probs
        else:
            return x_cur.detach()


class ParallelTemperingLangevinSampler(nn.Module):
    def __init__(self, dim, n_steps=10, n_chains=4, temps=None, step_size=0.2, mh=True, swap_interval=5):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.temps = temps if temps else [1.0] + [10 ** i for i in range(1, n_chains)]  # Default temperatures
        self.step_size = step_size
        self.mh = mh
        self.swap_interval = swap_interval
        self.entropy = False

        # Initialize Langevin samplers for each temperature
        self.samplers = [LangevinSampler(dim, n_steps, temp=2 * temp,fixed_proposal=False, approx=True, multi_hop=False, step_size=step_size, mh=mh) for temp in self.temps]

    def parallel_tempering_step(self, x, model, step_count):
        """
        Perform one step of parallel tempering for all chains and temperatures.
        :param x: Tensor of shape (n_chains, K, dim), where K is the number of temperatures.
        :param model: Energy function to evaluate the samples.
        :param step_count: Current iteration count.
        :return: Updated tensor of shape (n_chains, K, dim).
        """
        n_chains, K, dim = x.shape

        # Step 1: Perform Langevin sampling for all chains and temperatures
        new_samples = []
        for temp_idx, sampler in enumerate(self.samplers):
            temp_samples = sampler.step(x[:, temp_idx, :], model)  # Batched sampling for all chains at this temperature
            new_samples.append(temp_samples)
        new_samples = torch.stack(new_samples, dim=1)  # Shape: (n_chains, K, dim)

        # Step 2: Perform swaps between adjacent temperatures if needed
        if step_count % self.swap_interval == 0:
            beta = torch.tensor([1.0 / temp for temp in self.temps], device=x.device)  # Shape: (K,)
            for i in range(K - 1):
                x_i, x_j = new_samples[:, i, :], new_samples[:, i + 1, :]
                beta_i, beta_j = beta[i], beta[i + 1]

                # Compute Metropolis-Hastings acceptance probability
                delta_energy = (beta_i - beta_j) * (model(x_i) - model(x_j))
                delta_energy = torch.clamp(delta_energy, min=-100, max=100)  # Numerical stability
                acceptance_probs = torch.exp(delta_energy).squeeze()

                # Randomly accept/reject swaps
                swap_mask = (torch.rand(n_chains, device=x.device) < acceptance_probs).float()
                swap_mask = swap_mask.unsqueeze(1)  # Shape: (n_chains, 1)
                swapped_i = swap_mask * x_j + (1 - swap_mask) * x_i
                swapped_j = swap_mask * x_i + (1 - swap_mask) * x_j
                new_samples[:, i, :] = swapped_i
                new_samples[:, i + 1, :] = swapped_j

        return new_samples

    def step(self, x, model, step_count):
        """
        Perform parallel tempering for all independent chains.
        :param x: Tensor of shape (n_chains, dim), where n_chains is the batch size.
        :param model: Energy function to evaluate the samples.
        :param step_count: Current iteration count.
        :return: Tensor of updated samples (n_chains, dim) with only T=1 samples for each chain.
        """
        n_chains, dim = x.shape
        K = len(self.temps)

        # Step 1: Reshape x into individual chains with K temperatures
        x = x.unsqueeze(1).repeat(1, K, 1)  # Shape: (n_chains, K, dim)

        # Step 2: Perform parallel tempering
        updated_chains = self.parallel_tempering_step(x, model, step_count)

        # Step 3: Extract T=1 samples for each chain (first temperature)
        return updated_chains[:, 0, :]  # Shape: (n_chains, dim)
