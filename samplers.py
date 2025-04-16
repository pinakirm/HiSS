import torch
import torch.nn as nn
import torch.distributions as dists
import utils, math
import numpy as np
from tuning_components import *
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


class LangevinSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=0.2,
                 mh=True):
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
        self.entropy = False

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.mh = mh
        self.a_s = []
        self.hops = []
        # print('Langevin Sampler')

    def step(self, x, model):

        x_cur = x
        # print('instep')
        # print(x_cur)

        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            term2 =  1/(2 * self.step_size)  # for binary {0,1}, the L2 norm is always 1
            flip_prob = torch.exp(forward_delta - term2) / (torch.exp(forward_delta - term2) + 1)

            rr = torch.rand_like(x_cur)
            # print("rr:", rr)
            ind = (rr < flip_prob) * 1
            x_delta = (1. - x_cur) * ind + x_cur * (1. - ind)
            # print("x_delta", x_delta)

            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                reverse_delta = self.diff_fn(x_delta, model)
                flip_prob = torch.exp(reverse_delta - term2) / (torch.exp(reverse_delta - term2) + 1)
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)

                m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            else:
                x_cur = x_delta

        return x_cur



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
        self.entropy = False
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
        self.entropy = False
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp
        self.a_s = []
        self.hops = []

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
            # self._phops = (x_delta != x).float().sum(-1).mean().item()
            cur_hops = (x_cur[0] != x_delta[0]).float().sum(-1).item()
            self.hops.append(cur_hops)

            reverse_delta = self.diff_fn(x_delta, model)
            cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

            lp_reverse = cd_reverse.log_prob(changes_all).sum(0)

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            self.a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)
        # print(self._ar)
        self._hops = (x != x_cur).float().sum(-1).mean().item()
        return x_cur


class PerDimGibbsSampler(nn.Module):
    def __init__(self, dim, n, rand=False):
        super().__init__()
        self.dim = dim
        self.n_steps = n
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 1.
        self.rand = rand
        self.entropy = False

    def step(self, x, model):
        sample = x.clone()
        for i in range(self.n_steps):
            lp_keep = model(sample).squeeze()
            if self.rand:
                changes = dists.OneHotCategorical(logits=torch.zeros((self.dim,))).sample((x.size(0),)).to(x.device)
            else:
                changes = torch.zeros((x.size(0), self.dim)).to(x.device)
                changes[:, self._i] = 1.

            sample_change = (1. - changes) * sample + changes * (1. - sample)

            lp_change = model(sample_change).squeeze()

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
        self.entropy = False

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


class PerDimLB(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 0.
        self.rand = rand
        self.entropy = False

    def step(self, x, model):
        logits = []
        ndim = x.size(-1)
        fx = model(x).squeeze()
        for k in range(ndim):
            sample = x.clone()
            sample[:, k] = 1 - sample[:, k]
            lp_k = (model(sample).squeeze() - fx) / 2.
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        Z_forward = torch.sum(torch.exp(logits), dim=-1)
        dist = dists.OneHotCategorical(logits=logits)
        changes = dist.sample()
        x_delta = (1. - x) * changes + x * (1. - changes)
        fx_delta = model(x_delta)
        logits = []
        for k in range(ndim):
            sample = x_delta.clone()
            sample[:, k] = 1 - sample[:, k]
            lp_k = (model(sample).squeeze() - fx_delta) / 2.
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        Z_reverse = torch.sum(torch.exp(logits), dim=-1)
        la = Z_forward / Z_reverse
        a = (la > torch.rand_like(la)).float()
        x = x_delta * a[:, None] + x * (1. - a[:, None])
        # a_s.append(a.mean().item())
        # self._ar = np.mean(a_s)
        return x

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
        self.entropy = False
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
            constant = 1.
            forward_delta = self.diff_fn(x_cur, model)

            # make sure we dont choose to stay where we are!
            forward_logits = forward_delta - constant * x_cur
            # print(forward_logits)
            cd_forward = dists.OneHotCategorical(logits=forward_logits.view(x_cur.size(0), -1))
            changes = cd_forward.sample()
            # print(x_cur.shape,forward_delta.shape,changes.shape)
            # exit()
            # compute probability of sampling this change
            lp_forward = cd_forward.log_prob(changes)
            # reshape to (bs, dim, nout)
            changes_r = changes.view(x_cur.size())
            # get binary indicator (bs, dim) indicating which dim was changed
            changed_ind = changes_r.sum(-1)
            # mask out cuanged dim and add in the change
            x_delta = x_cur.clone() * (1. - changed_ind[:, :, None]) + changes_r

            reverse_delta = self.diff_fn(x_delta, model)
            reverse_logits = reverse_delta - constant * x_delta
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
        self.entropy = False

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

class PerDimGibbsSamplerOrd(nn.Module):
    def __init__(self, dim, max_val, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.0
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._ar = 0.0
        self._hops = 0.0
        self._phops = 1.0
        self.rand = rand
        self.max_val = max_val
        self.entropy = False

    def step(self, x, model):
        sample = x.clone()
        lp_keep = model(sample).squeeze()
        if self.rand:
            changes = (
                dists.OneHotCategorical(logits=torch.zeros((self.dim,)))
                .sample((x.size(0),))
                .to(x.device)
            )
        else:
            changes = torch.zeros((x.size(0), self.dim)).to(x.device)
            changes[:, self._i] = 1.0
        # need to calculate the energies of all the possible values
        sample_expanded = torch.repeat_interleave(sample, self.max_val, dim=0)
        values_to_test = torch.Tensor([[i] for i in range(self.max_val)]).repeat(
            (sample.size(0), 1)
        )
        sample_expanded[:, self._i] = values_to_test[:, 0].to(sample.device)
        energies = model(sample_expanded).squeeze()
        cat_dist = dists.categorical.Categorical(
            energies.reshape((sample.size(0), self.max_val)).exp()
        )
        new_coords = cat_dist.sample()
        sample[:, self._i] = new_coords
        self._i = (self._i + 1) % self.dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.0


class LangevinSamplerOrdinal(nn.Module):
    def __init__(
        self,
        dim,
        bal,
        max_val=3,
        n_steps=10,
        multi_hop=False,
        temp=1.0,
        step_size=0.2,
        mh=True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.multi_hop = multi_hop
        self.temp = temp
        # rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2
        self.a_s = []
        self.bal = bal
        self.mh = mh
        self.max_val = max_val  ### number of classes in each dimension
        self.step_size = (step_size * self.max_val) ** (self.dim**2)
        # self.step_size = step_size
        self.entropy = False

    def get_grad(self, x, model):
        x = x.requires_grad_()
        out = model(x)
        gx = torch.autograd.grad(out.sum(), x)[0]
        return gx.detach()

    def _calc_logits(self, x_cur, grad):
        # creating the tensor of discrete values to compute the probabilities for
        batch_size = x_cur.shape[0]
        disc_values = torch.tensor([i for i in range(self.max_val)])[None, None, :]
        disc_values = disc_values.repeat((batch_size, self.dim, 1)).to(x_cur.device)
        term1 = torch.zeros((batch_size, self.dim, self.max_val))
        term2 = torch.zeros((batch_size, self.dim, self.max_val))
        x_expanded = x_cur[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        grad_expanded = grad[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        term1 = self.bal * grad_expanded * (disc_values - x_expanded)
        term2 = (disc_values - x_expanded) ** 2 * (1 / (2 * self.step_size))
        return term1 - term2

    def step(self, x, model, use_dula=False):
        """
        input x : bs * dim, every dim contains a integer of 0 to (num_cls-1)
        """
        x_cur = x
        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            # batch size X dim
            grad = self.get_grad(x_cur.float(), model)
            logits = self._calc_logits(x_cur, grad)
            cat_dist = torch.distributions.categorical.Categorical(logits=logits)
            x_delta = cat_dist.sample()

            if self.mh:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                grad_delta = self.get_grad(x_delta.float(), model) / self.temp

                logits_reverse = self._calc_logits(x_delta, grad_delta)

                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=logits_reverse
                )
                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)

                m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                if use_dula:
                    x_cur = x_delta
                else:
                    x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
            else:
                x_cur = x_delta
            x_cur = x_cur.long()
        return x_cur




class ModifiedDiGsSamplerOrdinal(nn.Module):
    def __init__(
            self,
            dim,
            bal,
            max_val=3,
            n_steps=10,
            multi_hop=False,
            temp=1.0,
            step_size=0.2,
            eta=1,
            mh=True,
            device=None,
            score_sweeps=4
            ,K=1
    ):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.multi_hop = multi_hop
        self.temp = temp
        # rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2
        self.a_s = []
        self.bal = bal
        self.mh = mh
        self.max_val = max_val  ### number of classes in each dimension
        self.step_size = (step_size * self.max_val) ** (self.dim ** 2)
        self.eta = eta
        self.flag = 0
        self.entropy = True

        self.L = score_sweeps
        self.K = K
        self.mh = mh
        self.score_based_sampler = ModifiedDiGsScoreSampOrdinal(
            dim=self.dim,
            max_val=self.max_val,
            n_steps=self.L,
            mh=self.mh,
            step_size=self.step_size,
            bal=self.bal,
            device=device,
            eta=self.eta
        )


    def logistic_denoising_logits(self, x_a_cur):
        batch_size, dim = x_a_cur.shape[0], x_a_cur.shape[1]

        # Create a tensor of discrete values to compute the Gaussian probabilities for
        disc_values = torch.tensor([i for i in range(self.max_val)])[None, None, :]  # Shape: (1, 1, max_val)
        disc_values = disc_values.repeat((batch_size, self.dim, 1)).to(
            x_a_cur.device)  # Shape: (batch_size, dim, max_val)

        x_a_cur_expanded = x_a_cur[:, :, None].repeat(1, 1, self.max_val)
        diff=(x_a_cur_expanded-disc_values)/2*self.eta
        #diff = torch.clamp(diff, min=-50, max=50)  # Prevent extreme values for cosh

        # Compute logits, adding clamping for log(cosh(diff))
        cosh_diff = torch.cosh(diff)
        #cosh_diff = torch.clamp(cosh_diff, min=1e-10, max=1e10)  # Prevent extreme values

        logits = -2 * torch.log(cosh_diff)
        logits = torch.clamp(logits, min=-1e10, max=1e10)  # Final clamping for logits
        return logits

    def step(self, x, x_a, model, use_dula=False):
        """
        input x : bs * dim, every dim contains a integer of 0 to (num_cls-1)
        """

        x_cur = x
        x_a_cur = x_a
        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):

            for i in range(self.K):
                uniform = torch.rand_like(x_cur.float())
                logistic_noise = torch.log(uniform / (1 - uniform))
                x_a_cur = x_cur + self.eta * logistic_noise

                logits = self.logistic_denoising_logits(x_a_cur)
                cat_dist = torch.distributions.categorical.Categorical(logits=logits)
                x_delta = cat_dist.sample()

                #print((x_cur != x_delta).float().sum(-1).mean().item())

                if i != self.K - 1:
                    x_cur = x_delta

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

                x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
            else:
                x_cur = x_delta

            t = self.score_based_sampler.step(x_cur.long().detach(), x_a.long().detach(), model)
            x_cur = t[0].detach()

        return (x_cur, x_a_cur)

  # Update x, not x_a




class ModifiedDiGsScoreSampOrdinal(nn.Module):
    def __init__(
            self,
            dim,
            bal,
            max_val=3,
            n_steps=10,
            multi_hop=False,
            temp=1.0,
            step_size=0.2,
            eta=1,
            mh=True,
            device=None,
    ):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.0
        self._mt = 0.0
        self._pt = 0.0
        self._hops = 0.0
        self._phops = 0.0
        self.multi_hop = multi_hop
        self.temp = temp
        # rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2
        self.a_s = []
        self.bal = bal
        self.mh = mh
        self.max_val = max_val  ### number of classes in each dimension
        self.step_size = step_size
        self.eta = eta
        self.flag = 0
        self.entropy = True

    def agrad(self, x_cur, x_a_cur):
        return torch.tanh((x_a_cur - x_cur) / 2 * self.eta) / (self.eta )

    def get_grad(self, x, model):
        x = x.requires_grad_()
        out = model(x)
        gx = torch.autograd.grad(out.sum(), x)[0]
        return gx.detach()

    def _calc_logits(self, x_cur, grad):
        # creating the tensor of discrete values to compute the probabilities for
        batch_size = x_cur.shape[0]
        disc_values = torch.tensor([i for i in range(self.max_val)])[None, None, :]
        disc_values = disc_values.repeat((batch_size, self.dim, 1)).to(x_cur.device)
        term1 = torch.zeros((batch_size, self.dim, self.max_val))
        term2 = torch.zeros((batch_size, self.dim, self.max_val))
        x_expanded = x_cur[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        grad_expanded = grad[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        term1 = self.bal * grad_expanded * (disc_values - x_expanded)
        term2 = (disc_values - x_expanded) ** 2 * (1 / (2 * self.step_size))
        return term1 - term2

    def step(self, x, x_a, model, use_dula=False):
        """
        input x : bs * dim, every dim contains a integer of 0 to (num_cls-1)
        """

        x_cur = x
        x_a_cur = x_a
        m_terms = []
        prop_terms = []

        EPS = 1e-10
        for i in range(self.n_steps):
            # batch size X dim

            grad = self.get_grad(x_cur.float(), model) / self.temp
            grad_x_cur_a = self.agrad(x_cur.float(), x_a_cur.float()) / self.temp
            grad_x_cur_modified = grad + grad_x_cur_a

            logits = self._calc_logits(x_cur, grad_x_cur_modified)
            cat_dist = torch.distributions.categorical.Categorical(logits=logits)
            x_delta = cat_dist.sample()
            # x_delta_a = x_a_cur + (self.step_size_a / 2.) * grad_x_cur_a + ( (1 * self.step_size_a) ** 0.5) * torch.randn_like(x_a_cur, device=x_a_cur.device)

            if self.mh:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                grad_delta = self.get_grad(x_delta.float(), model) / self.temp
                grad_delta_a = self.agrad(x_delta.float(), x_a_cur) / self.temp
                grad_delta_modified = grad_delta + grad_delta_a

                logits_reverse = self._calc_logits(x_delta, grad_delta_modified)

                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=logits_reverse
                )
                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)

                m_term = (model(x_delta).squeeze() - 2 * torch.log(
                    torch.cosh((x_a_cur - x_delta) / (2 * self.eta))).sum(dim=1)) - (
                                 model(x_cur).squeeze() - 2 * torch.log(
                             torch.cosh((x_a_cur - x_cur) / (2 * self.eta))).sum(
                             dim=1))

                la = m_term + lp_reverse - lp_forward

                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())

                x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
            else:
                x_cur = x_delta

        x_cur = x_cur.long()
        return (x_cur, x_a_cur)

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
                print(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

            else:
                x_cur = x_delta

            t = self.score_based_sampler.step(x_cur.detach(), x_a_cur.detach(), model)
            x_cur = t[0].detach()

        return (x_cur, x_a_cur)


class DGLangevinSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=0.1,
                 alpha=0.2,
                 mh=True, eta=10 ** 4,n=1, score_sweeps=50):
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
        self.alpha = alpha
        self.eta = eta  # flatness
        self.entropy = True


        self.L = score_sweeps
        self.mh = mh
        self.score_based_sampler = DGScoreSamp(self.dim, self.L, fixed_proposal=False, approx=True, multi_hop=False,
                                               temp=2., step_size=self.step_size, alpha=self.alpha, eta=self.eta, mh=True)

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.a_s = []
        self.hops = []

    def step(self, x, x_a, model):

        x_cur = x
        x_a_cur = x_a
        EPS = 1e-10

        for i in range(self.n_steps):
            x_a_cur = self.alpha * x_cur + np.sqrt(self.eta) * torch.randn_like(x_a_cur, device=x_cur.device)

            """
            numbers = torch.randint(0, 2**self.dim -1 , (x_cur.shape[0],), dtype=torch.long)
            x_delta = ((numbers.unsqueeze(1) >> torch.arange(self.dim  -1, -1, -1)) & 1).float()
            """

            diff_0 = (x_a_cur / self.alpha - 0) ** 2  # Squared distance for binary state 0
            diff_1 = (x_a_cur / self.alpha - 1) ** 2  # Squared distance for binary state 1

            # Step 3: Compute unnormalized logits (negative energy)
            logits_0 = -((self.alpha ** 2) / (2 * self.eta)) * diff_0  # Logits for binary state 0
            logits_1 = -((self.alpha ** 2) / (2 * self.eta)) * diff_1  # Logits for binary state 1

            # Step 4: Stack logits and compute probabilities
            logits = torch.stack([logits_0, logits_1], dim=2)  # Shape: (batch_size, dim, 2)
            probs = torch.softmax(logits, dim=2)  # Normalize probabilities along binary states (last dimension)
            probs = torch.clamp(probs, min=EPS, max=1.0)

            # Step 5: Sample binary states for each coordinate
            categorical_dist = torch.distributions.Categorical(probs=probs)  # Tensorized categorical distribution
            sampled_indices = categorical_dist.sample()  # Shape: (batch_size, dim)

            # Step 6: Convert sampled indices (0 or 1) to binary states
            x_delta = sampled_indices.float()
            #print((x_cur != x_delta).float().sum(-1).mean().item())

            if self.mh:
                lp_forward = -((self.alpha ** 2) / (2 * self.eta)) * torch.sum(torch.pow(x_a_cur / self.alpha - x_cur, 2),
                                                                               dim=1)

                m_term = (model(x_delta).squeeze() - torch.sum(torch.pow(self.alpha * x_delta - x_a_cur, 2), dim=1) / (
                        2 * self.eta)) - (
                                 model(x_cur).squeeze() - torch.sum(torch.pow(self.alpha * x_cur - x_a_cur, 2),
                                                                    dim=1) / (
                                         2 * self.eta))

                lp_reverse = -((self.alpha ** 2) / (2 * self.eta)) * torch.sum(torch.pow(x_a_cur / self.alpha - x_delta, 2),
                                                                               dim=1)
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

            else:
                x_cur = x_delta

            t = self.score_based_sampler.step(x_cur.detach(), x_a_cur.detach(), model)
            x_cur = t[0].detach()

        return (x_cur, x_a_cur)


class DGScoreSamp(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, step_size=0.1, fixed_proposal=False, temp=2.,
                 alpha=0.2,
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
        self.step_size = step_size
        self.alpha = alpha
        self.eta = eta  # flatness
        self.entropy = True

        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp

        self.agrad = lambda eta, x, x_a: utils.auxillary_gradient_function(eta, self.alpha * x, x_a)

        self.mh = mh
        self.a_s = []
        self.hops = []

    def step(self, x, x_a, model):

        x_cur = x
        x_a_cur = x_a

        EPS = 1e-10
        for i in range(self.n_steps):

            grad_x_cur = self.diff_fn(x_cur, model)
            grad_x_cur_a = self.agrad(self.eta, x_cur, x_a_cur)
            grad_x_cur_modified = grad_x_cur - self.alpha * (grad_x_cur_a / self.temp) * -(2 * x_cur - 1)

            term2 = 1. / (2 * self.step_size)  # for binary {0,1}, the L2 norm is always 1
            flip_prob = torch.exp(grad_x_cur_modified - term2) / (torch.exp(grad_x_cur_modified - term2) + 1)  # softmax

            rr = torch.rand_like(x_cur)
            ind = (rr < flip_prob) * 1
            x_delta = (1. - x_cur) * ind + x_cur * (1. - ind)

            if self.mh:
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_forward = torch.sum(torch.log(probs + EPS), dim=-1)

                grad_x_delta = self.diff_fn(x_delta, model)
                grad_x_delta_a = self.agrad(self.eta, x_delta, x_a_cur)

                grad_x_delta_modified = grad_x_delta - self.alpha * (grad_x_delta_a / self.temp) * -(2 * x_delta - 1)

                flip_prob = torch.exp(grad_x_delta_modified - term2) / (torch.exp(grad_x_delta_modified - term2) + 1)
                probs = flip_prob * ind + (1 - flip_prob) * (1. - ind)
                lp_reverse = torch.sum(torch.log(probs + EPS), dim=-1)

                m_term = (model(x_delta).squeeze() - torch.sum(torch.pow(self.alpha * x_delta - x_a_cur, 2), dim=1) / (
                        2 * self.eta)) - (
                                 model(x_cur).squeeze() - torch.sum(torch.pow(self.alpha * x_cur - x_a_cur, 2),
                                                                    dim=1) / (
                                         2 * self.eta))

                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

            else:
                x_cur = x_delta

        return (x_cur, x_a_cur)


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
        self.samplers = [LangevinSampler(dim, n_steps, fixed_proposal=False, approx=True, multi_hop=False, temp=2 * temp, step_size=step_size, mh=mh) for temp in self.temps]

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

                if torch.equal(x_i, x_j)==False:
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


class AutomaticCyclicalSamplerOrdinal(nn.Module):
    def __init__(
            self,
            dim,
            device,
            max_val,
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
            adapt_alg="twostage_optim",
            sbc=False,
            big_step=None,
            big_bal=None,
            small_step=None,
            small_bal=None,
            iter_per_cycle=None,
            min_lr=False,
    ):
        super().__init__()
        self.device = device
        self.dim = dim
        self.max_val = max_val
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
        self.mh = mh
        self.entropy=False
        self.a_s = []
        self.hops = []
        self.initial_balancing_constant = initial_balancing_constant
        self.balancing_constant = initial_balancing_constant
        self.adapt_alg = adapt_alg
        if iter_per_cycle and sbc:
            self.iter_per_cycle = iter_per_cycle
        else:
            self.iter_per_cycle = math.ceil(self.num_iters / self.num_cycles)
        mean_stepsize_actual = torch.Tensor([mean_stepsize * self.max_val]).to(
            self.device
        ) ** (self.dim ** 2)
        self.step_sizes = self.calc_stepsizes(mean_stepsize)
        self.diff_values = []
        self.flip_probs = []
        self.balancing_constants = self.calc_balancing_constants(
            self.initial_balancing_constant
        )
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

    def get_name(self):
        if self.mh:
            base = "cyc_dmala"
        else:
            base = "cyc_dula"
        if self.burnin_adaptive:
            name = f"{base}_cycles_{self.num_cycles}"
            name = (
                    name
                    + f"_{self.adapt_alg}_budget_{self.burnin_budget}_lr_{self.burnin_lr}"
            )
        else:
            name = f"{base}_cycles_{self.num_cycles}"
            name = (
                    name
                    + f"_stepsize_{self.initial_step_size}_initbal_{self.balancing_constant}"
            )
        if self.sbc:
            name = (
                    base
                    + f"_cycle_length_{self.iter_per_cycle}_big_s_{self.big_step}_b_{self.big_bal}_small_s_{self.small_step}_b_{self.small_bal}"
            )
        if self.min_lr:
            name += "_min_lr"
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
        actual_val = torch.Tensor([self.min_lr * self.max_val]).to(self.device) ** (
                self.dim ** 2
        )
        # actual_val = self.min_lr
        for i in range(len(self.step_sizes)):
            if self.step_sizes[i] <= actual_val:
                self.step_sizes[i] = actual_val
                self.balancing_constants[i] = 0.5

    def calc_stepsizes(self, mean_step):
        res = []
        total_iter = self.iter_per_cycle
        for k_iter in range(total_iter):
            inner = (np.pi * k_iter) / total_iter
            cur_step_size = mean_step * (np.cos(inner) + 1)
            step_size = cur_step_size
            res.append(step_size)
        res = (torch.tensor(res, device=self.device) * self.max_val) ** (self.dim ** 2)
        return res

    def get_grad(self, x, model):
        x = x.requires_grad_()
        out = model(x)
        gx = torch.autograd.grad(out.sum(), x)[0]
        return gx.detach()

    def _calc_logits(self, x_cur, grad, step_size, bal):
        # creating the tensor of discrete values to compute the probabilities for
        batch_size = x_cur.shape[0]
        disc_values = torch.tensor([i for i in range(self.max_val)])[None, None, :]
        disc_values = disc_values.repeat((batch_size, self.dim, 1)).to(x_cur.device)
        term1 = torch.zeros((batch_size, self.dim, self.max_val))
        term2 = torch.zeros((batch_size, self.dim, self.max_val))
        x_expanded = x_cur[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        grad_expanded = grad[:, :, None].repeat((1, 1, self.max_val)).to(x_cur.device)
        term1 = grad_expanded * (disc_values - x_expanded) * bal
        term2 = (disc_values - x_expanded) ** 2 * (1 / (2 * step_size))
        return term1 - term2

    # def _calc_logits(self, x_cur, grad, step_size, bal):
    #     # creating the tensor of discrete values to compute the probabilities for
    #     batch_size = x_cur.shape[0]
    #     disc_values = torch.arange(start=1, end=self.max_val, step=1)
    #     # disc_values = torch.tensor([i for i in range(self.max_val)]).to(self.device)
    #     # first term of term 1 = nabla(theta) * theta'
    #     term1_1 = torch.einsum('bd, v -> bdv',  [grad, disc_values])
    #     term1_2 = torch.einsum('bd, bd -> bd', [grad, x_cur])
    #     term1 = term1_1 - term1_2[:, :, None]

    #     # term 2, expanded via foil
    #     term2_1 = disc_values ** 2
    #     term2_2 = torch.einsum('bd, v -> bdv', [x_cur, disc_values])
    #     term2_3 = x_cur ** 2

    #     term2 = term2_1[None, None, :] - 2 * term2_2 + term2_3[:, :, None]
    #     return term1 * bal - term2 * (1 / (2 * step_size))

    def step(self, x, model, k_iter):
        x_cur = x

        step_size = self.step_sizes[k_iter % self.iter_per_cycle]
        balancing_constant = self.balancing_constants[k_iter % self.iter_per_cycle]
        for i in range(self.n_steps):
            grad = self.get_grad(x_cur.float(), model)
            logits = self._calc_logits(
                x_cur, grad, step_size=step_size, bal=balancing_constant
            )
            cat_dist = torch.distributions.categorical.Categorical(logits=logits)
            x_delta = cat_dist.sample()
            if self.mh:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta), dim=1)
                grad_delta = self.get_grad(x_delta.float(), model) / self.temp

                logits_delta = self._calc_logits(
                    x_delta, grad_delta, step_size=step_size, bal=balancing_constant
                )

                cat_dist_delta = torch.distributions.categorical.Categorical(
                    logits=logits_delta
                )
                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur), dim=1)

                m_term = model(x_delta).squeeze() - model(x_cur).squeeze()
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                self.a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1.0 - a[:, None])
            else:
                x_cur = x_delta
            x_cur = x_cur.long()
        return x_cur


