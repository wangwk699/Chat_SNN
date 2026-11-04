# Extracted from the local PhaseSNN implementation; see NOTICE and LICENSE.
import torch
from torch import nn

class _HeavisideSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta: float):
        ctx.save_for_backward(x); ctx.beta = beta
        return (x > 0).to(x.dtype)
    @staticmethod
    def backward(ctx, grad_out):

        (x,) = ctx.saved_tensors; beta = ctx.beta
        sig = torch.sigmoid(beta * x)
        return grad_out * (beta * sig * (1 - sig)), None

class FSNeuron(nn.Module):
    def __init__(self, T: int, quantized_shape, quantized_item_stat, num_grains: int = 1, beta : float = 1.0, genotype : list = None, neuron_d : torch.Tensor = None, tau : float = 1.0, neuron_h : torch.Tensor = None, neuron_theta : torch.Tensor = None, neuron_v0 : float = 0.0, spike_one: bool = False):
        """
        Args:
            T: 总时间步数
            num_grains: 粒度数量 (默认2个粒度)
        """
        super().__init__()
        self.T = T
        self.num_grains = num_grains

        if neuron_d is not None:
            self.d = nn.Parameter(neuron_d)
            self.tau = nn.Parameter(torch.Tensor([tau]))
            self.h = nn.Parameter(neuron_h)
            self.theta = nn.Parameter(neuron_theta)
        else:
            self.d = nn.Parameter(torch.ones(num_grains)*2)

            self.group_size = quantized_shape[-1]
            self.find_activation_quant_param(quantized_item_stat)
            self.tau = nn.Parameter(torch.Tensor(self.tau))
            genotype = [0] * self.T
            self.h = nn.Parameter(torch.ones(num_grains)*2)
            self.theta = nn.Parameter(torch.ones(num_grains)*2)

        self.beta = beta

        self.grain_assignment = self._assign_grains(genotype)

        self.v = None

        self.v0 = nn.Parameter(torch.Tensor([neuron_v0]))


        self.spike_one = spike_one

    def _heaviside_ste(self, x, beta=10.0):
        return _HeavisideSTE.apply(x, beta)

    def _assign_grains(self, genotype=None):
        assignment = []
        steps_per_grain = self.T // self.num_grains
        remainder = self.T % self.num_grains

        if genotype is not None:
            assignment = genotype
            return assignment
        else:
            return [0, 0, 0, 0, 0, 0, 0, 0]

    def get_grain_power(self, t: int, mode='d'):
        grain_idx = self.grain_assignment[t]
        grain_start = 0
        position_in_grain = t - grain_start

        if mode == 'd':
            return torch.pow(self.d[grain_idx], -(position_in_grain + 1)) * self.tau
        if mode == 'h':
            self.tau = self.tau.to(self.h.device)
            return torch.pow(self.h[grain_idx], -(position_in_grain + 1)) * self.tau
        if mode == 'theta':
            return torch.pow(self.theta[grain_idx], -(position_in_grain + 1)) * self.tau

    def forward(self, x, spikes: torch.Tensor = None, return_spikes: bool = False):
        T, bs, n, dim1 = x.shape
        y = []

        if spikes is None and x is not None:
            self.v = torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)


            for t in range(1):
                self.v = self.v + x[t, ...]
            v_sign = torch.sign(self.v)


            self.v = self.v.abs() + self.v0.to(x.device)


            spikes = []
            for t in range(self.T):

                h = self.get_grain_power(t, mode='h')
                theta = self.get_grain_power(t, mode='theta')
                s_t = self._heaviside_ste(self.v - theta, self.beta)
                spikes.append(s_t)
                self.v = self.v - h * s_t


            spikes = torch.stack(spikes, dim=0).reshape(self.T, bs, n, dim1)

            if self.spike_one:
                nonzero_mask = (spikes != 0)
                cumsum_mask = torch.cumsum(nonzero_mask.int(), dim=0)
                first_nonzero_mask = ((cumsum_mask <= 2) & nonzero_mask)
                spikes = spikes * first_nonzero_mask


        for t in range(self.T):
            d = self.get_grain_power(t, mode='d')
            y.append(v_sign * d * spikes[t])

        y = torch.stack(y, dim=0).sum(0).reshape(T, bs, n, dim1)


        if return_spikes:
            return y, spikes
        return y

    def find_activation_quant_param(self, quantized_item_stat):
        x = quantized_item_stat.reshape(-1, self.group_size)
        xmax = x.amax([-1], keepdim=True)
        self.tau = xmax.squeeze(-1).float()
