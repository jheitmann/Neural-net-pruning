import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Make more general: add abstract class?


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, in_channels=0, bias=True, track_corr=False):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        w = self.weight.detach()
        self.register_buffer("mask", w.new_full(w.shape, 1.))
        self.in_channels = in_channels
        if track_corr:
            self.track_correlation()
        else:
            self.track_corr = False

    def get_mask(self):
        return self.mask.clone()

    def set_mask(self, mask):
        self.mask = mask.clone()

    def get_weights(self):
        w = self.weight.detach()
        w = w * self.mask
        return w

    def set_weights(self, weights):
        self.weight = nn.Parameter(self.weight.new_tensor(weights))

    def get_biases(self):
        b = self.bias.clone().detach()
        return b

    def track_correlation(self):
        self.track_corr = True
        self.n_inputs = 0
        input_params = self.in_channels if self.in_channels else self.weight.shape[1]
        self.input_sum = torch.zeros(input_params)
        self.input_dot = torch.zeros(input_params, input_params)

    def unpruned_parameters(self):
        first_col = self.mask[:, 0]
        return first_col.nonzero().flatten().tolist()

    def input_correlation(self):
        assert self.track_corr

        N = self.n_inputs
        if self.in_channels:
            feature_map_size = self.in_features // self.in_channels
            N *= feature_map_size

        input_mean = (self.input_sum[:, None] / N).numpy()
        input_dot_mean = (self.input_dot / N).numpy()
        input_squared_mean = np.diag(input_dot_mean)

        mean_prod = input_mean @ input_mean.T
        variances = (input_squared_mean - np.diag(mean_prod))[:, np.newaxis]
        var_prod = variances @ variances.T

        corr = (input_dot_mean - mean_prod) / np.sqrt(var_prod)

        self.track_corr = False

        return corr

    def forward(self, x):
        if self.track_corr:
            batch_size, in_features = x.shape[0], x.shape[1]
            if self.in_channels:
                x_flattened = x.view(batch_size, self.in_channels, -1)
            else:
                x_flattened = x.view(batch_size, in_features, -1)
            for i in range(batch_size):
                input_i = x_flattened[i]
                input_i_sum = input_i.sum(axis=1)
                self.input_sum += input_i_sum
                input_i_dot = torch.mm(x_flattened[i], x_flattened[i].t())
                self.input_dot += input_i_dot
            self.n_inputs += batch_size

        w = self.weight * self.mask
        return F.linear(x, w, self.bias)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, track_corr=False):
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding, dilation, groups, bias)
        w = self.weight.detach()
        self.register_buffer("mask", w.new_full(w.shape, 1.))
        self.in_channels = in_channels
        if track_corr:
            self.track_correlation()
        else:
            self.track_corr = False

    def get_mask(self):
        return self.mask.clone().view(self.mask.shape[0], -1)

    def set_mask(self, mask):
        self.mask = mask.clone().view(self.weight.shape)

    def get_weights(self):
        w = self.weight.detach()
        w = w * self.mask
        return w.view(w.shape[0], -1)

    def set_weights(self, weights):
        self.weight = nn.Parameter(self.weight.new_tensor(weights).view(self.weight.shape))

    def get_biases(self):
        b = self.bias.clone().detach()
        return b

    def track_correlation(self):
        self.track_corr = True
        self.n_inputs = 0
        self.input_sum = torch.zeros(self.in_channels)
        self.input_dot = torch.zeros(self.in_channels, self.in_channels)

    def unpruned_parameters(self):
        first_col = self.mask[:, 0, 0, 0]
        return first_col.nonzero().flatten().tolist()

    def input_correlation(self):
        assert self.track_corr

        N = self.n_inputs * self.feature_map_size

        input_mean = (self.input_sum[:, None] / N).numpy()
        input_dot_mean = (self.input_dot / N).numpy()
        input_squared_mean = np.diag(input_dot_mean)

        mean_prod = input_mean @ input_mean.T
        variances = (input_squared_mean - np.diag(mean_prod))[:, np.newaxis]
        var_prod = variances @ variances.T

        corr = (input_dot_mean - mean_prod) / np.sqrt(var_prod)

        self.track_corr = False

        return corr

    def forward(self, x):
        if self.track_corr:
            batch_size, in_channels = x.shape[0], x.shape[1]
            x_flattened = x.view(batch_size, in_channels, -1)
            if not self.feature_map_size:
                self.feature_map_size = x_flattened.shape[-1]
            for i in range(batch_size):
                input_i = x_flattened[i]
                input_i_sum = input_i.sum(axis=1)
                self.input_sum += input_i_sum
                input_i_dot = torch.mm(x_flattened[i], x_flattened[i].t())
                self.input_dot += input_i_dot
            self.n_inputs += batch_size

        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
