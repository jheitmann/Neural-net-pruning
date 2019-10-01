import torch
import torch.nn as nn
import torch.nn.functional as F


# Make more general: add abstract class?


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        w = self.weight.detach()
        self.register_buffer("mask", w.new_full(w.shape, 1.))


    # Careful! move mask to device
    def set_mask_old(self, mask):
        self.register_buffer("mask", mask)
        #mask_var = self.get_mask()
        #self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True


    def get_mask(self):
        return self.mask.clone()


    def set_mask(self, mask):
        self.mask = mask.clone()


    def get_weights(self):
        w = self.weight.detach()
        w = w * self.mask
        return w


    def unpruned_parameters(self):
        first_col = self.mask[:, 0]
        return first_col.nonzero().flatten().tolist()


    def forward(self, x):
        w = self.weight * self.mask
        return F.linear(x, w, self.bias)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias)
        w = self.weight.detach()
        self.register_buffer("mask", w.new_full(w.shape, 1.))
    

    def set_mask_old(self, mask):
        self.register_buffer('mask', mask.view(self.weight.shape))
        #mask_var = self.get_mask()
        #self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True


    def get_mask(self):
        return self.mask.clone().view(self.mask.shape[0], -1)


    def set_mask(self, mask):
        self.mask = mask.clone().view(self.weight.shape)


    def get_weights(self):
        w = self.weight.detach()
        w = w * self.mask
        return w.view(w.shape[0], -1)


    def unpruned_parameters(self):
        first_col = self.mask[:, 0, 0, 0]
        return first_col.nonzero().flatten().tolist()

    
    def forward(self, x):
        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)