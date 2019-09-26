import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False


    # Careful! move mask to device
    def set_mask(self, mask):
        self.register_buffer('mask', mask)
        #mask_var = self.get_mask()
        #self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True


    def get_weights(self):
        w = self.weight.data
        if self.mask_flag:
            w = w * self.mask
        return w


    def unpruned_parameters(self):
        if self.mask_flag:
            first_col = self.mask[:, 0]
            return first_col.nonzero().flatten().tolist()
        else:
            N = self.weight.shape[0]
            return torch.arange(N).tolist()


    def forward(self, x):
        if self.mask_flag:
            weight = self.weight * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)