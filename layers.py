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


    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)