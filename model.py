import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask):
        super(AffineCouplingLayer, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hid_dim),\
                                        nn.ReLU(),\
                                        nn.Linear(hid_dim, hid_dim),\
                                        nn.ReLU(),\
                                        nn.Linear(hid_dim, output_dim),\
                                        nn.Tanh())
                                        
        self.layer2 = nn.Sequential(nn.Linear(input_dim, hid_dim),\
                                         nn.ReLU(),\
                                         nn.Linear(hid_dim, hid_dim),\
                                         nn.ReLU(),\
                                         nn.Linear(hid_dim, output_dim))
        self.mask = mask
        
    
    def forward(self, x):
        x_a = x * self.mask
        x_b = x * (1 - self.mask)
        out1 = self.layer1(x_a.float())
        out2 = self.layer2(x_a.float()) # TODO: Why out2 can imporve performance?
        z = x_a + (x_b * torch.exp(out1) + out2 * (1-self.mask)) 
        log_det = out1.sum(dim=1) 

        return z, log_det

    def backward(self, z):
        z_a = z * self.mask
        z_b = z * (1 - self.mask)
        out1 = self.layer1(z_a.float())
        out2 = self.layer2(z_a.float())
        x = z_a + (torch.exp(-out1)) * (z_b - out2 * (1-self.mask))

        return x

class RealNVP(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, mask, n_layers):
        super(RealNVP, self).__init__()
        assert n_layers >=2

        self.modules = []
        for _ in range(n_layers):
            mask = 1 - mask
            self.modules.append(AffineCouplingLayer(input_dim, output_dim, hid_dim, mask))
            
        self.module_list = nn.ModuleList(self.modules)
    
    def forward(self, x):
        log_det_sum = 0.0
        for module in self.module_list:
            x, log_det = module(x)
            log_det_sum += log_det
        return x, log_det_sum
    
    def backward(self, z):
        for module in reversed(self.module_list):
            z = module.backward(z)
        return z