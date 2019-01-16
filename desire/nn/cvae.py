import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import matplotlib.pyplot as plt
import numpy as np

from desire.utils import *


class CVAE(nn.Module):

    def __init__(self,
                 params : CVAEParams):
        super(CVAE, self).__init__()
        self.params = params
        self.latent_size = params.latent_size
        self.encoder = CVAEEncoder(params.encoder_params) 
        self.decoder = CVAEDecoder(params.decoder_params) 


    def forward(self, x, c):
        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = to_var(torch.randn([batch_size, self.latent_size]))
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z
        

    def inference(self, c, batch_size=1):
        z = to_var(torch.randn([batch_size, self.latent_size]))
        recon_x = self.decoder(z, c)
        return recon_x
    

class CVAEEncoder(nn.Module):
    def __init__(self,
                 params : CVAEEncoderParams):
        super(CVAEEncoder, self).__init__()

        self.params = params
        
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(params.layer_sizes):
            self.MLP.add_module(name="L%i" % (i), module=nn.Linear(in_size, out_size))
            if params.layer_activations[i] is not None:
                self.MLP.add_module(name="A%i" % (i), module=params.layer_activations[i])                
        self.linear_means = nn.Linear(*params.layer_sizes[-1])
        self.linear_log_var = nn.Linear(*params.layer_sizes[-1])


    def forward(self, x, c):
        print("Here")
        x = torch.cat([x,c], dim=-1)
        print("Here 1")
        x = self.MLP(x)
        print("Here 1")
        means = self.linear_means(x)
        logvars = self.linear_log_var(x)
        
        return means, logvars


class CVAEDecoder(nn.Module):
    def __init__(self,
                 params : CVAEDecoderParams):
        super(CVAEDecoder, self).__init__()

        self.params = params
        
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(params.layer_sizes):
            self.MLP.add_module(name="L%i" % (i), module=nn.Linear(in_size, out_size))
            if params.layer_activations[i] is not None:
                self.MLP.add_module(name="A%i" % (i), module=params.layer_activations[i])



    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        print("Dimension of z is", z.size())
        x = self.MLP(z)
        return x



def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)



    
# default_enc_params = CVAEEncoderParams(layer_sizes=[(96, 48),
#                                                     (48,48),
#                                                     (48, 48)],
#                                        layer_activations=[nn.ReLU(),
#                                                           None,
#                                                           Exp()])
# default_dec_params = CVAEDecoderParams(layer_sizes=[(96, 48),
#                                                     (48,48)],
#                                        layer_activations=[None,
#                                                           nn.Softmax()])

# params = CVAEParams(encoder_params=default_enc_params,
#                     decoder_params=default_dec_params,
#                     cond_size=48,
#                     latent_size=48)

