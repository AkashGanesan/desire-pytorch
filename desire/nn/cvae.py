import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import matplotlib.pyplot as plt
import numpy as np

class CVAE(nn.Module):

    def __init__(self,
                 encoder_layer_sizes,
                 latent_size,
                 decoder_layer_sizes,
                 cond_size):
        super(CVAE, self).__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, latent_size, cond_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, cond_size)

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



class Encoder(nn.Module):
    def __init__(self,
                 layer_sizes,
                 latent_size,
                 cond_size):
        super(Encoder, self).__init__()

        layer_sizes[0] += cond_size

        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L%i" % (i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A%i" % (i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)


    def forward(self, x, c):
        x = torch.cat([x,c], dim=-1)
        x = self.MLP(x)

        means = self.linear_means(x)
        logvars = self.linear_log_var(x)

        return means, logvars



class Decoder(nn.Module):
    def __init__(self,
                 layer_sizes,
                 latent_size,
                 cond_size):
        super(Decoder, self).__init__()
        
        input_size = latent_size + cond_size
        
        self.MLP = nn.Sequential()
        
        for i, (in_size, out_size) in enumerate( zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A%i"%(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
                

    def forward(self, z, c):

        z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)
        return x


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
