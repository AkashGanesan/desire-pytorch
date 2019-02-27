import torch.nn as nn
import torch

def return_seq(params):
    seq_net = nn.Sequential()
    for i, (in_channel,
            out_channel,
            kernel_size,
            stride,
            nn_type,
            activation) in enumerate(params):
        self.seq_net.add_module(name="L%i" % (i),
                                module=nn_type(in_channel,
                                               out_channel,
                                               kernel_size,
                                               stride))
        if activation is not None:
            self.seq_net.add_module(name="A%i" % (i),
                                    module=activation)
    return seq_net.to

def get_fc_act(layers):
    seq_net = nn.Sequential()
    for i, (in_features, out_features, activation) in enumerate(layers):
        seq_net.add_module(name="L%i" % (i),
                           module=nn.Linear(in_features,
                                            out_features))
        if activation is not None:
            seq_net.add_module(name="A%i" % (i),
                               module=activation)
    
    return seq_net
