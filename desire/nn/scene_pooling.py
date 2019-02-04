import torch.nn as nn
from desire.utils import PoolingCNNParams


class ScenePoolingCNN(nn.Module):

    def __init__(self, params: PoolingCNNParams):
        super(PoolingCNN, self).__init__()
        self.MLP = nn.Sequential()
        for i, (in_channel,
                out_channel,
                kernel_size,
                stride,
                nn_type,
                activation) in enumerate(params.layer_configuration):
            self.MLP.add_module(name="L%i" % (i), module=nn_type(in_channel,
                                                                 out_channel,
                                                                 kernel_size,
                                                                 stride))
            self.MLP.add_module(name="A%i" % (i), module=activation)
            
    def forward(self, x):
        return self.MLP(x)
