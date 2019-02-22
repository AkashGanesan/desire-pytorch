from typing import NamedTuple
import torch.nn as nn


NUM_AGENTS = 16
NUM_LENGTH = 40

class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        return x.exp() * 0.5


class EncoderRNNParams(NamedTuple):
    input_size: int = 2
    intermediate_size: int = 16
    gru_hidden_size: int = 48
    kernel_size: int = 3
    n_layers: int = 40
    dropout: float = 0.0


class DecoderRNNParams(NamedTuple):
    gru_hidden_size: int = 48
    output_size: int = 2
    n_layers: int = 40
    dropout: float = 0.0


class CVAEEncoderParams(NamedTuple):
    layer_sizes: list = [(96, 48), (48, 48), (48, 48)]
    layer_activations: list = [nn.ReLU(), None, Exp()]


class CVAEDecoderParams(NamedTuple):
    layer_sizes: list = [(96, 48), (48, 48)]
    layer_activations: list = [None, nn.Softmax(dim=1)]

 
class CVAEParams(NamedTuple):
    encoder_params: CVAEEncoderParams = CVAEEncoderParams()
    decoder_params: CVAEDecoderParams = CVAEDecoderParams()
    cond_size: int = 48
    latent_size: int = 48


class SocialPoolingParams(NamedTuple):
    num_rings: int = 6
    num_wedges: int = 6
    hidden_size: int = 48
    num_agents: int = NUM_AGENTS
    rmin: int = 1
    rmax: int = 2
    fc_config: list = [num_rings * num_wedges * hidden_size, 48, nn.ReLU()]

    
class PoolingCNNParams(NamedTuple):
    layer_configuration: list = [(3, 16, (5, 5), 2, nn.Conv2d, nn.ReLU()),
                                 (16, 32, (5, 5), 1, nn.Conv2d, nn.ReLU()),
                                 (32, 32, (5, 5), 1, nn.Conv2d, nn.ReLU())]

    

class SCFParams(NamedTuple):
    velocity_fc: list = [(2, 16, nn.ReLU())]
    sp_params: SocialPoolingParams = SocialPoolingParams()


class SGMParams(NamedTuple):
    cvae_params: CVAEParams = CVAEParams()
    rnn_enc_params: EncoderRNNParams = EncoderRNNParams()
    rnn_dec_params: DecoderRNNParams = DecoderRNNParams()

class IOCParams(NamedTuple):
    scf_params: SCFParams = SCFParams()
    gru_params: dict = {'input_size': 96,
                        'hidden_size': 48,
                        'num_layers': 1,
                        'dropout': 0,
                        'bidirectional': False,
                        'batch_first': True}
    num_layers: int = NUM_LENGTH
    num_agents: int = NUM_AGENTS
    num_dims: int = 2
    scoring_fc: list = [(48, 1, nn.ReLU())]


# class SCFParams(NamedTuple):
#     feature_pooling: 
    
# class IOCParams(NamedTuple):
    
# _cvae_enc1_params = CVAEEncoderParams(
#     layer_sizes=[(96, 48), (48, 48), (48, 48)],
#     layer_activations=[nn.ReLU(), None, Exp()])


# _cvae_dec1_params = CVAEDecoderParams(
#     layer_sizes=[(96, 48), (48, 48)],
#     layer_activations=[None,
#                        nn.Softmax()])

# cvae_params = CVAEParams(
#     encoder_params=_cvae_enc1_params,
#     decoder_params=_cvae_dec1_params,
#     cond_size=48,
#     latent_size=48)


# sample_parameters = EncoderRNNParams(
#     input_size=2,
#     intermediate_size=20,
#     gru_hidden_size=48,
#     kernel_size=3,
#     n_layers=20,
#     dropout=0)
