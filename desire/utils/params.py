from typing import NamedTuple
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List

NUM_AGENTS = 16
NUM_LENGTH = 40

class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()

    def forward(self, x):
        return x.exp() * 0.5



# class EncoderRNNParams(NamedTuple):
#     input_size: int = 2
#     intermediate_size: int = 16
#     gru_hidden_size: int = 48
#     kernel_size: int = 3
#     n_layers: int = 40
#     dropout: float = 0.0


# class DecoderRNNParams(NamedTuple):
#     gru_hidden_size: int = 48
#     output_size: int = 48
#     n_layers: int = 12
#     dropout: float = 0.0


# class CVAEEncoderParams(NamedTuple):
#     layer_sizes: list = [(96, 48), (48, 48), (48, 48)]
#     layer_activations: list = [nn.ReLU(), None, Exp()]

# class CVAEDecoderParams(NamedTuple):
#     layer_sizes: list = [(96, 48), (48, 48)]
#     layer_activations: list = [None, nn.Softmax(dim=1)]


# class CVAEParams(NamedTuple):
#     encoder_params: CVAEEncoderParams = CVAEEncoderParams()
#     decoder_params: CVAEDecoderParams = CVAEDecoderParams()
#     cond_size: int = 48
#     latent_size: int = 48


# class SocialPoolingParams(NamedTuple):
#     num_rings: int = 6
#     num_wedges: int = 6
#     hidden_size: int = 48
#     num_agents: int = NUM_AGENTS
#     rmin: int = 1
#     rmax: int = 2
#     fc_config: list = [num_rings * num_wedges * hidden_size, 48, nn.ReLU()]


# class PoolingCNNParams(NamedTuple):
#     layer_configuration: list = [(3, 16, (5, 5), 2, nn.Conv2d, nn.ReLU()),
#                                  (16, 32, (5, 5), 1, nn.Conv2d, nn.ReLU()),
#                                  (32, 32, (5, 5), 1, nn.Conv2d, nn.ReLU())]

# class SCFParams(NamedTuple):
#     velocity_fc: list = [(2, 16, nn.ReLU())]
#     sp_params: SocialPoolingParams = SocialPoolingParams()

# class SGMParams(NamedTuple):
#     cvae_params: CVAEParams = CVAEParams()
#     rnn_enc_x_params: EncoderRNNParams = EncoderRNNParams(n_layers=8)
#     rnn_enc_y_params: EncoderRNNParams = EncoderRNNParams(n_layers=12)
#     rnn_dec_params: DecoderRNNParams = DecoderRNNParams(n_layers=12)
#     final_output_size: int = 2

# class IOCParams(NamedTuple):
#     scf_params: SCFParams = SCFParams()
#     gru_params: dict = {'input_size': 96,
#                         'hidden_size': 48,
#                         'num_layers': 1,
#                         'dropout': 0,
#                         'bidirectional': False,
#                         'batch_first': True}
#     num_layers: int = 12
#     num_agents: int = NUM_AGENTS
#     num_dims: int = 2
#     scoring_fc: list = [(48, 1, nn.ReLU())]



@dataclass
class EncoderRNNParams:
    input_size: int = 2
    intermediate_size: int = 16
    gru_hidden_size: int = 48
    kernel_size: int = 3
    n_layers: int = 40
    dropout: float = 0.0


@dataclass
class DecoderRNNParams:
    gru_hidden_size: int = 48
    output_size: int = 48
    n_layers: int = 12
    dropout: float = 0.0

@dataclass
class CVAEEncoderParams:
    cond_size: int = 48
    latent_size: int = 48

    def __post_init__(self):
        self.layer_sizes = [(self.cond_size + self.latent_size, 48),
                       (48, 48),
                       (48, 48)]
        self.layer_activations = [nn.ReLU(), None, Exp()]



@dataclass
class CVAEDecoderParams:
    cond_size: int = 48
    latent_size: int = 48

    def __post_init__(self):
        self.layer_sizes = [(self.cond_size + self.latent_size, 48),
                            (48, 48)]
        self.layer_activations = [None, nn.Softmax(dim=1)]


@dataclass
class CVAEParams:
    cond_size: int = 48
    latent_size: int = 48

    def __post_init__(self):
        self.encoder_params = CVAEEncoderParams(cond_size=self.cond_size,
                                                latent_size=self.latent_size)
        self.decoder_params = CVAEDecoderParams(cond_size=self.cond_size,
                                                latent_size=self.latent_size)


@dataclass
class SocialPoolingParams:
    num_rings: int = 6
    num_wedges: int = 6
    hidden_size: int = 48
    num_agents: int = 4
    rmin: int = 1
    rmax: int = 2

    def __post_init__(self):
        self.fc_config = [self.num_rings * self.num_wedges * self.hidden_size,
                          self.hidden_size,
                          nn.ReLU()]

@dataclass    
class PoolingCNNParams:
    layer_configuration: List = field(default_factory = lambda : [(3, 16, (5, 5), 2, nn.Conv2d, nn.ReLU()),
                                                                  (16, 32, (5, 5), 1, nn.Conv2d, nn.ReLU()),
                                                                  (32, 32, (5, 5), 1, nn.Conv2d, nn.ReLU())])
    

@dataclass
class SCFParams:
    scene_size: tuple = field(default_factory=lambda: (720, 576, 2))
    input_size: int = 2
    intermediate_size: int = 16
    sp_params: SocialPoolingParams = field(default_factory= lambda: SocialPoolingParams())

    def __post_init__(self):
        self.velocity_fc = [(self.input_size,
                             self.intermediate_size,
                             nn.ReLU())]


@dataclass
class SGMParams:
    final_output_size: int = 2
    n_layers_x: int = 8
    n_layers_y: int = 12
    n_layers_ypred: int = 12
    cvae_params: CVAEParams = CVAEParams()

    def __post_init__(self):
        self.rnn_enc_x_params = EncoderRNNParams(n_layers=self.n_layers_x)
        self.rnn_enc_y_params = EncoderRNNParams(n_layers=self.n_layers_y)
        self.rnn_dec_params = DecoderRNNParams(n_layers=self.n_layers_ypred)


@dataclass
class IOCParams:
    scene_size: tuple = field(default_factory=lambda: (720, 576, 2))
    num_layers: int = 12
    num_agents: int = 4
    num_dims: int = 2
    gru_params: dict = field(default_factory=lambda: {'input_size': 96,
                                                      'hidden_size': 48,
                                                      'num_layers': 1,
                                                      'dropout': 0,
                                                      'bidirectional': False,
                                                      'batch_first': True})

    scoring_fc: list = field(default_factory=lambda: [(48, 1, nn.ReLU())])

    def __post_init__(self):
        self.scf_params = SCFParams(scene_size=self.scene_size)
