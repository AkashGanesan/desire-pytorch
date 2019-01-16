import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple

from desire.nn.exp import Exp


class EncoderRNNParams(NamedTuple):
    input_size: int = 2
    intermediate_size: int = 20
    gru_hidden_size: int = 48
    kernel_size: int = 3
    n_layers: int = 20
    dropout: float = 0.0


class DecoderRNNParams(NamedTuple):
    gru_hidden_size: int = 48
    output_size: int = 2
    n_layers: int = 20
    dropout: float = 0.0


class CVAEEncoderParams(NamedTuple):
    layer_sizes: list = [(96, 48), (48, 48), (48, 48)]
    layer_activations: list = [nn.ReLU(), None, Exp()]


class CVAEDecoderParams(NamedTuple):
    layer_sizes: list = [(96, 48), (48, 48)]
    layer_activations: list = [None, nn.Softmax()]


class CVAEParams(NamedTuple):
    encoder_params: CVAEEncoderParams = CVAEEncoderParams()
    decoder_params: CVAEDecoderParams = CVAEDecoderParams()
    cond_size: int = 48
    latent_size: int = 48


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
