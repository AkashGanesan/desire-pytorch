import torch
import torch.nn as nn
import torch.nn.functional as F
from desire.nn import CVAE, EncoderRNN, DecoderRNN


class SGM:
    def __init__(self,
                 # Encoder Parameters
                 encoder_layer_sizes,
                 latent_size,
                 decoder_layer_sizes,

                 # Encoder Parameters
                 enc_hidden_size,
                 enc_n_layers,
                 enc_dropout,

                 # Decoder Parameters
                 attn_model,
                 dec_hidden_size,
                 dec_output_size,
                 dec_n_layers,
                 dec_dropout):

        self.cvae = CVAE(encoder_layer_sizes, latent_size, decoder_layer_sizes)

        self.enc_x = EncoderRNN(enc_hidden_size,
                                enc_n_layers,
                                enc_dropout)
        self.enc_y = EncoderRNN(enc_hidden_size,
                                enc_n_layers,
                                enc_dropout)

        self.dec = DecoderRNN(dec_hidden_size,
                              dec_output_size,
                              dec_n_layers,
                              dec_dropout)


    def forward(self, x, y):
        x_enc_out, x_enc_hidden = self.enc_x(x)
        y_enc_out, y_enc_hidden = self.enc_y(y)
        recon_y, means, log_var, z = self.cvae(y_enc_hidden, x_enc_hidden)
        masked_out = torch.mul(recon_y, x_enc_out)
        dec_out, dec_hidden  = dec(x, masked_out)
        return dec_out


    def inference(self, x):
        x_enc_out, x_enc_hidden = self.enc_x(x)
        recon_y = self.cvae.inference(x_enc_hidden, x_enc_hidden.size(0))
        masked_out = torch.mul(recon_y, x_enc_out)
        ypred = dec(x,
                    masked_out)
        return ypred
