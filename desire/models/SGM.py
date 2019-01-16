import torch
import torch.nn as nn
import torch.nn.functional as F
from desire.nn import CVAE, EncoderRNN, DecoderRNN
from desire.utils import CVAEParams, EncoderRNNParams, DecoderRNNParams


class SGM(nn.Module):
    def __init__(self,
                 cvae_params: CVAEParams,
                 rnn_enc_params: EncoderRNNParams,
                 rnn_dec_params: DecoderRNNParams):
        super(SGM, self).__init__()
        
        self.cvae = CVAE(cvae_params)

        self.enc_x = EncoderRNN(rnn_enc_params)
        self.enc_y = EncoderRNN(rnn_enc_params)

        self.dec = DecoderRNN(rnn_dec_params)

    def forward(self, x, y):
        x_enc_out, x_enc_hidden = self.enc_x(x)
        y_enc_out, y_enc_hidden = self.enc_y(y)
        recon_y, means, log_var, z = self.cvae(y_enc_hidden[-1], x_enc_hidden[-1])

        masked_out = torch.mul(recon_y, x_enc_hidden[-1])
        masked_out.unsqueeze_(1)

        batch_size = x_enc_hidden.size(1)
        n_layers = x_enc_hidden.size(0)
        hidden_size = x_enc_hidden.size(2)

        masked_out = torch.cat((masked_out, torch.zeros(batch_size,
                                                        n_layers - 1,
                                                        hidden_size)), dim=1)
        print("Masked out size", masked_out.shape)

        hidden_rnn_dec_input = torch.zeros_like(masked_out)    
        print("Hidden layer size to RNN decoder", hidden_rnn_dec_input.shape)
        dec_out, dec_hidden = self.dec(masked_out, hidden_rnn_dec_input)
        dec_out.transpose_(1, 2)  # Swap seq_length with no of dimensions
        return dec_out

    def inference(self, x):
        x_enc_out, x_enc_hidden = self.enc_x(x)
        recon_y = self.cvae.inference(x_enc_hidden, x_enc_hidden.size(0))
        masked_out = torch.mul(recon_y, x_enc_out)
        ypred = self.dec(x,
                         masked_out)
        return ypred


if __name__ == "__main__":
    model = SGM(CVAEParams(), EncoderRNNParams(), DecoderRNNParams())
    x = torch.rand(16, 2, 20)
    y = torch.rand(16, 2, 20)
    pred = model(x, y)
