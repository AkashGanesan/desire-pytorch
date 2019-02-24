import torch
import torch.nn as nn
import torch.nn.functional as F
from desire.nn import CVAE, EncoderRNN, DecoderRNN
from desire.utils import SGMParams


class SGM(nn.Module):
    def __init__(self,
                 params: SGMParams):
        super(SGM, self).__init__()

        self.params = params
        self.cvae = CVAE(params.cvae_params)

        self.enc_x = EncoderRNN(params.rnn_enc_x_params)
        self.enc_y = EncoderRNN(params.rnn_enc_y_params)

        self.dec = DecoderRNN(params.rnn_dec_params)
        self.dec_fc = nn.Linear(params.rnn_dec_params.gru_hidden_size,
                                params.final_output_size)

    def forward(self, x, y):
        x_enc_out, x_enc_hidden = self.enc_x(x)
        y_enc_out, y_enc_hidden = self.enc_y(y)
        recon_y, means, log_var, z = self.cvae(y_enc_hidden[-1], x_enc_hidden[-1])
        masked_out = torch.mul(recon_y, x_enc_hidden[-1])

        masked_out.unsqueeze_(1)
        hidden_rnn_dec_input = torch.zeros_like(masked_out)
        batch_size = masked_out.size(0)
        hidden_size = masked_out.size(2)
        masked_out = torch.cat([masked_out,
                               torch.zeros(batch_size,
                                           self.params.rnn_dec_params.n_layers - 1,
                                           hidden_size)], dim=1)
        dec_out, dec_hidden = self.dec(masked_out, hidden_rnn_dec_input)
        dec_out.transpose_(1, 2)  # Swap seq_length with no of dimensions
        dec_out_list = []
        for i in range(dec_out.size(2)):
            dec_out_list.append(self.dec_fc(dec_out[:, :, i]))

        return (torch.stack(dec_out_list, dim=2),
                x_enc_out[:, -1, :],
                means,
                log_var)
    
    def inference(self, x):
        x_enc_out, x_enc_hidden = self.enc_x(x)
        recon_y = self.cvae.inference(x_enc_hidden, x_enc_hidden.size(0))
        masked_out = torch.mul(recon_y, x_enc_out)
        ypred = self.dec(x,
                         masked_out)
        return ypred


if __name__ == "__main__":
    model = SGM(SGMParams())
    x = torch.rand(16, 2, 8)
    y = torch.rand(16, 2, 12)
    pred, last_hidden, means, log_var = model(x, y)



# x_enc_out, x_enc_hidden = model.SGM.enc_x(x_rel)
# y_enc_out, y_enc_hidden = model.SGM.enc_y(y_rel)

# recon_y, means, log_var, z = model.SGM.cvae(y_enc_hidden[-1], x_enc_hidden[-1])
# masked_out = torch.mul(recon_y, x_enc_hidden[-1])
# masked_out.unsqueeze_(1)

# batch_size = x_enc_hidden.size(1)
# n_layers = x_enc_hidden.size(0)
# hidden_size = x_enc_hidden.size(2)

# masked_out = torch.cat((masked_out, torch.zeros(batch_size,
#                                                 n_layers - 1,
#                                                 hidden_size)), dim=1)
        
