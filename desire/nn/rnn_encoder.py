import torch
import torch.nn as nn
import torch.nn.functional as F
from desire.utils import EncoderRNNParams


class EncoderRNN(nn.Module):
    '''Source :
    https://pytorch.org/tutorials/beginner/chatbot_tutorial.html This will
    be used in the front end.

    Removed the embedding layer as we it is directly encoded into
    positions (x,y)
    '''

    def __init__(self, params: EncoderRNNParams):
        # hidden_size, n_layers=20, dropout=0
        super(EncoderRNN, self).__init__()
        self.params = params
        self.n_layers = params.n_layers
        self.dropout = params.dropout
        self.intermediate_size = params.intermediate_size
        self.gru_hidden_size = params.gru_hidden_size
        self.input_size = params.input_size
        self.kernel_size = params.kernel_size

        self.in_conv1d = nn.Conv1d(self.input_size,
                                   self.intermediate_size,
                                   self.kernel_size,
                                   padding=True)

        self.gru = nn.GRU(self.intermediate_size,
                          self.gru_hidden_size,
                          self.n_layers,
                          dropout=(0 if self.n_layers == 1 else self.dropout),
                          bidirectional=False,
                          batch_first=True)

    def forward(self, input_seq, hidden=None):
        packed = F.relu(self.in_conv1d(input_seq))
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        return outputs, hidden


if __name__ == "__main__":
    ''' Sample parameters and simple forwar pass test'''
    sample_parameters = EncoderRNNParams(input_size=2,
                                         intermediate_size=20,
                                         gru_hidden_size=48,
                                         kernel_size=3,
                                         n_layers=20,
                                         dropout=0)
    sample_parameters = EncoderRNNParams()
    a = torch.rand(3, 20, 2)
    b = EncoderRNN(sample_parameters)
    o, h = b(a)
