import torch
import torch.nn as nn
import torch.nn.functional as F
from desire.utils import DecoderRNNParams


class DecoderRNN(nn.Module):
    def __init__(self,
                 params: DecoderRNNParams):
        
        super(DecoderRNN, self).__init__()
        self.params = params
        self.gru_hidden_size = params.gru_hidden_size
        self.output_size = params.output_size
        self.dropout = params.dropout
        self.n_layers = params.n_layers
        
        # self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(self.gru_hidden_size,
                          self.gru_hidden_size,
                          self.n_layers,
                          dropout=(0 if self.n_layers == 1 else self.dropout),
                          batch_first=True)
        self.out = nn.Linear(self.gru_hidden_size,
                             self.output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, hidden = self.gru(input)

        print ("INput size, output_Size", input.size(), output.size())
        output = self.out(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
