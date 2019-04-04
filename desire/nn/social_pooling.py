import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

from typing import NamedTuple
from desire.utils import SocialPoolingParams
import torch_scatter as ts

def ring_indices(ydash, params: SocialPoolingParams):
    ''' Returns the ring indices of each id given an id'''

    with torch.no_grad():
        num_rings = params.num_rings
        num_wedges = params.num_wedges
        rmin = params.rmin
        rmax = params.rmax
        rmax_by_rmin = params.rmax_by_rmin

        r = torch.norm(ydash[:, None] - ydash, dim=2, p=2)

        ring_ids = torch.ceil((num_rings-1) * (torch.log(r / rmin) /
                                               rmax_by_rmin))
        ring_ids[ring_ids == -float("Inf")] = 0
        ring_ids = ring_ids.long()
        x_diff = (ydash[:, 0] - ydash[:, 0, None])
        y_diff = (ydash[:, 1] - ydash[:, 1, None])

        theta = torch.atan2(y_diff, x_diff)
        wedge_ids = theta * num_wedges / (2 * np.pi)
        wedge_ids = (wedge_ids + (num_wedges // 2 - 1)).long()

        final_index = (ring_ids * num_wedges + wedge_ids)
        final_index[final_index < params.num_wedges] = 0
        final_index[final_index >= params.num_wedges ** 2] = 0

    return final_index.long()



def scatter_hidden(hidden, pool_indices, params: SocialPoolingParams):
    '''
    scatter_hidden :: hidden -> pool_indices -> SocialPoolingParams -> pooled_hidden_layers
    hidden :: #Agents -> hidden_size
    pool_indices::  #Agents -> #Agents -> Distance

    - Hidden layer contains the hidden layers of every agent.
    - pool_indices contains info into which (ring_id x wedge_id) hidden layers of neighoring pixels go into.

    '''

    batch_size, hidden_size = hidden.size()

    out = torch.zeros(batch_size, params.num_wedges * params.num_rings * 48, device=hidden.device)
    for cnt, idx in enumerate(pool_indices):
        hidden_per_agent = ts.scatter_mean(hidden, idx.long(), 0, dim_size=((params.num_wedges + 1) * params.num_rings))
        out[cnt, :] = hidden_per_agent[params.num_wedges:, :].flatten()

    return out


class SocialPool(nn.Module):

    def __init__(self, idx, params):
        super(SocialPool, self).__init__()
        self.params = params
        self.fc = nn.Linear(*params.fc_config[0:2])
        self.hidden_size = params.hidden_size

    def forward(self, y_pred, x_start, hidden, seq_start_end=None):
        device = x_start.device
        TOTAL_BATCH_SIZE, _ = y_pred.size()
        HIDDEN_SIZE = hidden.size(1)

        if seq_start_end is None:
            seq_start_end = ((0, y_pred.size(0)),)

        log_polar_hidden = torch.zeros(TOTAL_BATCH_SIZE,
                                       self.params.num_rings,
                                       self.params.num_wedges,
                                       HIDDEN_SIZE).to(device)


        out = torch.zeros(TOTAL_BATCH_SIZE, (self.params.num_wedges *
                                             self.params.num_rings *
                                             self.params.hidden_size), device=device)
        for (start, end) in seq_start_end:

            pool_indices = ring_indices(y_pred[start:end],
                                        self.params)
            out[start:end, :] = scatter_hidden(hidden[start:end], pool_indices, self.params)

        return F.relu(self.fc(out))



if __name__ == "__main__":
    input = torch.tensor([[0, 0],
                          [1, 1],
                          [-1, 1],
                          [1, -1],
                          [-1,-1],
                          [299, 299],
                          [300, 299],
                          [301, 301]],
                         dtype=torch.float)

    input = torch.tensor([[0, 0],
                          [0.9, 0.9],
                          [1, 1.5],
                          [-1, -1]],
                         dtype=torch.float)


    y_pred_rel = torch.randn(100, 2)
    x_start = torch.randn(100, 2)
    hidden = torch.randn(100,48)

    seq_start_end = torch.tensor([[0,10],
                                  [10,100],
                                  [100,1000]])

    c = ring_indices(input, SocialPoolingParams())

    model = SocialPool(0, SocialPoolingParams())
    model(y_pred_rel, x_start, hidden, seq_start_end)

    c = torch.zeros(params.num_agents,
                    params.num_rings,
                    params.num_wedges,
                    params.hidden_size)


    log_polar_hidden = torch.zeros(params.num_agents,
                                   params.num_rings,
                                   params.num_wedges,
                                   params.hidden_size)



    for (k, v) in valid_hidden_idx.items():
        log_polar_hidden[k] = h[v].mean(dim=0)



    src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]]).float()
    src = torch.randn(3, 7)

    index = torch.tensor([1,1,0])
    out = ts.scatter_mean(src, index)



