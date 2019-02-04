import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

from typing import NamedTuple
from desire.utils import SocialPoolingParams



def ring_indices(ydash, params : SocialPoolingParams):
    ''' Returns the ring indices of each id given an id'''

    with torch.no_grad():
        num_rings = params.num_rings
        num_wedges = params.num_wedges
        rmin = params.rmin
        rmax = params.rmax

        print("A")
        r = torch.norm(ydash[:, None] - ydash, dim=2, p=2)
        rmax_by_rmin = np.log(int(rmax / float(rmin)))
        
        ring_ids = torch.where(r < rmin,
                               torch.zeros_like(r) - 1,
                               torch.floor((num_rings - 1) * (torch.log(r / rmin) /
                                                              rmax_by_rmin)))
        
        x_diff = (ydash[:, 0] - ydash[:, 0, None])
        y_diff = (ydash[:, 1] - ydash[:, 1, None])

        theta = torch.atan2(y_diff, x_diff)
        wedge_ids = theta * num_wedges / (2 * np.pi)


    return ring_ids, (wedge_ids + (num_wedges // 2 - 1)).int()


def pool_layers(ring_id_all, wedge_id_all, params : SocialPoolingParams):

    num_rings = params.num_rings
    num_wedges = params.num_wedges

    valid_hidden_idx = defaultdict(list)

    with torch.no_grad():
        num_agents = ring_id_all.size(0)
        for agent_idx in range(num_agents):
            ring_ids = ring_id_all[agent_idx]
            wedge_ids = wedge_id_all[agent_idx]

            for idx, ring_id in enumerate(ring_ids):
                if ring_id >= 0 and ring_id < num_rings:
                    wedge_idx = wedge_ids[idx]
                    valid_hidden_idx[(agent_idx,
                                      int(ring_id.item()),
                                      wedge_idx.item())].append(idx)
    return valid_hidden_idx


class SocialPool(nn.Module):

    def __init__(self, idx, params):
        super(SocialPool, self).__init__()
        self.params = params
        self.fc = nn.Linear(*params.fc_config[0:2])
        self.hidden_size = params.hidden_size

    def forward(self, ypred, hidden):
        ridx, widx = ring_indices(ypred,
                                  self.params)

                
        valid_hidden_idx = pool_layers(ridx,
                                       widx,
                                       self.params)

        log_polar_hidden = torch.zeros(self.params.num_agents,
                                       self.params.num_rings,
                                       self.params.num_wedges,
                                       self.params.hidden_size)

        for (k, v) in valid_hidden_idx.items():
            log_polar_hidden[k] = hidden[v].mean(dim=0)

        log_polar_hidden = log_polar_hidden.flatten(start_dim=1, end_dim=-1)
        return F.relu(self.fc(log_polar_hidden))



if __name__ == "__main__":
    input = torch.tensor([[0, 0],
                          [1, 1],
                          [1, 1.5],
                          [-1, -1],
                          [-1, -1],
                          [2, 0],
                          [-2, 0],
                          [10, 10]],
                         dtype=torch.float)
    params = SocialPoolingParams()
    a, b = ring_indices(input, params)



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
