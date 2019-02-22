''' This contains the IOC related modules
'''
import torch
import torch.nn as nn
import torch.functional as F
from desire.utils import IOCParams
from desire.utils import get_fc_act
from desire.nn import SCF
import numpy as np
from desire.nn import ScenePoolingCNN

class IOC(nn.Module):
    def __init__(self, params: IOCParams):
        super(IOC, self).__init__()
        self.params = params

        self.scfs = []
        self.grus = []
        self.scoring_fcs = []
        for i in range(self.params.num_layers):
            self.grus.append(nn.GRU(**params.gru_params))
            self.scfs.append(SCF(i, params.scf_params))
            self.scoring_fcs.append(get_fc_act(params.scoring_fc))
        self.last_hidden_to_delta = nn.Linear(params.gru_params['hidden_size'],
                                              (params.num_layers
                                              * params.num_dims))

        self.scene_pooling_cnn = ScenePoolingCNN()

    def forward(self, ypred, prev_hidden, scene, x_start):
        velocity = np.gradient(ypred.detach().numpy(), axis=0)
        prev_hidden = prev_hidden.unsqueeze(0)
        out_scores = []
        scene = self.scene_pooling_cnn(scene).squeeze(0)
        print("Scene shape is", scene.size())
        for i in range(self.params.num_layers):
            prev_hidden.squeeze_(0)
            # print ("prev_hidden shape", prev_hidden.shape)
            scf_out = self.scfs[i](prev_hidden,
                                   ypred[:, :, i],
                                   velocity[:, :, i],
                                   scene,
                                   x_start)
            # print("scf dimensions", scf_out.size())
            gru_out, prev_hidden = self.grus[0](scf_out.unsqueeze(1),
                                                prev_hidden.unsqueeze(0))
            # print("gru_out shape", gru_out.shape)
            out_scores.append(self.scoring_fcs[0](gru_out.squeeze(1)))

        # print(prev_hidden.squeeze(0).shape)
        return (out_scores,
                self.last_hidden_to_delta(prev_hidden.squeeze(0)).view(-1,
                                                                       self.params.num_dims,
                                                                       self.params.num_layers))


if __name__ == "__main__":
    model = IOC(IOCParams())
    prev_hidden = torch.randn(16, 48)
    ypred = torch.randn(16, 2, 40)
    scene = torch.randn(1, 3, 640, 480)
    x_start = torch.rand(16,2)
    out_scores, prev_hidden = model(ypred, prev_hidden, scene, x_start)
