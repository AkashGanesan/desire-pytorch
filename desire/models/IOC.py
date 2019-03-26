'''This contains the IOC related modules

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

        self.grus = nn.ModuleList([nn.GRU(**params.gru_params) for i in range(params.num_layers)])
        self.scfs = nn.ModuleList([SCF(i, params.scf_params) for i in range(params.num_layers)])
        self.scoring_fcs = nn.ModuleList([get_fc_act(params.scoring_fc) for i in range(params.num_layers)])

        self.last_hidden_to_delta = nn.Linear(params.gru_params['hidden_size'],
                                              (params.num_layers
                                               * params.num_dims))

        self.scene_pooling_cnn = ScenePoolingCNN()

    def forward(self, ypred, prev_hidden, scene, x_start, seq_start_end=None):
        ypred_cpu = ypred.cpu()
        ypred_cpu = ypred_cpu.detach().numpy()
        velocity = np.gradient(ypred_cpu, axis=0)
        velocity = torch.Tensor(velocity).to(ypred.device)
        prev_hidden = prev_hidden.unsqueeze(0)
        out_scores = []
        scene = self.scene_pooling_cnn(scene).squeeze(0)


        prev_hidden = prev_hidden.clone()
        prev_hidden.detach()


        for i in range(self.params.num_layers):
            prev_hidden.squeeze_(0)
            # print ("prev_hidden shape", prev_hidden.shape,
            #        ypred.shape,
            #        velocity.shape,
            #        scene.shape,
            #        x_start.shape)

            # print("prev_hidden", prev_hidden.get_device())
            # print("ypred", ypred.get_device())
            # print("velocity", velocity.get_device())
            # print("scene", scene.get_device())
            # print("x_start", x_start.get_device())

            scf_out = self.scfs[i](prev_hidden,
                                   ypred[:, :, i],
                                   velocity[:, :, i],
                                   scene,
                                   x_start,
                                   seq_start_end)
            # print("scf_out", scf_out.get_device())
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IOC(IOCParams()).to(device)
    prev_hidden = torch.randn(16, 48).to(device)
    ypred = torch.randn(16, 2, 40).to(device)
    scene = torch.randn(1, 3, 640, 480).to(device)
    x_start = torch.rand(16, 2).to(device)
    out_scores, prev_hidden = model(ypred, prev_hidden, scene, x_start)
