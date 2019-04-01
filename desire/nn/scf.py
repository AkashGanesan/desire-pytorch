import torch
import torch.nn as nn
from desire.utils import SCFParams
from desire.nn import SocialPool
from desire.utils import SocialPoolingParams, get_scene, get_fc_act, SCFParams
import numpy as np


class SCF(nn.Module):
    def __init__(self, index, params: SCFParams):
        super(SCF, self).__init__()
        self.params = params
        self.index = index
        self.velocity_fc = get_fc_act(params.velocity_fc)
        self.sp_nn = SocialPool(index, SocialPoolingParams())

    def forward(self, hidden, y_pred, velocity, scene, x_start, seq_start_end=None):

        vel_out = self.velocity_fc(y_pred)
        # print(y_pred.device,
        #       x_start.device,
        #       hidden.device,
        #       vel_out.device)

        scene_out = get_scene(scene, y_pred, x_start, self.params.scene_size)

        # print("scene out device", scene_out.device)
        sp_out = self.sp_nn(y_pred, x_start, hidden, seq_start_end)
        # print("Shapes 1", y_pred.shape, x_start.shape, scene.shape)
        # print ("Shapes",
        #        sp_out.shape,
        #        vel_out.shape,
        #        scene_out.shape)
        return torch.cat((sp_out, vel_out, scene_out), 1)


if __name__ == "__main__":
    idx = 0
    num_agents = 4
    dimensions = 2
    length = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scf = SCF(idx, SCFParams()).to(device)
    print(scf)
    y = torch.randn(num_agents, dimensions)
    x_start = torch.randn(num_agents, dimensions).to(device)
    v = torch.Tensor(np.gradient(y, axis=0)).to(device)
    y = y.to(device)
    hidden = torch.randn(num_agents, 48).to(device)
    scene = torch.randn(32, 720 // 2, 576 // 2).to(device)

    m = scf(hidden, y, v, scene, x_start)

