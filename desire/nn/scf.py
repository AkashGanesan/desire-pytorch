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

    def forward(self, hidden, ypred, velocity, scene, x_start):

        vel_out = self.velocity_fc(ypred)
        scene_out = get_scene(scene, ypred, x_start, self.params.scene_size)
        sp_out = self.sp_nn(ypred, x_start, hidden)
        # print("Shapes 1", ypred.shape, x_start.shape, scene.shape)
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
    scf = SCF(idx, SCFParams())
    y = torch.randn(num_agents, dimensions)
    x_start = torch.randn(num_agents, dimensions)
    v = np.gradient(y, axis=0)
    hidden = torch.randn(num_agents, 48)
    scene = torch.randn(1, 32, 100, 120)
    m = scf(hidden, y, v, scene, x_start)
