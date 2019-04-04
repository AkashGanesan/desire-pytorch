from desire.models.IOC import IOC
from desire.models.SGM import SGM
from desire.utils import IOCParams, SGMParams

import torch
import torch.nn as nn


class DESIRE(nn.Module):

    def __init__(self,
                 ioc_params: IOCParams,
                 sgm_params: SGMParams):
        super(DESIRE, self).__init__()

        self.sgm_params = sgm_params
        self.ioc_params = ioc_params
        self.SGM = SGM(sgm_params)
        self.IOC = IOC(ioc_params)

    def forward(self,
                obs_traj_rel,
                pred_traj_gt_rel,
                x_start,
                scene,
                seq_start_end=None):
        pred_traj_rel, x_last_hidden, mean, log_var = self.SGM(obs_traj_rel,
                                                               pred_traj_gt_rel)

        obs_traj_rel_cum_last = obs_traj_rel.cumsum(dim=2)[:,:,-1]
        out_scores, pred_delta = self.IOC(pred_traj_rel=pred_traj_rel,
                                          prev_hidden=x_last_hidden,
                                          scene=scene,
                                          x_start=x_start,
                                          obs_traj_rel_cum_last=obs_traj_rel_cum_last,
                                          seq_start_end=seq_start_end)


        return pred_traj_rel, pred_delta, mean, log_var

    def inference(self,
                  obs_traj_rel,
                  scene,
                  x_start,
                  seq_start_end):
        pred_traj_rel, x_last_hidden = self.SGM.inference(obs_traj_rel)
        obs_traj_rel_cum_last = obs_traj_rel.cumsum(dim=2)[:,:,-1]

        out_scores, pred_delta = self.IOC(pred_traj_rel=pred_traj_rel,
                                          prev_hidden=x_last_hidden,
                                          scene=scene,
                                          x_start=x_start,
                                          obs_traj_rel_cum_last=obs_traj_rel_cum_last,
                                          seq_start_end=seq_start_end)

        return pred_traj_rel, pred_delta

if __name__ == "__main__":
    model = DESIRE(IOCParams(), SGMParams())
    obs_traj_rel = torch.rand(4, 2, 8)
    pred_traj_gt_rel = torch.rand(4, 2, 12)
    x_start = torch.rand(4, 2)
    scene = torch.randn(1, 3, 640, 480)
    pred_traj_rel, pred_delta, mean, log_var = model(obs_traj_rel,
                                                     pred_traj_gt_rel,
                                                     x_start,
                                                     scene)
