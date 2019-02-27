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
                x_rel,
                y_rel,
                x_start,
                scene):
        y_pred_rel, x_last_hidden, mean, log_var = self.SGM(x_rel, y_rel)
        
        
        # print("YPRED REL", y_pred_rel.shape)
        out_scores, pred_delta = self.IOC(y_pred_rel,
                                          x_last_hidden,
                                          scene,
                                          x_start)
        return y_pred_rel, pred_delta, mean, log_var

    def inference(self, x):
        ypred, x_last_hidden = self.SGM.inference(x)
        out_scores, pred_delta = self.IOC(ypred, x_last_hidden)


if __name__ == "__main__":
    model = DESIRE(IOCParams(), SGMParams())
    x_rel = torch.rand(4, 2, 8)
    y_rel = torch.rand(4, 2, 12)
    x_start = torch.rand(4, 2)
    scene = torch.randn(1, 3, 640, 480)
    y_pred_rel, pred_delta, mean, log_var = model(x_rel,
                                                  y_rel,
                                                  x_start,
                                                  scene)
