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
        self.ioc_params = ioc_params
        self.sgm_params = sgm_params

        self.IOC = IOC(ioc_params)
        self.SGM = SGM(sgm_params)

    def forward(self, x, y):
        ypred, x_last_hidden = self.SGM(x, y)
        out_scores, last_hidden = self.IOC(ypred, x_last_hidden)  # XXX.  ADD scene here
        return ypred, x_last_hidden, out_scores, last_hidden


if __name__ == "__main__":
    model = DESIRE(IOCParams(), SGMParams())
    x = torch.rand(16, 2, 40)
    y = torch.rand(16, 2, 40)
    ypred, x_last_hidden, out_scores, last_hidden = model(x, y)
