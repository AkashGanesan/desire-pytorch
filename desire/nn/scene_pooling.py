import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScenePoolingCNN(nn.Module):

    def __init__(self):
        super(ScenePoolingCNN, self).__init__()

        self.pad1 = nn.ReflectionPad2d(tuple(map(lambda x: int(np.ceil(x)),
                                                 ((0 + 5 - 2) / 2,
                                                  (0 + 5 - 2) / 2,
                                                  (0 + 5 - 2) / 2,
                                                  (0 + 5 - 2) / 2))))

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=16,
                               stride=2,
                               kernel_size=(5, 5))

        self.pad2 = nn.ReflectionPad2d((5 - 1) // 2)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               stride=1,
                               kernel_size=(5, 5))

        self.pad3 = nn.ReflectionPad2d((5 - 1) // 2)
        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               stride=1,
                               kernel_size=(5, 5))

    def forward(self, x):
        x = self.pad1(x)
        x = F.relu(self.conv1(x))
        x = self.pad2(x)
        x = F.relu(self.conv2(x))
        x = self.pad3(x)
        x = F.relu(self.conv3(x))
        return x
