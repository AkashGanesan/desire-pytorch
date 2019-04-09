import os
import logging
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import numpy as np

from PIL import Image
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn
import torch.optim as optim

from desire.data.loader import data_loader
from desire.utils.misc import relative_to_abs, get_dset_path
from desire.utils.misc import int_tuple, bool_flag, get_total_norm
from desire.models import DESIRE
from desire.utils.params import IOCParams, SGMParams
from desire.nn.loss import *
from PIL import Image



FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)



def absolute_to_rel_traj(abs_traj):
    rel_curr_ped_seq = torch.zeros_like(abs_traj)
    rel_curr_ped_seq[:, :, 1:] = \
                              abs_traj[:, :, 1:] - abs_traj[:, :, :-1]
    return rel_curr_ped_seq

class Annotator(object):
    def __init__(self, axes):
        self.axes = axes
        self.xdata = []
        self.ydata = []



    def mouse_press(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata

        self.xdata.append(x)
        self.ydata.append(y)
        line = Line2D(self.xdata,self.ydata)
        line.set_color('r')
        self.axes.add_line(line)
        plt.draw()

    def mouse_release(self, event):
        # Erase x and y data for new line
        self.xdata = []
        self.ydata = []

restore_model_path = '/home/akaberto/learn/desire-torch/weights/iter_490.pth'
path_of_static_image = '/home/akaberto/learn/desire-torch/zara01.background.png'
img = mpimg.imread(path_of_static_image)

fig, axes = plt.subplots()
cursor = Annotator(axes)
axes.imshow(img)
plt.axis("off")
plt.gray()
annotator = Annotator(axes)
plt.connect('button_press_event', cursor.mouse_press)
# plt.connect('button_release_event', cursor.mouse_release)

axes.plot()
plt.show()

traj = torch.tensor(list(zip(cursor.xdata, cursor.ydata)))
scene_size = (720, 576)
width = scene_size[0]
height = scene_size[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
traj_old = traj.clone().numpy()

traj[:, 0] = traj[:, 0] - width // 2
traj[:, 1] = traj[:, 1] - height // 2
traj = traj.unsqueeze(0)
traj = traj.permute(0, 2, 1)
traj = traj.to(device)
rel_traj = absolute_to_rel_traj(traj).to(device)


logger.info("Moving static image to the device")
image = Image.open(path_of_static_image)
scene = TF.to_tensor(image)
scene.unsqueeze_(0)
scene = scene.to(device)

# Create model.

desire = DESIRE(IOCParams(),
                SGMParams())
logger.info("Created model")
logger.debug(desire)
logger.info("Moving to device: {}".format(device))
desire.to(device)
logger.info("Loading state dict")
state_dict_checkpoint = torch.load(restore_model_path)
desire.load_state_dict(state_dict_checkpoint)
state_dict_checkpoint = torch.load(restore_model_path)
desire.load_state_dict(state_dict_checkpoint)



with torch.no_grad():
    pred_traj_fake_rel, pred_traj_fake_delta_rel = desire.inference(rel_traj, scene, traj[:,:,0], None)

    pred_traj_fake = relative_to_abs(pred_traj_fake_rel - pred_traj_fake_delta_rel,
                                     traj[:,:, -1])


pred_traj_fake[:, 0, :] = width // 2 + pred_traj_fake[:, 0, :]
pred_traj_fake[:, 1, :] = height // 2 + pred_traj_fake[:, 1, :]

pred_traj_fake = pred_traj_fake.cpu().detach().numpy()

print(pred_traj_fake.squeeze(0).transpose())

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, aspect='equal')
ax2.imshow(img)

ax2.add_patch(
    patches.Polygon(traj_old, closed=False, color='b'))

ax2.add_patch(
    patches.Polygon(pred_traj_fake.squeeze(0).transpose(), closed=False, color='r'))
fig2.show()
