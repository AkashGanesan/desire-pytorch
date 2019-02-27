import os
import logging
import sys
import time
from collections import defaultdict

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


from PIL import Image
import torchvision.transforms.functional as TF


FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def main(dataset_name,
         path_of_static_image,
         restore_path=None,
         checkpoint_name=None,
         batch_size=1,
         num_epochs=1):
    
    train_path = get_dset_path(dataset_name, 'train')
    val_path = get_dset_path(dataset_name, 'val')
    
    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(train_path)
    # logger.info("Initializing val dataset")
    # _, val_loader = data_loader(val_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device is %s", device)

    image = Image.open(path_of_static_image)
    scene = TF.to_tensor(image)
    scene.unsqueeze_(0)
    scene = scene.to(device)
    
    iterations_per_epoch = len(train_dset) / batch_size
    if num_epochs:
        num_iterations = int(iterations_per_epoch * num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    desire = DESIRE(IOCParams(),
                    SGMParams())
    desire = desire.to(device)
    lr = 1e-3
    optimizer = optim.Adam(desire.parameters(),lr=lr)

    # Maybe restore from checkpoint
    if restore_path is not None:
        restore_dict = torch.load(os.path.join(restore_path,
                                               checkpoint_name))

        desire.load_state_dict(restore_dict)

    curr_epoch = 0
    t = 0
    print("Num iterations", num_iterations)
    while curr_epoch < num_epochs:
        for batch in train_dset:
            optimizer.zero_grad()
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, _, _= batch
            obs_traj = obs_traj.to(device)
            obs_traj_rel = obs_traj_rel.to(device)
            pred_traj = pred_traj.to(device)
            pred_traj_rel = pred_traj_rel.to(device)
            # print("obs_traj size", obs_traj.size())
            # print("pred_traj size", pred_traj.size())
            # print("obs_traj_rel size", obs_traj_rel.size())
            # print("pred_traj_rel size", pred_traj_rel.size())
            # logging.info("scene device id %s", scene.get_device())
            # logging.info("OBS_TRAJ device id: %s", obs_traj.get_device())
            x_start = obs_traj[:, :, 0].clone().to(device)
            # logging.info("x_start device id: %s", x_start.get_device())
            y_pred_traj, pred_delta, mean, log_var = desire(obs_traj_rel,
                                                            pred_traj_rel,
                                                            x_start,
                                                            scene)


            tloss, all_loss = total_loss(y_pred_traj,
                                         pred_delta,
                                         pred_traj,
                                         mean,
                                         log_var)

            tloss.backward()
            optimizer.step()
            if t % 1000 == 0:
                logging.info("Total loss %s", str(tloss))
                break
            t +=1



if __name__ == "__main__":
    print(os.getcwd())
    dataset_name = os.path.abspath("./dataset/datasets/zara1/")
    path_of_static_image = os.path.abspath("./zara01.background.png")
    a, b = main(dataset_name, path_of_static_image)
