import os
import logging
import sys
import time
from collections import defaultdict
import numpy as np

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

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def train(dataset_name,
          path_of_static_image,
          restore_path=None,
          batch_size=32,
          num_epochs=700,
          norm_clip_value=1.0,
          lr = 5e-4):

    train_path = get_dset_path(dataset_name, 'train')
    val_path = get_dset_path(dataset_name, 'val')

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(train_path, batch_size=batch_size)
    # logger.info("Initializing val dataset")
    # _, val_loader = data_loader(val_path)


    device = torch.device("cuda:{}".format(get_freer_gpu()) if torch.cuda.is_available() else "cpu")
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

    optimizer = optim.Adam(desire.parameters(),lr=lr)

    # Maybe restore from checkpoint
    if restore_path is not None:
        restore_dict = torch.load(restore_path)


        desire.load_state_dict(restore_dict)

    curr_epoch = 0
    t = 0
    print("Num iterations", num_iterations)
    scene = scene.to(device)
    for epoch in range(num_epochs):

        for batch_idx, batch in enumerate(train_loader):
            logging.info("epoch {} :batch_idx {}, ".format(epoch,batch_idx))
            optimizer.zero_grad()

            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, _, _, _, _, seq_start_end) = batch

            obs_traj = obs_traj.permute(1,2,0)
            pred_traj_gt = pred_traj_gt.permute(1,2,0)

            x_start = obs_traj[:, :, 0].to(device)
            obs_traj_rel = obs_traj - obs_traj[:, :, 0].unsqueeze(2)
            pred_traj_rel = pred_traj_gt - pred_traj_gt[:, :, 0].unsqueeze(2)



            # logging.info("x_start device id: %s", x_start.get_device())

            y_pred_traj, pred_delta, mean, log_var = desire(obs_traj_rel,
                                                            pred_traj_rel,
                                                            x_start,
                                                            scene,
                                                            seq_start_end)
            tloss, (l2l,kld, cel,rl) = total_loss(y_pred_traj,
                                                  pred_delta,
                                                  pred_traj_rel,
                                                  mean,
                                                  log_var)



            l2l.backward(retain_graph=True)
            kld.backward(retain_graph=True)
            cel.backward(retain_graph=True)
            rl.backward(retain_graph=False)

            torch.nn.utils.clip_grad_norm_(desire.parameters(), norm_clip_value)
            optimizer.step()
            if t % 10 == 0:
                t = 0
                logging.info("Total loss {}; epoch = {}".format(str(tloss.item()), epoch))
                logging.info("L2L {}; RL {}; CEL {}; KLD {}; epoch = {}".format(l2l.item(),
                                                                                rl.item(),
                                                                                cel.item(),
                                                                                kld.item(),
                                                                                epoch))
            t +=1

        weight_save_path = "weights/iter_{}.pth".format(str(epoch).zfill(3))
        logging.info("Saving weights for epoch {} in {}".format(epoch, weight_save_path))
        torch.save(desire.state_dict(), weight_save_path)
        logging.info("Done saving weights for epoch {} in {}".format(epoch, weight_save_path))

if __name__ == "__main__":
    print(os.getcwd())
    dataset_name = os.path.abspath("./dataset/datasets/zara1/")
    path_of_static_image = os.path.abspath("./zara01.background.png")
    # restore_path = '/home/akaberto/learn/desire-torch/weights/iter_000.pth'
    train(dataset_name, path_of_static_image)
