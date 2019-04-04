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



def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    displacement = torch.cumsum(rel_traj, dim=2)
    start_pos = torch.unsqueeze(start_pos, dim=2)
    abs_traj = displacement + start_pos
    return abs_traj

def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (batch, 2, seq_len). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (batch, 2, seq_len). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    # _ , _,seq_len_ = pred_traj.size()
    loss = pred_traj_gt - pred_traj
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == "mean":
        return torch.mean(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    if mode == 'mean':
        return loss.sum()
    else:
        return torch.sum(loss)


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]

        _error = torch.sum(_error, dim=1)
        _error = torch.min(_error)
        sum_ += _error
    return sum_




def main(dataset_name,
         path_of_static_image,
         restore_model_path,
         num_samples=20):
    logger.info("Initializing train dataset")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Moving static image to the device")
    image = Image.open(path_of_static_image)
    scene = TF.to_tensor(image)
    scene.unsqueeze_(0)
    scene = scene.to(device)

    eval_path = get_dset_path(dataset_name, 'val')
    eval_dset, eval_loader = data_loader(eval_path)
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
    total_traj = 0
    ade_outer, fde_outer = [], []
    t = 0
    with torch.no_grad():
        for batch in eval_loader:
            dset, fde = [], []
            batch = [tensor.to(device) for tensor in batch]

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _, _, seq_start_end) = batch
            obs_traj = obs_traj.permute(1, 2, 0)
            pred_traj_gt = pred_traj_gt.permute(1, 2, 0)

            x_start = obs_traj[:, :, 0].to(device)
            obs_traj_rel = obs_traj_rel.permute(1, 2, 0)
            pred_traj_gt_rel = pred_traj_gt_rel.permute(1, 2, 0)


            total_traj += pred_traj_gt.size(0)
            ade, fde = [], []
            for i in range(num_samples):
                pred_traj_fake_rel, pred_traj_fake_delta_rel = desire.inference(
                    obs_traj_rel, scene, obs_traj[:, :, 0], seq_start_end
                )
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel - pred_traj_fake_delta_rel,
                                                 pred_traj_gt[:,:, 0])


                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[:, :, -1], pred_traj_gt[:, :, -1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

            if t % 10 == 0:
                t = 0
                logging.info("ade_outer {}".format(sum(ade_outer)))
                logging.info("fde_outer {}".format(sum(fde_outer)))

            t +=1


        ade = sum(ade_outer) / (total_traj * 12)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde



if __name__ == "__main__":
    print(os.getcwd())
    dataset_name = os.path.abspath("./dataset/datasets/zara1/")
    path_of_static_image = os.path.abspath("./zara01.background.png")
    restore_model_path =  os.path.abspath("/home/akaberto/learn/desire-torch/weights/iter_458.pth")
    batch = main(dataset_name, path_of_static_image, restore_model_path)
