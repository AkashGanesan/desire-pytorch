import torch
import torch.nn.functional as F
import torch.nn as nn

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def l2_loss(pred_traj, pred_traj_gt, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    batch, _, seq_len = pred_traj.size()
    loss = (pred_traj_gt - pred_traj).norm(dim=1)
    # if mode == 'sum':
    return torch.sum(loss, dim=1) / seq_len
    # elif mode == 'average':
    #     return torch.sum(loss) / torch.numel(loss_mask.data)
    # elif mode == 'raw':
    #     return loss.sum(dim=2).sum(dim=1)


def kld_loss(mean,
             log_var):
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return KLD


def cross_entropy_loss(pred_traj,
                       pred_traj_gt):
    d = (pred_traj_gt - pred_traj).norm(dim=1)
    return F.softmax(-d.max(dim=1)[0], dim=0)



def reg_loss(pred_traj,
             pred_delta,
             pred_traj_gt):
    batch, _, seq_len = pred_traj.size()
    d = (pred_traj_gt - pred_traj - pred_delta).norm(dim=1)
    return d.sum(dim=1) / seq_len


# class total_loss(nn.Module):
#     def __init__(self):
#         super(total_loss, self).__init__()

#     def forward(pred_traj,
#                 pred_delta,
#                 pred_traj_gt,
#                 mean,
#                 logvar):
#         print("Print this from total loss ", pred_traj)
#         batch, _, seq_len = pred_traj.size()
#         l2l = l2_loss(pred_traj, pred_traj_gt)
#         kld = kld_loss(mean, logvar)
#         cel = cross_entropy_loss(pred_traj, pred_traj_gt)
#         rl = reg_loss(pred_traj, pred_delta, pred_traj_gt)
        
#         tloss = (l2l + kld + cel + rl).sum() / batch
#         return tloss, (l2l, kld, cel, rl)


def total_loss(pred_traj,
               pred_delta,
               pred_traj_gt,
               mean,
               logvar):
    batch, _, seq_len = pred_traj.size()
    l2l = l2_loss(pred_traj, pred_traj_gt)
    kld = kld_loss(mean, logvar)
    cel = cross_entropy_loss(pred_traj, pred_traj_gt)
    rl = reg_loss(pred_traj, pred_delta, pred_traj_gt)

    tloss = (l2l + kld + cel + rl).sum() / batch
    return tloss, (l2l, kld, cel, rl)
