import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

Unfolder = nn.Unfold(kernel_size=(16,16), stride=16)
#奥行き画像を分割する関数
def img_to_patch(imgs, patch_size=(16,16), stride=16):
    Unfolder = nn.Unfold(kernel_size=patch_size, stride=stride)
    unfolded = Unfolder(imgs)#[bactch_size, 256(16*16*1), 196]
    unfolded = unfolded.permute(0, 2, 1)#[batch_size, 196, 256(=16*16)]

    return unfolded

def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan

    img = torch.nan_to_num(img, nan=0.5)
    bs = img.shape[0]#img shape :[bs, 1, *, *]
    flatten_imgs = img.flatten(2,3)
    maxs = flatten_imgs.max(dim=2).values
    mins = flatten_imgs.min(dim=2).values
    dist = maxs-mins
    maxs = maxs.reshape(bs, 1, 1, 1)
    mins = mins.reshape(bs, 1, 1, 1)
    dist = dist.reshape(bs, 1, 1, 1)
    img = (img-mins)/dist

    return img