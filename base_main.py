import numpy as np
import torch
from torch import nn
from Captioning_models.Base_caption_model.base_train import train_base_soft, train_base_hard
from Captioning_models.Base_caption_model.nic import train_nic
import sys

def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    np.random.seed(seed)

def main():
    torch_seed()
    exp_time = 3
    datas = ["coco", "original"]
    args = sys.argv
    if len(args)==1:
        print("input {soft/hard} {coco/original} or only nic")
        return
    elif args[1] == "soft":
        useData = args[2]
        if useData in datas:
            for i in range(exp_time):
                train_base_soft(i, useData)
        else:
            print("input coco or original")
            return
    elif args[1] == "hard":
        useData == args[2]
        if useData in datas:
            for i in range(exp_time):
                train_base_hard(i, useData)
        else:
            print("input coco or original")
            return

    elif args[1] == "nic":
        for i in range(exp_time):
            train_nic(i)


if __name__ == "__main__":
    main()
