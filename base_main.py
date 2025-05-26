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
    args = sys.argv
    if args[1] == "soft":
        for i in range(exp_time):
            train_base_soft(i)
    elif args[1] == "hard":
        for i in range(exp_time):
            train_base_hard(i)

    elif args[1] == "nic":
        for i in range(exp_time):
            train_nic(i)


if __name__ == "__main__":
    main()
