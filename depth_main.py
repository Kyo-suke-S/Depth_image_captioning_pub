import numpy as np

import torch
from Captioning_models.Depth_caption_model.depth_train import train_Cdepth_soft, train_Mdepth_soft, train_Cdepth_hard, train_Mdepth_hard
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
    if args[1] == "soft" and args[2] == "cnn":
        for i in range(exp_time):
            train_Cdepth_soft(i)
    elif args[1] == "soft" and args[2] == "mlp":
        for i in range(exp_time):
            train_Mdepth_soft(i)

    elif args[1] == "hard" and args[2] == "cnn":
        for i in range(exp_time):
            train_Cdepth_hard(i)
    elif args[1] == "hard" and args[2] == "mlp":
        for i in range(exp_time):
            train_Mdepth_hard(i)


if __name__ == "__main__":
    main()