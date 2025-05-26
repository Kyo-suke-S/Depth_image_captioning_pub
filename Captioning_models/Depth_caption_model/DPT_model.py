import torch
from torch import nn
from torchvision import transforms

import PIL

#from Captioning_models.Depth_caption_model.modules.unet import UNet
from Captioning_models.Depth_caption_model.modules.midas.dpt_depth import DPTDepthModel
#from Captioning_models.Depth_caption_model.data.transforms import get_transform

#DPT-Hybridのモデルクラス

map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
image_size = 384

class DPT_Depthestimator(nn.Module):
    def __init__(self):
        super().__init__()

        #DPT入力画像サイズ
        #self.image_size = 384
        #DPT Pretrainedパラメータのパス
        self.pretrained_weight_path = "/home/shirota/omnidata/torch/pretrained_models/omnidata_dpt_depth_v2.ckpt"
        #DPT本体
        self.model = DPTDepthModel(backbone='vitb_rn50_384')
        #tranform
        self.trans = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=0.5, std=0.5)])

    def load_weight(self):
        checkpoint = torch.load(self.pretrained_weight_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            #print("True side")
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def standardize_depth_map(self, img): #mask_valid=None, trunc_value=0.1):
        #depth mapの正規化
        #値域を0-1にする
        #if mask_valid is not None:
            #img[~mask_valid] = torch.nan

        img = torch.nan_to_num(img, nan=0.5)
        bs = img.shape[0]#img shape : [bs, 1, *, *]
        flatten_imgs = img.flatten(2,3)
        maxs = flatten_imgs.max(dim=2).values
        mins = flatten_imgs.min(dim=2).values
        dist = maxs-mins
        maxs = maxs.reshape(bs, 1, 1, 1)
        mins = mins.reshape(bs, 1, 1, 1)
        dist = dist.reshape(bs, 1, 1, 1)
        img = (img-mins)/dist
 
        return img
    
    @torch.no_grad()
    def forward(self, imgs):
        depth_maps = self.model(imgs)

        return depth_maps