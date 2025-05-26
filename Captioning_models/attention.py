import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class Gumbel_softmax(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k= k
        #self.final_temp = final_temp

    def forward(self, logits, device, temp):
        batch_size = logits.size(0)
        #print(batch_size)
        #k = 196
        #with torch.no_grad():
        u = torch.rand(batch_size, self.k)
        g = -torch.log(-torch.log(u)).detach().to(device)#Gumbel分布からの乱数
            #g.to(device)
        #print(g.shape)
        #tau = torch.tensor([self.tau]).float()
        z = (logits+g)/temp
        alpha = z.softmax(dim=1)

        return alpha
    
    #@torch.no_grad()
    #def temp_scheduler(self, step):
        #t1 = torch.exp(-(10**(-1.8))*step)
        #comp = torch.tensor([t1, self.final_temp])
        #t = torch.max(comp)
        #return t
    
    @torch.no_grad()
    def Gumbel_maxtrick(self, logits, device):
        batch_size = logits.size(0)
        #print(batch_size)
        #k = 196
        #with torch.no_grad():
        u = torch.rand(batch_size, self.k)
        g = -torch.log(-torch.log(u)).detach().to(device)#Gumbel分布からの乱数
        #print(g.shape)
        #tau = torch.tensor([self.tau]).float()
        z = logits+g
        pos = torch.argmax(z, dim=1)
        alpha = F.one_hot(pos, num_classes=self.k)

        return alpha
        
    

class Soft_Attention(nn.Module):#soft attentionの実装
    '''
    アテンション機構 (Attention mechanism)
    dim_encoder  : エンコーダ出力の特徴次元
    dim_decoder  : デコーダ出力の次元
    dim_attention: アテンション機構の次元
    '''
    def __init__(self, dim_encoder: int, 
                 dim_decoder: int, dim_attention: int):
        super().__init__()

        # z: エンコーダ出力を変換する全結合層(Wz)
        self.encoder_att = nn.Linear(dim_encoder, dim_attention)#baseはdim_encoder=2048, depthではMLPの場合dim_encoder=2048+mlp_dim_out

        # h: デコーダ出力を変換する全結合層(Wh)
        self.decoder_att = nn.Linear(dim_decoder, dim_attention)

        # e: アライメントスコアを計算するための全結合層
        self.full_att = nn.Linear(dim_attention, 1)

        # α: アテンション重みを計算する活性化関数
        self.relu = nn.ReLU(inplace=True)

    '''
    アテンション機構の順伝播
    encoder_out   : エンコーダ出力,
                    [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
    decoder_hidden: デコーダ隠れ状態の次元
    '''
    def forward(self, encoder_out: torch.Tensor, 
                decoder_hidden: torch.Tensor):
        # e: アライメントスコア
        att1 = self.encoder_att(encoder_out)    # Wz * z [バッチサイズ, 196, D]=>[バッチサイズ, 196, att_D]
        att2 = self.decoder_att(decoder_hidden) # Wh * h_{t-1} [バッチサイズ, hidden_D]=>[バッチサイズ, att_D]
        att = self.full_att(
                self.relu(att1 + att2.unsqueeze(1))).squeeze(2) #[バッチサイズ, 196]

        # α: T個の部分領域ごとのアテンション重み
        alpha = att.softmax(dim=1)

        # c: コンテキストベクトル
        context_vector = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)#[バッチサイズ, 196, D]

        return context_vector, alpha
    
# ------------------------------------------------------------------------------------------------------------------

class Hard_Attention(nn.Module):#hard attentionの実装
    '''
    アテンション機構 (Attention mechanism)
    dim_encoder  : エンコーダ出力の特徴次元
    dim_decoder  : デコーダ出力の次元
    dim_attention: アテンション機構の次元
    '''
    def __init__(self, dim_encoder: int, 
                 dim_decoder: int, dim_attention: int,
                 k = 196):
        super().__init__()

        # z: エンコーダ出力を変換する全結合層(Wz)
        self.encoder_att = nn.Linear(dim_encoder, dim_attention)

        # h: デコーダ出力を変換する全結合層(Wh)
        self.decoder_att = nn.Linear(dim_decoder, dim_attention)

        # e: アライメントスコアを計算するための全結合層
        self.full_att = nn.Linear(dim_attention, 1)

        # α: アテンション重みを計算する活性化関数
        self.relu = nn.ReLU(inplace=True)

        #Gumbel softmax
        self.gumbel_softmax = Gumbel_softmax(k)

    '''
    アテンション機構の順伝播
    encoder_out   : エンコーダ出力,
                    [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
    decoder_hidden: デコーダ隠れ状態の次元
    '''
    def forward(self, encoder_out: torch.Tensor, 
                decoder_hidden: torch.Tensor,
                device: str,
                temp: torch.Tensor):
        # e: アライメントスコア
        att1 = self.encoder_att(encoder_out)    # Wz * z [バッチサイズ, 196, D]=>[バッチサイズ, 196, att_D]
        att2 = self.decoder_att(decoder_hidden) # Wh * h_{t-1} [バッチサイズ, hidden_D]=>[バッチサイズ, att_D]
        att = self.full_att(
                self.relu(att1 + att2.unsqueeze(1))).squeeze(2) #[バッチサイズ, 196]

        # α: T個の部分領域ごとのアテンション重み
        alpha = self.gumbel_softmax(att, device, temp)

        # c: コンテキストベクトル
        context_vector = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)#[バッチサイズ, 196, D]=>[バッチサイズ, D]

        return context_vector, alpha
    
    @torch.no_grad()
    def Hard_sample(self, encoder_out: torch.Tensor, 
                decoder_hidden: torch.Tensor,
                device: str):
        # e: アライメントスコア
        att1 = self.encoder_att(encoder_out)    # Wz * z [バッチサイズ, 196, D]=>[バッチサイズ, 196, att_D]
        att2 = self.decoder_att(decoder_hidden) # Wh * h_{t-1} [バッチサイズ, hidden_D]=>[バッチサイズ, att_D]
        att = self.full_att(
                self.relu(att1 + att2.unsqueeze(1))).squeeze(2) #[バッチサイズ, 196]

        # α: T個の部分領域ごとのアテンション重み(one-hot)
        alpha = self.gumbel_softmax.Gumbel_maxtrick(att, device)
        #print(alpha)

        # c: コンテキストベクトル
        context_vector = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)#[バッチサイズ, 196, D]

        return context_vector, alpha



