import os
import numpy as np
#import datetime
from tqdm import tqdm
import pickle
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.transforms as T
import torchvision.datasets as dataset
from Captioning_models.Base_caption_model.base_caption_models import CNNEncoder_Atten
from Captioning_models.Depth_caption_model.depth_models import Depth_CNN_endoder, Depth_MLP_endoder, CD_RNNDecoderWithSoftAttention, MD_RNNDecoderWithSoftAttention,\
CD_RNNDecoderWithHardAttention, MD_RNNDecoderWithHardAttention
from Captioning_models.Depth_caption_model.DPT_model import DPT_Depthestimator
from Captioning_models.config import ConfigTrain

import Captioning_models.util as util
import gc

#tqdmバーを表示するかしないか
tqdm_disable=False
lam = 0.7  # regularization parameter for 'doubly stochastic attention', as in the paper

def train_Cdepth_soft(ext):#CNNによる奥行きエンコーダを使った画像
    config = ConfigTrain()
    depth_dic = {}
    depth_dic_val = {}

    # 辞書（単語→単語ID）の読み込み
    with open(config.ori_word_to_id_file, 'rb') as f:
        word_to_id = pickle.load(f)

    # 辞書サイズを保存
    vocab_size = len(word_to_id)
        
    # モデル出力用のディレクトリを作成
    os.makedirs(config.save_directory_Cdep_s_ori, exist_ok=True)

    # 画像のtransformsを定義
    transforms = T.Compose([
        T.Resize((224, 224)),
        #T.RandomHorizontalFlip(),
        T.ToTensor()
        # ImageNetデータセットの平均と標準偏差
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])

    #通常のCNNエンコーダ入力用トランスフォーム
    #base_trandforms = T.Compose([
        #T.Resize((224,224)),
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #])

    depth_transforms = T.Compose([T.Resize((224,224))])

    # COCOデータロードの定義
    #train_dataset = dataset.CocoCaptions(root=config.img_directory, 
                                         #annFile=config.anno_file, 
                                         #transform=transforms)
    train_dataset = dataset.CocoCaptions(root=config.train_img_directory, 
                                         annFile=config.ori_train_anno_file, 
                                         transform=transforms)
    val_dataset = dataset.CocoCaptions(root=config.val_img_directory, 
                                         annFile=config.ori_val_anno_file, 
                                         transform=transforms)
    
    # Subset samplerの生成
    #val_set, train_set = util.generate_subset(
        #train_dataset, config.val_ratio)

    # 学習時にランダムにサンプルするためのサンプラー
    #train_sampler = SubsetRandomSampler(train_set)

    # DataLoaderを生成
    collate_func_lambda = lambda x: util.collate_func_for_dep(x, word_to_id)
    train_loader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True,
                        num_workers=config.num_workers, 
                        collate_fn=collate_func_lambda)
    val_loader = torch.utils.data.DataLoader(
                        val_dataset, 
                        batch_size=config.batch_size, 
                        num_workers=config.num_workers, 
                        collate_fn=collate_func_lambda)
    #train_loader = torch.utils.data.DataLoader(
                        #train_dataset, 
                        #batch_size=config.batch_size, 
                        #num_workers=config.num_workers, 
                        #sampler=train_sampler,
                        #collate_fn=collate_func_lambda)
    #val_loader = torch.utils.data.DataLoader(
                        #train_dataset, 
                        #batch_size=config.batch_size, 
                        #num_workers=config.num_workers, 
                        #sampler=val_set,
                        #collate_fn=collate_func_lambda)

    # モデルの定義
    encoder = CNNEncoder_Atten(config.enc_img_size)
    decoder = CD_RNNDecoderWithSoftAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size,
                                      config.dropout)
    dpt = DPT_Depthestimator()
    depth_encoder = Depth_CNN_endoder(config.enc_img_size)

    encoder.to(config.device)
    decoder.to(config.device)
    dpt.to(config.device)
    depth_encoder.to(config.device)
    
    dpt.load_weight()#DPT 学習済みパラメータのロード

    # 損失関数の定義
    loss_func = lambda x, y: F.cross_entropy(
        x, y, ignore_index=word_to_id.get('<null>', None))
    
    # 最適化手法の定義
    params = list(decoder.parameters())+list(depth_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr)
    
    # 学習率スケジューラの定義
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=config.lr_drop, gamma=0.1)

    # 学習経過の書き込み
    train_loss_file = '{}/depth_soft_train_loss_ori{}.csv'\
        .format(config.save_directory_Cdep_s_ori, ext)
    val_loss_file = '{}/depth_soft_val_loss_ori{}.csv'\
        .format(config.save_directory_Cdep_s_ori, ext)
    #now = datetime.datetime.now()
    #train_loss_file = '{}/base_soft_train_loss_{}.csv'\
        #.format(config.save_directory_soft, now.strftime('%Y%m%d_%H%M%S'))
    #val_loss_file = '{}/base_soft_val_loss_{}.csv'\
        #.format(config.save_directory_soft, now.strftime('%Y%m%d_%H%M%S'))

    # 学習
    val_loss_best = float('inf')
    for epoch in range(config.num_epochs):
        with tqdm(train_loader, disable=tqdm_disable) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')

            # 学習モードに設定
            encoder.train()
            decoder.train()
            depth_encoder.train()
            dpt.eval()

            train_losses = deque()
            train_losses_list = []
            for imgs, imgs_for_dep, captions, lengths, allcaps in pbar:
                # ミニバッチを設定
                imgs = imgs.to(config.device)
                imgs_for_dep = imgs_for_dep.to(config.device)
                captions = captions.to(config.device)

                #imgs = base_trandforms(imgs)
                #imgs_for_dep = dpt.trans(imgs_for_dep)

                optimizer.zero_grad()

                features = encoder(imgs)#カラー画像のエンコード

                depth_map = features.new_zeros((config.batch_size,1,1))

                # エンコーダ-デコーダモデル
                if epoch == 0:
                    #imgs_for_dep = imgs_for_dep.to(config.device)
                    depth_map = dpt(imgs_for_dep)
                    depth_map = depth_map.unsqueeze(1)
                    #depth_map = depth_map.detach()
                    depth_map = dpt.standardize_depth_map(depth_map)
                    depth_map = depth_transforms(depth_map)
                    #imgsc = imgs
                    dpc = depth_map.detach().cpu()
                    for i in range(len(imgs)):
                            depth_dic[allcaps[i]] = dpc[i]
                    
                else:
                    for i in range(len(imgs)):
                        if i==0:
                            depth_map = depth_dic[allcaps[i]].unsqueeze(0).to(config.device)
                        else:
                            dm = depth_dic[allcaps[i]].unsqueeze(0).to(config.device)
                            depth_map = torch.cat([depth_map, dm], dim=0)
                
                depth_map = depth_map.detach()
                
                depth_features = depth_encoder(depth_map)#depth mapのエンコード
                outputs, alphas = decoder(features, depth_features, captions, lengths)

                # 損失の計算
                captions = captions[:, 1:] 
                lengths = [length - 1 for length in lengths]
                targets = pack_padded_sequence(captions, lengths, 
                                               batch_first=True)
                loss = loss_func(outputs.data, targets.data)
                # Add doubly stochastic attention regularization
                loss += lam * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # 誤差逆伝播
                loss.backward()
                
                optimizer.step()

                # 学習時の損失をログに書き込み
                train_losses.append(loss.item())
                train_losses_list.append(loss.item())
                if len(train_losses) > config.moving_avg:
                    train_losses.popleft()
                pbar.set_postfix({
                    'loss': torch.Tensor(train_losses).mean().item()})
                
        train_loss = np.mean(train_losses_list)
        with open(train_loss_file, 'a') as f:
            print(f'{epoch}, {train_loss}', file=f)

        print(f"[epoch:{epoch}] train loss: {train_loss}")

        # 検証
        with tqdm(val_loader, disable=tqdm_disable) as pbar:
            pbar.set_description(f'[検証]')

            # 評価モード
            encoder.eval()
            decoder.eval()
            depth_encoder.eval()
            dpt.eval()

            val_losses = []
            for imgs, imgs_for_dep, captions, lengths, allcaps in pbar:

                # ミニバッチを設定
                imgs = imgs.to(config.device)
                imgs_for_dep = imgs_for_dep.to(config.device)
                captions = captions.to(config.device)

                # エンコーダ-デコーダモデル
                features = encoder(imgs)

                depth_map = features.new_zeros((config.batch_size,1,1))

                # エンコーダ-デコーダモデル
                if epoch == 0:
                    #imgs_for_dep = imgs_for_dep.to(config.device)
                    depth_map = dpt(imgs_for_dep)
                    depth_map = depth_map.unsqueeze(1)
                    #depth_map = depth_map.detach()
                    depth_map = dpt.standardize_depth_map(depth_map)
                    depth_map = depth_transforms(depth_map)
                    #imgsc = imgs
                    dpc = depth_map.detach().cpu()
                    for i in range(len(imgs)):
                            depth_dic_val[allcaps[i]] = dpc[i]

                else:
                    for i in range(len(imgs)):
                        if i==0:
                            depth_map = depth_dic_val[allcaps[i]].unsqueeze(0).to(config.device)
                        else:
                            dm = depth_dic_val[allcaps[i]].unsqueeze(0).to(config.device)
                            depth_map = torch.cat([depth_map, dm], dim=0)
                
                depth_map = depth_map.detach()
                depth_features = depth_encoder(depth_map)#depth mapのエンコード
                outputs, alphas = decoder(features, depth_features, captions, lengths)

                # 損失の計算
                captions = captions[:, 1:] 
                lengths = [length - 1 for length in lengths]
                targets = pack_padded_sequence(captions, lengths, 
                                               batch_first=True)
                loss = loss_func(outputs.data, targets.data)
                loss += lam * ((1. - alphas.sum(dim=1)) ** 2).mean()
                val_losses.append(loss.item())

                # Validation Lossをログに書き込み
                #with open(val_loss_file, 'a') as f:
                    #print(f'{epoch}, {loss.item()}', file=f)

        # Loss 表示
        val_loss = np.mean(val_losses)
        #print(f'Validation loss: {val_loss}')
        print(f'[epoch:{epoch}] Validation loss: {val_loss}')
        with open(val_loss_file, 'a') as f:
            print(f'{epoch}, {val_loss}', file=f)

        # より良い検証結果が得られた場合、モデルを保存
        if val_loss < val_loss_best:
            val_loss_best = val_loss

            # エンコーダモデルを保存
            torch.save(
                encoder.state_dict(),
                f'{config.save_directory_Cdep_s_ori}/depth_soft_encoder_best_ori{ext}.pth')

            # デコーダモデルを保存
            torch.save(
                decoder.state_dict(),
                f'{config.save_directory_Cdep_s_ori}/depth_soft_decoder_best_ori{ext}.pth')
            
            # 奥行きエンコーダを保存
            torch.save(
                depth_encoder.state_dict(),
                f'{config.save_directory_Cdep_s_ori}/depth_soft_D_encoder_best_ori{ext}.pth')
            
    del depth_dic, depth_dic_val, dpt, encoder, decoder, depth_encoder 
    gc.collect()

# --------------------------------------------------------------------------------------

def train_Mdepth_soft(ext):
    config = ConfigTrain()
    depth_dic = {}
    depth_dic_val = {}

    # 辞書（単語→単語ID）の読み込み
    with open(config.word_to_id_file, 'rb') as f:
        word_to_id = pickle.load(f)

    # 辞書サイズを保存
    vocab_size = len(word_to_id)
        
    # モデル出力用のディレクトリを作成
    os.makedirs(config.save_directory_Mdep_s, exist_ok=True)

    # 画像のtransformsを定義
    transforms = T.Compose([
        T.Resize((224, 224)),
        #T.RandomHorizontalFlip(),
        T.ToTensor()
        # ImageNetデータセットの平均と標準偏差
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])

    #通常のCNNエンコーダ入力用トランスフォーム
    #base_trandforms = T.Compose([
        #T.Resize((224,224)),
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #])

    depth_transforms = T.Compose([T.Resize((224,224))])

    # COCOデータロードの定義
    #train_dataset = dataset.CocoCaptions(root=config.img_directory, 
                                         #annFile=config.anno_file, 
                                         #transform=transforms)
    train_dataset = dataset.CocoCaptions(root=config.train_img_directory, 
                                         annFile=config.train_anno_file, 
                                         transform=transforms)
    val_dataset = dataset.CocoCaptions(root=config.val_img_directory, 
                                         annFile=config.val_anno_file, 
                                         transform=transforms)
    
    # Subset samplerの生成
    #val_set, train_set = util.generate_subset(
        #train_dataset, config.val_ratio)

    # 学習時にランダムにサンプルするためのサンプラー
    #train_sampler = SubsetRandomSampler(train_set)

    # DataLoaderを生成
    collate_func_lambda = lambda x: util.collate_func_for_dep(x, word_to_id)
    train_loader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True,
                        num_workers=config.num_workers, 
                        collate_fn=collate_func_lambda)
    val_loader = torch.utils.data.DataLoader(
                        val_dataset, 
                        batch_size=config.batch_size, 
                        num_workers=config.num_workers, 
                        collate_fn=collate_func_lambda)
    #train_loader = torch.utils.data.DataLoader(
                        #train_dataset, 
                        #batch_size=config.batch_size, 
                        #num_workers=config.num_workers, 
                        #sampler=train_sampler,
                        #collate_fn=collate_func_lambda)
    #val_loader = torch.utils.data.DataLoader(
                        #train_dataset, 
                        #batch_size=config.batch_size, 
                        #num_workers=config.num_workers, 
                        #sampler=val_set,
                        #collate_fn=collate_func_lambda)

    # モデルの定義
    encoder = CNNEncoder_Atten(config.enc_img_size)
    decoder = MD_RNNDecoderWithSoftAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.mlp_dim_encoder,
                                      config.dim_hidden,
                                      vocab_size,
                                      config.dropout)
    dpt = DPT_Depthestimator()
    depth_encoder = Depth_MLP_endoder(config.dim_l1, config.dim_l2, config.dim_out)

    encoder.to(config.device)
    decoder.to(config.device)
    dpt.to(config.device)
    depth_encoder.to(config.device)
    
    dpt.load_weight()#DPT 学習済みパラメータのロード

    # 損失関数の定義
    loss_func = lambda x, y: F.cross_entropy(
        x, y, ignore_index=word_to_id.get('<null>', None))
    
    # 最適化手法の定義
    params = list(decoder.parameters())+list(depth_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr)
    
    # 学習率スケジューラの定義
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=config.lr_drop, gamma=0.1)

    # 学習経過の書き込み
    train_loss_file = '{}/Mdepth_soft_train_loss_{}.csv'\
        .format(config.save_directory_Mdep_s, ext)
    val_loss_file = '{}/Mdepth_soft_val_loss_{}.csv'\
        .format(config.save_directory_Mdep_s, ext)
    #now = datetime.datetime.now()
    #train_loss_file = '{}/base_soft_train_loss_{}.csv'\
        #.format(config.save_directory_soft, now.strftime('%Y%m%d_%H%M%S'))
    #val_loss_file = '{}/base_soft_val_loss_{}.csv'\
        #.format(config.save_directory_soft, now.strftime('%Y%m%d_%H%M%S'))

    # 学習
    val_loss_best = float('inf')
    for epoch in range(config.num_epochs):
        with tqdm(train_loader, disable=tqdm_disable) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')

            # 学習モードに設定
            encoder.train()
            decoder.train()
            depth_encoder.train()
            dpt.eval()

            train_losses = deque()
            train_losses_list = []
            for imgs, imgs_for_dep, captions, lengths, allcaps in pbar:
                # ミニバッチを設定
                imgs = imgs.to(config.device)
                imgs_for_dep = imgs_for_dep.to(config.device)
                captions = captions.to(config.device)

                #imgs = base_trandforms(imgs)
                #imgs_for_dep = dpt.trans(imgs_for_dep)

                optimizer.zero_grad()

                features = encoder(imgs)#カラー画像のエンコード

                depth_map = features.new_zeros((config.batch_size,1,1))

                # エンコーダ-デコーダモデル
                if epoch == 0:
                    #imgs_for_dep = imgs_for_dep.to(config.device)
                    depth_map = dpt(imgs_for_dep)
                    depth_map = depth_map.unsqueeze(1)
                    #depth_map = depth_map.detach()
                    depth_map = dpt.standardize_depth_map(depth_map)
                    depth_map = depth_transforms(depth_map)
                    #imgsc = imgs
                    dpc = depth_map.detach().cpu()
                    for i in range(len(imgs)):
                            depth_dic[allcaps[i]] = dpc[i]
                    
                else:
                    for i in range(len(imgs)):
                        if i==0:
                            depth_map = depth_dic[allcaps[i]].unsqueeze(0).to(config.device)
                        else:
                            dm = depth_dic[allcaps[i]].unsqueeze(0).to(config.device)
                            depth_map = torch.cat([depth_map, dm], dim=0)
                
                depth_map = depth_map.detach()
                depth_map = depth_encoder.img_to_patch(depth_map)
                
                depth_features = depth_encoder(depth_map)#depth mapのエンコード
                outputs, alphas = decoder(features, depth_features, captions, lengths)

                # 損失の計算
                captions = captions[:, 1:] 
                lengths = [length - 1 for length in lengths]
                targets = pack_padded_sequence(captions, lengths, 
                                               batch_first=True)
                loss = loss_func(outputs.data, targets.data)
                # Add doubly stochastic attention regularization
                loss += lam * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # 誤差逆伝播
                loss.backward()
                
                optimizer.step()

                # 学習時の損失をログに書き込み
                train_losses.append(loss.item())
                train_losses_list.append(loss.item())
                if len(train_losses) > config.moving_avg:
                    train_losses.popleft()
                pbar.set_postfix({
                    'loss': torch.Tensor(train_losses).mean().item()})
                
        train_loss = np.mean(train_losses_list)
        with open(train_loss_file, 'a') as f:
            print(f'{epoch}, {train_loss}', file=f)

        print(f"[epoch:{epoch}] train loss: {train_loss}")

        # 検証
        with tqdm(val_loader, disable=tqdm_disable) as pbar:
            pbar.set_description(f'[検証]')

            # 評価モード
            encoder.eval()
            decoder.eval()
            depth_encoder.eval()
            dpt.eval()

            val_losses = []
            for imgs, imgs_for_dep, captions, lengths, allcaps in pbar:

                # ミニバッチを設定
                imgs = imgs.to(config.device)
                imgs_for_dep = imgs_for_dep.to(config.device)
                captions = captions.to(config.device)

                # エンコーダ-デコーダモデル
                features = encoder(imgs)

                depth_map = features.new_zeros((config.batch_size,1,1))

                # エンコーダ-デコーダモデル
                if epoch == 0:
                    #imgs_for_dep = imgs_for_dep.to(config.device)
                    depth_map = dpt(imgs_for_dep)
                    depth_map = depth_map.unsqueeze(1)
                    #depth_map = depth_map.detach()
                    depth_map = dpt.standardize_depth_map(depth_map)
                    depth_map = depth_transforms(depth_map)
                    #imgsc = imgs
                    dpc = depth_map.detach().cpu()
                    for i in range(len(imgs)):
                            depth_dic_val[allcaps[i]] = dpc[i]

                else:
                    for i in range(len(imgs)):
                        if i==0:
                            depth_map = depth_dic_val[allcaps[i]].unsqueeze(0).to(config.device)
                        else:
                            dm = depth_dic_val[allcaps[i]].unsqueeze(0).to(config.device)
                            depth_map = torch.cat([depth_map, dm], dim=0)
                
                depth_map = depth_map.detach()
                depth_map = depth_encoder.img_to_patch(depth_map)
                depth_features = depth_encoder(depth_map)#depth mapのエンコード
                outputs, alphas = decoder(features, depth_features, captions, lengths)

                # 損失の計算
                captions = captions[:, 1:] 
                lengths = [length - 1 for length in lengths]
                targets = pack_padded_sequence(captions, lengths, 
                                               batch_first=True)
                loss = loss_func(outputs.data, targets.data)
                loss += lam * ((1. - alphas.sum(dim=1)) ** 2).mean()
                val_losses.append(loss.item())

                # Validation Lossをログに書き込み
                #with open(val_loss_file, 'a') as f:
                    #print(f'{epoch}, {loss.item()}', file=f)

        # Loss 表示
        val_loss = np.mean(val_losses)
        #print(f'Validation loss: {val_loss}')
        print(f'[epoch:{epoch}] Validation loss: {val_loss}')
        with open(val_loss_file, 'a') as f:
            print(f'{epoch}, {val_loss}', file=f)

        # より良い検証結果が得られた場合、モデルを保存
        if val_loss < val_loss_best:
            val_loss_best = val_loss

            # エンコーダモデルを保存
            torch.save(
                encoder.state_dict(),
                f'{config.save_directory_Mdep_s}/Mdepth_soft_encoder_best{ext}.pth')

            # デコーダモデルを保存
            torch.save(
                decoder.state_dict(),
                f'{config.save_directory_Mdep_s}/Mdepth_soft_decoder_best{ext}.pth')
            
            # 奥行きエンコーダを保存
            torch.save(
                depth_encoder.state_dict(),
                f'{config.save_directory_Mdep_s}/Mdepth_soft_D_encoder_best{ext}.pth')
            
    del depth_dic, depth_dic_val, dpt, encoder, decoder, depth_encoder 
    gc.collect()
    
def temp_anneal(epoch_num):
    temp = np.array(np.cos(np.pi*(epoch_num/360)))
    if temp <= 0.5:
        temp = np.array(0.5)
    
    temp = temp.astype(np.float32)
    temp = torch.from_numpy(temp).clone()
    return temp            

def train_Cdepth_hard(ext):
    config = ConfigTrain()
    depth_dic = {}
    depth_dic_val = {}

    # 辞書（単語→単語ID）の読み込み
    with open(config.ori_word_to_id_file, 'rb') as f:
        word_to_id = pickle.load(f)

    # 辞書サイズを保存
    vocab_size = len(word_to_id)
        
    # モデル出力用のディレクトリを作成
    os.makedirs(config.save_directory_Cdep_h_ori, exist_ok=True)

    # 画像のtransformsを定義
    transforms = T.Compose([
        T.Resize((224, 224)),
        #T.RandomHorizontalFlip(),
        T.ToTensor()
        # ImageNetデータセットの平均と標準偏差
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])

    #通常のCNNエンコーダ入力用トランスフォーム
    #base_trandforms = T.Compose([
        #T.Resize((224,224)),
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #])

    depth_transforms = T.Compose([T.Resize((224,224))])

    # COCOデータロードの定義
    #train_dataset = dataset.CocoCaptions(root=config.img_directory, 
                                         #annFile=config.anno_file, 
                                         #transform=transforms)
    train_dataset = dataset.CocoCaptions(root=config.train_img_directory, 
                                         annFile=config.ori_train_anno_file, 
                                         transform=transforms)
    val_dataset = dataset.CocoCaptions(root=config.val_img_directory, 
                                         annFile=config.ori_val_anno_file, 
                                         transform=transforms)
    
    # Subset samplerの生成
    #val_set, train_set = util.generate_subset(
        #train_dataset, config.val_ratio)

    # 学習時にランダムにサンプルするためのサンプラー
    #train_sampler = SubsetRandomSampler(train_set)

    # DataLoaderを生成
    collate_func_lambda = lambda x: util.collate_func_for_dep(x, word_to_id)
    train_loader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True,
                        num_workers=config.num_workers, 
                        collate_fn=collate_func_lambda)
    val_loader = torch.utils.data.DataLoader(
                        val_dataset, 
                        batch_size=config.batch_size, 
                        num_workers=config.num_workers, 
                        collate_fn=collate_func_lambda)
    #train_loader = torch.utils.data.DataLoader(
                        #train_dataset, 
                        #batch_size=config.batch_size, 
                        #num_workers=config.num_workers, 
                        #sampler=train_sampler,
                        #collate_fn=collate_func_lambda)
    #val_loader = torch.utils.data.DataLoader(
                        #train_dataset, 
                        #batch_size=config.batch_size, 
                        #num_workers=config.num_workers, 
                        #sampler=val_set,
                        #collate_fn=collate_func_lambda)

    # モデルの定義
    encoder = CNNEncoder_Atten(config.enc_img_size)
    decoder = CD_RNNDecoderWithHardAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size,
                                      config.device,
                                      config.dropout)
    dpt = DPT_Depthestimator()
    depth_encoder = Depth_CNN_endoder(config.enc_img_size)

    encoder.to(config.device)
    decoder.to(config.device)
    dpt.to(config.device)
    depth_encoder.to(config.device)
    
    dpt.load_weight()#DPT 学習済みパラメータのロード

    # 損失関数の定義
    loss_func = lambda x, y: F.cross_entropy(
        x, y, ignore_index=word_to_id.get('<null>', None))
    
    # 最適化手法の定義
    params = list(decoder.parameters())+list(depth_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr)
    
    # 学習率スケジューラの定義
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=config.lr_drop, gamma=0.1)

    # 学習経過の書き込み
    train_loss_file = '{}/depth_hard_train_loss_ori{}.csv'\
        .format(config.save_directory_Cdep_h_ori, ext)
    val_loss_file = '{}/depth_hard_val_loss_ori{}.csv'\
        .format(config.save_directory_Cdep_h_ori, ext)
    #now = datetime.datetime.now()
    #train_loss_file = '{}/base_soft_train_loss_{}.csv'\
        #.format(config.save_directory_soft, now.strftime('%Y%m%d_%H%M%S'))
    #val_loss_file = '{}/base_soft_val_loss_{}.csv'\
        #.format(config.save_directory_soft, now.strftime('%Y%m%d_%H%M%S'))

    # 学習
    val_loss_best = float('inf')
    temp = torch.tensor(1.0).detach().to(config.device)
    for epoch in range(config.num_epochs):
        with tqdm(train_loader, disable=tqdm_disable) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')

            # 学習モードに設定
            encoder.train()
            decoder.train()
            depth_encoder.train()
            dpt.eval()

            train_losses = deque()
            train_losses_list = []
            if epoch != 0 and epoch%10==0:
                tempi = temp_anneal(epoch)
                temp = tempi.detach().to(config.device)
            for imgs, imgs_for_dep, captions, lengths, allcaps in pbar:
                # ミニバッチを設定
                imgs = imgs.to(config.device)
                imgs_for_dep = imgs_for_dep.to(config.device)
                captions = captions.to(config.device)

                #imgs = base_trandforms(imgs)
                #imgs_for_dep = dpt.trans(imgs_for_dep)

                optimizer.zero_grad()

                features = encoder(imgs)#カラー画像のエンコード

                depth_map = features.new_zeros((config.batch_size,1,1))

                # エンコーダ-デコーダモデル
                if epoch == 0:
                    #imgs_for_dep = imgs_for_dep.to(config.device)
                    depth_map = dpt(imgs_for_dep)
                    depth_map = depth_map.unsqueeze(1)
                    #depth_map = depth_map.detach()
                    depth_map = dpt.standardize_depth_map(depth_map)
                    depth_map = depth_transforms(depth_map)
                    #imgsc = imgs
                    dpc = depth_map.detach().cpu()
                    for i in range(len(imgs)):
                            depth_dic[allcaps[i]] = dpc[i]
                    
                else:
                    for i in range(len(imgs)):
                        if i==0:
                            depth_map = depth_dic[allcaps[i]].unsqueeze(0).to(config.device)
                        else:
                            dm = depth_dic[allcaps[i]].unsqueeze(0).to(config.device)
                            depth_map = torch.cat([depth_map, dm], dim=0)
                
                depth_map = depth_map.detach()
                
                depth_features = depth_encoder(depth_map)#depth mapのエンコード
                outputs = decoder(features, depth_features, captions, lengths, temp)

                # 損失の計算
                captions = captions[:, 1:] 
                lengths = [length - 1 for length in lengths]
                targets = pack_padded_sequence(captions, lengths, 
                                               batch_first=True)
                loss = loss_func(outputs.data, targets.data)
                # Add doubly stochastic attention regularization
                #loss += lam * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # 誤差逆伝播
                loss.backward()
                
                optimizer.step()

                # 学習時の損失をログに書き込み
                train_losses.append(loss.item())
                train_losses_list.append(loss.item())
                if len(train_losses) > config.moving_avg:
                    train_losses.popleft()
                pbar.set_postfix({
                    'loss': torch.Tensor(train_losses).mean().item()})
                
        train_loss = np.mean(train_losses_list)
        with open(train_loss_file, 'a') as f:
            print(f'{epoch}, {train_loss}', file=f)

        print(f"[epoch:{epoch}] train loss: {train_loss}")

        # 検証
        with tqdm(val_loader, disable=tqdm_disable) as pbar:
            pbar.set_description(f'[検証]')

            # 評価モード
            encoder.eval()
            decoder.eval()
            depth_encoder.eval()
            dpt.eval()

            val_losses = []
            for imgs, imgs_for_dep, captions, lengths, allcaps in pbar:

                # ミニバッチを設定
                imgs = imgs.to(config.device)
                imgs_for_dep = imgs_for_dep.to(config.device)
                captions = captions.to(config.device)

                # エンコーダ-デコーダモデル
                features = encoder(imgs)

                depth_map = features.new_zeros((config.batch_size,1,1))

                # エンコーダ-デコーダモデル
                if epoch == 0:
                    #imgs_for_dep = imgs_for_dep.to(config.device)
                    depth_map = dpt(imgs_for_dep)
                    depth_map = depth_map.unsqueeze(1)
                    #depth_map = depth_map.detach()
                    depth_map = dpt.standardize_depth_map(depth_map)
                    depth_map = depth_transforms(depth_map)
                    #imgsc = imgs
                    dpc = depth_map.detach().cpu()
                    for i in range(len(imgs)):
                            depth_dic_val[allcaps[i]] = dpc[i]

                else:
                    for i in range(len(imgs)):
                        if i==0:
                            depth_map = depth_dic_val[allcaps[i]].unsqueeze(0).to(config.device)
                        else:
                            dm = depth_dic_val[allcaps[i]].unsqueeze(0).to(config.device)
                            depth_map = torch.cat([depth_map, dm], dim=0)
                
                depth_map = depth_map.detach()
                depth_features = depth_encoder(depth_map)#depth mapのエンコード
                outputs = decoder.eval_forward(features, depth_features, captions, lengths)

                # 損失の計算
                captions = captions[:, 1:] 
                lengths = [length - 1 for length in lengths]
                targets = pack_padded_sequence(captions, lengths, 
                                               batch_first=True)
                loss = loss_func(outputs.data, targets.data)
                #loss += lam * ((1. - alphas.sum(dim=1)) ** 2).mean()
                val_losses.append(loss.item())

                # Validation Lossをログに書き込み
                #with open(val_loss_file, 'a') as f:
                    #print(f'{epoch}, {loss.item()}', file=f)

        # Loss 表示
        val_loss = np.mean(val_losses)
        #print(f'Validation loss: {val_loss}')
        print(f'[epoch:{epoch}] Validation loss: {val_loss}')
        with open(val_loss_file, 'a') as f:
            print(f'{epoch}, {val_loss}', file=f)

        # より良い検証結果が得られた場合、モデルを保存
        if val_loss < val_loss_best:
            val_loss_best = val_loss

            # エンコーダモデルを保存
            torch.save(
                encoder.state_dict(),
                f'{config.save_directory_Cdep_h_ori}/depth_hard_encoder_best_ori{ext}.pth')

            # デコーダモデルを保存
            torch.save(
                decoder.state_dict(),
                f'{config.save_directory_Cdep_h_ori}/depth_hard_decoder_best_ori{ext}.pth')
            
            # 奥行きエンコーダを保存
            torch.save(
                depth_encoder.state_dict(),
                f'{config.save_directory_Cdep_h_ori}/depth_hard_D_encoder_best_ori{ext}.pth')
            
    del depth_dic, depth_dic_val, dpt, encoder, decoder, depth_encoder 
    gc.collect()

    return

def train_Mdepth_hard(ext):
    config = ConfigTrain()
    depth_dic = {}
    depth_dic_val = {}

    # 辞書（単語→単語ID）の読み込み
    with open(config.word_to_id_file, 'rb') as f:
        word_to_id = pickle.load(f)

    # 辞書サイズを保存
    vocab_size = len(word_to_id)
        
    # モデル出力用のディレクトリを作成
    os.makedirs(config.save_directory_Mdep_h, exist_ok=True)

    # 画像のtransformsを定義
    transforms = T.Compose([
        T.Resize((224, 224)),
        #T.RandomHorizontalFlip(),
        T.ToTensor()
        # ImageNetデータセットの平均と標準偏差
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])

    #通常のCNNエンコーダ入力用トランスフォーム
    #base_trandforms = T.Compose([
        #T.Resize((224,224)),
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #])

    depth_transforms = T.Compose([T.Resize((224,224))])

    # COCOデータロードの定義
    #train_dataset = dataset.CocoCaptions(root=config.img_directory, 
                                         #annFile=config.anno_file, 
                                         #transform=transforms)
    train_dataset = dataset.CocoCaptions(root=config.train_img_directory, 
                                         annFile=config.train_anno_file, 
                                         transform=transforms)
    val_dataset = dataset.CocoCaptions(root=config.val_img_directory, 
                                         annFile=config.val_anno_file, 
                                         transform=transforms)
    
    # Subset samplerの生成
    #val_set, train_set = util.generate_subset(
        #train_dataset, config.val_ratio)

    # 学習時にランダムにサンプルするためのサンプラー
    #train_sampler = SubsetRandomSampler(train_set)

    # DataLoaderを生成
    collate_func_lambda = lambda x: util.collate_func_for_dep(x, word_to_id)
    train_loader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True,
                        num_workers=config.num_workers, 
                        collate_fn=collate_func_lambda)
    val_loader = torch.utils.data.DataLoader(
                        val_dataset, 
                        batch_size=config.batch_size, 
                        num_workers=config.num_workers, 
                        collate_fn=collate_func_lambda)
    #train_loader = torch.utils.data.DataLoader(
                        #train_dataset, 
                        #batch_size=config.batch_size, 
                        #num_workers=config.num_workers, 
                        #sampler=train_sampler,
                        #collate_fn=collate_func_lambda)
    #val_loader = torch.utils.data.DataLoader(
                        #train_dataset, 
                        #batch_size=config.batch_size, 
                        #num_workers=config.num_workers, 
                        #sampler=val_set,
                        #collate_fn=collate_func_lambda)

    # モデルの定義
    encoder = CNNEncoder_Atten(config.enc_img_size)
    decoder = MD_RNNDecoderWithHardAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.mlp_dim_encoder,
                                      config.dim_hidden,
                                      vocab_size,
                                      config.device,
                                      config.dropout)
    dpt = DPT_Depthestimator()
    depth_encoder = Depth_MLP_endoder(config.dim_l1, config.dim_l2, config.dim_out)

    encoder.to(config.device)
    decoder.to(config.device)
    dpt.to(config.device)
    depth_encoder.to(config.device)
    
    dpt.load_weight()#DPT 学習済みパラメータのロード

    # 損失関数の定義
    loss_func = lambda x, y: F.cross_entropy(
        x, y, ignore_index=word_to_id.get('<null>', None))
    
    # 最適化手法の定義
    params = list(decoder.parameters())+list(depth_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr)
    
    # 学習率スケジューラの定義
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=config.lr_drop, gamma=0.1)

    # 学習経過の書き込み
    train_loss_file = '{}/Mdepth_hard_train_loss_{}.csv'\
        .format(config.save_directory_Mdep_h, ext)
    val_loss_file = '{}/Mdepth_hard_val_loss_{}.csv'\
        .format(config.save_directory_Mdep_h, ext)
    #now = datetime.datetime.now()
    #train_loss_file = '{}/base_soft_train_loss_{}.csv'\
        #.format(config.save_directory_soft, now.strftime('%Y%m%d_%H%M%S'))
    #val_loss_file = '{}/base_soft_val_loss_{}.csv'\
        #.format(config.save_directory_soft, now.strftime('%Y%m%d_%H%M%S'))

    # 学習
    val_loss_best = float('inf')
    temp = torch.tensor(1.0).detach().to(config.device)
    for epoch in range(config.num_epochs):
        with tqdm(train_loader, disable=tqdm_disable) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')

            # 学習モードに設定
            encoder.train()
            decoder.train()
            depth_encoder.train()
            dpt.eval()

            train_losses = deque()
            train_losses_list = []
            if epoch != 0 and epoch%10==0:
                tempi = temp_anneal(epoch)
                temp = tempi.detach().to(config.device)
            for imgs, imgs_for_dep, captions, lengths, allcaps in pbar:
                # ミニバッチを設定
                imgs = imgs.to(config.device)
                imgs_for_dep = imgs_for_dep.to(config.device)
                captions = captions.to(config.device)

                #imgs = base_trandforms(imgs)
                #imgs_for_dep = dpt.trans(imgs_for_dep)

                optimizer.zero_grad()

                features = encoder(imgs)#カラー画像のエンコード

                depth_map = features.new_zeros((config.batch_size,1,1))

                # エンコーダ-デコーダモデル
                if epoch == 0:
                    #imgs_for_dep = imgs_for_dep.to(config.device)
                    depth_map = dpt(imgs_for_dep)
                    depth_map = depth_map.unsqueeze(1)
                    #depth_map = depth_map.detach()
                    depth_map = dpt.standardize_depth_map(depth_map)
                    depth_map = depth_transforms(depth_map)
                    #imgsc = imgs
                    dpc = depth_map.detach().cpu()
                    for i in range(len(imgs)):
                            depth_dic[allcaps[i]] = dpc[i]
                    
                else:
                    for i in range(len(imgs)):
                        if i==0:
                            depth_map = depth_dic[allcaps[i]].unsqueeze(0).to(config.device)
                        else:
                            dm = depth_dic[allcaps[i]].unsqueeze(0).to(config.device)
                            depth_map = torch.cat([depth_map, dm], dim=0)
                
                depth_map = depth_map.detach()
                depth_map = depth_encoder.img_to_patch(depth_map)#奥行き画像の各領域をベクトル化
                
                depth_features = depth_encoder(depth_map)#depth mapのエンコード
                outputs = decoder(features, depth_features, captions, lengths, temp)

                # 損失の計算
                captions = captions[:, 1:] 
                lengths = [length - 1 for length in lengths]
                targets = pack_padded_sequence(captions, lengths, 
                                               batch_first=True)
                loss = loss_func(outputs.data, targets.data)
                # Add doubly stochastic attention regularization
                #loss += lam * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # 誤差逆伝播
                loss.backward()
                
                optimizer.step()

                # 学習時の損失をログに書き込み
                train_losses.append(loss.item())
                train_losses_list.append(loss.item())
                if len(train_losses) > config.moving_avg:
                    train_losses.popleft()
                pbar.set_postfix({
                    'loss': torch.Tensor(train_losses).mean().item()})
                
        train_loss = np.mean(train_losses_list)
        with open(train_loss_file, 'a') as f:
            print(f'{epoch}, {train_loss}', file=f)

        print(f"[epoch:{epoch}] train loss: {train_loss}")

        # 検証
        with tqdm(val_loader, disable=tqdm_disable) as pbar:
            pbar.set_description(f'[検証]')

            # 評価モード
            encoder.eval()
            decoder.eval()
            depth_encoder.eval()
            dpt.eval()

            val_losses = []
            for imgs, imgs_for_dep, captions, lengths, allcaps in pbar:

                # ミニバッチを設定
                imgs = imgs.to(config.device)
                imgs_for_dep = imgs_for_dep.to(config.device)
                captions = captions.to(config.device)

                # エンコーダ-デコーダモデル
                features = encoder(imgs)

                depth_map = features.new_zeros((config.batch_size,1,1))

                # エンコーダ-デコーダモデル
                if epoch == 0:
                    #imgs_for_dep = imgs_for_dep.to(config.device)
                    depth_map = dpt(imgs_for_dep)
                    depth_map = depth_map.unsqueeze(1)
                    #depth_map = depth_map.detach()
                    depth_map = dpt.standardize_depth_map(depth_map)
                    depth_map = depth_transforms(depth_map)
                    #imgsc = imgs
                    dpc = depth_map.detach().cpu()
                    for i in range(len(imgs)):
                            depth_dic_val[allcaps[i]] = dpc[i]

                else:
                    for i in range(len(imgs)):
                        if i==0:
                            depth_map = depth_dic_val[allcaps[i]].unsqueeze(0).to(config.device)
                        else:
                            dm = depth_dic_val[allcaps[i]].unsqueeze(0).to(config.device)
                            depth_map = torch.cat([depth_map, dm], dim=0)
                
                depth_map = depth_map.detach()
                depth_map = depth_encoder.img_to_patch(depth_map)

                depth_features = depth_encoder(depth_map)#depth mapのエンコード
                outputs = decoder.eval_forward(features, depth_features, captions, lengths)

                # 損失の計算
                captions = captions[:, 1:] 
                lengths = [length - 1 for length in lengths]
                targets = pack_padded_sequence(captions, lengths, 
                                               batch_first=True)
                loss = loss_func(outputs.data, targets.data)
                #loss += lam * ((1. - alphas.sum(dim=1)) ** 2).mean()
                val_losses.append(loss.item())

                # Validation Lossをログに書き込み
                #with open(val_loss_file, 'a') as f:
                    #print(f'{epoch}, {loss.item()}', file=f)

        # Loss 表示
        val_loss = np.mean(val_losses)
        #print(f'Validation loss: {val_loss}')
        print(f'[epoch:{epoch}] Validation loss: {val_loss}')
        with open(val_loss_file, 'a') as f:
            print(f'{epoch}, {val_loss}', file=f)

        # より良い検証結果が得られた場合、モデルを保存
        if val_loss < val_loss_best:
            val_loss_best = val_loss

            # エンコーダモデルを保存
            torch.save(
                encoder.state_dict(),
                f'{config.save_directory_Mdep_h}/Mdepth_hard_encoder_best{ext}.pth')

            # デコーダモデルを保存
            torch.save(
                decoder.state_dict(),
                f'{config.save_directory_Mdep_h}/Mdepth_hard_decoder_best{ext}.pth')
            
            # 奥行きエンコーダを保存
            torch.save(
                depth_encoder.state_dict(),
                f'{config.save_directory_Mdep_h}/Mdepth_hard_D_encoder_best{ext}.pth')
            
    del depth_dic, depth_dic_val, dpt, encoder, decoder, depth_encoder 
    gc.collect()

    return