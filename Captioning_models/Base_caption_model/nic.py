import os
import numpy as np
from tqdm import tqdm
import pickle
from collections import deque

import torch
from torch import nn
from torch.utils.data import Subset
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models
import torchvision.transforms as T
import torchvision.datasets as dataset

from Captioning_models.config import ConfigTrain, ConfigEval
from Captioning_models.evaluate_metrix import load_textfiles, score
import Captioning_models.util as util

tqdm_disable = True


class NIC_CNNEncoder(nn.Module):
    '''
    Show and tellのエンコーダ
    dim_embedding: 埋め込み次元
    '''
    def __init__(self, dim_embedding: int):
        super().__init__()

        # ImageNetで事前学習された
        # ResNet152モデルをバックボーンネットワークとする
        resnet = models.resnet152(weights="IMAGENET1K_V2") 

        # 特徴抽出器として使うため全結合層を削除
        #ResNet152は最後に(2048,1000)の全結合層があり、このPytorchのResNet152はまとめられた層が10個
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        # デコーダへの出力
        self.linear = nn.Linear(resnet.fc.in_features, dim_embedding)

    '''
    エンコーダの順伝播
    imgs: 入力画像, [バッチサイズ, チャネル数, 高さ, 幅]
    '''
    def forward(self, imgs: torch.Tensor):
        # 特徴抽出 -> [バッチサイズ, 2048]
        # 今回はバックボーンネットワークは学習させない
        with torch.no_grad():
            features = self.backbone(imgs)#この時点では(B,C,1,1)
            features = features.flatten(1)#(B,C) C=2048

        # 全結合 dim_embedding次元への写像を学習したいだけなので活性化関数はいらない
        features = self.linear(features)

        return features
    
#--------------------------------------------------------------------------

class NIC_RNNDecoder(nn.Module):
    '''
    Show and tellのデコーダ
    dim_embedding: 埋め込み次元（単語埋め込み次元）
    dim_hidden   : 隠れ層次元
    vocab_size   : 辞書サイズ
    num_layers   : レイヤー数
    dropout      : ドロップアウト確率
    '''
    def __init__(self, dim_embedding: int, dim_hidden: int, 
                 vocab_size: int, num_layers: int, dropout: int=0.1):
        super().__init__()

        # 単語埋め込み
        self.embed = nn.Embedding(vocab_size, dim_embedding)

        # LSTM
        self.lstm = nn.LSTM(dim_embedding, dim_hidden, 
                            num_layers, batch_first=True)

        # 全結合層
        self.linear = nn.Linear(dim_hidden, vocab_size)

        # ドロップアウト
        self.dropout = nn.Dropout(dropout)

    '''
    デコーダの順伝播
    features: エンコーダ出力特徴, [バッチサイズ, 埋め込み次元]
    captions: 画像キャプション,   [バッチサイズ, 系列長]
    lengths : 系列長のリスト
    '''
    def forward(self, features: torch.Tensor, captions: torch.Tensor,
                lengths: list):
        
        # 単語埋め込み -> [バッチサイズ, 系列長, 埋め込み次元]
        embeddings = self.embed(captions)

        # 画像埋め込みと単語埋め込みとを連結
        # features.unsqueeze(1) -> [バッチサイズ, 1, 埋め込み次元]
        # 連結後embeddings -> [バッチサイズ, 系列長 + 1, 埋め込み次元]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        # パディングされたTensorを可変長系列に戻してパック
        # packed.data() -> [実際の系列長, 埋め込み次元]
        packed = pack_padded_sequence(embeddings,
                                      lengths, batch_first=True)

        # LSTM
        hiddens, cell = self.lstm(packed)

        # ドロップアウト
        output = self.dropout(hiddens[0])

        # ロジットを取得
        outputs = self.linear(output)

        return outputs

    '''
    サンプリングによる説明文出力（貪欲法）
    features  : エンコーダ出力特徴, [バッチサイズ, 埋め込み次元]
    states    : LSTM隠れ状態
    max_length: キャプションの最大系列長
    '''
    @torch.no_grad()
    def sample(self, features: torch.Tensor, 
               states: torch.Tensor=None, max_length: int=30):

        inputs = features.unsqueeze(1)
        word_idx_list = []

        # 最大系列長まで再帰的に単語をサンプリング予測
        for step_t in range(max_length):
            # LSTM隠れ状態を更新
            hiddens, states = self.lstm(inputs, states)

            # 単語予測
            outputs = self.linear(hiddens.squeeze(1))
            outputs = outputs.softmax(dim=1)
            preds = outputs.argmax(dim=1)
            word_idx_list.append(preds[0].item())
            
            # t+1の入力を作成
            inputs = self.embed(preds)
            inputs = inputs.unsqueeze(1)  

        return word_idx_list
    
    @torch.no_grad()
    def batch_sample(self, features: torch.Tensor, 
               states: torch.Tensor=None, max_length: int=30):
        
        bs = features.size(0)
        inputs = features.unsqueeze(1)
        word_idx_lists = [[] for _ in range(bs)]

        # 最大系列長まで再帰的に単語をサンプリング予測
        for step_t in range(max_length):
            # LSTM隠れ状態を更新
            hiddens, states = self.lstm(inputs, states)

            # 単語予測
            outputs = self.linear(hiddens.squeeze(1))
            outputs = outputs.softmax(dim=1)
            preds = outputs.argmax(dim=1)
            preds_list = preds.tolist()
            for i, pred in enumerate(preds_list):
                word_idx_lists[i].append(pred)
            
            # t+1の入力を作成
            inputs = self.embed(preds)
            inputs = inputs.unsqueeze(1)  

        return word_idx_lists
    
#学習関数
def train_nic(et):
    config = ConfigTrain()

    # 辞書（単語→単語ID）の読み込み
    with open(config.word_to_id_file, 'rb') as f:
        word_to_id = pickle.load(f)

    # 辞書サイズを保存
    vocab_size = len(word_to_id)
        
    # モデル出力用のディレクトリを作成
    os.makedirs(config.save_directory_nic, exist_ok=True)

    # 画像のtransformsを定義
    transforms = T.Compose([
        T.Resize((224, 224)),
        #T.RandomHorizontalFlip(),
        T.ToTensor(),
        # ImageNetデータセットの平均と標準偏差
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])

    # COCOデータロードの定義
    train_dataset = dataset.CocoCaptions(root=config.train_img_directory, 
                                         annFile=config.train_anno_file, 
                                         transform=transforms)
    
    val_dataset = dataset.CocoCaptions(root=config.val_img_directory, 
                                         annFile=config.val_anno_file, 
                                         transform=transforms)
    
    # Subset samplerの生成
    #val_set, train_set = util.generate_subset(
        #train_dataset, config.val_ratio) #多分インデックスが入ったリスト

    # 学習時にランダムにサンプルするためのサンプラー
    #train_sampler = SubsetRandomSampler(train_set)

    # DataLoaderを生成
    collate_func_lambda = lambda x: util.collate_func(x, word_to_id)
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

    # モデルの定義
    encoder = NIC_CNNEncoder(config.nic_dim_embedding)
    decoder = NIC_RNNDecoder(
        config.nic_dim_embedding, config.dim_hidden, vocab_size, 
        config.num_layers, config.dropout)
    encoder.to(config.device)
    decoder.to(config.device)
    
    # 損失関数の定義
    loss_func = lambda x, y: F.cross_entropy(
        x, y, ignore_index=word_to_id.get('<null>', None))
    
    # 最適化手法の定義
    params = list(decoder.parameters()) \
             + list(encoder.linear.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.lr)

    # 学習率スケジューラの定義
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=config.lr_drop, gamma=0.1)
    
    # 学習経過の書き込み
    #now = datetime.datetime.now()
    train_loss_file = '{}/nic_train_loss{}.csv'\
        .format(config.save_directory_nic, et)
    val_loss_file = '{}/nic_val_loss{}.csv'\
        .format(config.save_directory_nic, et)

    # 学習
    val_loss_best = float('inf')
    for epoch in range(config.num_epochs):
        with tqdm(train_loader, disable=tqdm_disable) as pbar:
            pbar.set_description(f'[エポック {epoch + 1}]')

            # 学習モードに設定
            encoder.train()
            decoder.train()

            train_losses = deque()
            train_losses_list = []
            for imgs, captions, lengths in pbar:
                # ミニバッチを設定
                imgs = imgs.to(config.device)
                captions = captions.to(config.device)

                optimizer.zero_grad()

                # エンコーダ・デコーダモデル
                features = encoder(imgs)
                outputs = decoder(features, captions, lengths)

                # 損失の計算
                targets = pack_padded_sequence(captions, 
                                               lengths, 
                                               batch_first=True)[0]
                loss = loss_func(outputs, targets)

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

            val_losses = []
            for imgs, captions, lengths in pbar:

                # ミニバッチを設定
                imgs = imgs.to(config.device)
                captions = captions.to(config.device)

                # エンコーダ-デコーダモデル
                features = encoder(imgs)
                outputs = decoder(features, captions, lengths)

                # 損失の計算
                targets = pack_padded_sequence(captions, 
                                               lengths, 
                                               batch_first=True)[0]
                loss = loss_func(outputs, targets)
                val_losses.append(loss.item())

                # Validation Lossをログに書き込み
                with open(val_loss_file, 'a') as f:
                    print(f'{epoch}, {loss.item()}', file=f)

        # Loss 表示
        val_loss = np.mean(val_losses)
        #print(f'Validation loss: {val_loss}')
        print(f'[epoch:{epoch}] Validation loss: {val_loss}')

        # より良い検証結果が得られた場合、モデルを保存
        if val_loss < val_loss_best:
            val_loss_best = val_loss

            # エンコーダモデルを保存
            torch.save(
                encoder.state_dict(),
                f'{config.save_directory_nic}/nic_encoder_best{et}.pth')

            # デコーダモデルを保存
            torch.save(
                decoder.state_dict(),
                f'{config.save_directory_nic}/nic_decoder_best{et}.pth')
            
            print("best model parameters are changed")

#評価スコア------------------------------------------------------------------------------------

def evaluation_nic():
    config = ConfigEval()

    # 辞書（単語→単語ID）の読み込み
    with open(config.word_to_id_file, 'rb') as f:
        word_to_id = pickle.load(f)

    # 辞書（単語ID→単語）の読み込み
    with open(config.id_to_word_file, 'rb') as f:
        id_to_word = pickle.load(f)

    # 辞書サイズを保存
    vocab_size = len(id_to_word)
    
    # 画像のtransformsを定義
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # ImageNetデータセットの平均と標準偏差
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])
    #validationデータセット
    #ランダムな4000個を使用
    val_dataset = dataset.CocoCaptions(root=config.val_img_directory, 
                                         annFile=config.val_anno_file, 
                                         transform=transforms)
    
    make_refs_lambda = lambda x: util.make_refs(x, word_to_id)
    npy_file = config.index_dir
    indeces = np.load(npy_file).tolist()
    subcoco = Subset(val_dataset, indeces)
    #print(f"coco4k : {len(coco4k)}")
    val_loader = torch.utils.data.DataLoader(
                        subcoco, 
                        batch_size=config.batch_size, 
                        num_workers=config.num_workers, 
                        collate_fn=make_refs_lambda)
    
    # モデルの定義
    encoder = NIC_CNNEncoder(config.nic_dim_embedding)
    decoder = NIC_RNNDecoder(
        config.nic_dim_embedding, config.dim_hidden, vocab_size, 
        config.num_layers, config.dropout)
    
    encoder.to(config.device)
    decoder.to(config.device)
    encoder.eval()
    decoder.eval()

    scores = {"Bleu_1":[], "Bleu_2":[], "Bleu_3":[],"Bleu_4":[], "METEOR":[], "ROUGE_L":[],"CIDEr":[]}
    for key, enc_dec  in config.nic_parameter_files.items():
        # モデルの学習済み重みパラメータをロード
        encoder.load_state_dict(
            torch.load(f'{config.save_directory_nic}/{enc_dec[0]}'))
        decoder.load_state_dict(
            torch.load(f'{config.save_directory_nic}/{enc_dec[1]}'))

        #coco4kでキャプションを評価
        ref_caps = []
        hypos_id = []
        for imgs, captions in tqdm(val_loader):
            #captions: [[],[],..,[]]
            ref_caps.extend(captions)
            imgs = imgs.to(config.device)

            # エンコーダ・デコーダモデルによる予測
            feature = encoder(imgs)
            sampled_ids = decoder.batch_sample(feature)
            hypos_id.append(sampled_ids)

        hypos_id = np.concatenate(hypos_id)
        hypos_word = []
        for ids in hypos_id:
            line = []
            for id in ids:
                w = id_to_word[id]
                if w == "<end>":
                    break
                if w != "<start>":
                    line.append(w)
            hypos_word.append(" ".join(line))

        #print(f"ref: {len(ref_caps)}, hypo: {len(hypos_word)}")
        print(hypos_word[100])
        ref, hypo = load_textfiles(ref_caps,hypos_word)
        score_result = score(ref, hypo)
        #print(ref[10])
        #print(hypo[10])
        print(score_result)
        for mt, sc in score_result.items():
            scores[mt].append(sc)
        
    dire = "/home/shirota/Depth_image_caption_git/nic_test"+"/nic_scores.pkl"
    #dire = config.save_directory_nic+"/nic_scores.pkl"
    with open(dire, "wb") as f:
        pickle.dump(scores, f)