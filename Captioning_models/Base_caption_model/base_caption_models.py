import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models
from Captioning_models.attention import Soft_Attention, Hard_Attention
#import depth_fuction

import Captioning_models.util as util

#画像エンコーダ
class CNNEncoder_Atten(nn.Module):
    '''
    Show, attend and tellのエンコーダ
    encoded_img_size: 画像部分領域サイズ
    '''
    def __init__(self, encoded_img_size: int):
        super().__init__()

        # ImageNetで事前学習された
        # ResNet152モデルをバックボーンネットワークとする
        resnet = models.resnet152(weights="IMAGENET1K_V2") 

        # AdaptiveAvgPool2dで部分領域(14x14)を作成        
        resnet.avgpool = nn.AdaptiveAvgPool2d(encoded_img_size)#(バッチサイズ、チャンネル数、14、14)にする
        
        # 特徴抽出器として使うため全結合層を削除
        modules = list(resnet.children())[:-1]
        self.backbone = nn.Sequential(*modules)

    '''
    エンコーダの順伝播
    imgs : 入力画像, [バッチサイズ, チャネル数, 高さ, 幅]
    '''
    @torch.no_grad()
    def forward(self, imgs: torch.Tensor):
        # 特徴抽出
        features = self.backbone(imgs)

        # 並び替え
        # -> [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
        features = features.permute(0, 2, 3, 1).flatten(1, 2)#[バッチサイズ, 196, チャンネル数]

        return features
    
    #-----------------------------------------------------------------------------#
    #キャプション生成デコーダ
class RNNDecoderWithSoftAttention(nn.Module):
    '''
    アテンション機構付きデコーダネットワーク
    dim_attention: アテンション機構の次元
    dim_embedding: 埋込み次元
    dim_encoder  : エンコーダ出力の特徴量次元
    dim_decoder  : デコーダの次元
    vocab_size   : 辞書の次元
    dropout      : ドロップアウト確率
    '''
    def __init__(self, dim_attention: int, dim_embedding: int, 
                 dim_encoder: int, dim_decoder: int,
                 vocab_size: int, dropout: float=0.5):
        super().__init__()

        self.vocab_size = vocab_size

        # アテンション機構
        self.attention = Soft_Attention(dim_encoder, dim_decoder, 
                                   dim_attention)

        # 単語の埋め込み
        self.embed = nn.Embedding(vocab_size, dim_embedding)
        self.dropout = nn.Dropout(dropout)

        # LSTMセル
        self.decode_step = nn.LSTMCell(dim_embedding + dim_encoder, 
                                       dim_decoder, bias=True)

        # LSTM隠れ状態/メモリセルの初期値を生成する全結合層
        self.init_linear = nn.Linear(dim_encoder, dim_decoder * 2)

         # シグモイド活性化前の全結合層
        self.f_beta = nn.Linear(dim_decoder, dim_encoder)

        # 単語出力用の全結合層
        self.linear = nn.Linear(dim_decoder, vocab_size)

        # 埋め込み層、全結合層の重みを初期化
        self._reset_parameters()
        
    '''
    パラメータの初期化関数
    '''
    def _reset_parameters(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)#埋め込み層
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)#単語分類全結合層
        nn.init.constant_(self.linear.bias, 0)

    '''
    アテンション機構付きデコーダの順伝播
    features: エンコーダ出力,
              [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
    captions: キャプション, [バッチサイズ, 最大系列長]
    lengths : 系列長のリスト
    '''
    def forward(self, features: torch.Tensor, captions: torch.Tensor,
                lengths: list):
        # バッチサイズの取得
        bs = features.shape[0]
        num_pixels = features.shape[1]

        # 単語埋込み
        embeddings = self.embed(captions)#テンソルの要素(整数)それぞれがエンベッドされる

        # 隠れ状態ベクトル、メモリセルの初期値を生成
        mean_features = features.mean(dim=1)#[バッチサイズ, dim_encorder]
        init_state = self.init_linear(mean_features)#[バッチサイズ, dim_decorder*2]
        h, c = init_state.chunk(2, dim=1)#show and tellでは隠れ状態とセルは初期化されなかったというか一回LSTMに画像特徴ベクトルを入れて獲得していた

        # 最大系列長（<start>を除く）
        dec_lengths = [length - 1 for length in lengths]

        # キャプショニング結果を保持するためのテンソル
        preds = features.new_zeros(
            (bs, max(dec_lengths), self.vocab_size))
        alphas = features.new_zeros(bs, max(dec_lengths), num_pixels)

        # 文予測処理
        for t in range(max(dec_lengths)):
            # dec_lengths には単語数が降順にソートされている
            # 先頭から単語数tより多い文の個数をbooleanの和で算出
            bs_valid = sum([l > t for l in dec_lengths])

            # コンテキストベクトル, アテンション重み
            context_vector, alpha = self.attention(
                features[:bs_valid], h[:bs_valid])

            # LSTMセル
            gate = self.f_beta(h[:bs_valid]).sigmoid()
            context_vector = gate * context_vector
            context_vector = torch.cat(
                (embeddings[:bs_valid, t], context_vector), dim=1)
            h, c = self.decode_step(
                context_vector, (h[:bs_valid], c[:bs_valid]))#h, cは更新される
            
            # 単語予測
            pred = self.linear(self.dropout(h))

            # 情報保持
            preds[:bs_valid, t] = pred
            alphas[:bs_valid, t, :] = alpha

        # パディングされたTensorを可変長系列に戻してパック
        preds = pack_padded_sequence(preds, dec_lengths, 
                                     batch_first=True)

        return preds, alphas

    '''
    サンプリングによる説明文出力（貪欲法）
    features  : エンコーダ出力特徴,
                [1, 特徴マップの幅 * 高さ, 埋め込み次元]
    word_to_id: 単語->単語ID辞書
    max_length: キャプションの最大系列長
    '''    
    @torch.no_grad()
    def sample(self, features: torch.Tensor, word_to_id: list,
               max_length=30):        
        # 隠れ状態ベクトル、メモリセルの初期値を生成
        mean_features = features.mean(dim=1)
        init_state = self.init_linear(mean_features)
        h, c = init_state.chunk(2, dim=1)

        # 文生成の初期値として<start>を埋め込み
        id_start = word_to_id['<start>']
        prev_word = features.new_tensor((id_start,),
                                        dtype=torch.int64)

        # サンプリングによる文生成
        preds = []
        alphas = []
        for _ in range(max_length):
            # 単語埋め込み
            embeddings = self.embed(prev_word)

            # コンテキストベクトル, アテンション重み
            context_vector, alpha = self.attention(features, h)
            
            # LSTMセル
            gate = self.f_beta(h).sigmoid()
            context_vector = gate * context_vector
            context_vector = torch.cat(
                (embeddings, context_vector), dim=1)
            h, c = self.decode_step(context_vector, (h, c))

            # 単語予測
            pred = self.linear(h)
            pred = pred.softmax(dim=1)
            prev_word = pred.argmax(dim=1)

            # 予測結果とアテンション重みを保存
            preds.append(prev_word[0].item())
            alphas.append(alpha)

        return preds, alphas
    
    @torch.no_grad()#バッチ単位でのサンプル
    def batch_sample(self, features: torch.Tensor, word_to_id: list,
               max_length=30):        
        #バッチサイズの取得
        bs = features.shape[0]
        # 隠れ状態ベクトル、メモリセルの初期値を生成
        mean_features = features.mean(dim=1)
        init_state = self.init_linear(mean_features)
        h, c = init_state.chunk(2, dim=1)

        # 文生成の初期値として<start>を埋め込み
        id_start = word_to_id['<start>']
        id_start = [id_start for _ in range(bs)]
        prev_word = features.new_tensor(id_start,
                                        dtype=torch.int64)

        # サンプリングによる文生成
        preds = np.zeros((bs, max_length), dtype=np.int64)
        #alphas = []
        for step in range(max_length):
            # 単語埋め込み
            embeddings = self.embed(prev_word)#[bs, embed_dim]

            # コンテキストベクトル, アテンション重み
            context_vector, alpha = self.attention(features, h)
            
            # LSTMセル
            gate = self.f_beta(h).sigmoid()
            context_vector = gate * context_vector
            context_vector = torch.cat(
                (embeddings, context_vector), dim=1)
            h, c = self.decode_step(context_vector, (h, c))

            # 単語予測
            pred = self.linear(h)#[bs, vocab_size]
            pred = pred.softmax(dim=1)
            prev_word = pred.argmax(dim=1)#[bs]
            np_prev_word = prev_word.detach()
            np_prev_word = np_prev_word.to("cpu").numpy().copy()

            # 予測結果とアテンション重みを保存
            preds[:,step] = np_prev_word.astype(np.int64)
            #alphas.append(alpha)

        return preds #alphas
    
    


#--------------------------------------------------------------#
#Hard-Attentionキャプション生成デコーダ
class RNNDecoderWithHardAttention(nn.Module):
    '''
    アテンション機構付きデコーダネットワーク
    dim_attention: アテンション機構の次元
    dim_embedding: 埋込み次元
    dim_encoder  : エンコーダ出力の特徴量次元
    dim_decoder  : デコーダの次元
    vocab_size   : 辞書の次元
    dropout      : ドロップアウト確率
    '''
    def __init__(self, dim_attention: int, dim_embedding: int, 
                 dim_encoder: int, dim_decoder: int,
                 vocab_size: int, device:str, dropout: float=0.5):
        super().__init__()

        self.vocab_size = vocab_size
        self.device = device

        # アテンション機構
        self.attention = Hard_Attention(dim_encoder, dim_decoder, 
                                   dim_attention)

        # 単語の埋め込み
        self.embed = nn.Embedding(vocab_size, dim_embedding)
        self.dropout = nn.Dropout(dropout)

        # LSTMセル
        self.decode_step = nn.LSTMCell(dim_embedding + dim_encoder, 
                                       dim_decoder, bias=True)

        # LSTM隠れ状態/メモリセルの初期値を生成する全結合層
        self.init_linear = nn.Linear(dim_encoder, dim_decoder * 2)

         # シグモイド活性化前の全結合層
        self.f_beta = nn.Linear(dim_decoder, dim_encoder)

        # 単語出力用の全結合層
        self.linear = nn.Linear(dim_decoder, vocab_size)

        # 埋め込み層、全結合層の重みを初期化
        self._reset_parameters()
        
    '''
    パラメータの初期化関数
    '''
    def _reset_parameters(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)#埋め込み層
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)#単語分類全結合層
        nn.init.constant_(self.linear.bias, 0)

    '''
    アテンション機構付きデコーダの順伝播
    features: エンコーダ出力,
              [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
    captions: キャプション, [バッチサイズ, 最大系列長]
    lengths : 系列長のリスト
    '''
    def forward(self, features: torch.Tensor, captions: torch.Tensor,
                lengths: list, temp:torch.Tensor):
        # バッチサイズの取得
        bs = features.shape[0]

        # 単語埋込み
        embeddings = self.embed(captions)

        # 隠れ状態ベクトル、メモリセルの初期値を生成
        mean_features = features.mean(dim=1)#[バッチサイズ, dim_encorder]
        init_state = self.init_linear(mean_features)#[バッチサイズ, dim_decorder*2]
        h, c = init_state.chunk(2, dim=1)#show and tellでは隠れ状態とセルは初期化されなかったというか一回LSTMに画像特徴ベクトルを入れて獲得していた

        # 最大系列長（<start>を除く）
        dec_lengths = [length - 1 for length in lengths]

        # キャプショニング結果を保持するためのテンソル
        preds = features.new_zeros(
            (bs, max(dec_lengths), self.vocab_size))

        # 文予測処理
        for t in range(max(dec_lengths)):
            # dec_lengths には単語数が降順にソートされている
            # 先頭から単語数tより多い文の個数をbooleanの和で算出
            bs_valid = sum([l > t for l in dec_lengths])

            # コンテキストベクトル, アテンション重み
            context_vector, _ = self.attention(
                features[:bs_valid], h[:bs_valid], self.device, temp)

            # LSTMセル
            gate = self.f_beta(h[:bs_valid]).sigmoid()
            context_vector = gate * context_vector
            context_vector = torch.cat(
                (embeddings[:bs_valid, t], context_vector), dim=1)
            h, c = self.decode_step(
                context_vector, (h[:bs_valid], c[:bs_valid]))#h, cは更新される
            
            # 単語予測
            pred = self.linear(self.dropout(h))

            # 情報保持
            preds[:bs_valid, t] = pred

        # パディングされたTensorを可変長系列に戻してパック
        preds = pack_padded_sequence(preds, dec_lengths, 
                                     batch_first=True)

        return preds
    
    #one_hotサンプルでのforward, evalationはone_hot
    @torch.no_grad()
    def eval_forward(self, features: torch.Tensor, captions: torch.Tensor,
                     lengths: list):
        # バッチサイズの取得
        bs = features.shape[0]

        # 単語埋込み
        embeddings = self.embed(captions)

        # 隠れ状態ベクトル、メモリセルの初期値を生成
        mean_features = features.mean(dim=1)#[バッチサイズ, dim_encorder]
        init_state = self.init_linear(mean_features)#[バッチサイズ, dim_decorder*2]
        h, c = init_state.chunk(2, dim=1)#show and tellでは隠れ状態とセルは初期化されなかったというか一回LSTMに画像特徴ベクトルを入れて獲得していた

        # 最大系列長（<start>を除く）
        dec_lengths = [length - 1 for length in lengths]

        # キャプショニング結果を保持するためのテンソル
        preds = features.new_zeros(
            (bs, max(dec_lengths), self.vocab_size))

        # 文予測処理
        for t in range(max(dec_lengths)):
            # dec_lengths には単語数が降順にソートされている
            # 先頭から単語数tより多い文の個数をbooleanの和で算出
            bs_valid = sum([l > t for l in dec_lengths])

            # コンテキストベクトル, アテンション重み
            context_vector, _ = self.attention.Hard_sample(
                features[:bs_valid], h[:bs_valid], self.device)

            # LSTMセル
            gate = self.f_beta(h[:bs_valid]).sigmoid()
            context_vector = gate * context_vector
            context_vector = torch.cat(
                (embeddings[:bs_valid, t], context_vector), dim=1)
            h, c = self.decode_step(
                context_vector, (h[:bs_valid], c[:bs_valid]))#h, cは更新される
            
            # 単語予測
            pred = self.linear(self.dropout(h))

            # 情報保持
            preds[:bs_valid, t] = pred

        # パディングされたTensorを可変長系列に戻してパック
        preds = pack_padded_sequence(preds, dec_lengths, 
                                     batch_first=True)

        return preds

    '''
    サンプリングによる説明文出力（貪欲法）
    features  : エンコーダ出力特徴,
                [1, 特徴マップの幅 * 高さ, 埋め込み次元]
    word_to_id: 単語->単語ID辞書
    max_length: キャプションの最大系列長
    '''    
    @torch.no_grad()
    def sample(self, features: torch.Tensor, word_to_id: list,
               max_length=30):        
        # 隠れ状態ベクトル、メモリセルの初期値を生成
        mean_features = features.mean(dim=1)
        init_state = self.init_linear(mean_features)
        h, c = init_state.chunk(2, dim=1)

        # 文生成の初期値として<start>を埋め込み
        id_start = word_to_id['<start>']
        prev_word = features.new_tensor((id_start,),
                                        dtype=torch.int64)

        # サンプリングによる文生成
        preds = []
        alphas = []
        for _ in range(max_length):
            # 単語埋め込み
            embeddings = self.embed(prev_word)

            # コンテキストベクトル, アテンション重み
            context_vector, alpha = self.attention.Hard_sample(features, h, self.device)
            
            # LSTMセル
            gate = self.f_beta(h).sigmoid()
            context_vector = gate * context_vector
            context_vector = torch.cat(
                (embeddings, context_vector), dim=1)
            h, c = self.decode_step(context_vector, (h, c))

            # 単語予測
            pred = self.linear(h)
            pred = pred.softmax(dim=1)
            prev_word = pred.argmax(dim=1)

            # 予測結果とアテンション重みを保存
            preds.append(prev_word[0].item())
            alphas.append(alpha)

        return preds, alphas
    
    @torch.no_grad()#バッチ単位でのサンプル
    def batch_sample(self, features: torch.Tensor, word_to_id: list,
               max_length=30):        
        #バッチサイズの取得
        bs = features.shape[0]
        # 隠れ状態ベクトル、メモリセルの初期値を生成
        mean_features = features.mean(dim=1)
        init_state = self.init_linear(mean_features)
        h, c = init_state.chunk(2, dim=1)

        # 文生成の初期値として<start>を埋め込み
        id_start = word_to_id['<start>']
        id_start = [id_start for _ in range(bs)]
        prev_word = features.new_tensor(id_start,
                                        dtype=torch.int64)

        # サンプリングによる文生成
        preds = np.zeros((bs, max_length), dtype=np.int64)
        #alphas = []
        for step in range(max_length):
            # 単語埋め込み
            embeddings = self.embed(prev_word)#[bs, embed_dim]

            # コンテキストベクトル, アテンション重み
            context_vector, alpha = self.attention.Hard_sample(features, h, self.device)
            
            # LSTMセル
            gate = self.f_beta(h).sigmoid()
            context_vector = gate * context_vector
            context_vector = torch.cat(
                (embeddings, context_vector), dim=1)
            h, c = self.decode_step(context_vector, (h, c))

            # 単語予測
            pred = self.linear(h)#[bs, vocab_size]
            pred = pred.softmax(dim=1)
            prev_word = pred.argmax(dim=1)#[bs]
            np_prev_word = prev_word.detach()
            np_prev_word = np_prev_word.to("cpu").numpy().copy()

            # 予測結果とアテンション重みを保存
            preds[:,step] = np_prev_word.astype(np.int64)
            #alphas.append(alpha)

        return preds #alphas
