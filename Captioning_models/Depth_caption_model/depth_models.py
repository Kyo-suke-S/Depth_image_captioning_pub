import numpy as np
#from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from Captioning_models.attention import Soft_Attention, Hard_Attention

#import util

class Depth_CNN_endoder(nn.Module):
    '''
    奥行き画像のCNNエンコーダ
    '''
    def __init__(self, encoded_img_size: int):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 128, 7, stride=3)#224->73->24
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 512, 3)#24->22->7
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 2048, 1)#7->7
        self.bn3 = nn.BatchNorm2d(2048)
        #self.conv4 = nn.Conv2d(512, 2048, 1)#7->7
        #self.bn4 = nn.BatchNorm2d(2048)
        #self.conv5 = nn.Conv2d(512, 1024, 3)
        #self.bn5 = nn.BatchNorm2d(1024)
        #self.conv6 = nn.Conv2d(1024, 2048, 3)
        #self.bn6 = nn.BatchNorm2d(2048)

        self.avg_pool = nn.AdaptiveAvgPool2d(encoded_img_size)
        #self.max_pool1 = nn.MaxPool2d((2,2))
        self.max_pool = nn.MaxPool2d((3,3))
        self.relu = nn.ReLU(inplace=True)
        self.features = nn.Sequential(self.conv1,#224->73
                                      self.bn1,
                                      self.relu,
                                      self.max_pool,#73->24
                                      self.conv2,#24->22
                                      self.bn2,
                                      self.relu,
                                      self.max_pool,#22->7
                                      self.conv3,#7->7
                                      self.bn3,
                                      self.relu,
                                      self.avg_pool)
        
    def forward(self, depth_imgs: torch.Tensor):
        outputs = self.features(depth_imgs)

        # 並び替え
        # -> [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
        outputs = outputs.permute(0, 2, 3, 1).flatten(1, 2)#[バッチサイズ, 196, チャンネル数]

        return outputs

class Depth_MLP_endoder(nn.Module):
    '''
    奥行き画像のMLPエンコーダ
    奥行き画像の各部分をベクトル化した次元は256
    '''
    def __init__(self, dim_l1:int, dim_l2:int, dim_out:int):
        super().__init__()

        self.l1 = nn.Linear(256, dim_l1)#256->128
        self.l2 = nn.Linear(dim_l1, dim_l2)#128->64
        self.l3 = nn.Linear(dim_l2, dim_out)#64->32

        self.relu = nn.ReLU(inplace=True)

        self.mlp_encoder = nn.Sequential(self.l1,
                                         self.relu,
                                         self.l2,
                                         self.relu,
                                         self.l3,
                                         self.relu)
        
        self.Unfolder = nn.Unfold(kernel_size=(16,16), stride=16)
        
    def forward(self, depth_img_vec: torch.Tensor):
        depth_features = self.mlp_encoder(depth_img_vec)#[batchサイズ, 196, 32]
        
        return depth_features
    
    #奥行き画像を分割する関数
    def img_to_patch(self, imgs): # patch_size=(16,16), stride=16)
        unfolded = self.Unfolder(imgs)#[bactch_size, 256(16*16*1), 196]
        unfolded = unfolded.permute(0, 2, 1)#[batch_size, 196, 256(=16*16)]

        return unfolded

    
#------------------------------------------------------------------------------------------------

class CD_RNNDecoderWithSoftAttention(nn.Module):
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
    depth_features:奥行き画像エンコーダ出力, [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
    captions: キャプション, [バッチサイズ, 最大系列長]
    lengths : 系列長のリスト
    '''
    def forward(self, features: torch.Tensor, depth_features: torch.Tensor, captions: torch.Tensor,
                lengths: list):
        # バッチサイズの取得
        bs = features.shape[0]
        num_pixels = features.shape[1]

        # 単語埋込み
        embeddings = self.embed(captions)#テンソルの要素(整数)それぞれがエンベッドされる

        # 奥行きの特徴量と合体
        features = features.add(depth_features)

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
    def sample(self, features: torch.Tensor, depth_features: torch.Tensor, word_to_id: list,
               max_length=30):  
        # 奥行きの特徴量と合体
        features = features.add(depth_features)      
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
    def batch_sample(self, features: torch.Tensor, depth_features: torch.Tensor, word_to_id: list,
               max_length=30): 
        # 奥行きの特徴量と合体
        features = features.add(depth_features)       
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
    


class MD_RNNDecoderWithSoftAttention(nn.Module):
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
                 mlp_dim_encoder: int, dim_decoder: int,
                 vocab_size: int, dropout: float=0.5):
        super().__init__()

        self.vocab_size = vocab_size

        # アテンション機構
        self.attention = Soft_Attention(mlp_dim_encoder, dim_decoder, 
                                   dim_attention)

        # 単語の埋め込み
        self.embed = nn.Embedding(vocab_size, dim_embedding)
        self.dropout = nn.Dropout(dropout)

        # LSTMセル
        self.decode_step = nn.LSTMCell(dim_embedding + mlp_dim_encoder, 
                                       dim_decoder, bias=True)

        # LSTM隠れ状態/メモリセルの初期値を生成する全結合層
        self.init_linear = nn.Linear(mlp_dim_encoder, dim_decoder * 2)

         # シグモイド活性化前の全結合層
        self.f_beta = nn.Linear(dim_decoder, mlp_dim_encoder)

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
    depth_features:奥行き画像エンコーダ出力, [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
    captions: キャプション, [バッチサイズ, 最大系列長]
    lengths : 系列長のリスト
    '''
    def forward(self, features: torch.Tensor, depth_features: torch.Tensor, captions: torch.Tensor,
                lengths: list):
        # バッチサイズの取得
        bs = features.shape[0]
        num_pixels = features.shape[1]

        # 単語埋込み
        embeddings = self.embed(captions)#テンソルの要素(整数)それぞれがエンベッドされる

        # 奥行きの特徴量と合体
        features = torch.cat((features, depth_features), dim=2)#[batch_size, 196, dim_encoder+mlp_dim_out]

        # 隠れ状態ベクトル、メモリセルの初期値を生成
        mean_features = features.mean(dim=1)#[バッチサイズ, mlp_dim_encorder]
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
    def sample(self, features: torch.Tensor, depth_features: torch.Tensor, word_to_id: list,
               max_length=30):        
        # 隠れ状態ベクトル、メモリセルの初期値を生成
        features = torch.cat((features, depth_features), dim=2)
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
    def batch_sample(self, features: torch.Tensor, depth_features: torch.Tensor, word_to_id: list,
               max_length=30):        
        #バッチサイズの取得
        bs = features.shape[0]

        # 隠れ状態ベクトル、メモリセルの初期値を生成
        features = torch.cat((features, depth_features), dim=2)
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
    

#------------------------------------------------------------------------------------------------

class CD_RNNDecoderWithHardAttention(nn.Module):
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
                 vocab_size: int, device: str, dropout: float=0.5):
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
    depth_features:奥行き画像エンコーダ出力, [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
    captions: キャプション, [バッチサイズ, 最大系列長]
    lengths : 系列長のリスト
    '''
    def forward(self, features: torch.Tensor, depth_features: torch.Tensor, captions: torch.Tensor,
                lengths: list, temp: torch.tensor):
        # バッチサイズの取得
        bs = features.shape[0]
        #num_pixels = features.shape[1]

        # 単語埋込み
        embeddings = self.embed(captions)#テンソルの要素(整数)それぞれがエンベッドされる

        # 奥行きの特徴量と合体
        features = features.add(depth_features)

        # 隠れ状態ベクトル、メモリセルの初期値を生成
        mean_features = features.mean(dim=1)#[バッチサイズ, dim_encorder]
        init_state = self.init_linear(mean_features)#[バッチサイズ, dim_decorder*2]
        h, c = init_state.chunk(2, dim=1)#show and tellでは隠れ状態とセルは初期化されなかったというか一回LSTMに画像特徴ベクトルを入れて獲得していた

        # 最大系列長（<start>を除く）
        dec_lengths = [length - 1 for length in lengths]

        # キャプショニング結果を保持するためのテンソル
        preds = features.new_zeros(
            (bs, max(dec_lengths), self.vocab_size))
        #alphas = features.new_zeros(bs, max(dec_lengths), num_pixels)

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
            #alphas[:bs_valid, t, :] = alpha

        # パディングされたTensorを可変長系列に戻してパック
        preds = pack_padded_sequence(preds, dec_lengths, 
                                     batch_first=True)

        return preds # alphas
    
    #one_hotサンプルでのforward, evalationはone_hot
    @torch.no_grad()
    def eval_forward(self, features: torch.Tensor, depth_features: torch.Tensor, captions: torch.Tensor,
                     lengths: list):
        # バッチサイズの取得
        bs = features.shape[0]

        # 単語埋込み
        embeddings = self.embed(captions)
        # 奥行きの特徴量と合体
        features = features.add(depth_features)


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
    def sample(self, features: torch.Tensor, depth_features: torch.Tensor, word_to_id: list,
               max_length=30):        
        # 奥行きの特徴量と合体
        features = features.add(depth_features)
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
            #print(alpha)
            
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
    def batch_sample(self, features: torch.Tensor, depth_features: torch.Tensor, word_to_id: list,
               max_length=30):        
        #バッチサイズの取得
        bs = features.shape[0]

        # 奥行きの特徴量と合体
        features = features.add(depth_features)
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
            context_vector, _ = self.attention.Hard_sample(features, h, self.device)
            
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
    

class MD_RNNDecoderWithHardAttention(nn.Module):
    def __init__(self, dim_attention: int, dim_embedding: int, 
                 mlp_dim_encoder: int, dim_decoder: int,
                 vocab_size: int, device: str, dropout: float=0.5):
        super().__init__()

        self.vocab_size = vocab_size
        self.device = device

        # アテンション機構
        self.attention = Hard_Attention(mlp_dim_encoder, dim_decoder, 
                                   dim_attention)

        # 単語の埋め込み
        self.embed = nn.Embedding(vocab_size, dim_embedding)
        self.dropout = nn.Dropout(dropout)

        # LSTMセル
        self.decode_step = nn.LSTMCell(dim_embedding + mlp_dim_encoder, 
                                       dim_decoder, bias=True)

        # LSTM隠れ状態/メモリセルの初期値を生成する全結合層
        self.init_linear = nn.Linear(mlp_dim_encoder, dim_decoder * 2)

         # シグモイド活性化前の全結合層
        self.f_beta = nn.Linear(dim_decoder, mlp_dim_encoder)

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
    depth_features:奥行き画像エンコーダ出力, [バッチサイズ, 特徴マップの幅 * 高さ, チャネル数]
    captions: キャプション, [バッチサイズ, 最大系列長]
    lengths : 系列長のリスト
    '''
    def forward(self, features: torch.Tensor, depth_features: torch.Tensor, captions: torch.Tensor,
                lengths: list, temp: torch.tensor):
        # バッチサイズの取得
        bs = features.shape[0]
        #num_pixels = features.shape[1]

        # 単語埋込み
        embeddings = self.embed(captions)#テンソルの要素(整数)それぞれがエンベッドされる

        # 奥行きの特徴量と合体
        features = torch.cat((features, depth_features), dim=2)#[batch_size, 196, dim_encoder+mlp_dim_out]

        # 隠れ状態ベクトル、メモリセルの初期値を生成
        mean_features = features.mean(dim=1)#[バッチサイズ, dim_encorder]
        init_state = self.init_linear(mean_features)#[バッチサイズ, dim_decorder*2]
        h, c = init_state.chunk(2, dim=1)#show and tellでは隠れ状態とセルは初期化されなかったというか一回LSTMに画像特徴ベクトルを入れて獲得していた

        # 最大系列長（<start>を除く）
        dec_lengths = [length - 1 for length in lengths]

        # キャプショニング結果を保持するためのテンソル
        preds = features.new_zeros(
            (bs, max(dec_lengths), self.vocab_size))
        #alphas = features.new_zeros(bs, max(dec_lengths), num_pixels)

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
            #alphas[:bs_valid, t, :] = alpha

        # パディングされたTensorを可変長系列に戻してパック
        preds = pack_padded_sequence(preds, dec_lengths, 
                                     batch_first=True)

        return preds # alphas
    
    #one_hotサンプルでのforward, evalationはone_hot
    @torch.no_grad()
    def eval_forward(self, features: torch.Tensor, depth_features: torch.Tensor, captions: torch.Tensor,
                     lengths: list):
        # バッチサイズの取得
        bs = features.shape[0]

        # 単語埋込み
        embeddings = self.embed(captions)
        # 奥行きの特徴量と合体
        features = torch.cat((features, depth_features), dim=2)#[batch_size, 196, dim_encoder+mlp_dim_out]


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
    def sample(self, features: torch.Tensor, depth_features: torch.Tensor, word_to_id: list,
               max_length=30):        
        # 奥行きの特徴量と合体
        features = torch.cat((features, depth_features), dim=2)#[batch_size, 196, dim_encoder+mlp_dim_out]
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
    def batch_sample(self, features: torch.Tensor, depth_features: torch.Tensor, word_to_id: list,
               max_length=30):        
        #バッチサイズの取得
        bs = features.shape[0]

        # 奥行きの特徴量と合体
        features = torch.cat((features, depth_features), dim=2)#[batch_size, 196, dim_encoder+mlp_dim_out]
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
            context_vector, _ = self.attention.Hard_sample(features, h, self.device)
            
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