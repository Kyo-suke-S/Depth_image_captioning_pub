import random
from typing import Sequence, Dict, Tuple, Union

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
#from DPT_model import DPT_Depthestimator

import PIL


image_size = 384
norm_trans = T.Compose([T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
dep_trans = T.Compose([T.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        T.CenterCrop(image_size),
                                        #T.ToTensor(),
                                        T.Normalize(mean=0.5, std=0.5)])
#device = "cuda:0"
#dpt = DPT_Depthestimator()
#dpt.to(device)
#dpt.load_weight()
#dpt.eval()

'''
データセットを分割するための2つの排反なインデックス集合を生成する関数
dataset    : 分割対称のデータセット
ratio      : 1つ目のセットに含めるデータ量の割合
random_seed: 分割結果を不変にするためのシード
'''
def generate_subset(dataset: Dataset, ratio: float,
                    random_seed: int=0):
    # サブセットの大きさを計算
    size = int(len(dataset) * ratio)

    indices = list(range(len(dataset)))

    # 二つのセットに分ける前にシャッフル
    random.seed(random_seed)
    random.shuffle(indices)

    # セット1とセット2のサンプルのインデックスに分割
    indices1, indices2 = indices[:size], indices[size:]

    return indices1, indices2


'''
サンプルからミニバッチを生成するcollate関数
batch     : CocoCaptionsからサンプルした複数の画像とラベルをまとめたもの
word_to_id: 単語->単語ID辞書
'''
def collate_func(batch: Sequence[Tuple[Union[torch.Tensor, str]]],
                 word_to_id: Dict[str, int]):
    imgs, captions = zip(*batch)

    # それぞれのサンプルの5個のキャプションの中から1つを選択してトークナイズ
    captions = [tokenize_caption(
        random.choice(cap), word_to_id) for cap in captions]

    # キャプションの長さが降順になるように並び替え
    batch = zip(imgs, captions)
    batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs)

    lengths = [cap.shape[0] for cap in captions]
    targets = torch.full((len(captions), max(lengths)),
                         word_to_id['<null>'], dtype=torch.int64)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return imgs, targets, lengths

'''
サンプルからミニバッチを生成するcollate関数
batch     : CocoCaptionsからサンプルした複数の画像とラベルをまとめたもの
word_to_id: 単語->単語ID辞書
'''
def collate_func_for_dep(batch: Sequence[Tuple[Union[torch.Tensor, str]]],
                 word_to_id: Dict[str, int]):
    imgs, captions = zip(*batch)

    # それぞれのサンプルの5個のキャプションの中から1つを選択してトークナイズ
    allcaps = [" ".join(cap) for cap in captions]
    #choicedcaps = [random.choice(cap) for cap in captions]
    captions = [tokenize_caption(
        random.choice(cap), word_to_id) for cap in captions]
    #captions = [tokenize_caption(
    #    cap, word_to_id) for cap in choicedcaps]

    # キャプションの長さが降順になるように並び替え
    #batch = zip(imgs, captions)
    batch = zip(imgs,captions,allcaps)
    batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    #imgs, captions = zip(*batch)
    imgs, captions, allcaps = zip(*batch)
    imgs = torch.stack(imgs)
    imgs_for_dep = imgs.detach().clone()
    imgs = norm_trans(imgs)
    imgs_for_dep = dep_trans(imgs_for_dep)

    lengths = [cap.shape[0] for cap in captions]
    targets = torch.full((len(captions), max(lengths)),
                         word_to_id['<null>'], dtype=torch.int64)
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return imgs, imgs_for_dep, targets, lengths, allcaps


'''
トークナイザ - 文章(caption)を単語IDのリスト(tokens_id)に変換
caption   : 画像キャプション
word_to_id: 単語->単語ID辞書
'''
def tokenize_caption(caption: str, word_to_id: Dict[str, int]):
    tokens = caption.lower().split()
    
    tokens_temp = []    
    # 単語についたピリオド、カンマを削除
    for token in tokens:
        if token == '.' or token == ',':
            continue

        token = token.rstrip('.')
        token = token.rstrip(',')
        
        tokens_temp.append(token)
    
    tokens = tokens_temp        
        
    # 文章(caption)を単語IDのリスト(tokens_id)に変換
    tokens_ext = ['<start>'] + tokens + ['<end>']
    tokens_id = []
    for k in tokens_ext:
        if k in word_to_id:
            tokens_id.append(word_to_id[k])
        else:
            tokens_id.append(word_to_id['<unk>'])
    
    return torch.Tensor(tokens_id)

def untokenize_caption(caption: str, word_to_id: Dict[str, int]):
    words = caption.lower().split()
    
    words_temp = []    
    # 単語についたピリオド、カンマを削除
    for word in words:
        if word == '.' or word == ',':
            continue

        word = word.rstrip('.')
        word = word.rstrip(',')
        
        if word in word_to_id:
            words_temp.append(word)
        else:
            words_temp.append("<unk>")
    
    words = words_temp
    #単語ごとに分割された文を元にもどす
    cap_back = " ".join(words)
    #cap_back = [cap_back]
    return cap_back

'''
サンプルからミニバッチを生成するmake_refs関数
batch     : CocoCaptionsからサンプルした複数の画像とラベルをまとめたもの
wordはid化せず、すべてのキャプションを使う
word_to_id: 単語->単語ID辞書
'''
def make_refs(batch: Sequence[Tuple[Union[torch.Tensor, str]]],
                 word_to_id: Dict[str, int]):
    imgs, captions = zip(*batch)

    # それぞれのサンプルの5個のキャプションの中から1つを選択してトークナイズ
    captions = [[untokenize_caption(
        one_cap, word_to_id) for one_cap in cap ] for cap in captions]

    # キャプションの長さが降順になるように並び替え
    #batch = zip(imgs, captions)
    #batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    #imgs, captions = zip(*batch)
    imgs = torch.stack(imgs)

    #lengths = [cap.shape[0] for cap in captions]
    #targets = torch.full((len(captions), max(lengths)),
    #                     word_to_id['<null>'], dtype=torch.int64)
    #for i, cap in enumerate(captions):
    #    end = lengths[i]
    #    targets[i, :end] = cap[:end]
    
    return imgs, captions #targets, lengths 一つのテンソルとなった画像バッチとすべて小文字でid化されてないキャプションのリストが返される

def make_refs_dep(batch: Sequence[Tuple[Union[torch.Tensor, str]]],
                 word_to_id: Dict[str, int]):
    imgs, captions = zip(*batch)

    # それぞれのサンプルの5個のキャプションの中から1つを選択してトークナイズ
    captions = [[untokenize_caption(
        one_cap, word_to_id) for one_cap in cap ] for cap in captions]

    # キャプションの長さが降順になるように並び替え
    #batch = zip(imgs, captions)
    #batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    #imgs, captions = zip(*batch)
    imgs = torch.stack(imgs)
    imgs_for_dep = imgs.detach().clone()
    imgs = norm_trans(imgs)
    imgs_for_dep = dep_trans(imgs_for_dep)

    #lengths = [cap.shape[0] for cap in captions]
    #targets = torch.full((len(captions), max(lengths)),
    #                     word_to_id['<null>'], dtype=torch.int64)
    #for i, cap in enumerate(captions):
    #    end = lengths[i]
    #    targets[i, :end] = cap[:end]
    
    return imgs, imgs_for_dep, captions #targets, lengths 一つのテンソルとなった画像バッチとすべて小文字でid化されてないキャプションのリストが返される

