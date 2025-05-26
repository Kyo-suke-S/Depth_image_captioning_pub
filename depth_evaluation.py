import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import pickle
import PIL
from PIL import Image
import skimage.transform #画像出力用ライブラリ

import torch
from torch.utils.data import Subset
import torchvision.transforms as T
import torchvision.datasets as dataset
from Captioning_models.Base_caption_model.base_caption_models import CNNEncoder_Atten
from Captioning_models.Depth_caption_model.depth_models import Depth_CNN_endoder, CD_RNNDecoderWithSoftAttention, MD_RNNDecoderWithSoftAttention, CD_RNNDecoderWithHardAttention
from Captioning_models.Depth_caption_model.DPT_model import DPT_Depthestimator
from Captioning_models.config import ConfigEval
from Captioning_models.evaluate_metrix import load_textfiles, score
import sys
import gc

import Captioning_models.util as util

def Cdepth_evaluation(atten, useData = "coco"):
    config = ConfigEval()
    word_to_id_pass = config.word_to_id_file #MSCOCOで学習する時に使用
    id_to_word_pass = config.id_to_word_file #MSCOCOで学習する時に使用
    anno_file_pass = config.val_anno_file #MSCOCOのアノテーションファイル
    if atten == "soft":
        save_directly = config.save_directory_Cdep_soft
        param_files = config.depth_soft_parameter_files
    elif atten == "hard":
        save_directly = config.save_directory_Cdep_hard
        param_files = config.depth_hard_parameter_files

    if useData == "rem_original" or useData=="rem_coco":
        word_to_id_pass = config.ori_word_to_id_file
        id_to_word_pass = config.ori_id_to_word_file
        if atten == "soft":
            param_files = config.depth_soft_ori_parameter_files
            save_directly=config.save_directory_Cdep_soft_ori
        elif atten == "hard":
            param_files = config.depth_hard_ori_parameter_files
            save_directly=config.save_directory_Cdep_hard_ori

        if useData=="rem_original":
            anno_file_pass = config.rem_ori_val_anno_file
        else:
            anno_file_pass = config.remCOCO_ori_val_anno_file

    # 辞書（単語→単語ID）の読み込み
    with open(word_to_id_pass, 'rb') as f:
        word_to_id = pickle.load(f)

    # 辞書（単語ID→単語）の読み込み
    with open(id_to_word_pass, 'rb') as f:
        id_to_word = pickle.load(f)

    # 辞書サイズを保存
    vocab_size = len(id_to_word)
    print("attention: "+atten)
    print("Evaluation on bellow config")
    print("word_to_id file: "+word_to_id_pass)
    print("id_to_word file: "+id_to_word_pass)
    print("annotation file: "+anno_file_pass)
    print(f"Vocab size: {vocab_size}")
    
    # 画像のtransformsを定義
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # ImageNetデータセットの平均と標準偏差
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])

    depth_transforms = T.Compose([T.Resize((224,224))])
    #validationデータセット
    #ランダムな4000個を使用
    val_dataset = dataset.CocoCaptions(root=config.val_img_directory, 
                                         annFile=anno_file_pass,
                                         #annFile=config.remCOCO_ori_val_anno_file, 
                                         transform=transforms)
    
    make_refs_lambda = lambda x: util.make_refs_dep(x, word_to_id)

    if useData=="coco"or useData=="rem_coco":
        if useData=="coco":
            npy_file = config.index_dir
        else:
            npy_file = config.remCOCO_500_ori_index_dir
        indeces = np.load(npy_file).tolist()#ランダムに取り出すためのインデックス
        #print(len(indeces))
        val_dataset = Subset(val_dataset, indeces)#オリジナルデータ500点だけではサブセットを作らない
        print(f"subset size : {len(val_dataset)}")
    
    val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        #subcoco, 
                        batch_size=config.batch_size, 
                        num_workers=config.num_workers, 
                        collate_fn=make_refs_lambda)
    
    # モデルの定義
    encoder = CNNEncoder_Atten(config.enc_img_size)
    if atten == "soft":
        decoder = CD_RNNDecoderWithSoftAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size)
    elif atten == "hard":
        decoder = CD_RNNDecoderWithHardAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size, config.device, config.dropout)

    depth_encoder = Depth_CNN_endoder(config.enc_img_size)
    dpt = DPT_Depthestimator()

    encoder.to(config.device)
    decoder.to(config.device)
    depth_encoder.to(config.device)
    dpt.to(config.device)
    dpt.load_weight()

    encoder.eval()
    decoder.eval()
    depth_encoder.eval()
    dpt.eval()


    scores = {"Bleu_1":[], "Bleu_2":[], "Bleu_3":[],"Bleu_4":[], "METEOR":[], "ROUGE_L":[],"CIDEr":[]}
    for key, enc_dec_depenc  in param_files.items():
        # モデルの学習済み重みパラメータをロード
        encoder.load_state_dict(
            torch.load(f'{save_directly}/{enc_dec_depenc[0]}'))
        decoder.load_state_dict(
            torch.load(f'{save_directly}/{enc_dec_depenc[1]}'))
        depth_encoder.load_state_dict(
            torch.load(f'{save_directly}/{enc_dec_depenc[2]}'))

        #coco4kでキャプションを評価
        ref_caps = []
        hypos_id = []
        for imgs, imgs_for_dep, captions in tqdm(val_loader):
            #captions: [[],[],..,[]]
            ref_caps.extend(captions)
            imgs = imgs.to(config.device)
            imgs_for_dep = imgs_for_dep.to(config.device)

            depth_maps = dpt(imgs_for_dep)
            depth_maps = depth_maps.unsqueeze(1)
            #depth_map = depth_map.detach()
            depth_maps = dpt.standardize_depth_map(depth_maps)
            depth_maps = depth_transforms(depth_maps)

            depth_features = depth_encoder(depth_maps)

            # エンコーダ・デコーダモデルによる予測
            feature = encoder(imgs)
            sampled_ids = decoder.batch_sample(feature, depth_features, word_to_id)
            hypos_id.append(sampled_ids)

        hypos_id = np.concatenate(hypos_id)
        hypos_word = []
        for ids in hypos_id:
            line = []
            for id in ids:
                w = id_to_word[id]
                if w == "<end>":
                    break
                line.append(w)
            hypos_word.append(" ".join(line))

        #print(f"ref: {len(ref_caps)}, hypo: {len(hypos_word)}")
        ref, hypo = load_textfiles(ref_caps,hypos_word)
        score_result = score(ref, hypo)
        #print(ref[10])
        #print(hypo[10])
        print(score_result)
        for mt, sc in score_result.items():
            scores[mt].append(sc)

    #dire = f"/home/shirota/Depth_image_caption_git/depth_hard_test/{useData}_scores.pkl"
    dire = dire = save_directly+f"/{useData}_scores.pkl"
    with open(dire, "wb") as f:
        pickle.dump(scores, f)
       
    del encoder, decoder, depth_encoder, dpt
    gc.collect()

    return

def Cdepth_sample(atten, sample_pic, useData="coco"):
    config = ConfigEval()

    word_to_id_pass = config.word_to_id_file #MSCOCOで学習する時に使用
    id_to_word_pass = config.id_to_word_file #MSCOCOで学習する時に使用
    if atten == "soft":
        save_directly = config.save_directory_Cdep_soft
        param_files = config.depth_soft_parameter_files
    elif atten == "hard":
        save_directly = config.save_directory_Cdep_hard
        param_files = config.depth_hard_parameter_files

    if useData == "original":
        word_to_id_pass = config.ori_word_to_id_file
        id_to_word_pass = config.ori_id_to_word_file
        if atten == "soft":
            save_directly = config.save_directory_Cdep_soft_ori
            param_files = config.depth_soft_ori_parameter_files
        elif atten == "hard":
            save_directly = config.save_directory_Cdep_hard_ori
            param_files = config.depth_hard_ori_parameter_files

    # 辞書（単語→単語ID）の読み込み
    with open(word_to_id_pass, 'rb') as f:
        word_to_id = pickle.load(f)

    # 辞書（単語ID→単語）の読み込み
    with open(id_to_word_pass, 'rb') as f:
        id_to_word = pickle.load(f)

    # 辞書サイズを保存
    vocab_size = len(id_to_word)
    
    # 画像のtransformsを定義
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # ImageNetデータセットの平均と標準偏差
        #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
    ])

    norm_trans = T.Compose([T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    depth_resize = T.Compose([T.Resize((224,224))])
    image_size = 384
    dep_trans = T.Compose([T.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        T.CenterCrop(image_size),
                                        #T.ToTensor(),
                                        T.Normalize(mean=0.5, std=0.5)])
    
    # モデルの定義
    encoder = CNNEncoder_Atten(config.enc_img_size)
    if atten == "soft":
        decoder = CD_RNNDecoderWithSoftAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size)
        
    elif atten == "hard":
        decoder = CD_RNNDecoderWithHardAttention(config.dim_attention,
                                      config.dim_embedding, 
                                      config.dim_encoder,
                                      config.dim_hidden,
                                      vocab_size, config.device, config.dropout)
        
    depth_encoder = Depth_CNN_endoder(config.enc_img_size)
    dpt = DPT_Depthestimator()

    encoder.to(config.device)
    decoder.to(config.device)
    depth_encoder.to(config.device)
    dpt.to(config.device)
    dpt.load_weight()

    encoder.eval()
    decoder.eval()
    depth_encoder.eval()
    dpt.eval()

    encoder.load_state_dict(
            torch.load(f'{save_directly}/{param_files[1][0]}'))
    decoder.load_state_dict(
            torch.load(f'{save_directly}/{param_files[1][1]}'))
    depth_encoder.load_state_dict(
            torch.load(f'{save_directly}/{param_files[1][2]}'))


    #scores = {"Bleu_1":[], "Bleu_2":[], "Bleu_3":[],"Bleu_4":[], "METEOR":[], "ROUGE_L":[],"CIDEr":[]}
    if sample_pic == "sample1":
        img_directry = config.sample1_dir
    elif sample_pic == "sample2":
        img_directry = config.sample2_dir
    elif sample_pic == "sample3":
        img_directry = config.sample3_dir
    elif sample_pic == "airbus":
        img_directry = config.airbus_dir
    elif sample_pic == "cycling":
        img_directry = config.cycling_dir
    elif sample_pic == "dog":
        img_directry = config.dog_dir
    elif sample_pic == "football":
        img_directry = config.football_dir
    elif sample_pic == "soccer":
        img_directry = config.soccer_dir
    elif sample_pic == "river":
        img_directry = config.river_dir
    elif sample_pic == "seagull":
        img_directry = config.seagull_dir
    elif sample_pic == "bird":
        img_directry = config.bird_dir
    else:
        print("Input correct name")
        return
    
    output_save_directly = img_directry+f"/depth_{atten}"
    os.makedirs(output_save_directly, exist_ok=True)

    for img_file in sorted(
        glob.glob(os.path.join(img_directry, '*.[jp][pn]g'))):

        img = Image.open(img_file).convert("RGB")
        #img = np.asarray(img)
        #print(type(img))
        img = transforms(img)
        img = img.unsqueeze(0)
        img_for_dep = img.detach().clone()
        img = norm_trans(img)
        img_for_dep = dep_trans(img_for_dep)
        img = img.to(config.device)
        img_for_dep = img_for_dep.to(config.device)

        depth_map = dpt(img_for_dep)
        depth_map = depth_map.unsqueeze(1)
        #depth_map = depth_map.detach()
        depth_map = dpt.standardize_depth_map(depth_map)
        depth_map = depth_resize(depth_map)

        feature = encoder(img)
        depth_features = depth_encoder(depth_map)
        sampled_ids, alphas = decoder.sample(feature, depth_features, word_to_id)

        img_plt = Image.open(img_file)
        img_plt = img_plt.resize([224, 224], Image.LANCZOS)
        plt.imshow(img_plt)
        plt.axis('off')
        #plt.show()
        plt.savefig(output_save_directly + f'/input.png', bbox_inches='tight')
        plt.clf()

        print(f'入力画像: {os.path.basename(img_file)}')

        sampled_caption = []
        c = 0
        for word_id, alpha in zip(sampled_ids, alphas):
            word = id_to_word[word_id]
            sampled_caption.append(word)

            alpha = alpha.view(
                config.enc_img_size, config.enc_img_size)
            alpha = alpha.to('cpu').numpy()
            alpha = skimage.transform.pyramid_expand(
                alpha, upscale=16, sigma=8)
            
            # タイムステップtの画像をプロット
            plt.imshow(img_plt)
            #plt.text(0, 1, f'{word}', color='black',
            #         backgroundcolor='white', fontsize=12)
            plt.imshow(alpha, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
            #plt.show()
            plt.savefig(output_save_directly + f'/depth_{atten}_{word}_p{c}.png', bbox_inches='tight')
            plt.clf()
            c += 1
            
            if word == '<end>':
                break
        
        sentence = ' '.join(sampled_caption)
        print(f'出力キャプション: {sentence}')

        # 推定結果を書き込み
        gen_sentence_out = output_save_directly + '/caption.txt'
        with open(gen_sentence_out, 'w') as f:
            print(sentence, file=f)

    del encoder, decoder, depth_encoder, dpt
    gc.collect()

    return


#-----------------------------------------------------------------------------------

def main():
    args = sys.argv
    if len(args)==4:
        atten = args[1]
        useData = args[3]
        Cdepth_evaluation(atten, useData)
    
    elif len(args)==5:
        atten = args[1]
        sample_pic = args[3]
        useData = args[4]
        Cdepth_sample(atten, sample_pic, useData)
    
    else:
        print("depth_evaluation.py {soft/hard} {score/sample sample_pic} {useData}")


if __name__ == "__main__":
    main()