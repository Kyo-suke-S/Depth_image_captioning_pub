import os

class ConfigTrain(object):
    '''
    ハイパーパラメータ、システム共通変数の設定
    '''  
    def __init__(self):
        self.cwd = os.getcwd()

        # ハイパーパラメータ
        self.enc_img_size = 14     # Attention計算用画像サイズ
        self.dim_attention = 128   # Attention層の次元
        self.dim_embedding = 128   # 埋め込み層の次元
        self.dim_encoder = 2048    # エンコーダの特徴マップのチャネル数
        self.dim_hidden = 128      # LSTM隠れ層の次元
        self.dim_l1 = 128
        self.dim_l2 = 64
        self.dim_out = 32
        self.mlp_dim_encoder =2080
        self.lr = 0.001            # 学習率
        self.dropout = 0.5         # dropout確率
        self.batch_size = 30       # ミニバッチ数
        self.num_epochs = 150
        #self.num_epochs = 100       # エポック数→Colab無料版でテストする際は10未満に修正を推奨
        self.lr_drop = [20]         # 学習率を減衰させるエポック
        self.temp_sch = 10         #softmaxの温度を更新するエポック毎数
        
        self.nic_dim_embedding = 300 # 埋め込み層の次元
        self.num_layers = 2
                
        # パスの設定
        self.train_img_directory = self.cwd+"/dataset/coco2014/train2014"
        self.val_img_directory = self.cwd+"/dataset/coco2014/val2014"
        
        #self.img_directory = 'val2014'
        self.train_anno_file = self.cwd+'/dataset/coco2014/captions_train2014.json'
        self.val_anno_file = self.cwd+'/dataset/coco2014/captions_val2014.json'

        self.ori_train_anno_file = self.cwd+'/dataset/original_dataset/original_dataset.json'
        self.ori_val_anno_file = self.cwd+'/dataset/original_dataset/original_val_dataset.json'

        self.word_to_id_file = self.cwd+'/dataset/coco2014/word_to_id.pkl'
        self.ori_word_to_id_file = self.cwd+'/dataset/original_dataset/ori_word_to_id.pkl'
        
        self.save_directory_soft = self.cwd+'/exp_result/base_soft'
        self.save_directory_soft_ori = self.cwd+'/exp_result/base_soft_ori'

        self.save_directory_Cdep_soft = self.cwd+'/exp_result/CNN_depth_soft'
        self.save_directory_Cdep_soft_ori = self.cwd+'/exp_result/CNN_depth_soft_ori'
        #self.save_directory_Mdep_s = self.cwd+'/exp_result/MLP_depth_soft'
        self.save_directory_hard = self.cwd+'/exp_result/base_hard'
        self.save_directory_hard_ori = self.cwd+'/exp_result/base_hard_ori'
        self.save_directory_Cdep_hard = self.cwd+'/exp_result/CNN_depth_hard'
        self.save_directory_Cdep_hard_ori = self.cwd+'/exp_result/CNN_depth_hard_ori'
        #self.save_directory_Mdep_h = self.cwd+'/exp_result/MLP_depth_hard'

        self.save_directory_nic = self.cwd+'/exp_result/NIC'



        # 検証に使う学習セット内のデータの割合
        #self.val_ratio = 0.3

        # データローダーに使うCPUプロセスの数
        self.num_workers = 4

        # 学習に使うデバイス
        self.device = 'cuda:0'

        # 移動平均で計算する損失の値の数
        self.moving_avg = 100


class ConfigEval(object):
    '''
    ハイパーパラメータ、システム共通変数の設定
    '''  
    def __init__(self):
        
        self.cwd = os.getcwd()

        # ハイパーパラメータ
        self.enc_img_size = 14     # Attention計算用画像サイズ
        self.dim_attention = 128   # Attention層の次元
        self.dim_embedding = 128   # 埋め込み層の次元
        self.dim_encoder = 2048    # エンコーダの特徴マップのチャネル数
        self.dim_hidden = 128      # LSTM隠れ層の次元
        self.lr = 0.001            # 学習率
        self.dropout = 0.5         # dropout確率
        self.batch_size = 50       # ミニバッチ数
        self.num_epochs = 150
        #self.num_epochs = 100       # エポック数→Colab無料版でテストする際は10未満に修正を推奨
        self.lr_drop = [20]         # 学習率を減衰させるエポック
        self.temp_sch = 10         #softmaxの温度を更新するエポック毎数

        self.nic_dim_embedding = 300 # 埋め込み層の次元
        self.num_layers = 2
                
        # パスの設定
        self.train_img_directory = self.cwd+"/dataset/coco2014/train2014"
        self.val_img_directory = self.cwd+"/dataset/coco2014/val2014"
        
        #self.img_directory = 'val2014'
        self.train_anno_file = self.cwd+'/dataset/coco2014/captions_train2014.json'
        self.val_anno_file = self.cwd+'/dataset/coco2014/captions_val2014.json'

        self.ori_train_anno_file = self.cwd+'/dataset/original_dataset/original_dataset.json'
        self.ori_val_anno_file = self.cwd+'/dataset/orignal_dataset/original_val_dataset.json'

        self.rem_ori_val_anno_file = self.cwd+'/dataset/original_dataset/rem_original_val_dataset.json'
        self.remCOCO_ori_val_anno_file = self.cwd+'/dataset/original_dataset/remCOCO_original_val_dataset.json'

        #self.anno_file = 'drive/MyDrive/python_image_recognition/data/coco2014/captions_val2014.json'
        self.word_to_id_file = self.cwd+'/dataset/coco2014/model/word_to_id.pkl'
        self.ori_word_to_id_file = self.cwd+'/dataset/original_dataset/ori_word_to_id.pkl'
        #self.word_to_id_file = 'drive/MyDrive/python_image_recognition/6_img_captioning/model/word_to_id.pkl'
        self.id_to_word_file = self.cwd+'/dataset/coco2014/id_to_word.pkl'
        self.ori_id_to_word_file = self.cwd+'/dataset/original_dataset/ori_id_to_word.pkl'

        self.save_directory_soft = self.cwd+'/exp_result/base_soft'
        self.base_soft_parameter_files = {1:["base_soft_encoder_best_coco0.pth","base_soft_decoder_best_coco0.pth"],
                                          2:["base_soft_encoder_best_coco1.pth","base_soft_decoder_best_coco1.pth"],
                                          3:["base_soft_encoder_best_coco2.pth","base_soft_decoder_best_coco2.pth"]}
        
        self.save_directory_soft_ori = self.cwd+'/exp_result/base_soft_ori'
        self.base_soft_ori_parameter_files = {1:["base_soft_encoder_best_original0.pth","base_soft_decoder_best_original0.pth"],
                                              2:["base_soft_encoder_best_original1.pth","base_soft_decoder_best_original1.pth"],
                                              3:["base_soft_encoder_best_original2.pth","base_soft_decoder_best_original2.pth"]}

        self.save_directory_Cdep_soft = self.cwd+'/exp_result/CNN_depth_soft'
        self.depth_soft_parameter_files = {1:["depth_soft_encoder_best_coco0.pth","depth_soft_decoder_best_coco0.pth",
                                                 "depth_soft_D_encoder_best_coco0.pth"],
                                              2:["depth_soft_encoder_best_coco1.pth","depth_soft_decoder_best_coco1.pth",
                                                 "depth_soft_D_encoder_best_coco1.pth"],
                                              3:["depth_soft_encoder_best_coco2.pth","depth_soft_decoder_best_coco2.pth",
                                                 "depth_soft_D_encoder_best_coco2.pth"]}

        self.save_directory_Cdep_soft_ori = self.cwd+'/exp_result/CNN_depth_soft_ori'
        self.depth_soft_ori_parameter_files = {1:["depth_soft_encoder_best_original0.pth","depth_soft_decoder_best_original0.pth",
                                                 "depth_soft_D_encoder_best_original0.pth"],
                                              2:["depth_soft_encoder_best_original1.pth","depth_soft_decoder_best_original1.pth",
                                                 "depth_soft_D_encoder_best_original1.pth"],
                                              3:["depth_soft_encoder_best_original2.pth","depth_soft_decoder_best_original2.pth",
                                                 "depth_soft_D_encoder_best_original2.pth"]}

        #self.save_directory_Mdep_soft = self.cwd+'/exp_result/MLP_depth_soft'
        
        self.save_directory_hard = self.cwd+'/exp_result/base_hard'
        self.base_hard_parameter_files = {1:["base_hard_encoder_best_coco0.pth","base_hard_decoder_best_coco0.pth"],
                                          2:["base_hard_encoder_best_coco1.pth","base_hard_decoder_best_coco1.pth"],
                                          3:["base_hard_encoder_best_coco2.pth","base_hard_decoder_best_coco2.pth"]}

        self.save_directory_hard_ori = self.cwd+'/exp_result/base_hard_ori'
        self.base_hard_ori_parameter_files = {1:["base_hard_encoder_best_original0.pth","base_hard_decoder_best_original0.pth"],
                                              2:["base_hard_encoder_best_original1.pth","base_hard_decoder_best_original1.pth"],
                                              3:["base_hard_encoder_best_original2.pth","base_hard_decoder_best_original2.pth"]}

        self.save_directory_Cdep_hard = self.cwd+'/exp_result/CNN_depth_hard'
        self.depth_hard_parameter_files = {1:["depth_hard_encoder_best_coco0.pth","depth_hard_decoder_best_coco0.pth",
                                                 "depth_hard_D_encoder_best_coco0.pth"],
                                              2:["depth_hard_encoder_best_coco1.pth","depth_hard_decoder_best_coco1.pth",
                                                 "depth_hard_D_encoder_best_coco1.pth"],
                                              3:["depth_hard_encoder_best_coco2.pth","depth_hard_decoder_best_coco2.pth",
                                                 "depth_hard_D_encoder_best_coco2.pth"]}

        self.save_directory_Cdep_hard_ori = self.cwd+'/exp_result/CNN_depth_hard_ori'
        self.depth_hard_ori_parameter_files = {1:["depth_hard_encoder_best_original0.pth","depth_hard_decoder_best_original0.pth",
                                                 "depth_hard_D_encoder_best_original0.pth"],
                                              2:["depth_hard_encoder_best_original1.pth","depth_hard_decoder_best_original1.pth",
                                                 "depth_hard_D_encoder_best_original1.pth"],
                                              3:["depth_hard_encoder_best_original2.pth","depth_hard_decoder_best_original2.pth",
                                                 "depth_hard_D_encoder_best_original2.pth"]}

        #self.save_directory_Mdep_hard = self.cwd+'/exp_result/MLP_depth_hard'

        self.save_directory_nic = self.cwd+'/exp_result/NIC'
        self.nic_parameter_files = {1:["nic_encoder_best0.pth","nic_decoder_best0.pth"],
                                    2:["nic_encoder_best1.pth","nic_decoder_best1.pth"],
                                    3:["nic_encoder_best2.pth","nic_decoder_best2.pth"]}

        self.sample1_dir = self.cwd+"/sample_pic/sample1"
        self.sample2_dir = self.cwd+"/sample_pic/sample2"
        self.sample3_dir = self.cwd+"/sample_pic/sample3"

        self.airbus_dir = self.cwd+"/sample_pic/airbus"
        self.cycling_dir = self.cwd+"/sample_pic/cycling"
        self.dog_dir = self.cwd+"/sample_pic/dog"
        self.football_dir = self.cwd+"/sample_pic/football"
        self.soccer_dir = self.cwd+"/sample_pic/soccer"
        self.river_dir = self.cwd+"/sample_pic/river"
        self.seagull_dir = self.cwd+"/sample_pic/seagull"
        self.bird_dir = self.cwd+"/sample_pic/bird"

        self.index_dir = self.cwd+"/data_index/np_val_index.npy"
        self.Ori2000_index_dir = self.cwd+"/data_index/np_index_for_ori_val.npy"#originalのvalから2000個
        self.remCOCO_500_ori_index_dir = self.cwd+"/data_index/remCOCO_500_ori.npy"


        # 検証に使う学習セット内のデータの割合
        #self.val_ratio = 0.3

        # データローダーに使うCPUプロセスの数
        self.num_workers = 4

        # 学習に使うデバイス
        self.device = 'cuda:0'

        # 移動平均で計算する損失の値の数
        self.moving_avg = 100