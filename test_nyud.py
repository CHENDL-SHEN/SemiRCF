import os
import torch
import cv2
import random
import numpy as np
from src.edge_connect import EdgeConnect
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
import sys
from shutil import copyfile


a_path = 'predictions_nyud'  # work dir
# (gan_train_nyud&bsds)EdgeModel_gen.pth
# (mmd_train_nyud&bsds)EdgeModel_gen.pth
a_load_model_dir = 'final_model/(mmd_train_nyud&bsds)EdgeModel_gen.pth'  # file to resume
a_data_mode = 'rgb' # rgb for NYUD; gray for metal;
a_backebone = 'vgg16'  # vgg16, resnet50, resnext50, resnet101, resnext101

# domain adaptation block
a_Unsup = False
a_GAN = True
a_GAN_weight = 1
a_lambda_term = 10  # used in GAN loss
a_critic_iter = 5   # dis iter per gen
a_GAN_sample = 800
a_D2G_LR = 1 # gan lr ratio
a_MMD = False
a_MMD_weight = 1
a_MMD_sample = 600

a_noweight_pre = False
a_mask_expand = 3  # mask expande used for dele_pse num

# dataset dir
a_TARGET_LAB_IMG_FLIST = 'data_nyud/train_one_shot_image'
a_TARGET_LAB_MASK_FLIST = 'data_nyud/train_one_shot_mask_image'
a_TARGET_LAB_EDGE_FLIST = 'data_nyud/train_one_shot_gt'

a_SOURCE_LAB_IMG_FLIST = 'data_bsds/trans_image'
a_SOURCE_LAB_MASK_FLIST  = 'data_bsds/trans_mask'
a_SOURCE_LAB_EDGE_FLIST  = 'data_bsds/trans_label'

a_TARGET_UNLAB_IMG_FLIST = 'data_nyud/train_unlab_image'
a_TARGET_UNLAB_MASK_FLIST = 'data_nyud/train_unlab_mask_image'
a_TARGET_UNLAB_EDGE_FLIST =  'data_nyud/train_unlab_gt'
a_TARGET_UNLAB_EDGE_Pseudo_FLIST =  a_path + '/Pse_label_for_unlab'

a_TEST_IMG_FLIST = 'data_nyud/test_Images'

# mode
a_mode = 2 #1:train; 2:test；3: Pretrained Mode 4:test source 5: pre_train_unsup # 6: train_unsup

# learning strategy
a_lr = 1e-7
a_lr_mode = 'SGD' # SGD SGD_normal
a_lr_momentum = 0.9
a_lr_weight_decay = 2e-4
a_BATCH_SIZE_MERGE = 30

# train weight between tasks
a_target_lab_weight = 1
a_source_lab_weight = 1
a_target_unlab_weight = 1

# Configuration in training process
a_SAVE_INTERVAL = 10
a_SAMPLE_TEST = 10
a_EVAL_INTERVAL = 1
a_LOG_INTERVAL = 10
a_MAX_EPOCH = 51

# Configuration of Pseudo label
a_perued_num = 100 # The number of pseudo positive per epoch
a_weight_neg_pos = 1 # bias for pos Vs neg
a_negpos_ratio = 0.9523 # ratio for pos Vs neg
a_arlfa = 0.7  # weight for target lab
a_beta = 0.1  # weitht for source lab
a_gema = 0.2  # weight for target unlab
a_beta_unsup = 0.5  # weitht for source lab (unsup mode)
a_gema_unsup = 0.5  # weight for target unlab (unsup mode)
a_kmeans_num = 10  # number of kmeans cluster
a_conne_dis = 10  # threshold for connection
a_region_min = 50  # threshold for region merge
a_edge_dele_num = 10    # threshold for deletion
a_hard_distance = 6   # threshold for hard connection
a_threhold = 0.0 # minimal threhold to generate label
a_threhold_fugai = 0.05 # # minimal threhold to overrid assigned label
a_mask_ratio = 0.5 #  if mask bound > 0.5 * total; delete

# Configuration for Speeding up training
a_target_sample = 1 # sample ratio
a_source_sample = 1 # sample ratio
a_test_sample = 10 # sample ratio
a_neg_ample = 20 # sample ratio
a_target_fea_img = 1 # sample num
a_source_fea_img = 1 # sample num
a_multi_process_num = 20  # processor num

a_GAN_LOSS = 'nsgan'



def main(mode=None):
    if not os.path.exists(a_path):
        os.makedirs(a_path)
    copyfile(sys.argv[0], a_path +'/' + 'config_copy')

    config = load_config()
    if config.GAN:
        config.MMD = False
    else:
        config.MMD = True

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize

    model = EdgeConnect(config)
    model.load()

    THIS_DIR = abspath(dirname(__file__))
    TMP_DIR = join(THIS_DIR, a_path)
    if not isdir(TMP_DIR):
        os.makedirs(TMP_DIR)

    # model training
    if config.MODE == 1:
        if not a_Unsup:
            print('\nstart training...\n')
            model.train()
        else:
            print('\nstart training... on Unsupervised mode...\n')
            model.train(unsup = True)

    elif config.MODE == 2:
        if not a_Unsup:
            print('\nstart testing out...\n')
            model.real_test_out()
        else:
            print('\nstart testing out... on Unsupervised mode...\n')
            model.real_test_out(unsup = True)

    # eval mode
    elif config.MODE == 3:
        if not a_Unsup:
            print('\npre-training...\n')
            model.train(semi=False)
        else:
            print('\npre-training on Unsupervised mode...\n')
            model.train(unsup=True, semi=False)

    else:
        print('\nstart eval...\n')
        model.eval()




def load_config():
    # Some details and miscellaneous

    # limit uneffect for feature metric
    if a_data_mode == 'gray':
        a_arg_train = False  # arg mode
        a_pos_neibour_max = 5  # used for limit overrid
        a_min_distance = 25  # Seed point taboo  30
        a_region_min_second = 150  # used for merge region
        a_region_min_ratio = 0.5  # used for merge region
        a_edge_dele_ratio = 0.01  # used for delete boundary
        a_conne_ratio = 0.5  # used for connection
        a_dis_max_fine = 40  # used for fine-tune
        a_epoch_tune = 40  # termination epoch for pseudo label
        a_epoch_could_tune = 20 # tune epoch for neg
        a_epoch_could_pos_tune = 30 # tune epoch for pos

    elif a_data_mode == 'rgb':
        a_arg_train = True  # arg mode
        a_pos_neibour_max = 5  # 5 used for limit overrid
        a_min_distance = 30  # Seed point taboo
        a_region_min_second = 200  # used for merge region
        a_region_min_ratio = 0.5  # used for merge region
        a_edge_dele_ratio = 0.01  # used for delete boundary
        a_conne_ratio = 0.5  #  used for connection
        a_dis_max_fine = 30  # used for fine-tune
        a_epoch_tune = 25  # termination epoch for pseudo label
        a_epoch_could_tune = 12  # tune epoch for neg
        a_epoch_could_pos_tune = 15  # tune epoch for pos

    #if a_backebone != 'vgg16':
    #    a_pretrain_epoch = 16  # train multi_optimazation
    #    a_arg_train = True  # arg mode
    #else:
    a_pretrain_epoch = 0
    a_arg_train = True  # arg mode might be better for both task



    class con():
        def __init__(self):
            self.MODE = a_mode
            self.MODEL = 1
            self.SEED = 10
            self.GPU = [0]
            self.LR = a_lr
            self.BETA1 = 0.0
            self.BETA2 = 0.9
            self.min_distance = a_min_distance
            self.INPUT_SIZE = 256
            self.MAX_ITERS = a_MAX_EPOCH
            self.SAMPLE_TEST = a_SAMPLE_TEST
            self.conne_dis = a_conne_dis
            self.load_model_dir = a_load_model_dir
            self.target_fea_img = a_target_fea_img
            self.source_fea_img = a_source_fea_img
            self.epoch_could_tune = a_epoch_could_tune
            self.lr_mode = a_lr_mode
            self.arg_train = a_arg_train

            self.noweight_pre = a_noweight_pre

            self.region_min = a_region_min
            self.region_min_second = a_region_min_second
            self.region_min_ratio = a_region_min_ratio
            self.edge_dele_ratio = a_edge_dele_ratio
            self.edge_dele_num = a_edge_dele_num
            self.BATCH_SIZE_MERGE = a_BATCH_SIZE_MERGE
            self.dis_max_fine = a_dis_max_fine
            self.threhold_fugai = a_threhold_fugai
            self.backebone = a_backebone

            self.lr_momentum = a_lr_momentum
            self.lr_weight_decay = a_lr_weight_decay


            self.SAVE_INTERVAL = a_SAVE_INTERVAL
            self.EVAL_INTERVAL = a_EVAL_INTERVAL
            self.LOG_INTERVAL = a_LOG_INTERVAL
            self.EDGE_THRESHOLD = 0.5
            self.threhold = a_threhold
            self.weight_neg_pos = a_weight_neg_pos
            self.kmeans_num = a_kmeans_num
            self.pos_neibour_max = a_pos_neibour_max
            self.test_sample_num = a_test_sample
            self.hard_distance = a_hard_distance
            self.epoch_tune = a_epoch_tune
            self.negpos_ratio = a_negpos_ratio
            self.conne_ratio = a_conne_ratio

            self.multi_process_num = a_multi_process_num
            self.neg_ample = a_neg_ample
            self.mask_ratio = a_mask_ratio
            self.mask_expand = a_mask_expand



            self.PATH = a_path
            self.perued_num = a_perued_num
            self.data_mode = a_data_mode

            self.TARGET_LAB_IMG_FLIST = a_TARGET_LAB_IMG_FLIST
            self.TARGET_LAB_MASK_FLIST = a_TARGET_LAB_MASK_FLIST
            self.TARGET_LAB_EDGE_FLIST = a_TARGET_LAB_EDGE_FLIST

            self.SOURCE_LAB_IMG_FLIST = a_SOURCE_LAB_IMG_FLIST
            self.SOURCE_LAB_MASK_FLIST = a_SOURCE_LAB_MASK_FLIST
            self.SOURCE_LAB_EDGE_FLIST = a_SOURCE_LAB_EDGE_FLIST

            self.TARGET_UNLAB_IMG_FLIST = a_TARGET_UNLAB_IMG_FLIST
            self.TARGET_UNLAB_MASK_FLIST = a_TARGET_UNLAB_MASK_FLIST
            self.TARGET_UNLAB_EDGE_FLIST = a_TARGET_UNLAB_EDGE_FLIST
            self.TARGET_UNLAB_EDGE_Pseudo_FLIST = a_TARGET_UNLAB_EDGE_Pseudo_FLIST

            self.TEST_IMG_FLIST = a_TEST_IMG_FLIST


            self.target_lab_weight = a_target_lab_weight
            self.source_lab_weight = a_source_lab_weight
            self.target_unlab_weight = a_target_unlab_weight

            self.arlfa = a_arlfa
            self.beta = a_beta
            self.gema = a_gema
            self.beta_unsup = a_beta_unsup  # weitht for source lab (unsup mode)
            self.gema_unsup = a_gema_unsup  # weight for target unlab (unsup mode)
            self.Unsup = a_Unsup

            self.target_sample = a_target_sample
            self.source_sample = a_source_sample
            self.epoch_could_pos_tune = a_epoch_could_pos_tune

            self.D2G_LR = a_D2G_LR
            self.GAN_LOSS = a_GAN_LOSS
            self.GAN = a_GAN
            self.GAN_weight = a_GAN_weight
            self.lambda_term = a_lambda_term
            self.critic_iter = a_critic_iter
            self.GAN_sample = a_GAN_sample

            self.MMD = a_MMD
            self.MMD_weight = a_MMD_weight
            self.MMD_sample = a_MMD_sample

            self.pretrain_epoch = a_pretrain_epoch




    config = con()

    return config


if __name__ == "__main__":

    main()
