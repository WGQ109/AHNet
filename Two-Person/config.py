import os
from sacred import Experiment

ex = Experiment("AHNet", save_git_info=False)

@ex.config
def my_config():
    ############################## setting ##############################
    version = "pku_xsub_j"
    dataset = "pku"   # pku / pku / pku
    split = "xsub"
    sub = "joint"      # joint / motion / bone
    save_lp = False
    save_finetune = False
    save_semi = False
    pretrain_epoch = 300
    ft_epoch = 50
    lp_epoch = 50
    pretrain_lr = 5e-3
    lp_lr = 0.01
    ft_lr = 5e-3
    label_percent = 0.1
    weight_decay = 1e-5
    hidden_size = 512
    ######### ST-GCN ###############################
    in_channels = 3
    hidden_channels = 64
    hidden_dim = 512
    dropout = 0.5
    graph_args = {
    "layout" : 'ntu-rgb+d',
    "strategy" : 'spatial'
    }
    graph_args_1 = {
    "layout" : 'single-gcn',
    "strategy" : 'spatial'
    }
    edge_importance_weighting = True
    people_importance_weighting = True
    ############################ down stream ############################
    weight_path = './output/multi_model/xsub/vpku_xsub_j_epoch_300_pretrain.pt'
    train_mode = 'lp'
    # train_mode = 'finetune'
    # train_mode = 'pretrain'
    # train_mode = 'semi'
    log_path = './output/log/baseline2-lr1e-4'+'_'+train_mode+'300.log'
    ################################ GPU ################################
    gpus = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = "5"
    ########################## Skeleton Setting #########################
    batch_size = 128
    channel_num = 3
    person_num = 12
    joint_num = 17
    max_frame = 10
    # train_list = '/home/wanggq/datasets/action_dataset/'+dataset+'_frame50/'+split+'/train_position.npy'
    # test_list = '/home/wanggq/datasets/action_dataset/'+dataset+'_frame50/'+split+'/val_position.npy'
    # train_label = '/home/wanggq/datasets/action_dataset/'+dataset+'_frame50/'+split+'/train_label.pkl'
    # test_label = 'home/wanggq/datasets/action_dataset/'+dataset+'_frame50/'+split+'/val_label.pkl'
    # train_list = '/home/wanggq/datasets/action_dataset/pku_part1_frame50/xsub/train_joint_volleyball.npy'
    # test_list = '/home/wanggq/datasets/action_dataset/pku_part1_frame50/xsub/test_joint_volleyball.npy'
    # train_label = '/home/wanggq/datasets/action_dataset/pku_part1_frame50/xsub/train_label.pkl'
    # test_label = '/home/wanggq/datasets/action_dataset/pku_part1_frame50/xsub/test_label.pkl'
    train_list = '/home/wanggq/datasets/action_dataset/pku_part1_frame50/xsub/mtrain_position.npy'
    test_list = '/home/wanggq/datasets/action_dataset/pku_part1_frame50/xsub/mval_position.npy'
    train_label = '/home/wanggq/datasets/action_dataset/pku_part1_frame50/xsub/mtrain_label.pkl'
    test_label = '/home/wanggq/datasets/action_dataset/pku_part1_frame50/xsub/mval_label.pkl'
    ########################### Data Augmentation #########################
    temperal_padding_ratio = 6
    shear_amp = 1
    mask_joint = 8
    mask_frame = 20
    ############################ Barlow Twins #############################
    pj_size = 6144
    lambd = 2e-4
