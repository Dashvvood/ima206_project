import argparse

MEDMNIST_ROOT = "../data/medmnist2d/"
PathMNIST_MEAN = [0.73765225, 0.53090023, 0.70307171]
PathMNIST_STD = [0.12319908, 0.17607205, 0.12394462]
PathMNIST_HIST = [9366, 9510, 10362, 10404, 8010, 12187, 7892, 9408, 12893]


PathMNIST = {
    "ROOT": "../data/medmnist2d/",
    "MEAN": [0.73765225, 0.53090023, 0.70307171],
    "STD": [0.12319908, 0.17607205, 0.12394462],
    "HIST": [9366, 9510, 10362, 10404, 8010, 12187, 7892, 9408, 12893],
}

PathMNISTmeta = argparse.Namespace(**PathMNIST)


CKPT_DICT = {
    "resnet18_001": "./20240604-034156/last.ckpt",
    "resnet18_010": "./20240605-160349/last.ckpt",
    "resnet18_100": "./20240606-022215/last.ckpt",
    "bt_pretrain_001": "./20240607-055917/last.ckpt",
    "bt_pretrain_010": "./20240607-053715/last.ckpt",
    "bt_pretrain_100": "./20240607-115243/last.ckpt",
    "bt_pretrain_wo_rot_001": "./20240607-075832/last.ckpt",
    "bt_pretrain_wo_rot_010": "./20240607-063012/last.ckpt",
    "bt_pretrain_wo_rot_100": "./20240607-132913/last.ckpt",
    
    "bt_finetune_001": "./20240607-071946/last.ckpt",
    "bt_finetune_010": "./20240607-070031/last.ckpt",
    "bt_finetune_100": "./20240607-131352/epoch=177-step=7654.ckpt",
    
    "bt_finetune_001_wo_rot": "./20240607-084152/last.ckpt",
    "bt_finetune_010_wo_rot": "./20240607-083010/last.ckpt",
    "bt_finetune_100_wo_rot": "./20240607-143231/epoch=195-step=8428.ckpt",
    
    "bt_lr_100":  "./20240607-162429/epoch=192-step=8299.ckpt",
    "bt_lr_100_wo_rot": "./20240607-162356/",
}