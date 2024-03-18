# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.mydataloader import InfiniteDataLoader


def get_img_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist, tardatalist = [], [], []

    """
        args.img_dataset = {
        'office': ['amazon', 'dslr', 'webcam'],
        'office-caltech': ['amazon', 'dslr', 'webcam', 'caltech'],
        'office-home': ['Art', 'Clipart', 'Product', 'Real_World'],
        'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'],
        'dg5': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
        'VLCS': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    }
    
    """
    # names = args.img_dataset[args.dataset]
    names = ['art_painting', 'cartoon', 'photo', 'sketch']
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in args.test_envs:
            tardatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.draw_trans(args.dataset), test_envs=args.test_envs))
        else:
            tmpdatay = ImageDataset(args.dataset, args.task, args.data_dir,
                                    names[i], i, transform=imgutil.draw_trans(args.dataset), test_envs=args.test_envs).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1-rate, random_state=args.seed)
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l*rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]


            # 包含（1-rate）*train
            trdatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.draw_trans(args.dataset), indices=indextr, test_envs=args.test_envs))
            # 包含test+训练的域抽rate
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.draw_trans(args.dataset), indices=indexte, test_envs=args.test_envs))
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    
    # 包含（1-rate）*train,test,train * rate
    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in tedatalist]

    target_loaders = [DataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in tardatalist]


    return train_loaders, eval_loaders, target_loaders
