import os

import numpy as np
import torch as t
import torch.utils.data
import torchvision as tv
from sklearn.model_selection import train_test_split

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from os import path

class DomainNetSet(Dataset):
    def __init__(self, paths, transforms, dataset_path):
        super(DomainNetSet, self).__init__()
        self.paths = paths
        self.transforms = transforms
        self.data_paths = None
        self.data_labels = None
        self.dataset_path = dataset_path
        self.data_paths, self.data_labels = self.read_data(paths)
        
    def read_data(self, paths):
        data_paths = []
        data_labels = []
        with open(paths, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                data_path, label = line.split(' ')
                data_path = path.join(self.dataset_path, data_path)
                label = int(label)
                data_paths.append(data_path)
                data_labels.append(label)
        return data_paths, data_labels

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def __balance_val_split(dataset, val_split=0.):
    targets = np.array(dataset.data_labels)
    train_indices, val_indices = train_test_split(
        np.arange(targets.shape[0]),
        test_size=val_split,
        stratify=targets,
        random_state = 0
    )
    train_dataset = t.utils.data.Subset(dataset, indices=train_indices)
    val_dataset = t.utils.data.Subset(dataset, indices=val_indices)
    return train_dataset, val_dataset


def __deterministic_worker_init_fn(worker_id, seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)


def load_data2(cfg):
    if cfg.val_split < 0 or cfg.val_split >= 1:
        raise ValueError('val_split should be in the range of [0, 1) but got %.3f' % cfg.val_split)

    tv_normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    
    # 只处理DomainNet
    if cfg.dataset == 'DomainNet':
        train_transform = tv.transforms.Compose([
            # transforms.Resize((224,224)),
            # 转换大小，随机缩放
            tv.transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            # 随机翻转
            tv.transforms.RandomHorizontalFlip(),
            # 随机调整图像的亮度、对比度、饱和度和色调，增加数据的多样性。
            tv.transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            
            # 随机转换为灰度
            tv.transforms.RandomGrayscale(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transform = tv.transforms.Compose([
            tv.transforms.Resize((224,224)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_set = DomainNetSet(
            paths=os.path.join(cfg.datapath, 'train.txt'), transforms=train_transform, dataset_path=cfg.path)
        test_set = DomainNetSet(
            paths=os.path.join(cfg.datapath, 'test.txt'), transforms=test_transform, dataset_path=cfg.path)
    elif cfg.dataset == 'imagenet':
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        val_transform = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv_normalize
        ])

        train_set = tv.datasets.ImageFolder(
            root=os.path.join(cfg.path, 'train'), transform=train_transform)
        test_set = tv.datasets.ImageFolder(
            root=os.path.join(cfg.path, 'val'), transform=val_transform)

    elif cfg.dataset == 'cifar10':
        train_transform = tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, 4),
            tv.transforms.ToTensor(),
            tv_normalize
        ])
        val_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv_normalize
        ])

        train_set = tv.datasets.CIFAR10(cfg.path, train=True, transform=train_transform, download=True)
        test_set = tv.datasets.CIFAR10(cfg.path, train=False, transform=val_transform, download=True)

    else:
        raise ValueError('load_data does not support dataset %s' % cfg.dataset)

    if cfg.val_split != 0:
        train_set, val_set = __balance_val_split(train_set, cfg.val_split)
    else:
        # In this case, use the test set for validation
        val_set = test_set

    worker_init_fn = None
    if cfg.deterministic:
        worker_init_fn = __deterministic_worker_init_fn
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    
    
    world_size = torch.distributed.get_world_size()
    train_loader = t.utils.data.DataLoader(
        train_set, cfg.batch_size // world_size , pin_memory=True, worker_init_fn=worker_init_fn, num_workers=cfg.workers, sampler=train_sampler, shuffle=False)
    val_loader = t.utils.data.DataLoader(
        val_set, cfg.batch_size, pin_memory=True, worker_init_fn=worker_init_fn)
    test_loader = t.utils.data.DataLoader(
        test_set, cfg.batch_size, pin_memory=True, worker_init_fn=worker_init_fn)
    return train_loader, val_loader, test_loader



# from os import path
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# import sys
# sys.path.append("..")



# class DomainNetLoader:
#     def __init__(
#         self,
#         domain_name='clipart',
#         dataset_path=None,
#         batch_size=64,
#         num_workers=4,
#         use_gpu=False,
#         _C=None, 
#     ):
#         super(DomainNetLoader, self).__init__()
#         self.domain_name = domain_name
#         self.dataset_path = dataset_path
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.use_gpu = use_gpu
#         self._C = _C
#         # -------domainbed----------
#         # https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py
#         self.transforms_train = transforms.Compose([
#             # transforms.Resize((224,224)),
#             # 转换大小，随机缩放
#             transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
#             # 随机翻转
#             transforms.RandomHorizontalFlip(),
#             # 随机调整图像的亮度、对比度、饱和度和色调，增加数据的多样性。
#             transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            
#             # 随机转换为灰度
#             transforms.RandomGrayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#         self.transforms_test = transforms.Compose([
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])

#         # ------openmmlab------
#         # https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/datasets/imagenet_bs64_swin_224.py
#         # self.transforms_train = transforms.Compose([
#         #     transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC), 
#         #     transforms.RandomHorizontalFlip(p=0.5),
#         #     transforms.RandAugment(num_ops=2, magnitude=9, interpolation=transforms.InterpolationMode.BICUBIC), 
#         #     transforms.ToTensor(),
#         #     transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)), 
#         #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         # ])
#         # self.transforms_test = transforms.Compose([
#         #     transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC),
#         #     transforms.CenterCrop(224),
#         #     transforms.ToTensor(), 
#         #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
#         # ])
          
#         # -----------KD3A-------------
#         # self.transforms_train = transforms.Compose([
#         #     transforms.RandomResizedCrop(224, scale=(0.75, 1)),
#         #     transforms.RandomHorizontalFlip(),
#         #     transforms.ToTensor()
#         # ])
#         # self.transforms_test = transforms.Compose([
#         #     transforms.Resize((224,224)),
#         #     transforms.ToTensor(), 
#         # ])
        
    


#     def get_dloader(self):
#         '''
#         load target domain
#         return the training/val/test dataloader of the target domain
#         '''
#         print(f'==> Loading DomainNet {self.domain_name}...')



#         train_dataset = DomainNetSet(train_data_paths, train_data_labels, self.transforms_train)
#         val_dataset = DomainNetSet(val_data_paths, val_data_labels, self.transforms_test)
#         test_dataset = DomainNetSet(test_data_paths, test_data_labels, self.transforms_test)

#         train_dloader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
#                                 shuffle=True)
#         val_dloader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
#                                 shuffle=False)
#         test_dloader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
#                                 shuffle=False)
#         print(f'Train sample number: {len(train_data_paths)}\t Val sample number: {len(val_data_paths)}\t Test sample number: {len(test_data_paths)}')
#         return train_dloader, val_dloader, test_dloader