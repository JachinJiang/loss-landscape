# coding=utf-8
from alg.algs.ERM import ERM
from alg.algs.MMD import MMD
from alg.algs.CORAL import CORAL
from alg.algs.DANN import DANN
from alg.algs.RSC import RSC
from alg.algs.Mixup import Mixup
from alg.algs.MLDG import MLDG
from alg.algs.GroupDRO import GroupDRO
from alg.algs.ANDMask import ANDMask
from alg.algs.VREx import VREx
from alg.algs.DIFEX import DIFEX
from alg.algs.RSC_GT import RSC_GT
from alg.algs.RSC_RANDOM import RSC_RANDOM
from alg.algs.Combined_ANDMASK_MMD import Combined_ANDMASK_MMD
ALGORITHMS = [
    'ERM',
    'Mixup',
    'CORAL',
    'MMD',
    'DANN',
    'MLDG',
    'GroupDRO',
    'RSC',
    'ANDMask',
    'VREx',
    'DIFEX',
    'RSC_GT',
    'RSC_RANDOM',
    'Combined_ANDMASK_MMD'
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
