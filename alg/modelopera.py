# coding=utf-8
import torch
from network import img_network
import quan
import util
import pdb

def add_perturbation(model, perturbation_ratio=0.02):
    with torch.no_grad():
        for param in model.parameters():
            perturbation = torch.randn(param.size()) * perturbation_ratio
            param.data += perturbation


def add_perturbation_layer4_conv(model, perturbation_ratio=0.02):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'layer4' in name and 'conv' in name and 'weight' in name:
                perturbation = torch.randn(param.size()) * perturbation_ratio
                param.data += perturbation
def add_perturbation_conv1(model, perturbation_ratio=0.02):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'conv1.weight' == name:
                perturbation = torch.randn(param.size()) * perturbation_ratio
                param.data += perturbation
                print("conv1 success")
                break

def add_perturbation_conv(model, perturbation_ratio=0.02):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'conv' in name and 'weight' in name:
                perturbation = torch.randn(param.size()) * perturbation_ratio
                param.data +=  perturbation


def get_fea(args):
    # if args.dataset == 'dg5':
    #     net = img_network.DTNBase()
    # elif args.net.startswith('res'):
    net = img_network.ResBase(args.net, False)
    # pdb.set_trace()
    # add_perturbation_layer4_conv(net, perturbation_ratio=0.02)
    # add_perturbation_conv1(net, perturbation_ratio=0.02)
    if args.use_qat:
        new_args = util.get_config2(args.config_path)
        modules_to_replace = quan.find_modules_to_quantize(net, new_args.quan)
        net = quan.replace_module_by_names(net, modules_to_replace)
    
        # 等待替换成lsq版本
    # else:
    #     net = img_network.VGGBase(args.net)
    return net


def accuracy(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total
