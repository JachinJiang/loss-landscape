"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable

def eval_loss(net, criterion, loader, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    # num_batch = len(loader[0])
    # print(num_batch)
    if use_cuda:
        net.cuda()
    net.eval()
    # minibatches_iterator = zip(*loader)
    with torch.no_grad():
        # # traget
        # if isinstance(criterion, nn.CrossEntropyLoss):
        #     # for iter_num in range(1):
        #     #     minibatches = next(minibatches_iterator)
        #     for batch_idx, minibatches in enumerate(loader[0]):
        #         # all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        #         # all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        #         # print(minibatches[0].shape)
        #         all_x = minibatches[0].cuda().float()
        #         all_y = minibatches[1].cuda().long()
        #         # print(all_x.shape)
        #         batch_size = all_x.size(0)
        #         total += batch_size
        #         # all_x = Variable(all_x)
        #         # all_y = Variable(all_y)
        #         # if use_cuda:
        #         #     inputs, targets = inputs.cuda(), targets.cuda()
        #         outputs = net(all_x)
                
        #         print("out", outputs[0])
        #         loss = criterion(outputs, all_y)
                
        #         total_loss += loss.item()*batch_size
                
        #         _, predicted = torch.max(outputs.data, 1)
        #         # print(predicted, all_y)
        #         correct += predicted.eq(all_y).sum().item()
                
        if isinstance(criterion, nn.CrossEntropyLoss):
            for loader_single in loader:
                # for iter_num in range(1):
                #     minibatches = next(minibatches_iterator)
                for batch_idx, minibatches in enumerate(loader_single):
                    # all_x = torch.cat([data[0].cuda().float() for data in minibatches])
                    # all_y = torch.cat([data[1].cuda().long() for data in minibatches])
                    # print(minibatches[0].shape)
                    all_x = minibatches[0].cuda().float()
                    all_y = minibatches[1].cuda().long()
                    # print(all_x.shape)
                    batch_size = all_x.size(0)
                    total += batch_size
                    # all_x = Variable(all_x)
                    # all_y = Variable(all_y)
                    # if use_cuda:
                    #     inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = net(all_x)
                    
                    # print("out", outputs[0])
                    loss = criterion(outputs, all_y)
                    
                    total_loss += loss.item()*batch_size
                    
                    _, predicted = torch.max(outputs.data, 1)
                    # print(predicted, all_y)
                    correct += predicted.eq(all_y).sum().item()
        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader[0]):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    return total_loss/total, 100.*correct/total
