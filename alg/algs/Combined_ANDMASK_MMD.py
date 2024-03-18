# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM
from alg.algs.ANDMask import ANDMask
from alg.algs.MMD import MMD
import torch.autograd as autograd


class Combined_ANDMASK_MMD(ERM):
    def __init__(self, args):
        super(Combined_ANDMASK_MMD, self).__init__(args)
        self.tau = args.tau
        self.args = args
        self.kernel_type = "gaussian"
    def mask_grads(self, tau, gradients, params):

        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            mask = torch.mean(grad_signs, dim=0).abs() >= self.tau
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)

            mask_t = (mask.sum() / mask.numel())
            param.grad = mask * avg_grad
            param.grad *= (1. / (1e-10 + mask_t))

        return 0
    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy
    def update(self, minibatches, opt, sch, mmd_average=False):
        features = [self.featurizer(
            data[0].cuda().float()) for data in minibatches]
        nmb = len(minibatches)
        mmd_average = self.args.mmd_average
        param_gradients = [[] for _ in self.network.parameters()]

        total_loss = 0
        
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = [data[1].cuda().long() for data in minibatches]
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, data in enumerate(minibatches):
            x, y = data[0].cuda().float(), data[1].cuda().long()
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            penalty = 0
            env_loss = F.cross_entropy(logits, y) 
            for j in range(i + 1, nmb):
                if all_y[j] == all_y[i]:
                    penalty += self.args.mmd_gamma * self.mmd(features[i], features[j])
            total_loss += env_loss

            env_grads = autograd.grad(
                env_loss, self.network.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)

        opt.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        opt.step()
        # if sch:
        #     sch.step()
        

        if mmd_average==False:
            param_gradients2 = [[] for _ in self.featurizer.parameters()]
            for i in range(nmb):
                for j in range(i + 1, nmb):
                    penalty = self.args.mmd_gamma * self.mmd(features[i], features[j])
                        # else:
                        #     for grads, env_grad in zip(param_gradients2, env_grads2):
                        #         grads.append(env_grad)
                    
                    # 这里面只有部分参数被用到，特征层的参数
                    env_grads2 = autograd.grad(
                    penalty, self.featurizer.parameters(), retain_graph=True)
                    # Check if any gradient is unused
                    for grads, env_grad in zip(param_gradients2, env_grads2):
                        grads.append(env_grad)
            opt.zero_grad()
            self.mask_grads(self.tau, param_gradients2, self.featurizer.parameters())
        else:
            penalty = 0
            for i in range(nmb):
                # 算数平均，可以套andmask的几何平均
                # 让经过特征变换层的输出尽可能mmd相似程度变高
                # mmd可以简单看成两个向量做任意投影的最大差值
                # 为了计算MMD构建起来的这个空间，就是RKHS：reproducing kernel Hilbert space。
                # 经证明当F是RKHS中的单位球时，是最佳的。（单位球就是到原点模1的空间，在二维坐标系也就是单位圆）
                # 这里也可以改成几何平均
                for j in range(i + 1, nmb):
                    penalty += self.mmd(features[i], features[j])

            
            if nmb > 1:
                penalty /= (nmb * (nmb - 1) / 2)

            opt.zero_grad()
            (self.args.mmd_gamma*penalty).backward()
        opt.step()
        if sch:
            sch.step()
        return {'total': mean_loss.item()}



        

