# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM
import torch.autograd as autograd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# out1 = []
# out2 = []
# out3 = []
# out4 = []
# def hook_fn1(module, input, output):
#     out1.clear()
#     out1.append(output)
#     # print("1", output)
# def hook_fn2(module, input, output):
#     out2.clear()
#     out2.append(output)
#     # print("2", output)
# # def hook_fn3(module, input, output):
# #     out3.clear()
# #     out3.append(output)
# #     # print("3", output)
# # def hook_fn4(module, input, output):
# #     out4.clear()
# #     out4.append(output)
#     # print("4", output)
# def calculate_cosine_similarity(tensor1, tensor2):
#     return F.cosine_similarity(tensor1, tensor2)

# class ANDMask(ERM):
#     def __init__(self, args):
#         super(ANDMask, self).__init__(args)
#         self.tau = args.tau
#         self.featurizer.conv1.register_forward_hook(hook_fn1)
#         # self.featurizer.conv1.register_forward_hook(hook_fn2)
#         for name, module in self.featurizer.named_modules():
#             if 'layer4.1.bn2' in name:
#                 module.register_forward_hook(hook_fn2)
#                 # module.register_forward_hook(hook_fn4)
#                 break
            
#     # 先大多数再平均
#     def update(self, minibatches, opt, sch):
#         total_loss = 0
#         param_gradients = [[] for _ in self.network.parameters()]
#         # all_x = torch.cat([data[0].cuda().float() for data in minibatches])
#         # all_x = minibatches[0][0].cuda().float()
#         # print(all_x.shape)
#         all_x = torch.cat([data[0].cuda().float() for data in minibatches])
#         # print(all_x.shape)
#         all_logits = self.network(all_x)
#         all_logits_idx = 0
#         nmb = len(minibatches)
#         # 不进行聚类的逻辑
#         # print(out1[0].shape)
#         # print(out2[0].shape)
#         # print(out3[0].shape)
#         # print(out4[0].shape)
#         for i, data in enumerate(minibatches):
#             conv1_dist = 0
#             conv2_dist = 0
#             conv2_simi = 0
#             x, y = data[0].cuda().float(), data[1].cuda().long()
#             logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
#             all_logits_idx += x.shape[0]
#             env_loss = F.cross_entropy(logits, y)
            
#             for j in range(i + 1, nmb):
#                 conv1_dist += torch.dist(out1[0][i * x.shape[0]:(i + 1) * x.shape[0]], out1[0][j * x.shape[0]:(j + 1) * x.shape[0]])
#                 conv2_dist += torch.dist(out2[0][i * x.shape[0]:(i + 1) * x.shape[0]], out2[0][j * x.shape[0]:(j + 1) * x.shape[0]])
#                 conv2_simi += F.cosine_similarity(out2[0][i * x.shape[0]:(i + 1) * x.shape[0]].view(1, -1), out2[0][j * x.shape[0]:(j + 1) * x.shape[0]].view(1, -1))
#             if nmb > i + 1:
#                 conv2_simi = torch.squeeze(conv2_simi)
#                 conv1_dist /= (nmb - i - 1) * x.shape[0]
#                 conv2_dist /= (nmb - i - 1) * x.shape[0]
#                 # conv2_simi /= (nmb - i - 1) * x.shape[0]
#                 print(conv1_dist, conv2_dist, torch.squeeze(conv2_simi))
#                 # env_loss +=  1e-2 *  ((-conv1_dist / 200) + conv2_dist)
#                 env_loss += 0.5 * (0.1 * conv2_dist - conv2_simi)
#                 print("conv1_dist:", conv1_dist)
#                 print("conv2_dist:", conv2_dist)
#                 print("conv2_simi:", conv2_simi)
#                 print("conv loss", 0.1 * conv2_dist - conv2_simi)
#                 # print("conv loss",  1e-2 *  ((-conv1_dist / 200) + conv2_dist))
#             total_loss += env_loss

#             env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
#             for grads, env_grad in zip(param_gradients, env_grads):
#                 grads.append(env_grad)

#         mean_loss = total_loss / len(minibatches)

#         opt.zero_grad()
#         self.mask_grads(self.tau, param_gradients, self.network.parameters())
#         opt.step()
#         if sch:
#             sch.step()

#         return {'total': mean_loss.item()}
    
#     def mask_grads(self, tau, gradients, params):

#         for param, grads in zip(params, gradients):
#             grads = torch.stack(grads, dim=0)
#             grad_signs = torch.sign(grads)
#             # 判断一个参数的梯度要不要保留
#             mask = torch.mean(grad_signs, dim=0).abs() >= tau
#             # print(grad_signs.shape)
#             mask = mask.to(torch.float32)
#             avg_grad = torch.mean(grads, dim=0)
#             # 判断有多少个参数
#             mask_t = (mask.sum() / mask.numel())
#             # print(mask_t)
#             param.grad = mask * avg_grad
#             scale = 1e-10 + mask_t
#             param.grad *= 1 / scale

#         return 0

class ANDMask(ERM):
    def __init__(self, args):
        super(ANDMask, self).__init__(args)

        self.tau = args.tau
        # if self.use_disturb == True:
        
        # if self.use_disturb == True:
        #     add_perturbation_conv(self.network, perturbation_ratio=self.perturbation_ratio)
        #     print("add !")
        # add_perturbation_conv(self.network, perturbation_ratio=0.01)
        # print("add !")
    def update(self, minibatches, opt, sch):

        total_loss = 0
        param_gradients = [[] for _ in self.network.parameters()]
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, data in enumerate(minibatches):
            x, y = data[0].cuda().float(), data[1].cuda().long()
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]

            env_loss = F.cross_entropy(logits, y)
            total_loss += env_loss

            env_grads = autograd.grad(
                env_loss, self.network.parameters(), retain_graph=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                grads.append(env_grad)

        mean_loss = total_loss / len(minibatches)

        opt.zero_grad()
        self.mask_grads(self.tau, param_gradients, self.network.parameters())
        opt.step()
        if sch:
            sch.step()

        return {'total': mean_loss.item()}
    
    def mask_grads(self, tau, gradients, params):
        for param, grads in zip(params, gradients):
            grads = torch.stack(grads, dim=0)
            grad_signs = torch.sign(grads)
            # 判断一个参数的梯度要不要保留
            mask = torch.mean(grad_signs, dim=0).abs() >= tau
            # print(grad_signs.shape)
            mask = mask.to(torch.float32)
            avg_grad = torch.mean(grads, dim=0)
            # 判断有多少个参数
            mask_t = (mask.sum() / mask.numel())
            # print(mask_t)
            param.grad = mask * avg_grad
            scale = 1e-10 + mask_t
            param.grad *= 1 / scale

        return 0
#     # # # 先平均再听大多数
#     # # def update(self, minibatches, opt, sch):
#     # #     if self.tau > 1:
#     # #         self.cluster = True

#     # #     if self.cluster:
#     # #         # Clustering
#     # #         total_loss = 0
#     # #         param_gradients = [[] for _ in self.network.parameters()]
#     # #         all_x = torch.cat([data[0].cuda().float() for data in minibatches])
#     # #         all_y = torch.cat([data[1].cuda().long() for data in minibatches])
#     # #         all_logits = self.network(all_x)
#     # #         all_logits_idx = 0
#     # #         embeddings = all_logits.detach().cpu().numpy()
#     # #         similarity_matrix = cosine_similarity(embeddings)
#     # #         n_clusters = 10
#     # #         kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=3).fit(similarity_matrix)
#     # #         cluster_labels = kmeans.labels_

#     # #         cluster_gradients = [[] for _ in self.network.parameters()]

#     # #         cluster_labels_tensor = torch.tensor(cluster_labels)  # Convert cluster_labels to a tensor

#     # #         for cluster_label in range(10):
#     # #             cluster_indices = (cluster_labels_tensor == cluster_label).nonzero().view(-1)  # Use cluster_labels_tensor
#     # #             cluster_x = all_x[cluster_indices]
#     # #             cluster_y = all_y[cluster_indices]

#     # #             cluster_logits = self.network(cluster_x)
#     # #             cluster_loss = F.cross_entropy(cluster_logits, cluster_y)
#     # #             total_loss += cluster_loss

#     # #             cluster_grads = autograd.grad(cluster_loss, self.network.parameters(), retain_graph=True)
#     # #             for grads, cluster_grad in zip(cluster_gradients, cluster_grads):
#     # #                 grads.append(cluster_grad)

#     # #         mean_loss = total_loss / n_clusters  # 计算平均损失

#     # #         opt.zero_grad()

#     # #         # 计算每个簇的平均梯度
#     # #         # 将 grad_list 转换为张量，并计算每个簇的平均梯度
#     # #         avg_cluster_gradients = [[torch.mean(torch.stack(grad_list), dim=0)] for grad_list in cluster_gradients]
#     # #         self.mask_grads(0.6, avg_cluster_gradients, self.network.parameters())  # Pass cluster_labels_tensor
#     # #         opt.step()
#     # #         if sch:
#     # #             sch.step()

#     # #     else:
#     # #         total_loss = 0
#     # #         param_gradients = [[] for _ in self.network.parameters()]
#     # #         all_x = torch.cat([data[0].cuda().float() for data in minibatches])
#     # #         all_logits = self.network(all_x)
#     # #         all_logits_idx = 0
#     # #         # 不进行聚类的逻辑
#     # #         for i, data in enumerate(minibatches):
#     # #             x, y = data[0].cuda().float(), data[1].cuda().long()
#     # #             logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
#     # #             all_logits_idx += x.shape[0]

#     # #             env_loss = F.cross_entropy(logits, y)
#     # #             total_loss += env_loss

#     # #             env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
#     # #             for grads, env_grad in zip(param_gradients, env_grads):
#     # #                 grads.append(env_grad)

#     # #         mean_loss = total_loss / len(minibatches)

#     # #         opt.zero_grad()
#     # #         self.mask_grads(self.tau, param_gradients, self.network.parameters())
#     # #         opt.step()
#     # #         if sch:
#     # #             sch.step()

#     # #     return {'total': mean_loss.item()}
#     # 先大多数再平均
#     def update(self, minibatches, opt, sch):
#         if self.tau > 1:
#             self.cluster = True

#         if self.cluster:
#             # Clustering
#             total_loss = 0
#             param_gradients = [[] for _ in self.network.parameters()]
#             all_x = torch.cat([data[0].cuda().float() for data in minibatches])
#             all_y = torch.cat([data[1].cuda().long() for data in minibatches])
#             all_logits = self.network(all_x)
#             all_logits_idx = 0
#             embeddings = all_logits.detach().cpu().numpy()
#             similarity_matrix = cosine_similarity(embeddings)
#             n_clusters = 5
#             kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=3).fit(similarity_matrix)
#             cluster_labels = kmeans.labels_

#             cluster_gradients_all = []
#             for i in range(n_clusters):
#                 cluster_gradients = [[] for _ in self.network.parameters()]
#                 cluster_gradients_all.append(cluster_gradients)
#             cluster_labels_tensor = torch.tensor(cluster_labels)  # Convert cluster_labels to a tensor

#             for cluster_label in range(n_clusters):
#                 cluster_indices = (cluster_labels_tensor == cluster_label).nonzero().view(-1)  # Use cluster_labels_tensor
#                 cluster_x = all_x[cluster_indices]
#                 cluster_y = all_y[cluster_indices]

#                 cluster_logits = self.network(cluster_x)
#                 cluster_loss = F.cross_entropy(cluster_logits, cluster_y)
#                 total_loss += cluster_loss

#                 cluster_grads = autograd.grad(cluster_loss, self.network.parameters(), retain_graph=True)
#                 for grads, cluster_grad in zip(cluster_gradients_all[cluster_label], cluster_grads):
#                     grads.append(cluster_grad)

#             mean_loss = total_loss / n_clusters  # 计算平均损失

#             opt.zero_grad()
            
#             # 计算每个簇的平均梯度
#             # 将 grad_list 转换为张量，并计算每个簇的平均梯度
#             avg_cluster_gradients = [[] for _ in self.network.parameters()]
#             for cluster_gradients in  cluster_gradients_all:    
#                 for grads, grad_list in zip(avg_cluster_gradients, cluster_gradients):
#                     grads.append(torch.mean(torch.stack(grad_list), dim=0))
            
            
#             # avg_cluster_gradients相当于一个列表，元素是每个参数的梯度列表
#             self.mask_grads(0.6, [avg_cluster_gradients], self.network.parameters())  # Pass cluster_labels_tensor
#             opt.step()
#             if sch:
#                 sch.step()

#         else:
#             total_loss = 0
#             param_gradients = [[] for _ in self.network.parameters()]
#             all_x = torch.cat([data[0].cuda().float() for data in minibatches])
#             all_logits = self.network(all_x)
#             all_logits_idx = 0
#             # 不进行聚类的逻辑
#             for i, data in enumerate(minibatches):
#                 x, y = data[0].cuda().float(), data[1].cuda().long()
#                 logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
#                 all_logits_idx += x.shape[0]

#                 env_loss = F.cross_entropy(logits, y)
#                 total_loss += env_loss

#                 env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
#                 for grads, env_grad in zip(param_gradients, env_grads):
#                     grads.append(env_grad)

#             mean_loss = total_loss / len(minibatches)

#             opt.zero_grad()
#             self.mask_grads(self.tau, param_gradients, self.network.parameters())
#             opt.step()
#             if sch:
#                 sch.step()

#         return {'total': mean_loss.item()}
    
    # def mask_grads(self, tau, gradients, params):
    #     for param, grads in zip(params, gradients):
    #         grads = torch.stack(grads, dim=0)
    #         grad_signs = torch.sign(grads)
    #         # 判断一个参数的梯度要不要保留
    #         mask = torch.mean(grad_signs, dim=0).abs() >= tau
    #         # print(grad_signs.shape)
    #         mask = mask.to(torch.float32)
    #         avg_grad = torch.mean(grads, dim=0)
    #         # 判断有多少个参数
    #         mask_t = (mask.sum() / mask.numel())
    #         # print(mask_t)
    #         param.grad = mask * avg_grad
    #         scale = 1e-10 + mask_t
    #         param.grad *= 1 / scale

    #     return 0
        
# # 聚类梯度版本

# # coding=utf-8
# import torch
# import torch.nn.functional as F
# from alg.algs.ERM import ERM
# import torch.autograd as autograd
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_similarity


# class ANDMask(ERM):
#     def __init__(self, args):
#         super(ANDMask, self).__init__(args)

#         self.tau = args.tau
#         self.cluster = False
#     # def update(self, minibatches, opt, sch, combined=False, cluster=False):

#     #     total_loss = 0
#     #     param_gradients = [[] for _ in self.network.parameters()]
#     #     all_x = torch.cat([data[0].cuda().float() for data in minibatches])
#     #     all_logits = self.network(all_x)
#     #     all_logits_idx = 0
#     #     for i, data in enumerate(minibatches):
#     #         x, y = data[0].cuda().float(), data[1].cuda().long()
#     #         logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
#     #         all_logits_idx += x.shape[0]

#     #         env_loss = F.cross_entropy(logits, y)
#     #         total_loss += env_loss

#     #         env_grads = autograd.grad(
#     #             env_loss, self.network.parameters(), retain_graph=True)
#     #         for grads, env_grad in zip(param_gradients, env_grads):
#     #             grads.append(env_grad)

#     #     mean_loss = total_loss / len(minibatches)

#     #     if combined == False:
#     #         opt.zero_grad()
#     #     self.mask_grads(self.tau, param_gradients, self.network.parameters())
#     #     opt.step()
#     #     if sch:
#     #         sch.step()

#     #     return {'total': mean_loss.item()}

#     def update(self, minibatches, opt, sch):
#         if self.tau > 1:
#             self.cluster = True

#         total_loss = 0
#         param_gradients = [[] for _ in self.network.parameters()]
#         all_x = torch.cat([data[0].cuda().float() for data in minibatches])
#         all_logits = self.network(all_x)
#         all_logits_idx = 0
#         # 不进行聚类的逻辑
#         for i, data in enumerate(minibatches):
#             x, y = data[0].cuda().float(), data[1].cuda().long()
#             logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
#             all_logits_idx += x.shape[0]

#             env_loss = F.cross_entropy(logits, y)
#             total_loss += env_loss

#             env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True)
#             for grads, env_grad in zip(param_gradients, env_grads):
#                 grads.append(env_grad)

#         mean_loss = total_loss / len(minibatches)

#         opt.zero_grad()
#         self.mask_grads(self.tau, param_gradients, self.network.parameters())
#         opt.step()
#         if sch:
#             sch.step()

#         return {'total': mean_loss.item()}
#     def mask_grads(self, tau, gradients, params):
        
            
#         for param, grads in zip(params, gradients):
#             grads = torch.stack(grads, dim=0)
#             grad_signs = torch.sign(grads)
#             mask = torch.mean(grad_signs, dim=0).abs() >= tau
#             mask = mask.to(torch.float32)
#             avg_grad = torch.mean(grads, dim=0)

#             mask_t = (mask.sum() / mask.numel())
#             param.grad = mask * avg_grad
#             param.grad *= (1. / (1e-10 + mask_t))

#         return 0




