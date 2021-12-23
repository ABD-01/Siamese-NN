import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """

    def __init__(self, input_shape, hid_channels=64, out_channels=128):
        super(MNIST_CNN, self).__init__()

        self.conv1 = self.conv_block(input_shape[0], hid_channels)
        self.conv2 = self.conv_block(hid_channels, hid_channels)
        self.conv3 = self.conv_block(hid_channels, hid_channels)
        self.conv4 = self.conv_block(hid_channels, out_channels)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x

    def conv_block(self, in_channels,  out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

class Proto(nn.Module):

    def __init__(self, input_shape, num_classes, args, device) -> None:
        super(Proto, self).__init__()

        self.device = device
        self.n_support = args.num_support_tr
        self.ft_output_size = 128
        self.proto_size = int(self.ft_output_size * 0.25)
        self.feat_size = int(self.ft_output_size)

        self.prototyper = nn.Sequential(
            MNIST_CNN(input_shape),
            nn.Linear(in_features=self.ft_output_size, out_features=self.proto_size)
        )
        self.prototyper.to(self.device, non_blocking=True)

        self.optimizer = torch.optim.Adam(
            self.prototyper.parameters(), lr = args.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, args.lr_scheduler_step, gamma=args.lr_scheduler_gamma, verbose=False
        )


    def prototype_update(self, minibatch):
        """ Update to train prototypical network. """

        # all_x = torch.cat([x for x, y in minibatch])
        # all_d = torch.cat(
        #     [i * torch.ones((y.shape[0])) for i, (x, y) in enumerate(minibatch)]
        # )
        x, y = minibatch
        x, y = x.to(self.device), y.to(self.device)
        # x = x.to(self.device)

        # x_proto = self.prototyper(all_x)
        # num_domains_iter = len(minibatch)
        # td = num_domains_iter

        # loss, accuracy = prototypical_loss(
        #     x_proto, all_d, int(x_proto.shape[0] / (td * 1.0 / self.proto_frac))
        # )
        x_proto = self.prototyper(x)
        loss, accuracy = self.prototypical_loss(x_proto, y, self.n_support)
        del x

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

        if torch.is_tensor(loss):
            loss = loss.item()

        if torch.is_tensor(accuracy):
            accuracy = accuracy.item()

        return {"proto_loss": loss, "proto_acc": accuracy}


    def save_prototype(self, output_file):
        """ Write prototype to file. """
        return torch.save(self.prototyper, output_file)

    def load_prototype(self, output_file):
        """ Load prototype from file. """
        self.prototyper = torch.load(output_file)

    def compute_average_prototype(self, x):
        """ Compute prototype feature and average it."""
        x = self.prototyper(x)
        return torch.mean(x, dim=0).detach().cpu()

    def attach_prototypes(self, prototypes):
        """ Add prototypes to model. """
        self.prototypes = prototypes

    def init_prototype_training(self):
        """ Set up model to train prototype. """
        self.prototyper.to(self.device, non_blocking=True)
        self.prototyper = nn.parallel.DataParallel(self.prototyper).to(self.device, non_blocking=True)


    def prototypical_loss(self, input, target, n_support, n_query=None):

        classes = torch.unique(target)
        n_classes = len(classes)

        # n_q, selected_classes = [], []
        # for cx in classes:
        #     _nq_x = target.eq(cx.item()).sum().item() - n_support
        #     if _nq_x >= 0:
        #         n_q.append(_nq_x)
        #         selected_classes.append(cx.item())
        # # n_query = min(n_q + [n_query])
        # n_query = min(n_q)

        # classes = [cx for cx in classes if cx.item() in selected_classes]
        # n_classes = len(classes)

        n_query = target.eq(classes[0].item()).sum().item() - n_support

        support_idxs = list(map(lambda c: target.eq(c).nonzero()[:n_support].squeeze(1), classes))

        prototypes = torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])

        query_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[n_support:], classes))).view(-1)
                    #  torch.stack(list(map(lambda c: torch.where(target.eq(c))[0][n_support:], classes))).view(-1)
        query_samples = input[query_idxs]

        dists = self.euclidean_dist(query_samples, prototypes)

        target_inds = torch.arange(0, n_classes, device=self.device).view(n_classes, 1, 1).expand(n_classes, n_query, 1).long()
        # target_inds = torch.arange(0, n_classes).view(n_classes, 1, 1).expand(n_classes, n_query, 1).long()

        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
        loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss, acc_val
        
    def euclidean_dist(self, x, y):
        """
        Compute euclidean distance between two tensors
        """
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# def extract_sample(n_way, n_support, n_query, datax, datay):
#   """
#   Picks random sample of size n_support+n_querry, for n_way classes
#   Args:
#       n_way (int): number of classes in a classification task
#       n_support (int): number of labeled examples per class in the support set
#       n_query (int): number of labeled examples per class in the query set
#       datax (np.array): dataset of images
#       datay (np.array): dataset of labels
#   Returns:
#       (dict) of:
#         (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
#         (int): n_way
#         (int): n_support
#         (int): n_query
#   """
#   sample = []
#   K = np.random.choice(np.unique(datay), n_way, replace=False)
#   for cls in K:
#     datax_cls = datax[datay == cls]
#     perm = np.random.permutation(datax_cls)
#     sample_cls = perm[:(n_support+n_query)]
#     sample.append(sample_cls)
#   sample = np.array(sample)
#   sample = torch.from_numpy(sample).float()
#   sample = sample.permute(0,1,4,2,3)
#   return({
#       'images': sample,
#       'n_way': n_way,
#       'n_support': n_support,
#       'n_query': n_query
#       })


# def prototypical_loss(input, target, n_support):
#     '''
#     Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
#     Compute the barycentres by averaging the features of n_support
#     samples for each class in target, computes then the distances from each
#     samples' features to each one of the barycentres, computes the
#     log_probability for each n_query samples for each one of the current
#     classes, of appartaining to a class c, loss and accuracy are then computed
#     and returned
#     Args:
#     - input: the model output for a batch of samples
#     - target: ground truth for the above batch of samples
#     - n_support: number of samples to keep in account when computing
#       barycentres, for each one of the current classes
#     '''
#     target_cpu = target.to('cpu')
#     input_cpu = input.to('cpu')

#     def supp_idxs(c):
#         # FIXME when torch will support where as np
#         return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

#     # FIXME when torch.unique will be available on cuda too
#     classes = torch.unique(target_cpu)
#     n_classes = len(classes)
#     # FIXME when torch will support where as np
#     # assuming n_query, n_target constants
#     n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

#     support_idxs = list(map(supp_idxs, classes))

#     prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
#     # FIXME when torch will support where as np
#     query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

#     query_samples = input.to('cpu')[query_idxs]
#     dists = euclidean_dist(query_samples, prototypes)

#     log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

#     target_inds = torch.arange(0, n_classes)
#     target_inds = target_inds.view(n_classes, 1, 1)
#     target_inds = target_inds.expand(n_classes, n_query, 1).long()

#     loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
#     _, y_hat = log_p_y.max(2)
#     acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

#     return loss_val,  acc_val