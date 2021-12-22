import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import networks
from .lib.misc import cross_entropy, random_pairs_of_minibatches
from .lib.prototype import prototypical_loss


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__(input_shape)

        self.conv1 = self.conv_block(input_shape[0], 64)
        self.conv2 = self.conv_block(64, 64)
        self.conv3 = self.conv_block(64, 64)
        self.conv4 = self.conv_block(64, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

class Proto(nn.Module):

    def __init__(self, input_shape, num_classes) -> None:
        super(Proto, self).__init__()

        self.ft_output_size = 128
        self.proto_size = int(self.ft_output_size * 0.25)
        self.feat_size = int(self.ft_output_size)

        self.prototyper = nn.Sequential(
            MNIST_CNN(input_shape),
            nn.Linear(in_features=self.ft_output_size, out_features=self.proto_size)
        )
        # self.featurizer = nn.Sequential(
        #     MNIST_CNN(input_shape),
        #     nn.Linear(in_features=self.ft_output_size, out_features=self.feat_size)    
        # )
        # bottleneck_size = 1024
        # self.bottleneck = nn.Sequential(
        #     nn.Linear(self.feat_size+self.proto_size, bottleneck_size)
        # )
        # self.classifier = nn.Linear(bottleneck_size, num_classes)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr = 1e-3
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=0.5, last_epoch=-1
        )


    def prototype_update(self, minibatches):
        """ Update to train prototypical network. """

        all_x = torch.cat([x for x, y in minibatches])
        all_d = torch.cat(
            [i * torch.ones((y.shape[0])) for i, (x, y) in enumerate(minibatches)]
        )

        x_proto = self.prototyper(all_x)
        num_domains_iter = len(minibatches)
        td = num_domains_iter

        loss, accuracy = prototypical_loss(
            x_proto, all_d, int(x_proto.shape[0] / (td * 1.0 / self.proto_frac))
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if torch.is_tensor(loss):
            loss = loss.item()

        if torch.is_tensor(accuracy):
            accuracy = accuracy.item()

        return {"proto_loss": loss, "proto_acc": accuracy}



class Proto(nn.Module):
    """
    Domain-Aware Prototypical Domain Generalization (Proto)
    The abstract class for Proto builds on ERM
    The main feature of Proto abstract class is to load an additional
    "prototype" model (that is typically pretrained) in addition to the
    featurizer. For each input, the model concatenates the feature and
    prototype, followed by a bottleneck layer and then produces softmaxes.
    HYPERPARAMS:
    ============
    feat_size : dimensionality of prototyper feature (default 2048)
    bottleneck_size : dimensionality of bottleneck layer (default 512)
    data_parallel : use data-parallel processing (torch.Parallel.DataParallel,
        default=True)
    mixup : (only during prototype training) mixup samples from different
        domains, i.e., randomly select two domains and interpolate with random
        weight between 0.2 and 0.8.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, use_relu=True):
        super(Proto, self).__init__()

        self.hparams = hparams

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        # initializing constants
        self.nd = num_domains
        self.nc = num_classes

        # initializing architecture parameters
        self.ft_output_size = self.featurizer.n_outputs
        self.proto_size = int(self.ft_output_size * 0.25)
        self.feat_size = int(self.ft_output_size)

        # initializing hyperparameters
        self.proto_frac = hparams["proto_train_frac"]
        self.epochs = hparams["n_steps"]
        self.proto_epochs = hparams["n_steps_proto"]

        self.kernel_type = "gaussian"

        # initializing prototyper
        if use_relu:
            self.prototyper = nn.Sequential(
                networks.Featurizer(input_shape, self.hparams),
                nn.ReLU(inplace=False),
                nn.Linear(self.ft_output_size, self.proto_size),
                nn.ReLU(inplace=False),
            )
        else:
            self.prototyper = nn.Sequential(
                networks.Featurizer(input_shape, self.hparams),
                nn.Linear(self.ft_output_size, self.proto_size),
            )

        # initializing featurizer
        if use_relu:
            self.featurizer = nn.Sequential(
                networks.Featurizer(input_shape, self.hparams),
                nn.ReLU(inplace=False),
                nn.Linear(self.ft_output_size, self.feat_size),
                nn.ReLU(inplace=False),
            )
        else:
            self.featurizer = nn.Sequential(
                networks.Featurizer(input_shape, self.hparams),
                nn.Linear(self.ft_output_size, self.feat_size),
            )

        # initializing bottleneck architecture on top of prototyper
        if use_relu:
            self.bottleneck = nn.Sequential(
                nn.Linear(
                    self.feat_size + self.proto_size, self.hparams["bottleneck_size"]
                ),
                nn.ReLU(inplace=False),
            )
        else:
            self.bottleneck = nn.Sequential(
                nn.Linear(
                    self.feat_size + self.proto_size, self.hparams["bottleneck_size"]
                )
            )

        # initalizing classifier
        self.classifier = nn.Linear(self.hparams["bottleneck_size"], num_classes)

        # initialize parameters based on doing prototype training
        # or not

        do_prototype_training = (
            hparams["proto_model"] is None or hparams["train_prototype"]
        )

        if do_prototype_training:
            params = self.prototyper.parameters()
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.hparams["proto_lr"],
                weight_decay=self.hparams["proto_weight_decay"],
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.proto_epochs
            )
        else:
            params = (
                list(self.bottleneck.parameters())
                + list(self.classifier.parameters())
                + list(self.featurizer.parameters())
            )
            self.optimizer = torch.optim.Adam(
                params, lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"]
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.epochs
            )

    def prototype_update(self, minibatches):
        """ Update to train prototypical network. """

        all_x = torch.cat([x for x, y in minibatches])
        all_d = torch.cat(
            [i * torch.ones((y.shape[0])) for i, (x, y) in enumerate(minibatches)]
        )

        x_proto = self.prototyper(all_x)
        num_domains_iter = len(minibatches)
        td = num_domains_iter

        # if self.hparams["mixup"] > 0:
        #     n_mx_dom = int(self.hparams["mixup"] * num_domains_iter)
        #     n_iter = int(self.hparams["mixup"])

        #     mx_minibatches = random_pairs_of_minibatches(minibatches)[:n_mx_dom]

        #     for i in range(n_iter):
        #         _st = i * num_domains_iter
        #         _en = (i + 1) * num_domains_iter

        #         this_batch = None
        #         for (xi, yi), (xj, _) in mx_minibatches[_st:_en]:

        #             alpha = 0.2 + 0.6 * random.random()
        #             _x = alpha * xi + (1 - alpha) * xj

        #             all_d = torch.cat([all_d, td * torch.ones(yi.shape)])
        #             td += 1
        #             if this_batch is None:
        #                 this_batch = _x
        #             else:
        #                 this_batch = torch.cat([this_batch, _x])

        #         _px = self.prototyper(this_batch)
        #         x_proto = torch.cat([x_proto, _px])

        loss, accuracy = prototypical_loss(
            x_proto, all_d, int(x_proto.shape[0] / (td * 1.0 / self.proto_frac))
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

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

        self.bottleneck.to("cpu")
        self.featurizer.to("cpu")
        self.classifier.to("cpu")

        self.prototyper.to("cuda")
        self.prototyper = nn.parallel.DataParallel(self.prototyper).cuda()

    def init_main_training(self, hparams):
        """ Discard earlier optimizers and prepare for main training. """

        # first unload prototyper
        self.prototyper.to("cpu")

        self.bottleneck.to("cuda")
        self.featurizer.to("cuda")
        self.classifier.to("cuda")

        self.bottleneck = nn.parallel.DataParallel(self.bottleneck)
        self.featurizer = nn.parallel.DataParallel(self.featurizer)
        self.classifier = nn.parallel.DataParallel(self.classifier)

        params = (
            list(self.bottleneck.parameters())
            + list(self.classifier.parameters())
            + list(self.featurizer.parameters())
        )

        self.optimizer = torch.optim.Adam(
            params, lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"]
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.epochs
        )

    def predict(self, x, idx, device):
        """ Forward function to compute output. """

        bs = x.shape[0]
        proto_tile = self.prototypes[idx].to(device).unsqueeze(0).repeat(bs, 1)

        x = self.featurizer(x)
        x = torch.cat([x, proto_tile], dim=1)
        x = self.bottleneck(x)
        x = self.classifier(x)

        return x
