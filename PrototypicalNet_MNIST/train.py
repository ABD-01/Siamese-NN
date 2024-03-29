import random
import argparse
import os

from tqdm import tqdm
from termcolor import colored

import numpy as np
import torch
from torchsummary import summary
import torchvision.transforms as transforms
from torchvision.datasets import Omniglot

# import wandb

from model import *
from sampler import PrototypeSampler



def seed_init(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def get_dataset(args):
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.92206, std=0.08426)
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.92206, std=0.08426)
    ])

    train_data = Omniglot(root=args.dataset_root, background=True, transform = train_transform, download=True)
    val_data = Omniglot(root=args.dataset_root, background=False, transform = val_transform, download=True)

    return train_data, val_data

def get_loader(data, args, train=True):
    if train:
        targets = torch.tensor(list(map(lambda a: a[1], data._flat_character_images)))
        sampler = PrototypeSampler(targets, args.num_support_tr, args.num_query_tr, args.classes_per_it_tr, args.iterations)
        loader = torch.utils.data.DataLoader(data, batch_sampler=sampler, num_workers=1, pin_memory=True)
    else:
        targets = torch.tensor(list(map(lambda a: a[1], data._flat_character_images)))
        sampler = PrototypeSampler(targets, args.num_support_val, args.num_query_val, args.classes_per_it_val, args.iterations)
        loader = torch.utils.data.DataLoader(data, batch_sampler=sampler, num_workers=1, pin_memory=True)
    return loader

def get_model(input_shape, arg, d):
    model = Proto(input_shape, args.classes_per_it_tr, args, d)
    return model

@torch.inference_mode()
def validate(model, loader):
    for (x,y) in tqdm(loader, position=0, leave=True):
        x, y = x.to(DEVICE), y.to(DEVICE)
        val_loss, val_acuracy = AverageMeter("Val_Loss"), AverageMeter("Val_Accuracy")
        torch.cuda.empty_cache()

        x_hat = model.prototyper(x)
        loss, accuracy = model.prototypical_loss(x_hat, y, args.num_support_val)
        
        # wandb.log(dict(val_loss=loss, val_accuracy=accuracy))
        val_loss.update(loss)
        val_acuracy.update(accuracy)

    print(colored(val_loss, 'blue'))
    print(colored(val_acuracy, 'blue'))
    # wandb.log({"val_loss": val_loss.avg, "val_accuracy": val_acuracy.avg})



def main(args):
    global DEVICE
    DEVICE = torch.device('cpu')
    if torch.cuda.is_available() and args.cuda:
        DEVICE = torch.device('cuda')
        print(colored('Using GPU', 'green'))
    else:
        print(colored('WARNING: You are not using GPU','red'))

    seed_init(args.seed)

    # wandb.init(project="ProtoNets", config=args)

    train_data, val_data = get_dataset(args)
    train_loader = get_loader(train_data, args)
    val_loader = get_loader(val_data, args, train=False)

    ishape=train_data[0][0].shape
    model = get_model(ishape, args, DEVICE)
    # print(colored(summary(model.prototyper.cuda(), ishape), 'blue'))
    # model.prototyper.to(DEVICE)
    if args.resume is not None:
        model.load_prototype(args.resume)
        print(colored("Resumimg training", "blue"))

    for epoch in range(args.epochs):
        for batch in tqdm(train_loader, position=0, leave=True):
            epoch_loss, epoch_acuracy = AverageMeter("Loss"), AverageMeter("Accuracy")
            torch.cuda.empty_cache()

            result = model.prototype_update(batch)
            # wandb.log(result)
            epoch_loss.update(result['proto_loss'])
            epoch_acuracy.update(result['proto_acc'])

        model.scheduler.step()
        print(colored(f'[{epoch}] {epoch_loss}', 'blue'))
        print(colored(f'[{epoch}] {epoch_acuracy}', 'blue'))
        # wandb.log({"epoch_loss": epoch_loss.avg, "epoch_accuracy": epoch_acuracy.avg,  "lr":model.optimizer.state_dict()["param_groups"][0]['lr']})

        if epoch % 5 == 0:
            validate(model, val_loader)

        if epoch % 10 == 0:
            proto_chkpt = os.path.join(args.experiment_root, f"prototype_epoch_{epoch}.pth")
            model.save_prototype(proto_chkpt)     


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Prototypical Net")
    parser.add_argument('-root', '--dataset_root',
                        type=str,
                        help='path to dataset',
                        default='../Datasets')

    parser.add_argument('-exp', '--experiment_root',
                        type=str,
                        help='root where to store models, losses and accuracies',
                        default='./output')

    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20)

    parser.add_argument('-gamma', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)

    parser.add_argument('-ncTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=60',
                        default=60)

    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=5',
                        default=5)

    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=5',
                        default=5)

    parser.add_argument('-ncVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=5)

    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=5',
                        default=5)

    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=15',
                        default=15)
    
    parser.add_argument('--resume',
                        type=str,
                        help='path to checkpoint from where to resume',
                        default=None)

    parser.add_argument('--seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=0)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')
    args = parser.parse_args()
    main(args)