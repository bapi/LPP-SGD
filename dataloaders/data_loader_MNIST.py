"""
Create train, valid, test iterators for MNIST.
"""

import torch
from torchvision import datasets
from torchvision import transforms


def train_loader(args):
    dataset_loader = datasets.MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])
    trainset = dataset_loader(root=args.data_dir,
                              train=True,
                              download=True,
                              transform=transform)

    if args.partition:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset, num_replicas=args.commsize, rank=args.commrank)
    else:
        train_sampler = None

    trainloader = torch.utils.data.DataLoader(trainset,
                                              shuffle=(train_sampler is None),
                                              batch_size=args.train_bs,
                                              sampler=train_sampler,
                                              num_workers=args.workers,
                                              pin_memory=args.pm)

    return trainloader, train_sampler, len(trainset)


def test_loader(args):
    dataset_loader = datasets.MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])
    testset = dataset_loader(root=args.data_dir,
                             train=False,
                             download=True,
                             transform=transform)
    if args.partition:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            testset, num_replicas=args.commsize, rank=args.commrank)
    else:
        test_sampler = None

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.test_bs,
                                             num_workers=args.workers,
                                             sampler=test_sampler,
                                             pin_memory=args.pm)
    return testloader, len(testset)


def get_dataloader(args, testonly, trainonly):
    if trainonly:
        return train_loader(args)
    elif testonly:
        return test_loader(args)
    else:
        return train_loader(args), test_loader(args)