"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.

[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""
import os
import torch
from torchvision import datasets
from torchvision import transforms


def train_loader(args, normalize):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dir = os.path.join(args.imagenet_dir, 'train')
    trainset = datasets.ImageFolder(train_dir, transform=transform_train)

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


def test_loader(args, normalize):
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    test_dir = os.path.join(args.imagenet_dir, 'val')
    testset = datasets.ImageFolder(test_dir, transform_test)

    if args.partition:
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            testset, num_replicas=args.commsize, rank=args.commrank)
    else:
        test_sampler = None

    testloader = torch.utils.data.DataLoader(testset,
                                             shuffle=False,
                                             batch_size=args.test_bs,
                                             num_workers=args.workers,
                                             sampler=test_sampler,
                                             pin_memory=args.pm)
    return testloader, len(testset)


def get_dataloader(args, testonly, trainonly):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if trainonly:
        return train_loader(args, normalize)
    elif testonly:
        return test_loader(args, normalize)
    else:
        return train_loader(args, normalize), test_loader(args, normalize)