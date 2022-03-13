import torch
from torchvision import datasets
from torchvision import transforms


def train_loader(args, dataset_loader, normalize):
    transform_train = transforms.Compose([
        transforms.RandomCrop((32, 32), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    trainset = dataset_loader(root=args.data_dir,
                              train=True,
                              download=True,
                              transform=transform_train)
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


def test_loader(args, dataset_loader, normalize):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    testset = dataset_loader(root=args.data_dir,
                             train=False,
                             download=True,
                             transform=transform_test)
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
    if args.dataset == 'cifar10':
        dataset_loader = datasets.CIFAR10

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
    elif args.dataset == 'cifar100':
        dataset_loader = datasets.CIFAR100
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
    if trainonly:
        return train_loader(args, dataset_loader, normalize)
    elif testonly:
        return test_loader(args, dataset_loader, normalize)
    else:
        return train_loader(args, dataset_loader,
                            normalize), test_loader(args, dataset_loader,
                                                    normalize)
