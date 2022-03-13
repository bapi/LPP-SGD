# -*- coding: utf-8 -*-
# import sys
from torchvision import models
from .shufflenet import shufflenet
from .resnet import resnet
from .smallnet import SmallNet
from .densenet import densenet121
from .densenet import densenet169
from .mobnetv2 import mobilenetv2
from .mobilenetv3 import mobilenetv3_large
from .mobilenetv3 import mobilenetv3_small
from .wideresnet import Wide_ResNet


def get_model(args):
    # Model
    print('==> Building model..')
    if args.model == 'res18':
        net = models.resnet18(num_classes=args.num_classes)
    elif args.model == 'res34':
        net = models.resnet34(num_classes=args.num_classes)
    elif args.model == 'res50':
        net = models.resnet50(num_classes=args.num_classes)
    elif args.model == 'res20':
        net = resnet(args.dataset, 20)
    elif args.model == 'res32':
        net = resnet(args.dataset, 32)
    elif args.model == 'wnet28':
        net = Wide_ResNet(28, 10, 0.3, args.num_classes)
    elif args.model == 'wnet168':
        net = Wide_ResNet(16, 8, 0.3, args.num_classes)
    elif args.model == 'wnet34':
        net = Wide_ResNet(34, 10, 0.3, args.num_classes)
    elif args.model == 'shufflenet':
        net = models.shufflenet_v2_x1_0(num_classes=args.num_classes)
    elif args.model == 'resnext50':
        from .resnext import resnext50
        net = resnext50(num_classes=args.num_classes)  # ResNet50()
    elif args.model == 'small':
        net = SmallNet()
    elif args.model == 'squuezenet':
        from .squeezenet import squeezenet
        net = squeezenet(class_num=args.num_classes)
    elif args.model == 'nasnet':
        from .nasnet import nasnet
        net = nasnet(class_num=args.num_classes)
    elif args.model == 'densenet121':
        net = densenet121(num_classes=args.num_classes)
    elif args.model == 'densenet169':
        net = densenet169(num_classes=args.num_classes)
    elif args.model == 'mobilenetv2':
        net = models.mobilenet_v2(num_classes=args.num_classes)
    elif args.model == 'mobilenetv3l':
        net = mobilenetv3_large(num_classes=args.num_classes, width_mult=0.75)
    elif args.model == 'mobilenetv3s':
        net = mobilenetv3_small(num_classes=args.num_classes, width_mult=1.)
    elif args.model == 'efficientnetb0':
        from .efficientnet import efficientnet_b0
        net = efficientnet_b0(num_classes=args.num_classes)
    elif args.model == 'efficientnetb1':
        from .efficientnet import efficientnet_b1
        net = efficientnet_b1(num_classes=args.num_classes)
    elif args.model == 'efficientnetb2':
        from .efficientnet import efficientnet_b2
        net = efficientnet_b2(num_classes=args.num_classes)
    elif args.model == 'efficientnetb3':
        from .efficientnet import efficientnet_b3
        net = efficientnet_b3(num_classes=args.num_classes)
    else:
        raise NotImplementedError
    return net
