import math
import os
import random
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import defaultdict
from dataloaders import get_dataloader
from models import get_model
from utilities.utils import result_save
from utilities.utils import LocalMetric, accuracy
from utilities.utils import bar
from utilities.utils import test_epoch
from utilities.utils import CosineAnnealingLR
from utilities.utils import MultiStepLR
from utilities.communicator import communicate_to_all
from utilities.results_summary import results_summary
from utilities.utils import process_test_result
from utilities.utils import process_train_result


class opt(object):
    def __init__(self,
                 parameters,
                 lr,
                 momentum,
                 weight_decay,
                 nesterov,
                 eta=0.001,
                 lars=False):
        self.parameters = parameters
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.lr = lr
        self.eta = eta
        self.lars = lars
        self.state = defaultdict(dict)

    def step(self, grad):
        for p, d_p in zip(self.parameters, grad):
            if d_p is None:
                continue
            if self.lars:
                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)

                # Compute local learning rate for this layer
                local_lr = self.eta * weight_norm / \
                    (grad_norm + self.weight_decay * weight_norm)
                actual_lr = local_lr * self.lr
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = \
                            torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(self.momentum).add_(d_p + self.weight_decay * p.data,
                                             alpha=actual_lr)
                p.data.add_(-buf)
            else:
                if self.weight_decay != 0:
                    d_p.add_(p.data, alpha=self.weight_decay)

                if self.momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(self.momentum).add_(d_p)
                    if self.nesterov:
                        d_p = d_p.add(buf, alpha=self.momentum)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-self.lr)


def test_train(net, results, start, args, epoch_counter, minibatch_counter,
               best_acc, trainloader, train_sampler, testloader):
    torch.cuda.set_device(args.devicerank)
    print("PlSGD Training Started at Commrank ", args.commrank)
    torch.set_num_threads(args.num_threads)

    criterion = nn.CrossEntropyLoss()
    optimizer = opt(list(net.parameters()),
                    lr=args.baseline_lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=args.nesterov,
                    eta=args.eta,
                    lars=args.lars)

    if args.scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, args)
    else:
        scheduler = MultiStepLR(optimizer, args)
    epoch = 0
    last_averaged_at = [0]
    last_tested_at = [0]
    test_results = []
    train_results = []
    while epoch < args.epochs:
        sampling_epoch = epoch_counter.value
        epoch_counter.value += 1
        epoch = train_epoch(net, args, trainloader, optimizer, scheduler,
                            criterion, sampling_epoch, results, start,
                            minibatch_counter, train_sampler, best_acc,
                            testloader, last_averaged_at, last_tested_at,
                            test_results, train_results)
    results.append({'tag': 'testresult', 'val': test_results})
    results.append({'tag': 'trainresult', 'val': train_results})
    print("PlSGD Training Completed at Commrank ", args.commrank)


def train_epoch(net, args, trainloader, optimizer, scheduler, criterion,
                sampling_epoch, results, start, minibatch_counter,
                train_sampler, best_acc, testloader, last_averaged_at,
                last_tested_at, test_results, train_results):
    losses = LocalMetric('Loss')
    top1 = LocalMetric('Acc@1')
    if train_sampler is not None:
        train_sampler.set_epoch(sampling_epoch)
    net.train()
    b = bar(args.trainloaderlength, 30)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.cuda:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
        gr = None
        batch_loss = 0.0
        batch_acc = 0.0
        for i in range(0, len(inputs), args.train_processing_bs):
            net.train()
            data_batch = inputs[i:i + args.train_processing_bs]
            target_batch = targets[i:i + args.train_processing_bs]
            outputs = net(data_batch)
            acc = accuracy(outputs, target_batch)
            top1.update(acc.item(), data_batch.size(0))
            loss = criterion(outputs, target_batch)
            losses.update(loss.item(), data_batch.size(0))
            batch_loss += loss.item() * data_batch.size(0)
            batch_acc += acc.item() * data_batch.size(0)
            if len(inputs) != args.train_processing_bs:
                loss.div_(
                    math.ceil(float(len(inputs)) / args.train_processing_bs))
            grad = torch.autograd.grad(loss, optimizer.parameters)
            '''Gradient Accumulation'''
            if gr is None:
                gr = grad
            else:
                for g1, g2 in zip(gr, grad):
                    g1.add_(g2)
        '''Model update with gradient.'''
        minibatches = minibatch_counter.value
        minibatch_counter.value += 1
        epoch = minibatches * 1.0 / args.trainloaderlength
        scheduler.step(epoch)
        optimizer.step(gr)

        lr = optimizer.lr
        rightnow = time.perf_counter() - start
        banner_string = 'PID: {:d},Rank: {:d}|TrEp: {:.2f}|Loss: {:.4f}|Acc: {:4.3f}% ({:.0f}/{:.0f})|LR: {:.7f}'.format(
            os.getpid(), args.commrank, epoch, losses.avg, top1.avg * 100,
            top1.sum, top1.count, lr)
        b.progress_bar(batch_idx, rightnow, banner_string)
        train_results.append(
            (minibatches, rightnow, batch_loss, batch_acc, len(inputs)))
        avg_freq = 1 if epoch < args.pre_post_epochs else args.averaging_freq
        if minibatches - last_averaged_at[0] >= avg_freq:
            communicate_to_all(list(net.parameters()), args, minibatches)
            last_averaged_at[0] = minibatches
        if minibatches - last_tested_at[
                0] >= args.test_freq * args.trainloaderlength:
            test_epoch(net, args, start, testloader, criterion, best_acc,
                       test_results, epoch)
            last_tested_at[0] = minibatches
        if epoch >= args.epochs:
            print("Terminating at epoch ", epoch, " at commrank ",
                  args.commrank)
            break

    results.append({
        'tag': 'LR',
        'ep': sampling_epoch,
        'val': lr,
        'time': rightnow
    })

    return epoch


def run(args):
    args.devicerank = args.commrank % torch.cuda.device_count()
    print("CommRank=", args.commrank, "CommSize=", args.commsize, "DeviceRank=", \
    args.devicerank,  args.dist_url, args.dist_backend)
    torch.cuda.set_device(args.devicerank)
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.commsize,
                            rank=args.commrank)
    cudnn.deterministic = True
    cudnn.benchmark = False
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    net = get_model(args)
    net = net.cuda()

    start = time.perf_counter()
    manager = mp.Manager()
    results = manager.list()
    results.append({
        'tag': 'LR',
        'ep': 0,
        'val': args.baseline_lr,
        'time': time.perf_counter() - start
    })
    epoch_counter = mp.Value('i', 0)
    minibatch_counter = mp.Value('i', 0)
    best_acc = mp.Value('d', 0)
    (trainloader, train_sampler, _), (testloader, _) = get_dataloader(args)
    args.trainloaderlength = len(trainloader)
    args.testloaderlength = len(testloader)

    test_train(net, results, start, args, epoch_counter, minibatch_counter,
               best_acc, trainloader, train_sampler, testloader)

    process_test_result(results, args)
    process_train_result(results, args)
    results_summary(results, args)

    if args.storeresults:
        result_save(results, args)
    print("Run Complete!")
