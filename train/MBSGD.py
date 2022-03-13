import time
import random
import os
import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.distributed as dist
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
from utilities.utils import get_current_lr
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

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for p in self.parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue
            d_p = p.grad
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


def test_train(start, args, best_acc, results, trainloader, train_sampler,
               testloader):
    print("MBSGD Training Started at Commrank ", args.commrank)
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.commsize,
                            rank=args.commrank)
    torch.set_num_threads(args.num_threads)
    net = get_model(args)
    net = net.to('cuda')
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.devicerank])

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

    sampling_epoch_counter = [0]
    epoch = 0
    minibatch_counter = [0]
    last_tested_at = [0]
    test_results = []
    train_results = []
    while epoch < args.epochs:
        sampling_epoch = sampling_epoch_counter[0]
        sampling_epoch_counter[0] += 1
        epoch = train_epoch(net, args, trainloader, optimizer, scheduler,
                            criterion, sampling_epoch, results, start,
                            minibatch_counter, train_sampler, best_acc,
                            testloader, last_tested_at, test_results,
                            train_results)
    results.append({'tag': 'testresult', 'val': test_results})
    results.append({'tag': 'trainresult', 'val': train_results})
    print("MBSGD Training Completed at Commrank ", args.commrank)


def train_epoch(net, args, trainloader, optimizer, scheduler, criterion,
                sampling_epoch, results, start, minibatch_counter,
                train_sampler, best_acc, testloader, last_tested_at,
                test_results, train_results):
    # print('\nEpoch: %d' % epoch)
    losses = LocalMetric('Loss')
    top1 = LocalMetric('Acc@1')
    if train_sampler is not None:
        train_sampler.set_epoch(sampling_epoch)
    b = bar(args.trainloaderlength, 30)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        net.train()
        optimizer.zero_grad()
        outputs = net(inputs)
        acc = accuracy(outputs, targets)
        top1.update(acc.item(), targets.size(0))
        loss = criterion(outputs, targets)
        losses.update(loss.item(), targets.size(0))
        loss.backward()

        minibatches = minibatch_counter[0]
        minibatch_counter[0] += 1
        epoch = minibatches * 1.0 / args.trainloaderlength
        scheduler.step(epoch)
        optimizer.step()
        rightnow = time.perf_counter() - start
        lr = get_current_lr(optimizer)  #
        banner_string = 'PID: {:d},Rank: {:d}|TrEp: {:.2f}|Loss: {:.4f}|Acc: {:4.3f}% ({:.0f}/{:.0f})|LR: {:.7f}'.format(
            os.getpid(), args.commrank, epoch, losses.avg, top1.avg * 100,
            top1.sum, top1.count, lr)
        b.progress_bar(batch_idx, rightnow, banner_string)
        train_results.append((minibatches, rightnow, loss.item() * len(inputs),
                              acc.item() * len(inputs), len(inputs)))
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    start = time.perf_counter()
    best_acc = mp.Value('d', 0)
    manager = mp.Manager()
    results = manager.list()
    results.append({
        'tag': 'LR',
        'ep': 0,
        'val': args.baseline_lr,
        'time': time.perf_counter() - start
    })
    (trainloader, train_sampler, _), (testloader, _) = get_dataloader(args)
    args.trainloaderlength = len(trainloader)
    args.testloaderlength = len(testloader)
    test_train(start, args, best_acc, results, trainloader, train_sampler,
               testloader)

    process_test_result(results, args)
    process_train_result(results, args)
    results_summary(results, args)

    if args.storeresults:
        result_save(results, args)

    print("Run Complete!")
