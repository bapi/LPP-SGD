import json
import math
import os
import shutil
import time
import torch
import sys
import torch.distributed as dist
# from collections import defaultdict


class bar(object):
    def __init__(self, total, total_bar_length):
        self.total = total
        self.total_bar_length = total_bar_length

    def progress_bar(self, current, cur_time, msg=None):
        global last_time, begin_time
        if current == 0:
            begin_time = time.time()  # Reset for new bar.

        cur_len = int(self.total_bar_length * current / self.total)
        rest_len = int(self.total_bar_length - cur_len) - 1

        sys.stdout.write('[{:d}/{:d}'.format(current + 1, self.total))
        for i in range(cur_len):
            sys.stdout.write('-')
        sys.stdout.write('>')
        for i in range(rest_len):
            sys.stdout.write('.')
        sys.stdout.write('@{:.3f} Sec]'.format(cur_time))
        sys.stdout.write(msg)
        if current < self.total - 1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()


def sync_performance(loss, acc, mygroup=dist.group.WORLD):
    loss_acc = torch.tensor([loss, -acc]).float().cuda()
    dist.all_reduce(loss_acc, op=dist.ReduceOp.MAX, group=mygroup)
    return loss_acc[0].item(), -loss_acc[1].item()


class LocalMetric(object):
    def __init__(self, name):
        self.name = name
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count


class Metric(object):
    def __init__(self, name, rank=0):
        self.name = name
        self.sum = 0.0
        self.count = 0.0
        self.rank = rank

    def update(self, val, n=1, mygroup=dist.group.WORLD):
        val_t = torch.tensor([val * n, n]).float().cuda()
        dist.all_reduce(val_t, op=dist.ReduceOp.SUM, group=mygroup)
        self.sum += val_t[0].item()
        self.count += val_t[1].item()

    @property
    def avg(self):
        try:
            return self.sum / self.count
        except ZeroDivisionError:
            print("Rank: ", self.rank, self.count, self.sum)
            return 0


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).float().mean()


def get_current_lr(optimizer):
    if hasattr(optimizer, 'param_groups'):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    else:
        return optimizer.lr


def set_current_lr(optimizer, lr):
    if hasattr(optimizer, 'param_groups'):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        optimizer.lr = lr


def result_save(results, args):
    result_dict = {(i): r for i, r in enumerate(results)}
    with open(args.results_dir + "/results.json", 'w') as outfile:
        json.dump(result_dict, outfile)
    with open(args.results_dir + "/args.json", 'w') as outfile:
        json.dump(args.__dict__, outfile)


def assign_groups(args):
    args.commsize = len(args.gpus)
    args.num_workers = args.commsize - 1  #commsize here is the number of GPUs
    args.workers_per_process = int(args.num_workers / args.num_processes)
    if args.workers_per_process < 1:
        print("Not enough workers! ", args.commsize, args.num_workers,
              args.workers_per_process, args.num_processes)
        exit(1)
    args.working_nodes = 1 + args.workers_per_process * args.num_processes
    args.workforce = (args.workers_per_process + 1) * args.num_processes
    group_lists = []
    groups = {}
    distranks = args.num_processes
    allocated_gpus = 1
    for i in range(args.num_processes):
        my_group = [i]
        sampling_ranks = [-1]
        gpus = [0]
        for j in range(args.workers_per_process):
            my_group.append(distranks)
            sampling_ranks.append(j)
            gpus.append(allocated_gpus)
            allocated_gpus += 1
            distranks += 1
        for r, s, g in zip(my_group, sampling_ranks, gpus):
            groups.update(
                {str(r): {
                     'g': g,
                     'm': i,
                     's': s,
                     'sd': args.seed + i
                 }})
        if my_group not in group_lists:
            group_lists.append(my_group)

    return groups, group_lists


class MultiStepLR(object):
    def __init__(self, optimizer, args, milestones_travelled=0):
        self.optimizer = optimizer
        self.milestones_travelled = milestones_travelled
        self.milestones = args.lrmilestone
        self.gamma = args.gamma
        self.target_lr = args.lr
        self.init_lr = get_current_lr(optimizer)
        self.factor = self.target_lr / self.init_lr
        self.warm_up_epochs = args.warm_up_epochs
        if self.warm_up_epochs > 0:
            assert self.factor >= 1, "The target LR should be >= baseline_lr!"

    def step(self, epoch):
        if epoch < self.warm_up_epochs:
            lr = self.target_lr * 1 / self.factor * (
                epoch * (self.factor - 1) / self.warm_up_epochs + 1)
            set_current_lr(self.optimizer, lr)
        elif epoch >= self.warm_up_epochs and epoch < self.milestones[0]:
            set_current_lr(self.optimizer, self.target_lr)
        elif self.milestones_travelled < len(
                self.milestones) and epoch >= self.milestones[
                    self.milestones_travelled]:
            print("Dampening LR at ", epoch)
            self.milestones_travelled += 1
            set_current_lr(self.optimizer,
                           get_current_lr(self.optimizer) * self.gamma)


class CosineAnnealingLR(object):
    def __init__(self, optimizer, args, eta_min=0):
        self.optimizer = optimizer
        self.T_max = args.epochs + 0.1 - args.warm_up_epochs
        self.T_i = args.warm_up_epochs
        self.eta_min = eta_min
        self.gamma = args.gamma
        self.base_lr = args.lr
        self.init_lr = args.baseline_lr
        self.factor = self.base_lr / self.init_lr
        self.warm_up_epochs = args.warm_up_epochs
        if self.warm_up_epochs > 0:
            assert self.factor >= 1, "The target LR {:.3f} should be >= baseline_lr {:.2f}!".format(
                self.base_lr, self.init_lr)

    def step(self, epoch):
        if epoch < self.warm_up_epochs:
            lr = self.base_lr * 1 / self.factor * (
                epoch * (self.factor - 1) / self.warm_up_epochs + 1)
            set_current_lr(self.optimizer, lr)
        else:
            lr = self.eta_min + (self.base_lr - self.eta_min) * (
                1 + math.cos(math.pi * (epoch - self.T_i) / self.T_max)) / 2
            set_current_lr(self.optimizer, lr)


def sync_group(isMaster,
               net,
               communicator_tensor,
               mymaster,
               mygroup=dist.group.WORLD):
    if isMaster:
        tensors = [p.data.detach() for p in list(net.parameters())]
        copy_to_communicator(communicator_tensor, tensors)

    # dist.barrier(group=mygroup)
    dist.broadcast(communicator_tensor, mymaster, group=mygroup)

    if not isMaster:
        copy_from_communicator_to_parameters(communicator_tensor,
                                             list(net.parameters()))
    if isMaster:
        tensors = [p.data.detach() for p in list(net.buffers())]
        copy_to_communicator(communicator_tensor, tensors)

    # dist.barrier(group=mygroup)
    dist.broadcast(communicator_tensor, mymaster, group=mygroup)

    if not isMaster:
        copy_from_communicator_to_parameters(communicator_tensor,
                                             list(net.buffers()))
    # del communicator_tensor
    # torch.cuda.empty_cache()


def save_model(model, rightnow, snapdir, epoch, isbest, savedict=True):
    torch.save(model.state_dict(), snapdir + "/snap" + str(epoch) + ".pkl")
    # if savedict:
    #     snap = model.state_dict()
    # else:
    #     snap = [m.data for m in model.parameters()]
    # torch.save({'m': snap, 't': (rightnow), 'ep': epoch}, snapdir + "/snap.pt")
    # if isbest:
    #     shutil.copyfile(snapdir + "/snap.pt", snapdir + "/bestsnap.pt")


def test_epoch(net,
               args,
               start,
               testloader,
               criterion,
               best_acc,
               test_results,
               epoch,
               rank=0):
    loaderlength = len(testloader)
    b = bar(loaderlength, 30)
    losses = LocalMetric('Loss')
    top1 = LocalMetric('Acc@1')
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        with torch.no_grad():
            net.eval()
            outputs = net(inputs)
        loss = criterion(outputs, targets).item()
        acc = accuracy(outputs, targets).item()
        target_batch_size = targets.size(0)
        losses.update(loss, target_batch_size)
        top1.update(acc, target_batch_size)
        rightnow = time.perf_counter() - start
        banner_string = 'PID: {:d},CommRank: {:d},Rank: {:d}| TstEp: {:.2f} | Loss: {:.3f} | Acc: {:.3f}% ({:.0f}/{:.0f})'.format(
            os.getpid(), args.commrank, rank, epoch, losses.avg,
            top1.avg * 100, top1.sum, top1.count)
        b.progress_bar(batch_idx, rightnow, banner_string)
        test_results.append(
            (epoch, batch_idx, rightnow, loss, acc, target_batch_size))


    if args.storeresults and epoch >= args.pre_post_epochs:
        with best_acc.get_lock():
            ba = best_acc.value
            save_model(net, rightnow, args.snap_dir, epoch,
                       top1.avg * 100 > ba)
            if top1.avg > ba:
                best_acc.value = top1.avg


def process_test_result(results, args):
    test_results = []
    for r in results:
        if 'testresult' in r['tag']:
            test_results += r['val']
    len_test_results = len(test_results) * 1.0
    maxelements = torch.tensor([len_test_results]).cuda()
    dist.all_reduce(maxelements, op=dist.ReduceOp.MAX)
    assert len_test_results == maxelements[0].item(), "The number of \
        validation minibatches are not equal across the clusters"

    test_results.sort(key=lambda tup: (tup[0], tup[1]))
    # print(args.commrank, test_results)
    # return
    num_test_epochs = int(max(test_results)[0])
    for i in range(num_test_epochs):
        losses = Metric('Loss', args.commrank)
        top1 = Metric('Acc@1', args.commrank)
        for j in range(args.testloaderlength):
            losses.update(test_results[i * args.testloaderlength + j][3],
                          test_results[i * args.testloaderlength + j][5])
            top1.update(test_results[i * args.testloaderlength + j][4],
                        test_results[i * args.testloaderlength + j][5])

        print("Ep: ", i + 1, " TestLoss: ", losses.avg, " TestAcc@1: ",
              top1.avg * 100, " Time: ",
              test_results[(i + 1) * args.testloaderlength - 1][2])
        results.append({
            'tag':
            'TestLoss',
            'ep':
            i + 1,
            'val':
            losses.avg,
            'time':
            test_results[(i + 1) * args.testloaderlength - 1][2]
        })
        results.append({
            'tag':
            'TestAcc@1',
            'ep':
            i + 1,
            'val':
            top1.avg * 100,
            'time':
            test_results[(i + 1) * args.testloaderlength - 1][2]
        })
    for r in results:
        if 'testresult' in r['tag']:
            results.remove(r)
    # print("Rank: ", args.commrank, "Results: ", results)
    # print(" ")
    # print(" ")


def process_train_result(results, args):
    train_results = []
    for r in results:
        if 'trainresult' in r['tag']:
            train_results += r['val']
    len_train_results = len(train_results) * 1.0
    maxelements = torch.tensor([len_train_results]).cuda()
    dist.all_reduce(maxelements, op=dist.ReduceOp.MAX)
    assert len_train_results == maxelements[0].item(), "The number of \
        validation minibatches are not equal across the clusters"

    train_results.sort(key=lambda tup: (tup[0]))
    # print(args.commrank, test_results)
    # return
    for i in range(args.epochs):
        losses = Metric('Loss', args.commrank)
        top1 = Metric('Acc@1', args.commrank)
        for j in range(args.trainloaderlength):
            losses.update(
                train_results[i * args.trainloaderlength + j][2] /
                train_results[i * args.trainloaderlength + j][4],
                train_results[i * args.trainloaderlength + j][4])
            top1.update(
                train_results[i * args.trainloaderlength + j][3] /
                train_results[i * args.trainloaderlength + j][4],
                train_results[i * args.trainloaderlength + j][4])

        print("Ep: ", i + 1, " TrainLoss: ", losses.avg, " TrainAcc@1: ",
              top1.avg * 100, " Time: ",
              train_results[(i + 1) * args.trainloaderlength - 1][1])
        results.append({
            'tag':
            'TrainLoss',
            'ep':
            i + 1,
            'val':
            losses.avg,
            'time':
            train_results[(i + 1) * args.trainloaderlength - 1][1]
        })
        results.append({
            'tag':
            'TrainAcc@1',
            'ep':
            i + 1,
            'val':
            top1.avg * 100,
            'time':
            train_results[(i + 1) * args.trainloaderlength - 1][1]
        })
    for r in results:
        if 'trainresult' in r['tag']:
            results.remove(r)
    # print("Rank: ", args.commrank, "Results: ", results)
    # print(" ")
    # print(" ")
