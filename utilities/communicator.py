import time
import torch
import torch.distributed as dist
from torch._utils import (_flatten_dense_tensors, _unflatten_dense_tensors,
                          _take_tensors)


def communicate_to_all(param_list, args, minibatches):
    # print("Averaging at minibatch ", minibatches, " at rank ", args.commrank)
    t = time.perf_counter()
    communication_tensor = torch.cat(
        [t.contiguous().view(-1) for t in param_list], dim=0)
    buffer_model = torch.clone(communication_tensor).detach()
    packing_time = time.perf_counter() - t
    t = time.perf_counter()
    dist.all_reduce(communication_tensor, op=dist.ReduceOp.SUM)
    allreduce_time = time.perf_counter() - t
    t = time.perf_counter()
    communication_tensor.div_(args.commsize)
    communication_tensor.add_(buffer_model, alpha=-1)
    offset = 0
    for tensor in param_list:
        numel = tensor.numel()
        tensor.data.add_(
            communication_tensor.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    unpacking_time = time.perf_counter() - t
    return packing_time, allreduce_time, unpacking_time
