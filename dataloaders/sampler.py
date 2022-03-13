import math
import torch
from torch.utils.data import Sampler


class DistributedPercentageSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_workers (optional): Number of processes participating in
            distributed training.
        worker_rank (optional): worker_rank of the current process within num_workers.
        shuffle (optional): If true (default), sampler will shuffle the indices

    .. warning::
        In distributed mode, calling the ``set_epoch`` method is needed to
        make shuffling work; each process will use the same random seed
        otherwise.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
    """
    def __init__(self,
                 args,
                 dataset,
                 num_workers,
                 worker_rank,
                 masters_share=4,
                 shuffle=True):
        self.dataset = dataset
        self.num_workers = num_workers
        self.worker_rank = worker_rank
        self.seed = 0
        self.num_samples = int(
            math.ceil(
                len(self.dataset) * (args.masters_share * 1.0 - 1.0) /
                args.masters_share * (1.0 / args.workers_per_process)))
        self.masters_samples = int(
            math.ceil(len(self.dataset) - self.num_samples * self.num_workers))
        self.total_size = self.num_samples * self.num_workers + self.masters_samples
        self.shuffle = shuffle
        print("Commrank ", args.commrank, " MSamples ", self.masters_samples,
              " Samples ", self.num_samples, " BS ", args.bs)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.seed)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        masters_indices = indices[:self.masters_samples]
        workers_indices = indices[self.masters_samples:]
        indices = workers_indices[self.worker_rank:self.total_size:self.
                                  num_workers]
        if self.worker_rank == -1:
            assert len(masters_indices) == self.masters_samples
            return iter(masters_indices)
        else:
            assert len(indices) == self.num_samples
            return iter(indices)

    def __len__(self):
        if self.worker_rank == -1:
            return self.masters_samples
        else:
            return self.num_samples

    def set_epoch(self, seed):
        self.seed = seed
