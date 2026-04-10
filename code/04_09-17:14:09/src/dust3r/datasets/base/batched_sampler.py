import numpy as np
import torch
from accelerate import Accelerator
import torch.utils
from torch.utils.data import BatchSampler, Sampler
import torch.utils.data


class CustomRandomSampler(Sampler):
    """Random sampling under a constraint: each sample in the batch has the same feature,
    which is chosen randomly from a known pool of 'features' for each batch.

    For instance, the 'feature' could be the image aspect-ratio.

    The index returned is a tuple (sample_idx, feat_idx).
    This sampler ensures that each series of `batch_size` indices has the same `feat_idx`.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        pool_size,
        min_view_size,
        max_view_size,
        world_size,
        rank=0,
        warmup=1,
        drop_last=True,
    ):
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.min_view_size = min_view_size
        self.max_view_size = max_view_size
        self.drop_last = drop_last
        self.world_size = max(int(world_size), 1)
        self.rank = int(rank)
        if not (0 <= self.rank < self.world_size):
            raise ValueError(
                f"Invalid rank={self.rank} for world_size={self.world_size}"
            )
        self.len_dataset = N = len(dataset)

        if self.drop_last:
            self.num_samples = N // self.world_size
        else:
            self.num_samples = (N + self.world_size - 1) // self.world_size
        self.padded_total_size = self.num_samples * self.world_size
        self.total_size = self.num_samples

        self.epoch = None
        self.epochf = 0.0

    def __len__(self):
        return self.total_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.epoch is None:
            raise ValueError(
                "Epoch number not set. Please call 'set_epoch(epoch)' before iterating."
            )

        seed = self.epoch + 788
        rng = np.random.default_rng(seed=seed)
        sample_idxs = np.arange(self.len_dataset)
        rng.shuffle(sample_idxs)

        if self.drop_last:
            sample_idxs = sample_idxs[: self.padded_total_size]
        elif self.padded_total_size > self.len_dataset:
            pad_count = self.padded_total_size - self.len_dataset
            sample_idxs = np.concatenate([sample_idxs, sample_idxs[:pad_count]])

        sample_idxs = sample_idxs[self.rank : self.padded_total_size : self.world_size]

        # random feat_idxs (same across each batch)
        n_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        if self.pool_size > 1:
            p = np.ones(self.pool_size)
            p[: self.pool_size // 2] *= 2
            p = p / p.sum()
            _feat_idxs = rng.choice(self.pool_size, size=n_batches, p=p)
        else:
            _feat_idxs = rng.integers(self.pool_size, size=n_batches)
        _feat_idxs = np.broadcast_to(_feat_idxs[:, None], (n_batches, self.batch_size))
        _feat_idxs = _feat_idxs.ravel()[: self.num_samples]
        _view_idxs = rng.integers(
            self.min_view_size, self.max_view_size + 1, size=n_batches
        )
        _view_idxs = np.broadcast_to(_view_idxs[:, None], (n_batches, self.batch_size))
        _view_idxs = _view_idxs.ravel()[: self.num_samples]

        idxs = np.c_[sample_idxs, _feat_idxs, _view_idxs]
        yield from (tuple(idx) for idx in idxs)


class BatchedRandomSampler(BatchSampler):
    """Batch sampler that groups indices from RandomSampler into batches."""

    def __init__(self, sampler: CustomRandomSampler, batch_size, drop_last=True):
        self.sampler = sampler  # An instance of RandomSampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)


def round_by(total, multiple, up=False):
    if up:
        total = total + multiple - 1
    return (total // multiple) * multiple
