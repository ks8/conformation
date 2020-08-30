""" Class for subsampling in PyTorch data loading. """
import math
import random
from random import Random
from typing import Iterator

from torch.utils.data import Sampler

from conformation.dataset import GraphDataset


class MoleculeSampler(Sampler):
    """ Class for subsampling in PyTorch data loading. """

    def __init__(self,
                 dataset: GraphDataset,
                 fraction: float = 1.0,
                 shuffle: bool = True,
                 seed: int = 0):
        """
        :param seed: Random seed. Only needed if shuffle is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.fraction = fraction
        self.shuffle = shuffle
        self._random = Random(seed)
        self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """ Creates an iterator over indices to sample. """
        indices = self._random.sample(list(range(self.length)), math.ceil(self.fraction*self.length))
        if self.shuffle:
            self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        :return: Length of the iterator.
        """
        indices = self._random.sample(list(range(self.length)), math.ceil(self.fraction * self.length))
        return len(indices)
