""" PyTorch dataset class for atomic pairwise distance matrix data. """
import numpy as np
from typing import List, Dict

import torch
# noinspection PyUnresolvedReferences
from torch.utils.data import Dataset


class MolDataset(Dataset):
    """
    Dataset class for loading atomic pairwise distance information for molecules.
    """

    def __init__(self, metadata: List[Dict[str, str]]):
        """
        :param metadata: Metadata JSON file.
        """
        super(Dataset, self).__init__()
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> torch.Tensor:
        distmat = np.loadtxt(self.metadata[idx]['path'])
        data = []
        num_atoms = distmat.shape[0]
        for m in range(num_atoms):
            for n in range(1, num_atoms):
                if n > m:
                    data.append(distmat[m][n])
        data = torch.from_numpy(np.array(data))
        data = data.type(torch.float32)

        return data

    def __repr__(self) -> str:
        return '{}({})'.format(self.__class__.__name__, len(self))
