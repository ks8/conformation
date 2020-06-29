import torch
import numpy as np
# noinspection PyUnresolvedReferences
from torch.utils.data import Dataset


class MolDataset(Dataset):

    def __init__(self, metadata):
        """
        Custom dataset for pairwise distance info
        :param metadata: Metadata contents
        """
        super(Dataset, self).__init__()
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        distmat = np.loadtxt(self.metadata[idx]['path'])
        data = []
        num_atoms = distmat.shape[0]
        for m in range(num_atoms):
            for n in range(1, num_atoms):
                if n > m:
                    if m == 3 and n == 7 or m == 0 and n == 1 or m == 0 and n == 2 or m == 2 and n == 5 or m == 0 and \
                            n == 3 or m == 2 and n == 6:
                        data.append(distmat[m][n])
        data = torch.from_numpy(np.array(data))
        data = data.type(torch.float32)

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))
