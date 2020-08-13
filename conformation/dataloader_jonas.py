""" PyTorch dataloader for graph-structured data. """
from typing import List, Tuple

import torch.utils.data


def from_data_list(data_list: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batching function for improved relational architecture.
    :param data_list: List of data samples.
    :return:
    """
    v_in = None
    e_in = None
    mask = None
    target = None
    for i, data in enumerate(data_list):
        if v_in is None:
            v_in = data[0].unsqueeze(0)
        else:
            v_in = torch.cat([v_in, data[0].unsqueeze(0)])

        if e_in is None:
            e_in = data[1].unsqueeze(0)
        else:
            e_in = torch.cat([e_in, data[1].unsqueeze(0)])

        if mask is None:
            mask = data[2].unsqueeze(0)
        else:
            mask = torch.cat([mask, data[2].unsqueeze(0)])

        if target is None:
            target = data[3].unsqueeze(0)
        else:
            target = torch.cat([target, data[3].unsqueeze(0)])

    return v_in, e_in, mask, target


class DataLoader(torch.utils.data.DataLoader):
    """
    Dataloader class.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int = 1, shuffle: bool = True, **kwargs):
        """
        Initialize a Dataloader object using the custom collate_fn from Batch class.
        :param dataset: Dataset to be used.
        :param batch_size: Batch size.
        :param shuffle: Whether or not to shuffle data.
        :param kwargs: Other arguments.
        """
        super(DataLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda batch: from_data_list(batch),
            **kwargs)
