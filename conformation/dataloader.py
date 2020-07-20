""" PyTorch dataloader for graph-structured data. """
import torch.utils.data
from conformation.batch import Batch


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
            collate_fn=lambda batch: Batch.from_data_list(batch),
            **kwargs)
