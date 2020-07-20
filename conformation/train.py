""" Train function to execute training for one epoch. """
from logging import Logger
from typing import Tuple

# noinspection PyUnresolvedReferences
from torch.optim import Adam
# noinspection PyUnresolvedReferences
from torch.utils.data.dataloader import DataLoader
# noinspection PyUnresolvedReferences
from tqdm import tqdm

from conformation.flows import NormalizingFlowModel
from conformation.train_args import Args
from conformation.utils import loss_func


def train(model: NormalizingFlowModel, optimizer: Adam, data: DataLoader, args: Args, logger: Logger,
          n_iter: int) -> Tuple[int, float]:
    """
    Function for training a normalizing flow model.
    :param model: nn.Module neural network.
    :param optimizer: PyTorch optimizer.
    :param data: DataLoader.
    :param args: System args.
    :param logger: Logger.
    :param n_iter: Number of training iterations completed so far.
    :return: Total number of iterations completed.
    """

    # Set up logger
    debug, info = logger.debug, logger.info

    model.train()

    total_loss = 0.0
    loss_sum, iter_count = 0, 0
    for batch in tqdm(data, total=len(data)):
        if args.cuda:
            batch = batch.cuda()
        model.zero_grad()
        # noinspection PyCallingNonCallable
        z, log_jacobians = model(batch)
        loss = loss_func(z, log_jacobians, model.base_dist)
        loss_sum += loss.item()
        total_loss += loss_sum
        iter_count += len(batch)
        n_iter += len(batch)

        loss.backward()
        optimizer.step()

        if (n_iter // args.batch_size) % args.log_frequency == 0:
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0
            debug("Loss avg = {:.4e}".format(loss_avg))

    debug("Total loss = {:.4e}".format(total_loss))

    return n_iter, total_loss
