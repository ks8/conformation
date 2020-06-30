""" Train function to execute training for one epoch. """
from argparse import Namespace
from typing import Tuple

# noinspection PyUnresolvedReferences
from torch.optim import Adam
# noinspection PyUnresolvedReferences
from torch.utils.data.dataloader import DataLoader
# noinspection PyUnresolvedReferences
from tqdm import tqdm

from conformation.flows import NormalizingFlowModel
from conformation.utils import loss_func


def train(model: NormalizingFlowModel, optimizer: Adam, data: DataLoader, args: Namespace, n_iter: int) -> \
        Tuple[int, float]:
    """
    Function for training a normalizing flow model.
    :param n_iter: Number of training iterations completed so far.
    :param model: nn.Module neural network.
    :param optimizer: PyTorch optimizer.
    :param data: DataLoader.
    :param args: System args.
    :return: Total number of iterations completed.
    """
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
            print("Loss avg = {:.4e}".format(loss_avg))

    print("Total loss = {:.4e}".format(total_loss))

    return n_iter, total_loss
