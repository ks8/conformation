""" Run neural network training. """
import json
from logging import Logger
import os
from pprint import pformat
from typing import Tuple

import torch
# noinspection PyUnresolvedReferences
from torch.optim import Adam
# noinspection PyUnresolvedReferences
from torch.utils.data import DataLoader
# noinspection PyUnresolvedReferences
from tqdm import tqdm, trange

from conformation.dataset import MolDataset, CNFDataset
from conformation.flows import NormalizingFlowModel
from conformation.model import build_model
from conformation.train_args import Args
from conformation.utils import save_checkpoint, load_checkpoint, param_count, loss_func, loss_func_cnf


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
            with torch.cuda.device(args.gpu_device):
                if args.conditional:
                    batch = (batch[0].cuda(), batch[1].cuda(), batch[2].cuda())
                else:
                    batch = batch.cuda()
        model.zero_grad()
        if args.conditional:
            z, log_jacobians, means = model(batch[0], batch[1])
            loss = loss_func_cnf(z, log_jacobians, means, args.gpu_device)
        else:
            z, log_jacobians = model(batch)
            loss = loss_func(z, log_jacobians, model.base_dist)
        loss_sum += loss.item()
        total_loss += loss_sum
        iter_count += args.batch_size
        n_iter += args.batch_size

        loss.backward()
        optimizer.step()

        if (n_iter // args.batch_size) % args.log_frequency == 0:
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0
            debug("Loss avg = {:.4e}".format(loss_avg))

    debug("Total loss = {:.4e}".format(total_loss))

    return n_iter, total_loss


def run_training(args: Args, logger: Logger) -> None:
    """
    Perform neural network training.
    :param args: System parameters.
    :param logger: Logging.
    :return: None.
    """

    os.makedirs(os.path.join(args.save_dir, "checkpoints"))
    args.cuda = torch.cuda.is_available()

    # Set up logger
    debug, info = logger.debug, logger.info  # TODO: verbose log looks nasty via Tap - fix this

    debug(args)

    # Load datasets
    debug('Loading data')
    metadata = json.load(open(args.data_path))
    if args.conditional:
        train_data = CNFDataset(metadata, args.input_dim)
    else:
        train_data = MolDataset(metadata)

    # Dataset lengths
    train_data_length = len(train_data)
    debug('train size = {:,}'.format(train_data_length))

    # Convert to iterators
    train_data = DataLoader(train_data, args.batch_size)

    # Load/build model
    if args.checkpoint_path is not None:
        debug('Loading model from {}'.format(args.checkpoint_path))
        model = load_checkpoint(args.checkpoint_path, args.cuda, args.gpu_device)
    else:
        debug('Building model')
        model = build_model(args)

    debug(model)
    debug('Number of parameters = {:,}'.format(param_count(model)))

    if args.cuda:
        with torch.cuda.device(args.gpu_device):
            debug('Moving model to cuda')
            model = model.cuda()

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_epoch, n_iter = 0, 0
    for epoch in trange(args.num_epochs):
        n_iter, total_loss = train(model, optimizer, train_data, args, logger, n_iter)
        save_checkpoint(model, args, os.path.join(args.save_dir, "checkpoints", 'model-' + str(epoch) + '.pt'))
