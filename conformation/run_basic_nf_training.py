""" Run training for NF on non-molecule data. """
import json
from logging import Logger
import os
from typing import Tuple

from tensorboardX import SummaryWriter
import torch
# noinspection PyUnresolvedReferences
from torch.optim import Adam
# noinspection PyUnresolvedReferences
from torch.utils.data import DataLoader
# noinspection PyUnresolvedReferences
from tqdm import tqdm, trange

from conformation.dataset import BasicDataset
from conformation.flows import NormalizingFlowModel
from conformation.model import build_model
from conformation.train_args import Args
from conformation.utils import save_checkpoint, load_checkpoint, param_count, loss_func, loss_func_cnf


def train(model: NormalizingFlowModel, optimizer: Adam, data: DataLoader, args: Args, logger: Logger,
          n_iter: int, summary_writer: SummaryWriter) -> Tuple[int, float]:
    """
    Function for training a normalizing flow model.
    :param model: nn.Module neural network.
    :param optimizer: PyTorch optimizer.
    :param data: DataLoader.
    :param args: System args.
    :param logger: Logger.
    :param n_iter: Number of training iterations completed so far.
    :param summary_writer: TensorboardX logging.
    :return: Total number of iterations completed.
    """

    # Set up logger
    debug, info = logger.debug, logger.info

    model.train()

    total_loss = 0.0
    loss_sum, iter_count = 0, 0
    for batch in tqdm(data, total=len(data)):
        if args.cuda:
            # noinspection PyUnresolvedReferences
            with torch.cuda.device(args.gpu_device):
                if args.conditional or args.conditional_concat:
                    batch = (batch[0].cuda(), batch[1].cuda())
                else:
                    batch = batch.cuda()
        model.zero_grad()
        if args.conditional:
            z, log_jacobians, means = model(batch[0], batch[1])
            loss = loss_func_cnf(z, log_jacobians, means, args.gpu_device)
        elif args.conditional_concat:
            z, log_jacobians = model(batch[0], batch[1])
            loss = loss_func(z, log_jacobians, model.base_dist)
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
            debug("Train loss avg = {:.4e}".format(loss_avg))
            summary_writer.add_scalar("Avg Train Loss", loss_avg, n_iter)

    return n_iter, total_loss


def run_basic_nf_training(args: Args, logger: Logger) -> None:
    """
    Perform neural network training.
    :param args: System parameters.
    :param logger: Logging.
    :return: None.
    """
    os.makedirs(os.path.join(args.save_dir, "checkpoints"))
    args.cuda = torch.cuda.is_available()

    # Set up logger
    debug, info = logger.debug, logger.info

    debug(args)

    # Load datasets
    debug('Loading data')
    metadata = json.load(open(args.data_path))
    train_data = BasicDataset(metadata, args.conditional or args.conditional_concat)

    # Dataset lengths
    train_data_length = len(train_data)
    debug('train size = {:,}'.format(train_data_length))

    # Convert to iterators
    train_data = DataLoader(train_data, args.batch_size, shuffle=True)

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
        # noinspection PyUnresolvedReferences
        with torch.cuda.device(args.gpu_device):
            debug('Moving model to cuda')
            model = model.cuda()

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    summary_writer = SummaryWriter(logdir=args.save_dir)
    best_epoch, n_iter = 0, 0
    best_loss = float('inf')
    for epoch in trange(args.num_epochs):
        n_iter, total_loss = train(model, optimizer, train_data, args, logger, n_iter, summary_writer)
        debug(f"Epoch {epoch} total loss = {total_loss:.4e}")
        summary_writer.add_scalar("Total Train Loss", total_loss, epoch)
        save_checkpoint(model, args, os.path.join(args.save_dir, "checkpoints", 'model-' + str(epoch) + '.pt'))
        if total_loss < best_loss:
            save_checkpoint(model, args, os.path.join(args.save_dir, "checkpoints", 'best.pt'))
            best_loss = total_loss
            best_epoch = epoch

    debug(f"Best epoch: {best_epoch} with total loss = {best_loss:.4e}")
