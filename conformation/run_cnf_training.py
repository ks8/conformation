""" Run neural network training. """
import json
from logging import Logger
import os
from pprint import pformat

import torch
# noinspection PyUnresolvedReferences
from torch.optim import Adam
# noinspection PyUnresolvedReferences
from torch.utils.data import DataLoader
# noinspection PyUnresolvedReferences
from tqdm import trange, tqdm

from conformation.dataset import CNFDataset
from conformation.model import build_model
from conformation.train_args import Args
from conformation.utils import save_checkpoint, param_count
from conformation.utils import loss_func_cnf


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
    debug, info = logger.debug, logger.info

    debug(pformat(vars(args)))

    # Load datasets
    debug('Loading data')
    metadata = json.load(open(args.data_path))
    train_data = CNFDataset(metadata, args.input_dim)

    # Dataset lengths
    train_data_length = len(train_data)
    debug('train size = {:,}'.format(train_data_length))

    # Convert to iterators
    train_data = DataLoader(train_data, args.batch_size)

    debug('Building model')
    model = build_model(args)

    debug(model)
    debug('Number of parameters = {:,}'.format(param_count(model)))

    if args.cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_epoch, n_iter = 0, 0
    for epoch in trange(args.num_epochs):

        model.train()

        total_loss = 0.0
        loss_sum, iter_count = 0, 0
        for batch in tqdm(train_data, total=len(train_data)):
            if args.cuda:
                batch = (batch[0].cuda(), batch[1].cuda(), batch[2].cuda())
            model.zero_grad()
            # noinspection PyCallingNonCallable
            z, log_jacobians, means = model(batch[0], batch[1])
            loss = loss_func_cnf(z, log_jacobians, means)
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

        save_checkpoint(model, args, os.path.join(args.save_dir, "checkpoints", 'model-' + str(epoch) + '.pt'))
