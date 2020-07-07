""" Run neural network training. """
import argparse
from argparse import Namespace
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
from tqdm import trange

from conformation.create_logger import create_logger
from conformation.dataset import MolDataset
from conformation.evaluate import evaluate
from conformation.model import build_model
from conformation.train import train
from conformation.utils import save_checkpoint, load_checkpoint, param_count


def run_training(args: Namespace, logger: Logger) -> None:
    """
    Perform neural network training.
    :param args: System parameters.
    :param logger: Logging.
    :return: None.
    """

    # Set up logger
    debug, info = logger.debug, logger.info

    debug(pformat(vars(args)))

    # Load datasets
    debug('Loading data')
    metadata = json.load(open(args.input))
    train_data = MolDataset(metadata)

    # Dataset lengths
    train_data_length = len(train_data)
    debug('train size = {:,}'.format(train_data_length))

    # Convert to iterators
    train_data = DataLoader(train_data, args.batch_size)

    # Load/build model
    if args.checkpoint_path is not None:
        debug('Loading model from {}'.format(args.checkpoint_path))
        model = load_checkpoint(args.checkpoint_path, args.save_dir, args.cuda)
    else:
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
        n_iter, total_loss = train(model, optimizer, train_data, args, logger, n_iter)
        save_checkpoint(model, args, os.path.join(args.save_dir, "checkpoints", 'model-' + str(epoch) + '.pt'))


def main():
    """
    Parse arguments and run run_training function.
    :return: None.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest='input', default=1, help='Data folder')
    parser.add_argument('--num_epochs', type=int, dest='num_epochs', default=10, help='# training epochs')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=10, help='training batch size')
    parser.add_argument('--lr', type=float, dest='lr', default=1e-4, help='Learning rate')
    parser.add_argument('--input_dim', type=int, dest='input_dim', default=28, help='Input dimension')
    parser.add_argument('--num_atoms', type=int, dest='num_atoms', default=8, help='Number of atoms')
    parser.add_argument('--hidden_size', type=str, dest='hidden_size', default=256, help='Hidden size')
    parser.add_argument('--num_layers', type=int, dest='num_layers', default=6, help='# RealNVP layers')
    parser.add_argument('--log_frequency', type=int, dest='log_frequency', default=10, help='Log frequency')
    parser.add_argument('--save_dir', type=str, dest='save_dir', default=None, help='Save directory')
    parser.add_argument('--checkpoint_path', type=str, dest='checkpoint_path',
                        default=None, help='Directory of checkpoint')
    args = parser.parse_args()

    os.makedirs(args.save_dir)
    os.makedirs(os.path.join(args.save_dir, "checkpoints"))
    args.cuda = torch.cuda.is_available()

    logger = create_logger(name='train', save_dir=args.save_dir)
    run_training(args, logger)


if __name__ == '__main__':
    main()
