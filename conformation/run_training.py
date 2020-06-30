""" Run neural network training."""
import argparse
from argparse import Namespace
import json
import os

import torch
# noinspection PyUnresolvedReferences
from torch.optim import Adam
# noinspection PyUnresolvedReferences
from torch.utils.data import DataLoader
# noinspection PyUnresolvedReferences
from tqdm import trange

from conformation.dataset import MolDataset
from conformation.evaluate import evaluate
from conformation.model import build_model
from conformation.train import train
from conformation.utils import save_checkpoint


def run_training(args: Namespace) -> None:
    """
    Perform neural network training.
    :param args: System parameters.
    :return: None.
    """
    model = build_model(args)
    if args.cuda:
        model = model.cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)

    metadata = json.load(open(args.input))
    train_data = MolDataset(metadata)
    train_data = DataLoader(train_data, args.batch_size)

    best_epoch, n_iter = 0, 0
    for _ in trange(args.num_epochs):
        n_iter, total_loss = train(model, optimizer, train_data, args, n_iter)
        save_checkpoint(model, args, os.path.join(args.save_dir, 'model.pt'))

    evaluate(model, args)


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
    parser.add_argument('--num_test_samples', type=int, dest='num_test_samples', default=10000, help='# test samples')
    parser.add_argument('--save_dir', type=str, dest='save_dir', default=None, help='Save directory')
    args = parser.parse_args()

    os.makedirs(args.save_dir)
    args.cuda = torch.cuda.is_available()

    run_training(args)


if __name__ == '__main__':
    main()
