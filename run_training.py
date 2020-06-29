import argparse
from model import build_model
# noinspection PyUnresolvedReferences
from torch.optim import Adam
from train import train
import torch
from evaluate import evaluate
import json
from dataset import MolDataset
# noinspection PyUnresolvedReferences
from torch.utils.data import DataLoader
# noinspection PyUnresolvedReferences
from tqdm import trange


def run_training(args):
    """
    Perform neural network training
    :param args: System parameters
    :return: None
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

    evaluate(model, args.num_test_samples, args.num_layers)


def main():
    """
    Parse arguments and run run_training function
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest='input', default=1, help='Data folder')
    parser.add_argument('--num_epochs', type=int, dest='num_epochs', default=10, help='# training epochs')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=10, help='training batch size')
    parser.add_argument('--num_batch_iterations', type=int, dest='num_batch_iterations', default=1000)
    parser.add_argument('--lr', type=float, dest='lr', default=1e-4, help='Learning rate')
    parser.add_argument('--input_dim', type=int, dest='input_dim', default=28, help='Input dimension')
    parser.add_argument('--hidden_size', type=str, dest='hidden_size', default=256, help='Hidden size')
    parser.add_argument('--num_layers', type=int, dest='num_layers', default=6, help='# RealNVP layers')
    parser.add_argument('--log_frequency', type=int, dest='log_frequency', default=10, help='Log frequency')
    parser.add_argument('--num_test_samples', type=int, dest='num_test_samples', default=10000, help='# test samples')
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    run_training(args)


if __name__ == '__main__':
    main()
