import argparse
from model import build_model
# noinspection PyUnresolvedReferences
from torch.optim import Adam
from train import train
import torch
from evaluate import evaluate


def run_training(args):
    """
    Perform neural network training
    :param args: System parameters
    :return: None
    """
    cuda = torch.cuda.is_available()
    model = build_model(args)
    if cuda:
        model = model.cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)



    train(model, optimizer, args.batch_size, args.num_epochs, args.num_batch_iterations, args.input)

    evaluate(model, 10000, 6)


def main():
    """
    Parse arguments and run run_training function
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, dest='input', default=1, help='Data folder')
    parser.add_argument('--num_epochs', type=int, dest='num_epochs', default=50, help='# training epochs')
    parser.add_argument('--batch_size', type=int, dest='batch_size', default=10, help='training batch size')
    parser.add_argument('--num_batch_iterations', type=int, dest='num_batch_iterations', default=1000)
    parser.add_argument('--lr', type=float, dest='lr', default=1e-4, help='Learning rate')
    parser.add_argument('--input_dim', type=str, dest='input_dim', default=28, help='Input dimension')
    parser.add_argument('--hidden_size', type=str, dest='hidden_size', default=256, help='Hidden size')
    parser.add_argument('--num_layers', type=int, dest='num_layers', default=6, help='# RealNVP layers')
    args = parser.parse_args()

    run_training(args)


if __name__ == '__main__':
    main()
