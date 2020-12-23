""" Neural network module definitions for normalizing flows. """
import numpy as np

import torch
# noinspection PyUnresolvedReferences
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn

from conformation.flows import NormalizingFlowModel, GNFFlowModel
from conformation.flows import RealNVP, GRevNet
from conformation.relational import RelationalNetwork
from conformation.train_args import Args
from conformation.train_args_relational import Args as TrainArgsRelational


def nets(input_dim: int, hidden_size: int, output_dim: int = None) -> nn.Sequential:
    """
    RealNVP "s" neural network definition.
    :param input_dim: Data input dimension.
    :param hidden_size: Neural network hidden size.
    :param output_dim: Output dimension.
    :return: nn.Sequential neural network.
    """
    if output_dim is not None:
        out = output_dim
    else:
        out = input_dim
    return nn.Sequential(nn.Linear(input_dim, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, hidden_size),
                         nn.LeakyReLU(), nn.Linear(hidden_size, out), nn.Tanh())


def nett(input_dim: int, hidden_size: int, output_dim: int = None) -> nn.Sequential:
    """
    RealNVP "t" neural network definition.
    :param input_dim: Data input dimension.
    :param hidden_size: Neural network hidden size.
    :param output_dim: Output dimension.
    :return: nn.Sequential neural network.
    """
    if output_dim is not None:
        out = output_dim
    else:
        out = input_dim
    return nn.Sequential(nn.Linear(input_dim, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, hidden_size),
                         nn.LeakyReLU(), nn.Linear(hidden_size, out))


def build_model(args: Args) -> NormalizingFlowModel:
    """
    Function to build a RealNVP normalizing flow.
    :param args: System parameters.
    :return: nn.Module defining the normalizing flow.
    """
    if args.cuda:
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    # Define the base distribution
    if args.conditional_base:
        base_dist = None  # Base distribution will be specified inside the model
    else:
        base_dist = MultivariateNormal(torch.zeros(args.input_dim, device=device),
                                       torch.eye(args.input_dim, device=device))

    # Form the network layers
    if args.conditional_concat:
        nn_input_dim = args.input_dim + args.condition_dim
        nn_output_dim = args.input_dim
    else:
        nn_input_dim = args.input_dim
        nn_output_dim = None

    biject = []
    for i in range(args.num_layers):
        mask = torch.from_numpy(np.array([j < int(args.input_dim / 2) if i % 2 == 0 else j >= int(args.input_dim / 2)
                                          for j in range(args.input_dim)]).astype(np.float32))

        biject.append(RealNVP(nets(nn_input_dim, args.hidden_size, nn_output_dim),
                              nett(nn_input_dim, args.hidden_size, nn_output_dim),
                              mask))

    return NormalizingFlowModel(biject, base_dist, args.conditional_base, args.input_dim, args.condition_dim,
                                args.base_hidden_size, args.base_output_dim, args.conditional_concat, args.padding)


def build_gnf_model(args: TrainArgsRelational) -> GNFFlowModel:
    """
    Build a GNF normalizing flow.
    :param args: System parameters.
    :return: Module defining the normalizing flow.
    """
    if args.cuda:
        device = torch.device(args.gpu_device)
    else:
        device = torch.device('cpu')

    # Define the base distribution
    base_dist = MultivariateNormal(torch.zeros(1, device=device), torch.eye(1, device=device))

    # Form the network layers
    biject = []
    for i in range(args.num_layers):
        s = RelationalNetwork(int(args.hidden_size / 2), 1, int(args.hidden_size / 2), int(args.hidden_size / 2),
                              args.final_linear_size, args.final_output_size, gnf=True)
        t = RelationalNetwork(int(args.hidden_size / 2), 1, int(args.hidden_size / 2), int(args.hidden_size / 2),
                              args.final_linear_size, args.final_output_size, gnf=True)
        mask = i % 2
        biject.append(GRevNet(s, t, mask))

    return GNFFlowModel(biject, base_dist)
