""" Neural network module definitions for normalizing flows. """
import numpy as np

import torch
# noinspection PyUnresolvedReferences
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn

from conformation.flows import NormalizingFlowModel
from conformation.flows import RealNVP
from conformation.train_args import Args


def nets(input_dim: int, hidden_size: int) -> nn.Sequential:
    """
    RealNVP "s" neural network definition.
    :param input_dim: Data input dimension.
    :param hidden_size: Neural network hidden size.
    :return: nn.Sequential neural network.
    """
    return nn.Sequential(nn.Linear(input_dim, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, hidden_size),
                         nn.LeakyReLU(), nn.Linear(hidden_size, input_dim), nn.Tanh())


# def nets_cnf(input_dim: int, condition_dim: int, hidden_size: int) -> nn.Sequential:
#     """
#     RealNVP "s" neural network definition.
#     :param condition_dim: Dimension of embeddings.
#     :param input_dim: Data input dimension.
#     :param hidden_size: Neural network hidden size.
#     :return: nn.Sequential neural network.
#     """
#     return nn.Sequential(nn.Linear((condition_dim + 1), hidden_size), nn.LeakyReLU(),
#                          nn.Linear(hidden_size, hidden_size),
#                          nn.LeakyReLU(), nn.Linear(hidden_size, input_dim), nn.Tanh())


def nett(input_dim: int, hidden_size: int) -> nn.Sequential:
    """
    RealNVP "t" neural network definition.
    :param input_dim: Data input dimension.
    :param hidden_size: Neural network hidden size.
    :return: nn.Sequential neural network.
    """
    return nn.Sequential(nn.Linear(input_dim, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, hidden_size),
                         nn.LeakyReLU(), nn.Linear(hidden_size, input_dim))


# def nett_cnf(input_dim: int, condition_dim: int, hidden_size: int) -> nn.Sequential:
#     """
#     RealNVP "t" neural network definition.
#     :param condition_dim: Dimension of embeddings.
#     :param input_dim: Data input dimension.
#     :param hidden_size: Neural network hidden size.
#     :return: nn.Sequential neural network.
#     """
#     return nn.Sequential(nn.Linear((condition_dim + 1), hidden_size), nn.LeakyReLU(),
#                          nn.Linear(hidden_size, hidden_size),
#                          nn.LeakyReLU(), nn.Linear(hidden_size, input_dim))


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
    base_dist = MultivariateNormal(torch.zeros(args.input_dim, device=device), torch.eye(args.input_dim, device=device))

    # Form the network layers
    biject = []
    for i in range(args.num_layers):
        if i % 2 == 0:
            biject.append(RealNVP(nets(args.input_dim, args.hidden_size), nett(args.input_dim, args.hidden_size),
                                  torch.from_numpy(np.array([j < int(args.input_dim/2) for j in
                                                             range(args.input_dim)]).astype(np.float32))))
        else:
            biject.append(RealNVP(nets(args.input_dim, args.hidden_size), nett(args.input_dim, args.hidden_size),
                                  torch.from_numpy(np.array([j >= int(args.input_dim/2) for j in
                                                             range(args.input_dim)]).astype(np.float32))))

    if args.conditional:
        return NormalizingFlowModel(biject, conditional=True)
    else:
        return NormalizingFlowModel(biject, base_dist)


# def build_model_cnf(args: Args) -> NormalizingFlowModel:
#     """
#     Function to build a RealNVP normalizing flow.
#     :param args: System parameters.
#     :return: nn.Module defining the normalizing flow.
#     """
#     # Form the network layers
#     biject = []
#     for i in range(args.num_layers):
#         if i % 2 == 0:
#             biject.append(RealNVP(nets(args.input_dim, args.hidden_size), nett(args.input_dim, args.hidden_size),
#                                   torch.from_numpy(np.array([j < int(args.input_dim/2) for j in
#                                                              range(args.input_dim)]).astype(np.float32))))
#         else:
#             biject.append(RealNVP(nets(args.input_dim, args.hidden_size), nett(args.input_dim, args.hidden_size),
#                                   torch.from_numpy(np.array([j >= int(args.input_dim/2) for j in
#                                                              range(args.input_dim)]).astype(np.float32))))
#
#     return NormalizingFlowModel(biject, conditional=True)
