import numpy as np
import torch
# noinspection PyUnresolvedReferences
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
from flows import RealNVP
from flows import NormalizingFlowModel


def nets(input_dim, hidden_size):
    """
    RealNVP neural network definition
    :param input_dim: Data input dimension
    :param hidden_size: Neural network hidden size
    :return: nn.Sequential neural network
    """
    return nn.Sequential(nn.Linear(input_dim, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, hidden_size),
                         nn.LeakyReLU(), nn.Linear(hidden_size, input_dim), nn.Tanh())


def nett(input_dim, hidden_size):
    """
    RealNVP neural network definition
    :param input_dim: Data input dimension
    :param hidden_size: Neural network hidden size
    :return: nn.Sequential neural network
    """
    return nn.Sequential(nn.Linear(input_dim, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, hidden_size),
                         nn.LeakyReLU(), nn.Linear(hidden_size, input_dim))


def build_model(args):
    """
    Function to build a RealNVP normalizing flow
    :param args: Network parameters
    :return: nn.Module defining the normalizing flow
    """
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    base_dist = MultivariateNormal(torch.zeros(args.input_dim, device=device), torch.eye(args.input_dim, device=device))

    biject = []
    for i in range(args.num_layers):
        if i % 2 == 0:
            biject.append(RealNVP(nets(args.input_dim, args.hidden_size), nett(args.input_dim, args.hidden_size),
                                  torch.from_numpy(np.array([j < int(args.input_dim/2) for j in
                                                             range(args.input_dim)]).astype(np.float32)), base_dist))
        else:
            biject.append(RealNVP(nets(args.input_dim, args.hidden_size), nett(args.input_dim, args.hidden_size),
                                  torch.from_numpy(np.array([j >= int(args.input_dim/2) for j in
                                                             range(args.input_dim)]).astype(np.float32)), base_dist))

    return NormalizingFlowModel(base_dist, biject)
