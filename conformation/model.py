""" Neural network module definitions for normalizing flows. """
import numpy as np
from typing_extensions import Literal

import torch
# noinspection PyUnresolvedReferences
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn

from conformation.flows import NormalizingFlowModel, GNFFlowModel
from conformation.flows import RealNVP, GRevNet
from conformation.relational import RelationalNetwork
from conformation.train_args import Args
from conformation.train_args_relational import Args as TrainArgsRelational


class Net(nn.Module):
    """
    RealNVP neural network definitions.
    """

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int = None, num_layers: int = 3,
                 layer_type: Literal["s", "t"] = "s", output_activation: Literal["tanh", "lrelu"] = "tanh",
                 internal_batch_norm: bool = False, skip_connection: bool = False) -> None:
        """
        RealNVP neural network definition.
        :param input_dim: Data input dimension.
        :param hidden_size: Neural network hidden size.
        :param output_dim: Output dimension.
        :param num_layers: Number of linear layers.
        :param layer_type: "s" vs "t" type layer.
        :param output_activation: "s" network output activation function.
        :param internal_batch_norm: Whether or not to apply BatchNorm to "s" and "t" networks.
        :param skip_connection: Whether or not to add a skip connection.
        """
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.output_activation = output_activation
        self.internal_batch_norm = internal_batch_norm
        self.skip_connection = skip_connection
        self.featurize = nn.Linear(self.input_dim, self.hidden_size)
        if self.internal_batch_norm:
            self.featurize_bn = nn.BatchNorm1d(self.hidden_size)
        if self.output_dim is not None:
            out = self.output_dim
        else:
            out = self.input_dim
        ffn = []
        for i in range(self.num_layers - 2):
            ffn.extend([nn.LeakyReLU(), nn.Linear(self.hidden_size, self.hidden_size)])
            if self.internal_batch_norm:
                ffn.extend([nn.BatchNorm1d(self.hidden_size)])
        ffn.extend([nn.LeakyReLU()])
        self.internal_layers = nn.Sequential(*ffn)
        self.final_internal_layer = nn.Linear(self.hidden_size, out)
        if self.layer_type == "s":
            if self.output_activation == "tanh":
                self.output_act = nn.Tanh()
            elif self.output_activation == "lrelu":
                self.output_act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param x: Input.
        :return: Output.
        """
        x_in = self.featurize(x)
        if self.internal_batch_norm:
            u = self.featurize_bn(x_in)
        else:
            u = x_in
        u = self.layers(u)
        if self.skip_connection:
            u += x_in
        u = self.final_internal_layer(u)
        if self.layer_type == "s":
            u = self.output_act(u)

        return u


def net(input_dim: int, hidden_size: int, output_dim: int = None, num_layers: int = 3,
        layer_type: Literal["s", "t"] = "s", output_activation: Literal["tanh", "lrelu"] = "tanh",
        internal_batch_norm: bool = False) -> nn.Sequential:
    """
    RealNVP neural network definition.
    :param input_dim: Data input dimension.
    :param hidden_size: Neural network hidden size.
    :param output_dim: Output dimension.
    :param num_layers: Number of linear layers.
    :param layer_type: "s" vs "t" type layer.
    :param output_activation: "s" network output activation function.
    :param internal_batch_norm: Whether or not to apply BatchNorm to "s" and "t" networks.
    :return: nn.Sequential neural network.
    """
    if output_dim is not None:
        out = output_dim
    else:
        out = input_dim

    ffn = [nn.Linear(input_dim, hidden_size)]
    if internal_batch_norm:
        ffn.extend([nn.BatchNorm1d(hidden_size)])
    for i in range(num_layers - 1):
        if i == num_layers - 2:
            ffn.extend([nn.LeakyReLU(), nn.Linear(hidden_size, out)])
        else:
            ffn.extend([nn.LeakyReLU(), nn.Linear(hidden_size, hidden_size)])
            if internal_batch_norm:
                ffn.extend([nn.BatchNorm1d(hidden_size)])
    if layer_type == "s":
        if output_activation == "tanh":
            ffn.extend([nn.Tanh()])
        elif output_activation == "lrelu":
            ffn.extend([nn.LeakyReLU()])

    return nn.Sequential(*ffn)


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
                                       args.covariance_factor * torch.eye(args.input_dim, device=device))

    biject = []
    for i in range(args.num_layers):
        mask = torch.from_numpy(np.array([j < int(args.input_dim / 2) if i % 2 == 0 else j >= int(args.input_dim / 2)
                                          for j in range(args.input_dim)]).astype(np.float32))

        # Form the network layers
        if args.conditional_concat:
            nn_input_dim = int(sum(mask).item()) + args.condition_dim
            nn_output_dim = int(sum(mask).item())
        else:
            nn_input_dim = int(sum(mask).item())
            nn_output_dim = None

        # noinspection PyArgumentEqualDefault
        biject.append(RealNVP(net(nn_input_dim, args.hidden_size, nn_output_dim, args.num_internal_layers, "s",
                                  args.s_output_activation, internal_batch_norm=args.internal_batch_norm),
                              net(nn_input_dim, args.hidden_size, nn_output_dim, args.num_internal_layers, "t",
                                  internal_batch_norm=args.internal_batch_norm),
                              mask))

    return NormalizingFlowModel(biject, base_dist, args.conditional_base, args.input_dim, args.condition_dim,
                                args.base_hidden_size, args.base_output_dim, args.conditional_concat, args.padding,
                                args.covariance_factor)


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
