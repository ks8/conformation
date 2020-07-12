""" Neural network auxiliary functions. """
from argparse import Namespace

import torch
import torch.nn as nn
# noinspection PyUnresolvedReferences
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import List

from conformation.flows import NormalizingFlowModel
from conformation.model import build_model


def loss_func(z: torch.Tensor, log_jacobians: List[torch.Tensor], base_dist: MultivariateNormal) -> torch.Tensor:
    """
    Loss function that computes the mean log probability of a training example by computing the log probability of its
    corresponding latent variable and the sum of the log abs det jacobians of the normalizing flow transformations.
    :param z: Inverse values.
    :param log_jacobians: Log abs det jacobians.
    :param base_dist: Base distribution
    :return: Average loss.
    """

    return -(base_dist.log_prob(z) - sum(log_jacobians)).mean()


def save_checkpoint(model: NormalizingFlowModel, args: Namespace, path: str) -> None:
    """
    Saves a model checkpoint.
    :param model: A PyTorch model.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args,
        'state_dict': model.state_dict()
    }
    torch.save(state, path)


def load_checkpoint(path: str, save_dir: str, cuda: bool) -> NormalizingFlowModel:
    """
    Loads a model checkpoint.
    :param path: Path where checkpoint is saved.
    :param save_dir: Directory to save checkpoints.
    :param cuda: Whether to move model to cuda.
    :return: The loaded model.
    """
    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']

    # Update args with current args
    args.cuda = cuda

    model = build_model(args)
    model.load_state_dict(loaded_state_dict)

    return model


def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.
    :param model: An nn.Module.
    :return: The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


# loop_PyTorch
def contains_self_loops(edge_index):
    """Returns a boolean for existence of self-loops in the graph"""
    row, col = edge_index
    mask = row == col
    return mask.sum().item() > 0


def remove_self_loops(edge_index, edge_attr=None):
    """Remove self-loops from the edge_index and edge_attr attributes"""
    row, col = edge_index
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    mask = mask.expand_as(edge_index)
    edge_index = edge_index[mask].view(2, -1)

    return edge_index, edge_attr


# isolated_PyTorch
def contains_isolated_nodes(edge_index, num_nodes):
    """Check if there are any isolated nodes"""
    (row, _), _ = remove_self_loops(edge_index)
    return torch.unique(row).size(0) < num_nodes


def to_undirected(edge_index, num_nodes):
    """
    Returns an undirected (bidirectional) COO format connectivity matrix from an original matrix given by edge_index
    """

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    unique, inv = torch.unique(row*num_nodes + col, sorted=True, return_inverse=True)
    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
    perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
    index = torch.stack([row[perm], col[perm]])

    return index
