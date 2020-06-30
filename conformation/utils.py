""" Neural network auxiliary functions. """
from argparse import Namespace

import torch
# noinspection PyUnresolvedReferences
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import List

from conformation.flows import NormalizingFlowModel


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
