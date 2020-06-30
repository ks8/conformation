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
