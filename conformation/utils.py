""" Neural network auxiliary functions. """
import torch
import torch.nn as nn
# noinspection PyUnresolvedReferences
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import List

from conformation.model import build_model
from conformation.train_args import Args


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


def density_func(z: torch.Tensor, log_jacobians: List[torch.Tensor], base_dist: MultivariateNormal) -> torch.Tensor:
    """
    Density function that computes the probability of a target space point by computing the log probability of its
    corresponding latent variable and the sum of the log abs det jacobians of the normalizing flow transformations.
    :param z: Inverse values.
    :param log_jacobians: Log abs det jacobians.
    :param base_dist: Base distribution
    :return: Model density.
    """

    return torch.exp(base_dist.log_prob(z) - sum(log_jacobians))


def loss_func_cnf(z: torch.Tensor, log_jacobians: List[torch.Tensor], means: torch.Tensor, cuda: bool = False,
                  covariance_factor: float = 1.0) -> \
        torch.Tensor:
    """
    Loss function that computes the mean log probability of a training example by computing the log probability of its
    corresponding latent variable and the sum of the log abs det jacobians of the normalizing flow transformations.
    :param z: Inverse values.
    :param log_jacobians: Log abs det jacobians.
    :param means: Base distribution means.
    :param cuda: Whether or not to use GPU.
    :param covariance_factor: Multiplicative factor for the base distribution covariance matrix.
    :return: Average loss.
    """
    if cuda:
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    base_log_probs = torch.zeros(len(means), device=device)
    if covariance_factor == 1.0:
        for i in range(len(means)):
            base_log_probs[i] = MultivariateNormal(means[i],
                                                   scale_tril=torch.eye(means[i].shape[0], device=device)).\
                log_prob(z[i])
    else:
        for i in range(len(means)):
            base_log_probs[i] = MultivariateNormal(means[i],
                                                   covariance_matrix=covariance_factor * torch.eye(means[i].shape[0],
                                                                                                   device=device)).\
                log_prob(z[i])

    return -(base_log_probs - sum(log_jacobians)).mean()


def density_func_cnf(z: torch.Tensor, log_jacobians: List[torch.Tensor], means: torch.Tensor, cuda: bool = False,
                     covariance_factor: float = 1.0) -> \
        torch.Tensor:
    """
    Loss function that computes the mean log probability of a training example by computing the log probability of its
    corresponding latent variable and the sum of the log abs det jacobians of the normalizing flow transformations.
    :param z: Inverse values.
    :param log_jacobians: Log abs det jacobians.
    :param means: Base distribution means.
    :param cuda: Whether or not to use GPU.
    :param covariance_factor: Multiplicative factor for the base distribution covariance matrix.
    :return: Average loss.
    """
    if cuda:
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    base_log_probs = torch.zeros(len(means), device=device)
    if covariance_factor == 1.0:
        for i in range(len(means)):
            base_log_probs[i] = MultivariateNormal(means[i],
                                                   scale_tril=torch.eye(means[i].shape[0], device=device)).\
                log_prob(z[i])
    else:
        for i in range(len(means)):
            base_log_probs[i] = MultivariateNormal(means[i],
                                                   covariance_matrix=covariance_factor * torch.eye(means[i].shape[0],
                                                                                                   device=device)).\
                log_prob(z[i])

    return torch.exp(base_log_probs - sum(log_jacobians))


def save_checkpoint(model: nn.Module, args: Args, path: str) -> None:
    """
    Saves a model checkpoint.
    :param model: A PyTorch model.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args.as_dict(),
        'state_dict': model.state_dict()
    }
    torch.save(state, path)


def load_checkpoint(path: str, cuda: bool) -> nn.Module:
    """
    Loads a model checkpoint.
    :param path: Path where checkpoint is saved.
    :param cuda: Whether to move model to cuda.
    :return: The loaded model.
    """
    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args = Args().from_dict(state['args'])
    loaded_state_dict = state['state_dict']

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
