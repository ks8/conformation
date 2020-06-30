""" Normalizing flows class definitions. """
from typing import List, Tuple

import torch
import torch.nn as nn
# noinspection PyUnresolvedReferences
from torch.distributions.multivariate_normal import MultivariateNormal


class RealNVP(nn.Module):
    """
    Performs a single layer of the RealNVP flow.
    """

    def __init__(self, nets: nn.Sequential, nett: nn.Sequential, mask: torch.Tensor, prior: MultivariateNormal) -> None:
        """
        :param nets: "s" neural network definition.
        :param nett: "t" neural network definition.
        :param mask: Mask identifying which components of the vector will be processed together in any given layer.
        :param prior: Base distribution.
        """
        super(RealNVP, self).__init__()
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = nett
        self.s = nets

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform a sample from the base distribution or previous layer.
        :param z: Sample from the base distribution or previous layer.
        :return: Processed sample (in the direction towards the target distribution).
        """
        x = z
        x_ = x * self.mask
        s = self.s(x_) * (1 - self.mask)
        t = self.t(x_) * (1 - self.mask)
        x = x_ + (1 - self.mask) * (x * torch.exp(s) + t)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse of a target sample or a sample from the next layer.
        :param x: Sample from the target distribution or the next layer.
        :return: Inverse sample (in the direction towards the base distribution).
        """
        log_det_j, z = x.new_zeros(x.shape[0]), x
        z_ = self.mask * z
        s = self.s(z_) * (1 - self.mask)
        t = self.t(z_) * (1 - self.mask)
        z = (1 - self.mask) * (z - t) * torch.exp(-s) + z_
        return z

    def log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the logarithm of the absolute value of the determinant of the Jacobian for a sample in the forward
        direction.
        :param x: Sample.
        :return: log abs det jacobian.
        """
        log_det_j, z = x.new_zeros(x.shape[0]), x
        z_ = self.mask * z
        s = self.s(z_) * (1 - self.mask)
        log_det_j += s.sum(dim=1)
        return log_det_j


class NormalizingFlowModel(nn.Module):
    """
    Normalizing flow class. The forward function computes the cumulative log absolute value of the determinant of the
    jacobians of the transformations, and the sample function samples from the flow starting at the base distribution.
    """

    def __init__(self, base_dist: MultivariateNormal, biject: List[RealNVP]):
        """
        :param base_dist: Base distribution
        :param biject: List of flow layers
        """
        super(NormalizingFlowModel, self).__init__()
        self.biject = biject  # List of transformations, each of which is nn.Module
        self.base_dist = base_dist  # Base distribution
        self.bijectors = nn.ModuleList(self.biject)
        self.log_det = []

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute the inverse of a target distribution sample as well as the log abs det jacobians of the transformations
        :param x: Target sample
        :return: Inverse, log abs det jacobians
        """
        self.log_det = []  # Accumulate the log abs det jacobians of the transformations
        for b in range(len(self.bijectors) - 1, -1, -1):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(x))
            x = self.bijectors[b].inverse(x)
        return x, self.log_det

    def sample(self, sample_layers: int) -> torch.Tensor:
        """
        Produce samples by processing a sample from the base distribution through the normalizing flow.
        :param sample_layers: Number of layers to use for sampling
        :return: Sample from the approximate target distribution.
        """
        x = self.base_dist.sample()
        for b in range(sample_layers):
            x = self.bijectors[b](x)  # Process a sample through the flow
        return x
