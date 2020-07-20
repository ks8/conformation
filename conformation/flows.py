""" Normalizing flows class definitions. """
import numpy as np
from typing import List, Tuple, Union

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn


class RealNVP(nn.Module):
    """
    Performs a single layer of the RealNVP flow.
    """

    def __init__(self, nets: nn.Sequential, nett: nn.Sequential, mask: torch.Tensor) -> None:
        """
        :param nets: "s" neural network definition.
        :param nett: "t" neural network definition.
        :param mask: Mask identifying which components of the vector will be processed together in any given layer.
        :return: None.
        """
        super(RealNVP, self).__init__()
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


# class CNF(nn.Module):
#     """
#     Performs a single layer of the RealNVP flow.
#     """
#
#     def __init__(self, nets: nn.Sequential, nett: nn.Sequential, mask: torch.Tensor, prior: MultivariateNormal,
#                  padding_dim: int) -> None:
#         """
#         :param nets: "s" neural network definition.
#         :param nett: "t" neural network definition.
#         :param mask: Mask identifying which components of the vector will be processed together in any given layer.
#         :param prior: Base distribution.
#         :return: None.
#         """
#         super(CNF, self).__init__()
#         self.prior = prior
#         self.mask = nn.Parameter(mask, requires_grad=False)
#         self.t = nett
#         self.s = nets
#         self.padding_dim = padding_dim
#
#     def forward(self, z: torch.Tensor, c: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
#         """
#         Transform a sample from the base distribution or previous layer.
#         :param num: Number of molecules in each atom of a batch
#         :param c: Condition tensor.
#         :param z: Sample from the base distribution or previous layer.
#         :return: Processed sample (in the direction towards the target distribution).
#         """
#         if self.mask[0] == 1.0:
#             mask = [torch.from_numpy(np.array([j < int(num[i] / 2) for j in range(num[i].item())]).astype(np.float32))
#                     for i in range(len(num))]
#         else:
#             mask = [torch.from_numpy(np.array([j >= int(num[i] / 2) for j in
#             range(num[i].item())]).astype(np.float32))
#                     for i in range(len(num))]
#
#         for i in range(len(mask)):
#             padding = np.zeros(self.padding_dim)
#             padding[:mask[i].shape[0]] = mask[i]
#             # noinspection PyTypeChecker
#             mask[i] = padding
#         mask = nn.Parameter(torch.tensor(mask, dtype=torch.float32), requires_grad=False)
#         if torch.cuda.is_available():
#             mask = mask.cuda()
#         x = z
#         x_ = x * mask
#         c_ = c * mask.unsqueeze(2).repeat(1, 1, c.shape[2])
#         # noinspection PyArgumentList
#         combine = torch.cat((c_, x_.unsqueeze(2)), axis=2)
#         combine_ = combine
#         # combine_ = combine.view(combine.shape[0], -1)
#         s = self.s(combine_).sum(dim=2) * (1 - mask)
#         t = self.t(combine_).sum(dim=2) * (1 - mask)
#         x = x_ + (1 - mask) * (x * torch.exp(s) + t)
#         return x
#
#     def inverse(self, x: torch.Tensor, c: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the inverse of a target sample or a sample from the next layer.
#         :param num: Number of atoms in each molecule of a batch.
#         :param c: Condition tensor.
#         :param x: Sample from the target distribution or the next layer.
#         :return: Inverse sample (in the direction towards the base distribution).
#         """
#         if self.mask[0] == 1.0:
#             mask = [torch.from_numpy(np.array([j < int(num[i] / 2) for j in range(num[i].item())]).astype(np.float32))
#                     for i in range(len(num))]
#         else:
#             mask = [torch.from_numpy(np.array([j >= int(num[i] / 2) for j in
#             range(num[i].item())]).astype(np.float32))
#                     for i in range(len(num))]
#
#         for i in range(len(mask)):
#             padding = np.zeros(self.padding_dim)
#             padding[:mask[i].shape[0]] = mask[i]
#             # noinspection PyTypeChecker
#             mask[i] = padding
#
#         mask = nn.Parameter(torch.tensor(mask, dtype=torch.float32), requires_grad=False)
#         if torch.cuda.is_available():
#             mask = mask.cuda()
#         log_det_j, z = x.new_zeros(x.shape[0]), x
#         z_ = mask * z
#         c_ = c * mask.unsqueeze(2).repeat(1, 1, c.shape[2])
#         # noinspection PyArgumentList
#         combine = torch.cat((c_, z_.unsqueeze(2)), axis=2)
#         combine_ = combine
#         # combine_ = combine.view(combine.shape[0], -1)
#         s = self.s(combine_).sum(dim=2) * (1 - mask)
#         t = self.t(combine_).sum(dim=2) * (1 - mask)
#         z = (1 - mask) * (z - t) * torch.exp(-s) + z_
#         return z
#
#     def log_abs_det_jacobian(self, x: torch.Tensor, c: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
#         """
#         Compute the logarithm of the absolute value of the determinant of the Jacobian for a sample in the forward
#         direction.
#         :param num: Number of atoms in each molecule in a batch
#         :param c: Condition tensor.
#         :param x: Sample.
#         :return: log abs det jacobian.
#         """
#         if self.mask[0] == 1.0:
#             mask = [torch.from_numpy(np.array([j < int(num[i] / 2) for j in range(num[i].item())]).astype(np.float32))
#                     for i in range(len(num))]
#         else:
#             mask = [torch.from_numpy(np.array([j >= int(num[i] / 2) for j in
#             range(num[i].item())]).astype(np.float32))
#                     for i in range(len(num))]
#
#         for i in range(len(mask)):
#             padding = np.zeros(self.padding_dim)
#             padding[:mask[i].shape[0]] = mask[i]
#             # noinspection PyTypeChecker
#             mask[i] = padding
#
#         mask = nn.Parameter(torch.tensor(mask, dtype=torch.float32), requires_grad=False)
#         if torch.cuda.is_available():
#             mask = mask.cuda()
#         log_det_j, z = x.new_zeros(x.shape[0]), x
#         z_ = mask * z
#         c_ = c * mask.unsqueeze(2).repeat(1, 1, c.shape[2])
#         # noinspection PyArgumentList
#         combine = torch.cat((c_, z_.unsqueeze(2)), axis=2)
#         combine_ = combine
#         # combine_ = combine.view(combine.shape[0], -1)
#         s = self.s(combine_).sum(dim=2) * (1 - mask)
#         log_det_j += s.sum(dim=1)
#         return log_det_j


class NormalizingFlowModel(nn.Module):
    """
    Normalizing flow class. The forward function computes the cumulative log absolute value of the determinant of the
    jacobians of the transformations, and the sample function samples from the flow starting at the base distribution.
    """

    def __init__(self, biject: List[RealNVP], base_dist: MultivariateNormal = None, conditional: bool = False,
                 padding_dim: int = 528, condition_dim: int = 256, hidden_size: int = 1024):
        """
        :param biject: List of flow layers.
        :param base_dist: Base distribution, specified for non-conditional flow.
        :param conditional: Whether or not the flow is conditional.
        :param padding_dim: Padding dimension for input data.
        :param condition_dim: Second dimension of the condition matrix of size [padding_dim, conditional_dim]
        :param hidden_size: Hidden size for the linear layer applied to the condition matrix for conditional flows.
        """
        super(NormalizingFlowModel, self).__init__()
        self.biject = biject
        self.base_dist = base_dist
        self.bijectors = nn.ModuleList(self.biject)
        self.log_det = []
        self.conditional = conditional
        if self.conditional:
            self.padding_dim = padding_dim
            self.condition_dim = condition_dim
            self.hidden_size = hidden_size
            self.linear_layer = torch.nn.Linear(self.condition_dim, self.hidden_size)
            self.output_layer = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> Union[Tuple[torch.Tensor, List[torch.Tensor]],
                                                                        Tuple[torch.Tensor, List[torch.Tensor],
                                                                              torch.Tensor]]:
        """
        Compute the inverse of a target distribution sample as well as the log abs det jacobians of the transformations
        and (optionally) the processed condition matrix, which is used to condition the base distribution mean vector
        and/or covariance matrix.
        :param c: Condition matrix.
        :param x: Target sample.
        :return: Inverse, log abs det jacobians, and (optionally) processed condition matrix.
        """
        self.log_det = []  # Accumulate the log abs det jacobians of the transformations
        for b in range(len(self.bijectors) - 1, -1, -1):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(x))
            x = self.bijectors[b].inverse(x)
        if self.conditional:
            u = self.output_layer(self.linear_layer(c))
            return x, self.log_det, u.squeeze(2)
        else:
            return x, self.log_det

    def sample(self, sample_layers: int, condition_path: str = None, device: torch.device = None) -> torch.Tensor:
        """
        Produce samples by processing a sample from the base distribution through the normalizing flow.
        :param sample_layers: Number of layers to use for sampling.
        :param condition_path: Path to condition numpy file.
        :param device: CPU or GPU for tensor conversion.
        :return: Sample from the approximate target distribution.
        """
        if condition_path is not None:
            condition = np.load(condition_path)
            condition = torch.from_numpy(condition)
            condition = condition.type(torch.float32)
            padding = torch.zeros([self.padding_dim, self.condition_dim], device=device)
            padding[0:condition.shape[0], :] = condition
            condition = padding

            u = self.output_layer(self.linear_layer(condition)).squeeze(1)
            base_dist = MultivariateNormal(u, torch.eye(self.padding_dim, device=device))

        else:
            base_dist = self.base_dist

        x = base_dist.sample()
        for b in range(sample_layers):
            x = self.bijectors[b](x)  # Process a sample through the flow
        return x


# class CNFFlowModel(nn.Module):
#     """
#     Normalizing flow class for conditional normalizing flows. The forward function computes the cumulative
#     log absolute
#     value of the determinant of the jacobians of the transformations, and the sample function samples from the flow
#     starting at the base distribution.
#     """
#
#     def __init__(self, biject: List[RealNVP], condition_dim: int = 256, hidden_size: int = 1024,
#                  padding_dim: int = 528):
#         """
#         :param biject: List of flow layers.
#         :param biject: List of flow layers.
#         :param biject: List of flow layers.
#         :param biject: List of flow layers.
#         """
#         super(CNFFlowModel, self).__init__()
#         self.biject = biject  # List of transformations, each of which is nn.Module
#         self.bijectors = nn.ModuleList(self.biject)
#         self.log_det = []
#         self.condition_dim = condition_dim
#         self.hidden_size = hidden_size
#         self.padding_dim = padding_dim
#         self.linear_layer = torch.nn.Linear(self.condition_dim, self.hidden_size)  # Final linear layer
#         self.output_layer = torch.nn.Linear(self.hidden_size, 1)  # Output layer
#
#     def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
#         """
#         Compute the inverse of a target distribution sample as well as the log abs det jacobians of the
#         transformations
#         :param c: Condition.
#         :param x: Target sample.
#         :return: Inverse, log abs det jacobians.
#         """
#         self.log_det = []  # Accumulate the log abs det jacobians of the transformations
#         for b in range(len(self.bijectors) - 1, -1, -1):
#             self.log_det.append(self.bijectors[b].log_abs_det_jacobian(x))
#             x = self.bijectors[b].inverse(x)
#         u = self.output_layer(self.linear_layer(c))
#         return x, self.log_det, u.squeeze(2)
#
#     def sample(self, sample_layers: int, condition_path: str) -> torch.Tensor:
#         """
#         Produce samples by processing a sample from the base distribution through the normalizing flow.
#         :param condition_path: Path to condition numpy file.
#         :param sample_layers: Number of layers to use for sampling.
#         :return: Sample from the approximate target distribution.
#         """
#         if torch.cuda.is_available():
#             device = torch.device(0)
#         else:
#             device = torch.device('cpu')
#
#         condition = np.load(condition_path)
#         condition = torch.from_numpy(condition)
#         condition = condition.type(torch.float32)
#         padding = torch.zeros([self.padding_dim, self.condition_dim], device=device)
#         padding[0:condition.shape[0], :] = condition
#         condition = padding
#
#         u = self.output_layer(self.linear_layer(condition)).squeeze(1)
#         base_dist = MultivariateNormal(u, torch.eye(self.padding_dim, device=device))
#         x = base_dist.sample()
#         for b in range(sample_layers):
#             x = self.bijectors[b](x)  # Process a sample through the flow
#         return x
