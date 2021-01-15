""" Normalizing flows class definitions. """
import math
import numpy as np
from typing import List, Tuple, Union

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
from tqdm import tqdm

from conformation.batch import Batch
from conformation.relational import RelationalNetwork


class GRevNet(nn.Module):
    """
    Performs a single (half) layer of the GRevNet flow.
    """

    def __init__(self, s: RelationalNetwork, t: RelationalNetwork, mask: int) -> None:
        """
        :param s: F1 (scaling) in GNF paper.
        :param t: F2 (translation) in GNF paper.
        :param mask: Binary variable indicating whether or not to update left half (0) or right half (1) of features.
        """
        super(GRevNet, self).__init__()
        self.s = s
        self.t = t
        self.mask = mask

    def construct_batch_halves(self, batch: Batch) -> Tuple[Batch, Batch, Batch, Batch]:
        """
        Construct Batch objects representing the first and second halves of the node/edge features
        :param batch: Graph batch
        :return:
        """
        # Construct Batch objects representing the first and second halves of the node/edge features
        batch_0 = Batch()
        batch_0.edge_index = batch.edge_index
        batch_0.y = batch.y
        batch_0.batch = batch.batch
        batch_0.edge_membership = batch.edge_membership

        batch_1 = Batch()
        batch_1.edge_index = batch.edge_index
        batch_1.y = batch.y
        batch_1.batch = batch.batch
        batch_1.edge_membership = batch.edge_membership

        # Split the node and edge features
        batch_0.x, batch_1.x = torch.split(batch.x, math.ceil(batch.x.shape[1] / 2), dim=1)
        batch_0.edge_attr, batch_1.edge_attr = torch.split(batch.edge_attr,
                                                           math.ceil(batch.edge_attr.shape[1] / 2), dim=1)

        # Update the features
        if self.mask == 0:
            input_batch = batch_1
            update_batch = batch_0
        else:
            input_batch = batch_0
            update_batch = batch_1

        return input_batch, update_batch, batch_0, batch_1

    def new_batch(self, batch_0: Batch, batch_1: Batch, update_batch: Batch) -> Batch:
        """
        Construct Batch object representing the updated node/edge features.
        :param batch_0: Left half.
        :param batch_1: Right half.
        :param update_batch: Updated batch (left or right, depending on mask value).
        :return: Concatenated batch.
        """
        if self.mask == 0:
            batch_0 = update_batch
        else:
            batch_1 = update_batch

        # Construct Batch object representing the updated node/edge features
        completed_batch = Batch()
        completed_batch.edge_index = update_batch.edge_index
        completed_batch.y = update_batch.y
        completed_batch.batch = update_batch.batch
        completed_batch.edge_membership = update_batch.edge_membership
        # noinspection PyArgumentList
        completed_batch.x = torch.cat([batch_0.x, batch_1.x], axis=1)
        # noinspection PyArgumentList
        completed_batch.edge_attr = torch.cat([batch_0.edge_attr, batch_1.edge_attr], axis=1)

        return completed_batch

    def forward(self, batch: Batch) -> Batch:
        """
        Transform a sample from the base distribution or previous layer.
        :param batch: Graph batch.
        :return: Updated batch (i.e., updated edge/node features) in the direction towards the target distribution.
        """
        input_batch, update_batch, batch_0, batch_1 = self.construct_batch_halves(batch)

        v_i_s, e_ij_s = self.s(input_batch)
        v_i_t, e_ij_t = self.t(input_batch)
        v_i_s = torch.exp(v_i_s)
        e_ij_s = torch.exp(e_ij_s)
        update_batch.x = update_batch.x * v_i_s + v_i_t
        update_batch.edge_attr = update_batch.edge_attr * e_ij_s + e_ij_t

        completed_batch = self.new_batch(batch_0, batch_1, update_batch)

        return completed_batch

    def inverse(self, batch: Batch) -> Batch:
        """
        Compute the inverse of a target sample or a sample from the next layer.
        :param batch: Graph batch.
        :return: Inverse sample (in the direction towards the base distribution).
        """
        input_batch, update_batch, batch_0, batch_1 = self.construct_batch_halves(batch)

        v_i_s, e_ij_s = self.s(input_batch)
        v_i_t, e_ij_t = self.t(input_batch)
        v_i_s = torch.exp(-v_i_s)
        e_ij_s = torch.exp(-e_ij_s)
        update_batch.x = (update_batch.x - v_i_t) * v_i_s
        update_batch.edge_attr = (update_batch.edge_attr - e_ij_t) * e_ij_s

        completed_batch = self.new_batch(batch_0, batch_1, update_batch)

        return completed_batch

    def log_abs_det_jacobian(self, batch: Batch) -> torch.Tensor:
        """
        Compute the logarithm of the absolute value of the determinant of the Jacobian for a sample in the forward
        direction.
        :param batch: Graph batch.
        :return: log abs det jacobian.
        """
        log_det_j = []
        input_batch, update_batch, batch_0, batch_1 = self.construct_batch_halves(batch)
        v_i_s, e_ij_s = self.s(input_batch)
        v_i_s = torch.exp(v_i_s)
        e_ij_s = torch.exp(e_ij_s)
        for i in range(batch.num_graphs):
            log_det_j.append(v_i_s[batch.batch == i, :].sum() + e_ij_s[batch.edge_membership == i, :].sum())

        return torch.tensor(log_det_j)


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
        self.mask = mask
        self.constant_mask = [i == 1 for i in self.mask]
        self.modified_mask = [i == 0 for i in self.mask]
        self.t = nett
        self.s = nets

    def forward(self, x: torch.Tensor, c: torch.Tensor = None, compute_log_det_j: bool = False) -> \
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Transform a sample from the base distribution or previous layer.
        :param x: Sample from the base distribution or previous layer.
        :param c: Condition vector for conditional concat flow.
        :param compute_log_det_j: Whether or not to compute log det Jacobian.
        :return: Processed sample (in the direction towards the target distribution).
        """
        constant = x[:, self.constant_mask]
        modified = x[:, self.modified_mask]
        if c is not None:
            concat = torch.cat((constant, c), 1)
            s = self.s(concat)
            t = self.t(concat)
        else:
            s = self.s(constant)
            t = self.t(constant)

        modified = modified*torch.exp(s) + t

        if self.mask[0] == 0:
            x = torch.cat((modified, constant), 1)
        else:
            x = torch.cat((constant, modified), 1)

        if compute_log_det_j:
            log_det_j = x.new_zeros(x.shape[0])
            log_det_j += s.sum(dim=1)
            return x, log_det_j
        else:
            return x

    def inverse(self, x: torch.Tensor, c: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the inverse of a target sample or a sample from the next layer.
        :param x: Sample from the target distribution or the next layer.
        :param c: Condition vector for conditional concat flow.
        :return: Inverse sample (in the direction towards the base distribution).
        """
        constant = x[:, self.constant_mask]
        modified = x[:, self.modified_mask]
        if c is not None:
            concat = torch.cat((constant, c), 1)
            s = self.s(concat)
            t = self.t(concat)
        else:
            s = self.s(constant)
            t = self.t(constant)

        modified = (modified - t)*torch.exp(-s)

        if self.mask[0] == 0:
            x = torch.cat((modified, constant), 1)
        else:
            x = torch.cat((constant, modified), 1)

        log_det_j = s.sum(dim=1)

        return x, log_det_j

    def log_abs_det_jacobian(self, x: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the logarithm of the absolute value of the determinant of the Jacobian for a sample in the forward
        direction.
        :param x: Sample.
        :param c: Condition vector for conditional concat flow.
        :return: log abs det jacobian.
        """
        log_det_j = x.new_zeros(x.shape[0])
        constant = x[:, self.constant_mask]
        if c is not None:
            s = self.s(torch.cat((constant, c), 1))
        else:
            s = self.s(constant)

        log_det_j += s.sum(dim=1)

        return log_det_j


class NormalizingFlowModel(nn.Module):
    """
    Normalizing flow class. The forward function computes the cumulative log absolute value of the determinant of the
    jacobians of the transformations, and the sample function samples from the flow starting at the base distribution.
    """

    def __init__(self, biject: List[RealNVP], base_dist: MultivariateNormal = None, conditional_base: bool = False,
                 input_dim: int = 528, condition_dim: int = 256, base_hidden_size: int = 1024, base_output_dim: int = 1,
                 conditional_concat: bool = False, padding: bool = False, covariance_factor: float = 1.0):
        """
        :param biject: List of flow layers.
        :param base_dist: Base distribution, specified for non-conditional flow.
        :param conditional_base: Whether or not the flow is conditional.
        :param input_dim: Padding dimension for input data.
        :param condition_dim: Second dimension of the condition matrix of size [input_dim, conditional_dim]
        :param base_hidden_size: Hidden size for the linear layer applied to the condition matrix for conditional flows.
        :param padding: Whether or not padding is to be used.
        :param covariance_factor: Multiplicative factor for the base distribution covariance matrix
        """
        super(NormalizingFlowModel, self).__init__()
        self.biject = biject
        self.base_dist = base_dist
        self.bijectors = nn.ModuleList(self.biject)
        self.log_det = []
        self.conditional_base = conditional_base
        self.conditional_concat = conditional_concat
        self.input_dim = input_dim
        self.covariance_factor = covariance_factor
        if self.conditional_base:
            self.condition_dim = condition_dim
            self.base_hidden_size = base_hidden_size
            self.base_output_dim = base_output_dim
            self.linear_layer = torch.nn.Linear(self.condition_dim, self.base_hidden_size)
            self.output_layer = torch.nn.Linear(self.base_hidden_size, self.base_output_dim)
            self.padding = padding

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
            if self.conditional_concat:
                x, log_det_j = self.bijectors[b].inverse(x, c)
            else:
                x, log_det_j = self.bijectors[b].inverse(x)
            self.log_det.append(log_det_j)
        if self.conditional_base:
            u = self.output_layer(self.linear_layer(c))
            return x, self.log_det, u
        else:
            return x, self.log_det

    def sample(self, num_samples: int = 1000, cuda: bool = False, condition: torch.Tensor = None) -> np.ndarray:
        """
        Produce samples by processing a sample from the base distribution through the normalizing flow.
        :param num_samples: Number of samples to generate.
        :param cuda: Whether or not to use GPU.
        :param condition: Path to condition numpy file.
        :return: Sample from the approximate target distribution.
        """
        if cuda:
            device = torch.device(0)
        else:
            device = torch.device('cpu')

        if condition is not None:
            if self.conditional_concat:
                base_dist = self.base_dist
            if self.conditional_base:
                if self.padding:
                    padding = torch.zeros([self.input_dim, self.condition_dim], device=device)
                    padding[0:condition.shape[0], :] = condition
                    condition = padding

                u = self.output_layer(self.linear_layer(condition))
                if len(u.shape) == 2:
                    u = u.squeeze(1)

                base_dist = MultivariateNormal(u, self.covariance_factor*torch.eye(u.shape[0], device=device))

        else:
            base_dist = self.base_dist

        samples = []
        for _ in tqdm(range(num_samples)):
            # noinspection PyUnboundLocalVariable
            x = base_dist.sample()
            x = x.unsqueeze(0)
            if self.conditional_concat:
                for b in range(len(self.bijectors)):
                    # noinspection PyUnboundLocalVariable
                    x = self.bijectors[b](x, condition.unsqueeze(0))  # Process a sample through the flow
            else:
                for b in range(len(self.bijectors)):
                    x = self.bijectors[b](x)  # Process a sample through the flow
            samples.append(x.cpu().numpy())
        samples = np.array(samples)
        return samples

    def forward_pass_with_log_abs_det_jacobian(self, x: torch.Tensor, condition: torch.Tensor = None) -> \
            Tuple[torch.Tensor, List]:
        """
        Process a sample from the underlying distribution, x, through the flow and return the transformed sample as well
        as the corresponding log abs det Jacobian value.
        :param x: Sample that was drawn from this flow's base distribution.
        :param condition: Path to condition numpy file.
        :return: Transformed sample and log abs det Jacobian.
        """
        self.log_det = []
        for b in range(len(self.bijectors)):
            if b == 0:
                x = self.bijectors[b](x, condition)
            else:
                x, log_det_j = self.bijectors[b](x, condition, compute_log_det_j=True)
                self.log_det.append(log_det_j)
        return x, self.log_det


class GNFFlowModel(nn.Module):
    """
    GNF class. The forward function computes the cumulative log absolute value of the determinant of the
    jacobians of the transformations, and the sample function samples from the flow starting at the base distribution.
    """

    def __init__(self, biject: List[GRevNet], base_dist: MultivariateNormal):
        """
        :param biject: List of flow layers.
        :param base_dist: Base distribution, specified for non-conditional flow.
        """
        super(GNFFlowModel, self).__init__()
        self.biject = biject
        self.base_dist = base_dist
        self.bijectors = nn.ModuleList(self.biject)
        self.log_det = []

    def forward(self, x: Batch) -> Tuple[Batch, List[torch.Tensor]]:
        """
        Compute the inverse of a target distribution sample as well as the log abs det jacobians of the transformations.
        :param x: Target sample.
        :return: Inverse, log abs det jacobians.
        """
        self.log_det = []  # Accumulate the log abs det jacobians of the transformations
        for b in range(len(self.bijectors) - 1, -1, -1):
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(x))
            x = self.bijectors[b].inverse(x)
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
